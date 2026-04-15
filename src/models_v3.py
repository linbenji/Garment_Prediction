"""
models_v3.py (Final Master Architecture)

MasterDrapeModel: DINOv2 (Frozen) + FiLM-Modulated MeshGraphNet + Physics Triad Loss
Architecture:
  StyleViT_DINO  : DINOv2-Small (frozen) -> projection head -> 128-dim style
  Context Inject : MLP -> FiLM (Feature-wise Linear Modulation) [Scale & Shift]
  MeshGraphNet   : encode-process-decode (10 layers) with FiLM residuals
  Losses         : Position MSE + Edge Strain + SMPL Collision Penalty (currently disabled) + Aux Cls
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# ── Dimensions ───────────────────────────────────────────────────────────────

NODE_POS_DIM     = 3
NODE_UV_DIM      = 2
NODE_NORMAL_DIM  = 3

STYLE_DIM        = 128
SMPL_DIM         = 10
PHYSICS_DIM      = 10   
SIZE_DIM         = 2

EDGE_IN_DIM  = 4    
LATENT_DIM   = 128
GNN_LAYERS   = 10

NUM_FABRIC_FAMILIES = 6

# [MASSIVE UPGRADE FROM V2]: Node input shrinks from 158 to 8
# Context (150-dim) now bypasses the graph and goes directly to the FiLM Controller.
NODE_IN_DIM = NODE_POS_DIM + NODE_UV_DIM + NODE_NORMAL_DIM 
GLOBAL_COND_DIM = STYLE_DIM + SMPL_DIM + PHYSICS_DIM + SIZE_DIM 


# ── Utility Builders ─────────────────────────────────────────────────────────

def build_mlp(in_dim, out_dim, hidden_dim=256):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


# ── Vision Backbone: Frozen DINOv2 ───────────────────────────────────────────

class StyleViT_DINO(nn.Module):
    """
    Frozen Meta DINOv2-Small + Trainable Projection Head
    Solves the Sim-to-Real gap by leveraging pre-trained depth/geometry logic
    """
    def __init__(self, embed_dim=STYLE_DIM):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.projection_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim),
        )

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)   # (B, 384) CLS token
        return self.projection_head(features)  # (B, 128)


# ── FiLM Modulated Message Passing Block ─────────────────────────────────────

class FiLMMeshBlock(MessagePassing):
    """
    MeshGraphNet layer modulated by global context (Body Size, Fabric, Style)
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__(aggr='sum')
        
        self.edge_mlp = build_mlp(latent_dim * 3, latent_dim)
        self.node_mlp = build_mlp(latent_dim * 2, latent_dim)
        
        # FiLM specific normalization to zero-center features before modulation
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, edge_index, edge_attr, gamma, beta):
        """
        gamma, beta: (Total_Nodes, latent_dim) - The broadcasted tuning dials
        """
        src, dst = edge_index
        
        # 1. Edge Update
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_attr  = edge_attr + self.edge_mlp(edge_input)
        
        # 2. Node Update (Message Aggregation)
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        node_update = self.node_mlp(torch.cat([x, agg], dim=-1))
        
        # 3. Apply FiLM (Scale & Shift)
        node_update = self.norm(node_update)
        node_update = (gamma * node_update) + beta
        
        # 4. Residual Connection
        x = x + node_update
        return x, edge_attr

    def message(self, edge_attr):
        return edge_attr


# ── Final Master Architecture ────────────────────────────────────────────────

class MasterDrapeModel(nn.Module):
    def __init__(self, embed_dim=STYLE_DIM, latent_dim=LATENT_DIM, gnn_layers=GNN_LAYERS):
        super().__init__()

        self.vit = StyleViT_DINO(embed_dim=embed_dim)

        self.node_encoder = build_mlp(NODE_IN_DIM, latent_dim)
        self.edge_encoder = build_mlp(EDGE_IN_DIM, latent_dim)
        
        # We need a dedicated FiLM generator MLP for EVERY layer
        self.film_generators = nn.ModuleList([
            nn.Linear(GLOBAL_COND_DIM, latent_dim * 2) for _ in range(gnn_layers)
        ])
        
        self.processor = nn.ModuleList([FiLMMeshBlock(latent_dim) for _ in range(gnn_layers)])
        
        self.decoder = nn.Linear(latent_dim, 3)
        self.fabric_classifier = nn.Linear(embed_dim, NUM_FABRIC_FAMILIES)

    def forward(self, data):
        # 1. Vision Forward
        style_emb = self.vit(data.image) # (B, 128)

        # 2. Build Global Conditioning Vector
        # Pack Style, SMPL, Physics, and Size into one Master Context (B, 150)
        global_cond = torch.cat([
            style_emb, 
            data.tgt_smpl.view(-1, SMPL_DIM),
            data.tgt_physics.view(-1, PHYSICS_DIM),
            data.tgt_size.view(-1, SIZE_DIM)
        ], dim=-1)

        # 3. Build Node Features (Pure Geometry Now)
        x = torch.cat([data.pos, data.uvs, data.normals], dim=-1)
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(data.edge_attr) 

        b = data.batch # Node-to-Batch mapping

        # 4. Process Graph with Layer-specific FiLM
        for i, layer in enumerate(self.processor):
            # Generate the tuning dials for this specific layer
            film_params = self.film_generators[i](global_cond) # (B, 256)
            
            # Broadcast from (Batch) to (Total Nodes)
            film_params_nodes = film_params[b]                 # (Total_Nodes, 256)
            
            # Split into Scale (Gamma) and Shift (Beta)
            gamma, beta = torch.chunk(film_params_nodes, 2, dim=-1)
            
            x, edge_attr = layer(x, data.edge_index, edge_attr, gamma, beta)

        predicted_delta = self.decoder(x)             
        fabric_logits = self.fabric_classifier(style_emb)  

        return predicted_delta, fabric_logits


# ── The Physics Triad Loss ───────────────────────────────────────────────────

def compute_edge_strain_loss(pred_pos, gt_pos, edge_index):
    row, col = edge_index
    pred_edge_lengths = torch.norm(pred_pos[row] - pred_pos[col], dim=1)
    gt_edge_lengths = torch.norm(gt_pos[row] - gt_pos[col], dim=1)
    return F.mse_loss(pred_edge_lengths, gt_edge_lengths)

def compute_collision_penalty(pred_pos, body_pos, body_normals, threshold=0.002):
    """
    Calculates if shirt vertices have clipped inside the SMPL body.
    Requires body vertices and normals to be passed from the DataLoader.
    """
    # Note: A true point-to-surface distance is complex in pure PyTorch.
    # This is a highly efficient proximity approximation:
    # 1. Find distance from garment vertices to the nearest body vertex
    distances = torch.cdist(pred_pos, body_pos)
    min_dist, nearest_idx = torch.min(distances, dim=1)
    
    # 2. Check if the movement vector goes against the body surface normal
    # (meaning it pushed inside the skin rather than hovering above it)
    nearest_normals = body_normals[nearest_idx]
    direction_vectors = pred_pos - body_pos[nearest_idx]
    # Dot product < 0 means the garment vertex is behind the body normal (inside)
    inside_mask = (torch.sum(direction_vectors * nearest_normals, dim=1) < 0)
    
    # 3. Penalize vertices that are inside AND closer than the safety threshold (2mm)
    collision_error = torch.relu(threshold - min_dist[inside_mask])
    
    if collision_error.numel() == 0:
        return torch.tensor(0.0, device=pred_pos.device)
        
    return collision_error.mean()

def drape_loss(predicted_delta, target_delta, template_pos, edge_index, loss_weight,
               fabric_logits, fabric_labels, body_pos=None, body_normals=None,
               cls_weight=0.1, strain_weight=0.1, collision_weight=1.0):
    
    # 1. Drape (Position MSE)
    sq_err = ((predicted_delta - target_delta) ** 2).sum(dim=-1)
    d_loss = (sq_err * loss_weight).mean()

    pred_pos = template_pos + predicted_delta
    gt_pos = template_pos + target_delta

    # 2. Edge Strain
    e_loss = compute_edge_strain_loss(pred_pos, gt_pos, edge_index)
    
    # 3. Collision Penalty
    # We conditionally trigger this if the dataloader provides the body data
    col_loss = torch.tensor(0.0, device=d_loss.device)
    if body_pos is not None and body_normals is not None:
        col_loss = compute_collision_penalty(pred_pos, body_pos, body_normals)

    # 4. Aux Classification
    c_loss = F.cross_entropy(fabric_logits, fabric_labels)

    total = d_loss + (strain_weight * e_loss) + (collision_weight * col_loss) + (cls_weight * c_loss)

    return total, d_loss, e_loss, col_loss, c_loss