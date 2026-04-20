"""
models_v3.5.py (Hierarchical Multi-Scale Architecture)

NonbelieverDrapeModel: 
DINOv2 + FiLM-GNN + U-Net Skip Fusion + 2D UV-Refinement + Physics Septet Loss

This model moves beyond global shape prediction to capture high-frequency 
surface details (folds, wrinkles) by decoupling global drape from local 
geometric refinement.

Key Architectural Features:
-------------------------------------------------------------------------------
1. Vision Backbone (Style & Texture):
   - StyleViT_DINO: Frozen DINOv2-Small (vits14) acts as a high-level feature 
     extractor. It provides a 128-dim global 'style' embedding that encodes 
     material properties (stiffness, weight) and visual context.

2. Conditioning Engine (FiLM):
   - Feature-wise Linear Modulation: Style, SMPL betas, Physics params, and 
     Size encodings are fused into a 150-dim global context vector.
   - Per-layer FiLM Generators: Every GNN block receives a unique scale (gamma) 
     and shift (beta) based on this context, allowing the global state to 
     dynamically 'steer' the local vertex updates.

3. Core Processor (Hierarchical GNN):
   - U-Net Style Skip Connections: Captures features at multiple scales. 
     Features from early (global drape), middle, and late (local detail) 
     layers are concatenated and fused via a Hierarchical Fuser MLP.
   - Deep Message Passing: 12 FiLMMeshBlocks enable long-range signal 
     propagation across the 14,117-node mesh.

4. 2D UV-Space Refinement Head (CNN Detailer):
   - Spatial Detail Projection: GNN node features are mapped onto a 128x128 
     2D grid based on vertex UV coordinates.
   - 2D ResNet Refiner: Uses 2D convolutions to 'paint' high-frequency 
     displacement maps, capturing sharp wrinkles that GNNs naturally smooth out.
   - Grid Sampling: Refined displacements are sampled back to vertices and 
     added to the base GNN prediction.

5. Physics Septet Loss (Training Objectives):
   - Position (MVE): Standard MSE for global coordinate accuracy.
   - Asymmetric Strain: Penalizes extension (stretching) 10x more than 
     compression, forcing the fabric to buckle into realistic folds.
   - Collision Penalty: Corrected logic that uses body surface normals to 
     force fabric outside the skin, creating physical pressure.
   - Normal Consistency and Bending Energy:
     High-priority cosine similarity to ensure wrinkles face the correct physical direction.
     Dihedral angle penalty to maintain soft fabric sweeps.
   - Laplacian Smoothness: MM-space penalty to eliminate 'orange-peel' mesh 
     noise and jittery vertices.
   - Aux Classification: Forces the backbone to distinguish material families.
-------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from pykeops.torch import LazyTensor
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency

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

NUM_FABRIC_FAMILIES = 6

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
            features = self.backbone(images)
        return self.projection_head(features)


# ── FiLM Modulated Message Passing Block ─────────────────────────────────────

class FiLMMeshBlock(MessagePassing):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__(aggr='sum')
        self.edge_mlp = build_mlp(latent_dim * 3, latent_dim)
        self.node_mlp = build_mlp(latent_dim * 2, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, edge_index, edge_attr, gamma, beta):
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_attr  = edge_attr + self.edge_mlp(edge_input)
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        node_update = self.node_mlp(torch.cat([x, agg], dim=-1))
        node_update = self.norm(node_update)
        node_update = (gamma * node_update) + beta
        x = x + node_update
        return x, edge_attr

    def message(self, edge_attr):
        return edge_attr


# ── Automatic Loss Weighter ──────────────────────────────────────────────────

class AutomaticLossWeighter(nn.Module):
    """
    Dynamically balances N loss terms using learned uncertainty weighting.
    Accepts variable number of losses via *args — number must match num_tasks.
    """
    def __init__(self, num_tasks, priors=None):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        if priors is None:
            priors = [1.0] * num_tasks
        self.register_buffer('priors', torch.tensor(priors, dtype=torch.float32))

    def forward(self, *losses):
        assert len(losses) == len(self.log_vars), \
            f"Expected {len(self.log_vars)} losses, got {len(losses)}"
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            task_loss = (precision * loss) + self.log_vars[i]
            total_loss += self.priors[i] * task_loss
        return total_loss
    
# ── UV Refinement Head ───────────────────────────────────────────────────────
class UVRefinementNet(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, grid_res=LATENT_DIM):
        super().__init__()
        self.res = grid_res
        hidden_dim = latent_dim // 2
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, NODE_POS_DIM, kernel_size=1) # Fine-grained Delta X, Y, Z
        )

    def forward(self, x, uvs, batch_idx):
        # x: (N, latent_dim), uvs: (N, 2), batch_idx: (N,)
        # This is a simplified projection for v3.5
        B = batch_idx.max().item() + 1
        grid = torch.zeros((B, x.size(1), self.res, self.res), device=x.device, dtype=x.dtype)
        
        # Map [0,1] UVs to grid indices
        coords = (uvs * (self.res - 1)).long()
        for i in range(B):
            mask = (batch_idx == i)
            y_idx = coords[mask, 1]
            x_idx = coords[mask, 0]
            
            # Create a tuple of indices for the (C, H, W) grid
            # We broadcast the channel dimension to match the nodes
            C = x.size(1)
            num_nodes = mask.sum().item()
            c_idx = torch.arange(C, device=x.device).view(C, 1).expand(C, num_nodes)
            y_idx_expand = y_idx.view(1, num_nodes).expand(C, num_nodes)
            x_idx_expand = x_idx.view(1, num_nodes).expand(C, num_nodes)
            
            # Safely accumulate features without overwriting overlapping UVs
            grid[i].index_put_((c_idx.flatten(), y_idx_expand.flatten(), x_idx_expand.flatten()),
                               x[mask].t().flatten(), accumulate=True)
            
        refinement_map = self.net(grid) # (B, 3, res, res)
        
        # Sample back to mesh vertices
        fine_delta = F.grid_sample(
            refinement_map, 
            (uvs.view(B, -1, 1, NODE_UV_DIM) * 2 - 1), # Norm to [-1, 1]
            align_corners=True
        ).view(B, NODE_POS_DIM, -1).permute(0, 2, 1).reshape(-1, NODE_POS_DIM)
        
        return fine_delta


# ── Final Master Architecture ────────────────────────────────────────────────

class NonbelieverDrapeModel(nn.Module):
    def __init__(self, gnn_layers, embed_dim=STYLE_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        # 1. Vision Backbone (Frozen DINOv2)
        self.vit = StyleViT_DINO(embed_dim=embed_dim)
        # 2. Geometry Encoders
        self.node_encoder = build_mlp(NODE_IN_DIM, latent_dim)
        self.edge_encoder = build_mlp(EDGE_IN_DIM, latent_dim)
        # 3. FiLM Processor
        self.processor = nn.ModuleList([FiLMMeshBlock(latent_dim) for _ in range(gnn_layers)])
        # Per-layer FiLM parameter generators
        self.film_generators = nn.ModuleList([nn.Linear(GLOBAL_COND_DIM, latent_dim * 2) for _ in range(gnn_layers)])

        # 4. Hierarchical Skip Connections (U-Net style)
        # Concatenates features from layers 4, 8 to blend global drape with local detail
        self.hierarchical_fuser = nn.Linear(latent_dim * 3, latent_dim)
        
        # 5. 2D vision-based refinement head
        # Specifically targets high-frequency wrinkles by processing GNN features in 2D image space
        self.uv_refiner = UVRefinementNet(latent_dim)

        # 6. Decoders
        self.decoder = nn.Linear(latent_dim, 3)
        self.fabric_classifier = nn.Linear(embed_dim, NUM_FABRIC_FAMILIES)

    def forward(self, data):
        # A. Vision Feature Extraction
        style_emb = self.vit(data.image) # (B, 128)

        # B. Context Injection (FiLM)
        # Fuses Style, SMPL, Physics, and Size into a steering vector
        global_cond = torch.cat([
            style_emb,
            data.tgt_smpl.view(-1, SMPL_DIM),
            data.tgt_physics.view(-1, PHYSICS_DIM),
            data.tgt_size.view(-1, SIZE_DIM)
        ], dim=-1)

        # C. Initial Mesh Encoding
        x = torch.cat([data.pos, data.uvs, data.normals], dim=-1)
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(data.edge_attr)
        b = data.batch

        # D. Processor Loop with Hierarchical Capture
        skip_features = []
        for i, layer in enumerate(self.processor):
            # FiLM modulation
            film_params = self.film_generators[i](global_cond)
            gamma, beta = torch.chunk(film_params[b], 2, dim=-1)
            
            # Message Passing
            x, edge_attr = layer(x, data.edge_index, edge_attr, gamma, beta)

            # Save intermediate outputs for the U-Net skip fusion
            if i in [3, 7]: # Layers 4 and 8
                skip_features.append(x)

        # E. Multi-Scale Fusion
        # Fuses the final output with earlier "global" snapshots
        x = self.hierarchical_fuser(torch.cat([x] + skip_features, dim=-1))

        # F. Dual-Head Prediction
        base_delta = self.decoder(x)
        
        # G. High-frequency wrinkle detail from the CNN head
        fine_delta = self.uv_refiner(x, data.uvs, data.batch)

        # Output classification for auxiliary loss
        fabric_logits = self.fabric_classifier(style_emb)
        
        return base_delta + fine_delta, fabric_logits


# ── Loss: Edge Strain (Asymmetric) ────────────────────────────────────────────

def compute_edge_strain_loss(pred_pos, gt_pos, edge_index, w_ext=5.0, w_comp=0.5):
    row, col = edge_index
    pred_len = torch.norm(pred_pos[row] - pred_pos[col], dim=1)
    gt_len   = torch.norm(gt_pos[row]   - gt_pos[col],   dim=1)
    
    diff = pred_len - gt_len
    # Penalize extension more heavily than compression
    strain_sq = torch.where(diff > 0, w_ext * (diff**2), w_comp * (diff**2))
    return strain_sq.mean()


# ── Loss: Collision Penalty ───────────────────────────────────────────────────

def compute_collision_penalty(pred_pos, body_pos, body_normals, threshold=0.002):
    """
    Memory-free collision penalty using PyKeOps.
    Operates in O(N) VRAM instead of O(N*M).
    """
    # 1. Create LazyTensors with virtual axes
    # pred_pos becomes an (N, 1, 3) virtual tensor (Axis i)
    x_i = LazyTensor(pred_pos.view(-1, 1, 3))
    
    # body_pos becomes a (1, M, 3) virtual tensor (Axis j)
    y_j = LazyTensor(body_pos.view(1, -1, 3))
    
    # 2. Symbolic distance calculation
    # This does NOT compute the matrix yet. It just defines the math.
    D_ij = ((x_i - y_j) ** 2).sum(-1) 
    
    # 3. Kernel Execution: Find the minimum and its index
    # KeOps compiles and runs the C++ kernel here.
    min_sq_dist, nearest_idx = D_ij.min_argmin(dim=1)
    
    # KeOps returns shapes (N, 1). Flatten them to match standard PyTorch.
    min_sq_dist = min_sq_dist.view(-1)
    nearest_idx = nearest_idx.view(-1).long() # Indices must be integers
    
    # Convert squared distance back to actual distance
    min_dist = torch.sqrt(min_sq_dist)
    
    # 4. Standard PyTorch logic for penetration depth
    nearest_normals = body_normals[nearest_idx]
    direction_vectors = pred_pos - body_pos[nearest_idx]
    
    # Dot product < 0 means the garment is inside the skin
    inside_mask = (torch.sum(direction_vectors * nearest_normals, dim=1) < 0)
    
    if not inside_mask.any():
        return (pred_pos.sum() * 0.0)
    
    # Penalize deeper penetrations more heavily
    collision_error = min_dist[inside_mask] + threshold
    
    return collision_error.mean()


# ── Combined Loss Function ────────────────────────────────────────────────────

def drape_loss(predicted_delta, target_delta, template_pos, edge_index, loss_weight,
               fabric_logits, fabric_labels,
               batch_idx=None, faces=None,
               body_ids=None, get_body_data=None,
               use_normal_consistency=False, use_laplacian=False):
    """
    Calculates the raw mathematical losses. 
    Task balancing is handled externally by AutomaticLossWeighter.

    Returns: d_loss, e_loss, col_loss, n_loss, lap_loss, c_loss
    """

    # 1. Drape (Position MSE)
    sq_err = ((predicted_delta - target_delta) ** 2).sum(dim=-1)
    d_loss = (sq_err * loss_weight).mean()

    pred_pos = template_pos + predicted_delta
    gt_pos   = template_pos + target_delta

    # 2. Edge Strain
    e_loss = compute_edge_strain_loss(pred_pos, gt_pos, edge_index)

    # 3. Collision Penalty (AUTOMATED PER-SAMPLE LOOP)
    col_loss = torch.tensor(0.0, device=d_loss.device)
    if body_ids is not None and get_body_data is not None and batch_idx is not None:
        B = batch_idx.max().item() + 1
        batch_col_loss = 0.0
        for i in range(B):
            b_id = body_ids[i].item()
            b_pos, b_norm = get_body_data(b_id, d_loss.device)
            mask = (batch_idx == i)
            batch_col_loss += compute_collision_penalty(pred_pos[mask], b_pos, b_norm)
        col_loss = batch_col_loss / B

    # Initialize PyTorch3D Meshes. 
    # By passing lists of length 1, PyTorch3D treats the PyG batch as one giant mesh.
    garment_meshes = None
    if faces is not None and batch_idx is not None:
        B = batch_idx.max().item() + 1
        N = pred_pos.size(0) // B  # Number of vertices per garment (e.g., 14117)

        # Reshape from flat (B*N, 3) to batched (B, N, 3)
        pred_pos_batched = pred_pos.view(B, N, 3)
        
        # Expand template faces from (F, 3) to (B, F, 3)
        faces_batched = faces.unsqueeze(0).expand(B, -1, -1)

        garment_meshes = Meshes(verts=pred_pos_batched, faces=faces_batched)

    # 4 & 5. Normal Consistency & Bending Energy
    # PyTorch3D's mesh_normal_consistency handles both curvature and dihedral angles 
    # in a single highly optimized C++ kernel.
    n_loss = torch.tensor(0.0, device=d_loss.device)    
    if use_normal_consistency and garment_meshes is not None:
        # This replaces both the previous normal and bending functions
        n_loss = mesh_normal_consistency(garment_meshes)

    # 6. Laplacian Smoothness (toggle via flag)
    lap_loss = torch.tensor(0.0, device=d_loss.device)
    if use_laplacian and garment_meshes is not None:
        # Replaces manual node-degree gathering with a sparse matrix multiplication
        lap_loss = mesh_laplacian_smoothing(garment_meshes, method="uniform")
    
    # 7. Aux Classification
    c_loss = F.cross_entropy(fabric_logits, fabric_labels)

    return d_loss, e_loss, col_loss, n_loss, lap_loss, c_loss