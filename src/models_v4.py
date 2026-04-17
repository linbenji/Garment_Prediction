"""
models_v4.py (Cross-Attention Architecture)

MasterDrapeModel: DINOv2 (Frozen) + FiLM-Modulated MeshGraphNet + CLS Cross-Attention
Architecture:
  StyleViT_DINO      : DINOv2-Small (frozen) -> projection head -> 128-dim style
  Context Inject     : MLP -> FiLM (Feature-wise Linear Modulation) [Scale & Shift]
  CrossAttentionLayer: Per-vertex style gating at GNN layers 3 and 6
  MeshGraphNet       : encode-process-decode with FiLM residuals + cross-attention injection
  Losses             : Position MSE + Edge Strain + SMPL Collision Penalty (disabled) + Aux Cls

Changes from v3:
  - Added CrossAttentionLayer class (per-vertex gated attention over CLS style embedding)
  - Injected at GNN layers 3 and 6 (after FiLM modulation, additive residual)
  - FiLM conditioning unchanged — cross-attention is additive on top, not replacing it
  - All loss functions, constants, and other classes unchanged from teammate's v3
  - MasterDrapeModel.__init__ signature matches teammate's v3: gnn_layers is first positional arg
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

NUM_FABRIC_FAMILIES = 6

# [MASSIVE UPGRADE FROM V2]: Node input shrinks from 158 to 8
# Context (150-dim) now bypasses the graph and goes directly to the FiLM Controller.
NODE_IN_DIM     = NODE_POS_DIM + NODE_UV_DIM + NODE_NORMAL_DIM
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


# ── Cross-Attention Layer ─────────────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    """
    Per-vertex gated cross-attention over the CLS style embedding.

    FiLM broadcasts the same gamma/beta to every node uniformly.
    CrossAttentionLayer lets each node decide how much of the style
    embedding is relevant to its specific location on the mesh.

    - Hem vertices learn to gate heavily on drape/weight style signals
    - Collar vertices learn to gate low (structurally fixed regardless of fabric)
    - Sleeve ends learn to gate on stiffness/fold signals

    Uses sigmoid (not softmax) because each node attends to a single vector,
    not a sequence — softmax over one element always returns 1.0.

    Sits between selected GNN layers — additive residual on top of FiLM,
    never replacing it.
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.norm   = nn.LayerNorm(latent_dim)
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.scale  = latent_dim ** -0.5

    def forward(self, x, style_emb, batch):
        """
        x:         (total_nodes, latent_dim) — current node features
        style_emb: (B, latent_dim)           — CLS style embedding per graph
        batch:     (total_nodes,)            — node-to-graph index
        """
        x_norm = self.norm(x)
        Q      = self.q_proj(x_norm)              # (total_nodes, latent_dim)
        K      = self.k_proj(style_emb)[batch]    # (total_nodes, latent_dim)
        V      = self.v_proj(style_emb)[batch]    # (total_nodes, latent_dim)

        # Scaled dot-product — scalar gate per node
        attn = (Q * K).sum(dim=-1, keepdim=True) * self.scale  # (total_nodes, 1)
        attn = torch.sigmoid(attn)                               # gate in [0, 1]

        # Residual — if sigmoid learns to zero out, layer has no effect (safe)
        return x + attn * V


# ── Automatic Loss Weighter ───────────────────────────────────────────────────

class AutomaticLossWeighter(nn.Module):
    '''
    Find the perfect balance between the loss terms to maintain balance
    - Drape loss, edge strain loss, and classification loss are all scaled based on their values
    - Does not allow a single loss term to dominate during training
    - Can set priority values to assign priority between the loss terms
    '''
    def __init__(self, num_tasks=3, priors=None):
        super().__init__()
        # Initialize log(sigma^2) to 0.
        # Use log variance for numerical stability (prevents division by zero)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        # If no priors are provided, default to equal importance (1.0)
        if priors is None:
            priors = [1.0] * num_tasks
        # Store the priors as a buffer so they move to the GPU with the model,
        # but the optimizer knows NOT to train them
        self.register_buffer('priors', torch.tensor(priors, dtype=torch.float32))

    def forward(self, d_loss, e_loss, c_loss):
        # Gather the raw losses
        losses = [d_loss, e_loss, c_loss]
        total_loss = 0

        for i, loss in enumerate(losses):
            # Calculate precision: exp(-log(sigma^2)) = 1 / sigma^2
            precision = torch.exp(-self.log_vars[i])

            # Apply the uncertainty weighting formula
            task_loss = (precision * loss) + self.log_vars[i]
            # Multiply the dynamically balanced loss by the static priority weight
            total_loss += self.priors[i] * task_loss

        return total_loss


# ── V4 Master Architecture ────────────────────────────────────────────────────

class MasterDrapeModel(nn.Module):
    """
    V4 upgrade from v3:
    - CrossAttentionLayer injected at configurable GNN layer indices
    - Defaults to midpoint and final layer so it works correctly at any gnn_layers value
    - FiLM conditioning unchanged at all layers
    - Identical interface to teammate's v3 — gnn_layers is first positional arg

    cross_attn_layers: list of layer indices (0-indexed) where cross-attention is injected.
        Default None → automatically set to [gnn_layers // 2 - 1, gnn_layers - 1]
        Example with gnn_layers=6:  [2, 5]  (after layers 3 and 6)
        Example with gnn_layers=8:  [3, 7]  (after layers 4 and 8)
        Example with gnn_layers=10: [4, 9]  (after layers 5 and 10)
        Custom example:             [1, 4, 9] (inject at 3 points)

    One CrossAttentionLayer module is created per injection point — each learns
    independent Q/K/V projections so different layers can specialize.
    """
    def __init__(self, gnn_layers, embed_dim=STYLE_DIM, latent_dim=LATENT_DIM,
                 cross_attn_layers=None):
        super().__init__()

        self.vit = StyleViT_DINO(embed_dim=embed_dim)

        self.node_encoder = build_mlp(NODE_IN_DIM, latent_dim)
        self.edge_encoder = build_mlp(EDGE_IN_DIM, latent_dim)

        # Dedicated FiLM generator MLP for every layer — unchanged from v3
        self.film_generators = nn.ModuleList([
            nn.Linear(GLOBAL_COND_DIM, latent_dim * 2) for _ in range(gnn_layers)
        ])

        self.processor = nn.ModuleList([FiLMMeshBlock(latent_dim) for _ in range(gnn_layers)])

        # [V4 NEW] Configurable cross-attention injection points
        # Default: midpoint layer and final layer — works correctly at any gnn_layers value
        if cross_attn_layers is None:
            cross_attn_layers = [gnn_layers // 2 - 1, gnn_layers - 1]

        # Validate — all indices must be valid layer indices
        for idx in cross_attn_layers:
            assert 0 <= idx < gnn_layers, (
                f"cross_attn_layers index {idx} is out of range for gnn_layers={gnn_layers}. "
                f"Valid range: 0 to {gnn_layers - 1}."
            )

        # Store as a set for O(1) lookup in forward()
        self.cross_attn_layers = set(cross_attn_layers)

        # One independent CrossAttentionLayer per injection point
        # Stored in a dict keyed by layer index so forward() can look them up
        self.cross_attn_modules = nn.ModuleDict({
            str(idx): CrossAttentionLayer(latent_dim)
            for idx in cross_attn_layers
        })

        self.decoder = nn.Linear(latent_dim, 3)
        self.fabric_classifier = nn.Linear(embed_dim, NUM_FABRIC_FAMILIES)

        # Print injection points on init for visibility
        human_readable = [idx + 1 for idx in sorted(cross_attn_layers)]
        print(f"  CrossAttention injected after GNN layers: {human_readable} "
              f"(0-indexed: {sorted(cross_attn_layers)}) "
              f"out of {gnn_layers} total layers")

    def forward(self, data):
        # 1. Vision Forward
        style_emb = self.vit(data.image)  # (B, 128)

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

        b = data.batch  # Node-to-Batch mapping

        # 4. Process Graph with Layer-specific FiLM + Cross-Attention at configured layers
        for i, layer in enumerate(self.processor):

            # FiLM: generate per-layer scale and shift from global context
            film_params       = self.film_generators[i](global_cond)  # (B, 256)
            film_params_nodes = film_params[b]                          # (total_nodes, 256)
            gamma, beta       = torch.chunk(film_params_nodes, 2, dim=-1)

            # FiLM-modulated message passing
            x, edge_attr = layer(x, data.edge_index, edge_attr, gamma, beta)

            # [V4 NEW] Cross-attention at configured injection points
            # FiLM said the same thing to every node — cross-attention lets each
            # node decide how much of the style signal applies to its location
            if i in self.cross_attn_layers:
                x = self.cross_attn_modules[str(i)](x, style_emb, b)

        predicted_delta = self.decoder(x)
        fabric_logits   = self.fabric_classifier(style_emb)

        return predicted_delta, fabric_logits


# ── The Physics Triad Loss ───────────────────────────────────────────────────

def compute_edge_strain_loss(pred_pos, gt_pos, edge_index):
    row, col = edge_index
    pred_edge_lengths = torch.norm(pred_pos[row] - pred_pos[col], dim=1)
    gt_edge_lengths   = torch.norm(gt_pos[row]   - gt_pos[col],   dim=1)
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
    nearest_normals   = body_normals[nearest_idx]
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
    """
    Note: When using AutomaticLossWeighter, cls_weight and strain_weight are
    passed as 1.0 and dynamic scaling is handled externally.
    """

    # 1. Drape (Position MSE)
    sq_err = ((predicted_delta - target_delta) ** 2).sum(dim=-1)
    d_loss = (sq_err * loss_weight).mean()

    pred_pos = template_pos + predicted_delta
    gt_pos   = template_pos + target_delta

    # 2. Edge Strain
    e_loss = compute_edge_strain_loss(pred_pos, gt_pos, edge_index)

    # 3. Collision Penalty
    # Conditionally triggered if the dataloader provides body data
    col_loss = torch.tensor(0.0, device=d_loss.device)
    if body_pos is not None and body_normals is not None:
        col_loss = compute_collision_penalty(pred_pos, body_pos, body_normals)

    # 4. Aux Classification
    c_loss = F.cross_entropy(fabric_logits, fabric_labels)

    total = d_loss + (strain_weight * e_loss) + (collision_weight * col_loss) + (cls_weight * c_loss)

    return total, d_loss, e_loss, col_loss, c_loss
