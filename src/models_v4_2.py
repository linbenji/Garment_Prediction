"""
models_v4_2.py

MasterDrapeModel: DINOv2 (LoRA) + FiLM-Modulated MeshGraphNet + CLS Cross-Attention

CHANGE FROM models_v4.py: One thing only — the vision backbone.
    models_v4.py   : StyleViT_DINO      — fully frozen, torch.no_grad(), 0 trainable ViT params
    models_v4_2.py : StyleViT_DINO_LoRA — LoRA on last 4 blocks, ~67K trainable ViT params

Everything else is identical to models_v4.py:
    - Same CrossAttentionLayer (sigmoid gate, CLS vector)
    - Same FiLMMeshBlock
    - Same AutomaticLossWeighter (variadic *losses interface)
    - Same drape_loss (normal consistency + bending energy + all physics losses)
    - Same build_face_adjacency, compute_face_normals, all helpers
    - Same MasterDrapeModel forward() logic

Why LoRA on the backbone:
    Frozen DINOv2 produces a generic CLS embedding. LoRA fine-tunes only the
    Q/K/V and output projection in the last 4 transformer blocks (~67K params
    out of 22M total), letting high-level semantic layers adapt to fabric-specific
    features (stiff canvas vs soft jersey, drape stiffness, fold sharpness)
    without losing the geometric pre-training from earlier blocks.

    lora_B is initialised to zeros — at init, LoRA contribution is exactly zero
    and the model starts identical to frozen DINOv2. Adaptation grows from there.

    torch.no_grad() is removed from forward() because LoRA matrices inside the
    backbone need gradients. Frozen weights get no gradient automatically via
    requires_grad=False — PyTorch handles this correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# ── Dimensions ────────────────────────────────────────────────────────────────

NODE_POS_DIM    = 3
NODE_UV_DIM     = 2
NODE_NORMAL_DIM = 3

STYLE_DIM   = 128
SMPL_DIM    = 10
PHYSICS_DIM = 10
SIZE_DIM    = 2

EDGE_IN_DIM = 4
LATENT_DIM  = 128

NUM_FABRIC_FAMILIES = 6

NODE_IN_DIM     = NODE_POS_DIM + NODE_UV_DIM + NODE_NORMAL_DIM  # 8
GLOBAL_COND_DIM = STYLE_DIM + SMPL_DIM + PHYSICS_DIM + SIZE_DIM  # 150


# ── Utility ───────────────────────────────────────────────────────────────────

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


# ── LoRA Linear ───────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a low-rank trainable correction.

    output = W_frozen(x)  +  lora_B(lora_A(x)) * scale
             └─ frozen ─┘    └──── trainable ────┘

    Parameter count for a 384-dim attention projection:
        Full matrix : 384 × 384 = 147,456 params
        LoRA rank=4 : (384×4) + (4×384) = 3,072 params  ← 48x fewer

    Init: lora_A ~ kaiming_uniform, lora_B = zeros
    At init LoRA output is exactly zero — model starts identical to
    frozen DINOv2 and adapts from there during training.
    """
    def __init__(self, frozen_linear, rank=4, alpha=8):
        super().__init__()
        in_dim  = frozen_linear.in_features
        out_dim = frozen_linear.out_features

        self.frozen = frozen_linear
        self.frozen.weight.requires_grad = False
        if self.frozen.bias is not None:
            self.frozen.bias.requires_grad = False

        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        self.scale = alpha / rank

    def forward(self, x):
        return self.frozen(x) + self.lora_B(self.lora_A(x)) * self.scale


# ── Vision Backbone: DINOv2 + LoRA ───────────────────────────────────────────

class StyleViT_DINO_LoRA(nn.Module):
    """
    DINOv2-Small with LoRA on the last `lora_blocks` transformer blocks.
    Returns (B, 128) CLS embedding — identical interface to StyleViT_DINO.

    Blocks 0-7  (early): low-level edges/textures/geometry → keep frozen
    Blocks 8-11 (last 4): high-level semantics → inject LoRA

    No torch.no_grad() in forward — LoRA matrices need gradients.
    Frozen weights automatically receive no gradient (requires_grad=False).
    """
    def __init__(self, embed_dim=STYLE_DIM, lora_rank=4, lora_alpha=8, lora_blocks=4):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        for param in self.backbone.parameters():
            param.requires_grad = False

        n_blocks   = len(self.backbone.blocks)  # 12
        lora_start = n_blocks - lora_blocks      # 8 for last 4 blocks

        for block_idx in range(lora_start, n_blocks):
            block = self.backbone.blocks[block_idx]
            attn  = block.attn
            # qkv: fused Q/K/V projection — in=384, out=1152
            attn.qkv  = LoRALinear(attn.qkv,  rank=lora_rank, alpha=lora_alpha)
            # proj: output projection — in=384, out=384
            attn.proj = LoRALinear(attn.proj, rank=lora_rank, alpha=lora_alpha)

        self.projection_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim),
        )

        lora_params = sum(
            p.numel() for name, p in self.named_parameters()
            if p.requires_grad and 'projection_head' not in name
        )
        print(f"  LoRA on last {lora_blocks} ViT blocks (rank={lora_rank}, alpha={lora_alpha})")
        print(f"  Trainable LoRA params: {lora_params:,}")

    def forward(self, images):
        # No torch.no_grad() — LoRA matrices need gradients
        features = self.backbone(images)        # (B, 384) CLS token
        return self.projection_head(features)   # (B, 128)


# ── FiLM Mesh Block ───────────────────────────────────────────────────────────

class FiLMMeshBlock(MessagePassing):
    """Unchanged from models_v4.py."""
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


# ── Cross-Attention Layer ─────────────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    """
    Unchanged from models_v4.py.
    Per-vertex sigmoid gate over the single CLS style embedding.
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.norm   = nn.LayerNorm(latent_dim)
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.scale  = latent_dim ** -0.5

    def forward(self, x, style_emb, batch):
        x_norm = self.norm(x)
        Q = self.q_proj(x_norm)
        K = self.k_proj(style_emb)[batch]
        V = self.v_proj(style_emb)[batch]
        attn = (Q * K).sum(dim=-1, keepdim=True) * self.scale
        attn = torch.sigmoid(attn)
        return x + attn * V


# ── Automatic Loss Weighter ───────────────────────────────────────────────────

class AutomaticLossWeighter(nn.Module):
    """
    Unchanged from models_v4.py.
    Variadic *losses interface — num_tasks must match number of losses passed.
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
            precision  = torch.exp(-self.log_vars[i])
            task_loss  = (precision * loss) + self.log_vars[i]
            total_loss += self.priors[i] * task_loss
        return total_loss


# ── Master Model ──────────────────────────────────────────────────────────────

class MasterDrapeModel(nn.Module):
    """
    v4.2 — LoRA DINOv2 + FiLM + CLS CrossAttention

    Identical to models_v4.py MasterDrapeModel except:
        self.vit = StyleViT_DINO_LoRA(...)  instead of StyleViT_DINO(...)

    Adds lora_rank, lora_alpha, lora_blocks to __init__.
    forward() is completely unchanged.
    """
    def __init__(self, gnn_layers, embed_dim=STYLE_DIM, latent_dim=LATENT_DIM,
                 cross_attn_layers=None,
                 lora_rank=4, lora_alpha=8, lora_blocks=4):
        super().__init__()

        # ONLY CHANGE vs models_v4.py
        self.vit = StyleViT_DINO_LoRA(
            embed_dim=embed_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_blocks=lora_blocks,
        )

        self.node_encoder = build_mlp(NODE_IN_DIM, latent_dim)
        self.edge_encoder = build_mlp(EDGE_IN_DIM, latent_dim)

        self.film_generators = nn.ModuleList([
            nn.Linear(GLOBAL_COND_DIM, latent_dim * 2) for _ in range(gnn_layers)
        ])
        self.processor = nn.ModuleList([
            FiLMMeshBlock(latent_dim) for _ in range(gnn_layers)
        ])

        if cross_attn_layers is None:
            cross_attn_layers = [gnn_layers // 2 - 1, gnn_layers - 1]
        for idx in cross_attn_layers:
            assert 0 <= idx < gnn_layers, \
                f"cross_attn_layers index {idx} out of range for gnn_layers={gnn_layers}"

        self.cross_attn_layers  = set(cross_attn_layers)
        self.cross_attn_modules = nn.ModuleDict({
            str(idx): CrossAttentionLayer(latent_dim) for idx in cross_attn_layers
        })

        self.decoder           = nn.Linear(latent_dim, 3)
        self.fabric_classifier = nn.Linear(embed_dim, NUM_FABRIC_FAMILIES)

        human_readable = [idx + 1 for idx in sorted(cross_attn_layers)]
        print(f"  CrossAttention injected after GNN layers: {human_readable}")

    def forward(self, data):
        # Identical to models_v4.py — LoRA is inside self.vit, interface unchanged
        style_emb = self.vit(data.image)  # (B, 128)

        global_cond = torch.cat([
            style_emb,
            data.tgt_smpl.view(-1, SMPL_DIM),
            data.tgt_physics.view(-1, PHYSICS_DIM),
            data.tgt_size.view(-1, SIZE_DIM),
        ], dim=-1)

        x = torch.cat([data.pos, data.uvs, data.normals], dim=-1)
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(data.edge_attr)
        b = data.batch

        for i, layer in enumerate(self.processor):
            film_params       = self.film_generators[i](global_cond)
            film_params_nodes = film_params[b]
            gamma, beta       = torch.chunk(film_params_nodes, 2, dim=-1)
            x, edge_attr      = layer(x, data.edge_index, edge_attr, gamma, beta)

            if i in self.cross_attn_layers:
                x = self.cross_attn_modules[str(i)](x, style_emb, b)

        predicted_delta = self.decoder(x)
        fabric_logits   = self.fabric_classifier(style_emb)
        return predicted_delta, fabric_logits


# ── Face Adjacency ────────────────────────────────────────────────────────────

def build_face_adjacency(faces):
    """Unchanged from models_v4.py."""
    edge_to_face = {}
    adjacency    = []
    shared       = []

    for fi in range(len(faces)):
        for j in range(3):
            v0 = int(faces[fi][j])
            v1 = int(faces[fi][(j + 1) % 3])
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key in edge_to_face:
                adjacency.append([edge_to_face[edge_key], fi])
                shared.append([edge_key[0], edge_key[1]])
            else:
                edge_to_face[edge_key] = fi

    return (torch.tensor(adjacency, dtype=torch.long),
            torch.tensor(shared,    dtype=torch.long))


# ── Loss Functions ────────────────────────────────────────────────────────────

def compute_face_normals(verts, faces):
    """Unchanged from models_v4.py."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return F.normalize(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)


def compute_edge_strain_loss(pred_pos, gt_pos, edge_index):
    """Unchanged from models_v4.py."""
    row, col = edge_index
    pred_len = torch.norm(pred_pos[row] - pred_pos[col], dim=1)
    gt_len   = torch.norm(gt_pos[row]   - gt_pos[col],   dim=1)
    return F.mse_loss(pred_len, gt_len)


def compute_normal_consistency_loss(pred_pos, gt_pos, faces, face_adj, batch_idx):
    """Unchanged from models_v4.py."""
    B = batch_idx.max().item() + 1
    total_loss = 0.0
    f0, f1 = face_adj[:, 0], face_adj[:, 1]
    for i in range(B):
        mask = (batch_idx == i)
        pred_normals = compute_face_normals(pred_pos[mask], faces)
        gt_normals   = compute_face_normals(gt_pos[mask],   faces)
        pred_dots = (pred_normals[f0] * pred_normals[f1]).sum(dim=1)
        gt_dots   = (gt_normals[f0]   * gt_normals[f1]).sum(dim=1)
        total_loss += F.mse_loss(pred_dots, gt_dots)
    return total_loss / B


def compute_bending_energy_loss(pred_pos, gt_pos, faces, face_adj, shared_edges, batch_idx):
    """Unchanged from models_v4.py."""
    B = batch_idx.max().item() + 1
    total_loss = 0.0
    f0, f1 = face_adj[:, 0], face_adj[:, 1]
    for i in range(B):
        mask = (batch_idx == i)
        p_pos = pred_pos[mask]
        g_pos = gt_pos[mask]

        pred_normals = compute_face_normals(p_pos, faces)
        n1_p, n2_p  = pred_normals[f0], pred_normals[f1]
        e_vec_p     = p_pos[shared_edges[:, 1]] - p_pos[shared_edges[:, 0]]
        e_dir_p     = F.normalize(e_vec_p, dim=1)
        sin_p       = (torch.cross(n1_p, n2_p, dim=1) * e_dir_p).sum(dim=1)
        cos_p       = (n1_p * n2_p).sum(dim=1)
        theta_pred  = torch.atan2(sin_p, cos_p)

        gt_normals  = compute_face_normals(g_pos, faces)
        n1_g, n2_g  = gt_normals[f0], gt_normals[f1]
        e_vec_g     = g_pos[shared_edges[:, 1]] - g_pos[shared_edges[:, 0]]
        e_dir_g     = F.normalize(e_vec_g, dim=1)
        sin_g       = (torch.cross(n1_g, n2_g, dim=1) * e_dir_g).sum(dim=1)
        cos_g       = (n1_g * n2_g).sum(dim=1)
        theta_gt    = torch.atan2(sin_g, cos_g)

        total_loss += F.mse_loss(theta_pred, theta_gt)
    return total_loss / B


def compute_collision_penalty(pred_pos, body_pos, body_normals, threshold=0.002):
    """Unchanged from models_v4.py."""
    distances = torch.cdist(pred_pos, body_pos)
    min_dist, nearest_idx = torch.min(distances, dim=1)
    nearest_normals   = body_normals[nearest_idx]
    direction_vectors = pred_pos - body_pos[nearest_idx]
    inside_mask       = (torch.sum(direction_vectors * nearest_normals, dim=1) < 0)
    collision_error   = torch.relu(threshold - min_dist[inside_mask])
    if collision_error.numel() == 0:
        return torch.tensor(0.0, device=pred_pos.device)
    return collision_error.mean()


def drape_loss(predicted_delta, target_delta, template_pos, edge_index, loss_weight,
               fabric_logits, fabric_labels,
               batch_idx=None, faces=None, face_adj=None, shared_edges=None,
               body_pos=None, body_normals=None,
               use_normal_consistency=False, use_bending_energy=False,
               cls_weight=0.1, strain_weight=0.1, collision_weight=1.0,
               normal_weight=0.1, bending_weight=0.1):
    """Unchanged from models_v4.py."""

    sq_err   = ((predicted_delta - target_delta) ** 2).sum(dim=-1)
    d_loss   = (sq_err * loss_weight).mean()
    pred_pos = template_pos + predicted_delta
    gt_pos   = template_pos + target_delta

    e_loss = compute_edge_strain_loss(pred_pos, gt_pos, edge_index)

    col_loss = torch.tensor(0.0, device=d_loss.device)
    if body_pos is not None and body_normals is not None:
        col_loss = compute_collision_penalty(pred_pos, body_pos, body_normals)

    n_loss = torch.tensor(0.0, device=d_loss.device)
    if use_normal_consistency and faces is not None and face_adj is not None \
            and batch_idx is not None:
        n_loss = compute_normal_consistency_loss(
            pred_pos, gt_pos, faces, face_adj, batch_idx)

    b_loss = torch.tensor(0.0, device=d_loss.device)
    if use_bending_energy and faces is not None and face_adj is not None \
            and shared_edges is not None and batch_idx is not None:
        b_loss = compute_bending_energy_loss(
            pred_pos, gt_pos, faces, face_adj, shared_edges, batch_idx)

    c_loss = F.cross_entropy(fabric_logits, fabric_labels)

    total = (d_loss
             + strain_weight    * e_loss
             + collision_weight * col_loss
             + normal_weight    * n_loss
             + bending_weight   * b_loss
             + cls_weight       * c_loss)

    return total, d_loss, e_loss, col_loss, n_loss, b_loss, c_loss
