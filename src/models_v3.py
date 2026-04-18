"""
models_v3.py (Final Master Architecture)

MasterDrapeModel: DINOv2 (Frozen) + FiLM-Modulated MeshGraphNet + Physics Triad Loss
Architecture:
  StyleViT_DINO  : DINOv2-Small (frozen) -> projection head -> 128-dim style
  Context Inject : MLP -> FiLM (Feature-wise Linear Modulation) [Scale & Shift]
  MeshGraphNet   : encode-process-decode with FiLM residuals
  Losses         : Position MSE + Edge Strain + Normal Consistency + Bending Energy + Collision + Aux Cls
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


# ── Final Master Architecture ────────────────────────────────────────────────

class MasterDrapeModel(nn.Module):
    def __init__(self, gnn_layers, embed_dim=STYLE_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.vit = StyleViT_DINO(embed_dim=embed_dim)
        self.node_encoder = build_mlp(NODE_IN_DIM, latent_dim)
        self.edge_encoder = build_mlp(EDGE_IN_DIM, latent_dim)
        self.film_generators = nn.ModuleList([
            nn.Linear(GLOBAL_COND_DIM, latent_dim * 2) for _ in range(gnn_layers)
        ])
        self.processor = nn.ModuleList([FiLMMeshBlock(latent_dim) for _ in range(gnn_layers)])
        self.decoder = nn.Linear(latent_dim, 3)
        self.fabric_classifier = nn.Linear(embed_dim, NUM_FABRIC_FAMILIES)

    def forward(self, data):
        style_emb = self.vit(data.image)
        global_cond = torch.cat([
            style_emb,
            data.tgt_smpl.view(-1, SMPL_DIM),
            data.tgt_physics.view(-1, PHYSICS_DIM),
            data.tgt_size.view(-1, SIZE_DIM)
        ], dim=-1)
        x = torch.cat([data.pos, data.uvs, data.normals], dim=-1)
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(data.edge_attr)
        b = data.batch
        for i, layer in enumerate(self.processor):
            film_params = self.film_generators[i](global_cond)
            film_params_nodes = film_params[b]
            gamma, beta = torch.chunk(film_params_nodes, 2, dim=-1)
            x, edge_attr = layer(x, data.edge_index, edge_attr, gamma, beta)
        predicted_delta = self.decoder(x)
        fabric_logits = self.fabric_classifier(style_emb)
        return predicted_delta, fabric_logits


# ── Face Adjacency (precompute once at training startup) ─────────────────────

def build_face_adjacency(faces):
    """
    Finds all pairs of faces that share an edge.
    Returns:
        face_adj:     (num_adj_pairs, 2) — adjacent face index pairs
        shared_edges: (num_adj_pairs, 2) — vertex indices of the shared edge
    """
    edge_to_face = {}
    adjacency = []
    shared = []

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

    face_adj     = torch.tensor(adjacency, dtype=torch.long)
    shared_edges = torch.tensor(shared, dtype=torch.long)
    return face_adj, shared_edges


# ── Loss Helper: Face Normals ─────────────────────────────────────────────────

def compute_face_normals(verts, faces):
    """Unit face normals for a single graph. verts: (N,3), faces: (F,3) long."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    return F.normalize(normals, dim=1)


# ── Loss: Edge Strain ─────────────────────────────────────────────────────────

def compute_edge_strain_loss(pred_pos, gt_pos, edge_index):
    row, col = edge_index
    pred_edge_lengths = torch.norm(pred_pos[row] - pred_pos[col], dim=1)
    gt_edge_lengths   = torch.norm(gt_pos[row]   - gt_pos[col],   dim=1)
    return F.mse_loss(pred_edge_lengths, gt_edge_lengths)


# ── Loss: Normal Consistency ──────────────────────────────────────────────────

def compute_normal_consistency_loss(pred_pos, gt_pos, faces, face_adj, batch_idx):
    """
    Penalizes differences in surface curvature between predicted and GT.
    For each pair of adjacent faces, computes dot product of their normals.
    MSE between predicted and GT dot products forces the model to reproduce
    the same fold pattern — smooth where GT is smooth, folded where GT folds.
    """
    B = batch_idx.max().item() + 1
    total_loss = 0.0
    f0, f1 = face_adj[:, 0], face_adj[:, 1]

    for i in range(B):
        mask  = (batch_idx == i)
        p_pos = pred_pos[mask]
        g_pos = gt_pos[mask]

        pred_normals = compute_face_normals(p_pos, faces)
        gt_normals   = compute_face_normals(g_pos, faces)

        pred_dots = (pred_normals[f0] * pred_normals[f1]).sum(dim=1)
        gt_dots   = (gt_normals[f0]   * gt_normals[f1]).sum(dim=1)

        total_loss += F.mse_loss(pred_dots, gt_dots)

    return total_loss / B


# ── Loss: Bending Energy ──────────────────────────────────────────────────────

def compute_bending_energy_loss(pred_pos, gt_pos, faces, face_adj, shared_edges, batch_idx):
    """
    Penalizes differences in signed dihedral angles between predicted and GT.

    Unlike normal consistency (which uses unsigned dot products), bending energy
    uses the full signed dihedral angle via atan2. This distinguishes between
    inward and outward folds — a concave fold and a convex fold at the same angle
    score differently, which is important for fabric that has a preferred bending direction.
    """
    B = batch_idx.max().item() + 1
    total_loss = 0.0
    f0, f1 = face_adj[:, 0], face_adj[:, 1]

    for i in range(B):
        mask  = (batch_idx == i)
        p_pos = pred_pos[mask]
        g_pos = gt_pos[mask]

        # ── Predicted dihedral angles ─────────────────────────────────────
        pred_normals = compute_face_normals(p_pos, faces)
        n1_p, n2_p = pred_normals[f0], pred_normals[f1]

        e_vec_p = p_pos[shared_edges[:, 1]] - p_pos[shared_edges[:, 0]]
        e_dir_p = F.normalize(e_vec_p, dim=1)

        sin_p = (torch.cross(n1_p, n2_p, dim=1) * e_dir_p).sum(dim=1)
        cos_p = (n1_p * n2_p).sum(dim=1)
        theta_pred = torch.atan2(sin_p, cos_p)

        # ── Ground truth dihedral angles ──────────────────────────────────
        gt_normals = compute_face_normals(g_pos, faces)
        n1_g, n2_g = gt_normals[f0], gt_normals[f1]

        e_vec_g = g_pos[shared_edges[:, 1]] - g_pos[shared_edges[:, 0]]
        e_dir_g = F.normalize(e_vec_g, dim=1)

        sin_g = (torch.cross(n1_g, n2_g, dim=1) * e_dir_g).sum(dim=1)
        cos_g = (n1_g * n2_g).sum(dim=1)
        theta_gt = torch.atan2(sin_g, cos_g)

        total_loss += F.mse_loss(theta_pred, theta_gt)

    return total_loss / B


# ── Loss: Collision Penalty ───────────────────────────────────────────────────

def compute_collision_penalty(pred_pos, body_pos, body_normals, threshold=0.002):
    distances = torch.cdist(pred_pos, body_pos)
    min_dist, nearest_idx = torch.min(distances, dim=1)
    nearest_normals = body_normals[nearest_idx]
    direction_vectors = pred_pos - body_pos[nearest_idx]
    inside_mask = (torch.sum(direction_vectors * nearest_normals, dim=1) < 0)
    collision_error = torch.relu(threshold - min_dist[inside_mask])
    if collision_error.numel() == 0:
        return torch.tensor(0.0, device=pred_pos.device)
    return collision_error.mean()


# ── Combined Loss Function ────────────────────────────────────────────────────

def drape_loss(predicted_delta, target_delta, template_pos, edge_index, loss_weight,
               fabric_logits, fabric_labels,
               batch_idx=None, faces=None, face_adj=None, shared_edges=None,
               body_pos=None, body_normals=None,
               use_normal_consistency=False, use_bending_energy=False,
               cls_weight=0.1, strain_weight=0.1, collision_weight=1.0,
               normal_weight=0.1, bending_weight=0.1):
    """
    Combined loss with configurable surface quality terms.

    When using AutomaticLossWeighter, set all weights to 1.0 and let
    the weighter handle dynamic scaling.

    Returns: total, d_loss, e_loss, col_loss, n_loss, b_loss, c_loss
    """

    # 1. Drape (Position MSE)
    sq_err = ((predicted_delta - target_delta) ** 2).sum(dim=-1)
    d_loss = (sq_err * loss_weight).mean()

    pred_pos = template_pos + predicted_delta
    gt_pos   = template_pos + target_delta

    # 2. Edge Strain
    e_loss = compute_edge_strain_loss(pred_pos, gt_pos, edge_index)

    # 3. Collision Penalty
    col_loss = torch.tensor(0.0, device=d_loss.device)
    if body_pos is not None and body_normals is not None:
        col_loss = compute_collision_penalty(pred_pos, body_pos, body_normals)

    # 4. Normal Consistency (toggle via flag)
    n_loss = torch.tensor(0.0, device=d_loss.device)
    if use_normal_consistency and faces is not None and face_adj is not None and batch_idx is not None:
        n_loss = compute_normal_consistency_loss(pred_pos, gt_pos, faces, face_adj, batch_idx)

    # 5. Bending Energy (toggle via flag)
    b_loss = torch.tensor(0.0, device=d_loss.device)
    if use_bending_energy and faces is not None and face_adj is not None and shared_edges is not None and batch_idx is not None:
        b_loss = compute_bending_energy_loss(pred_pos, gt_pos, faces, face_adj, shared_edges, batch_idx)

    # 6. Aux Classification
    c_loss = F.cross_entropy(fabric_logits, fabric_labels)

    total = (d_loss
             + strain_weight * e_loss
             + collision_weight * col_loss
             + normal_weight * n_loss
             + bending_weight * b_loss
             + cls_weight * c_loss)

    return total, d_loss, e_loss, col_loss, n_loss, b_loss, c_loss