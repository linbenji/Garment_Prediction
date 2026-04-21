"""
models_v3.py (Final Master Architecture) — PATCHED

MasterDrapeModel: DINOv2 (Frozen) + FiLM-Modulated MeshGraphNet + Physics Triad Loss

Patches relative to the original v3:
  1. compute_edge_strain_loss is now asymmetric — extension penalized 5x
     more than compression. Encourages realistic fabric buckling instead of
     unphysical stretching.

  2. compute_collision_penalty rewritten with PyKeOps for O(N) VRAM.
     Lets you raise batch_size without OOM on the cdist matrix. Also
     promotes inputs to float32 so it survives autocast(bfloat16/float16).

  3. drape_loss gains a body_ids + get_body_data interface so collision
     is computed per-sample against each garment's specific body, mirroring
     how v3.5 handles it. The legacy body_pos/body_normals interface is
     preserved for backward compatibility.

Architecture (unchanged):
  StyleViT_DINO  : DINOv2-Small (frozen) -> projection head -> 128-dim style
  Context Inject : MLP -> FiLM (Feature-wise Linear Modulation) [Scale & Shift]
  MeshGraphNet   : encode-process-decode with FiLM residuals
  Losses         : Position MSE + Asym Edge Strain + Normal Consistency
                 + Bending Energy + Laplacian Smoothness + Collision + Aux Cls
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from pykeops.torch import LazyTensor

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
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    return F.normalize(normals, dim=1)


# ── Loss: Asymmetric Edge Strain ─────────────────────────────────────────────

def compute_edge_strain_loss(pred_pos, gt_pos, edge_index, w_ext=5.0, w_comp=0.5):
    """
    Asymmetric edge-length penalty.

    Penalizes extension (stretching, diff > 0) `w_ext / w_comp` = 10x more
    heavily than compression. Real fabric buckles rather than stretching, so
    a model that over-extends edges produces unphysical drape; this loss
    reshapes the gradient to push it toward compression instead.

    NOTE: average loss magnitude is ~3-4x higher than the symmetric MSE
    version (most edges in a draped garment are slightly extended, not
    compressed). If you keep the same prior_strain you'll be effectively
    weighting strain 3-4x harder than before. Halve prior_strain to roughly
    preserve the previous effective weight.
    """
    row, col = edge_index
    pred_len = torch.norm(pred_pos[row] - pred_pos[col], dim=1)
    gt_len   = torch.norm(gt_pos[row]   - gt_pos[col],   dim=1)
    diff = pred_len - gt_len
    strain_sq = torch.where(diff > 0, w_ext * (diff ** 2), w_comp * (diff ** 2))
    return strain_sq.mean()


# ── Loss: Laplacian Smoothness ────────────────────────────────────────────────

def compute_laplacian_loss(pred_delta, gt_delta, edge_index):
    row, col = edge_index
    N = pred_delta.size(0)
    D = pred_delta.size(1)
    device = pred_delta.device

    ones = torch.ones(row.size(0), device=device)
    deg = torch.zeros(N, device=device).scatter_add_(0, row, ones).clamp(min=1)

    index = row.unsqueeze(1).expand(-1, D)
    pred_sum = torch.zeros_like(pred_delta).scatter_add_(0, index, pred_delta[col])
    gt_sum   = torch.zeros_like(gt_delta  ).scatter_add_(0, index, gt_delta[col])

    pred_neigh_mean = pred_sum / deg.unsqueeze(1)
    gt_neigh_mean   = gt_sum   / deg.unsqueeze(1)

    pred_lap = pred_delta - pred_neigh_mean
    gt_lap   = gt_delta   - gt_neigh_mean

    return F.mse_loss(pred_lap, gt_lap)


# ── Loss: Normal Consistency ──────────────────────────────────────────────────

def compute_normal_consistency_loss(pred_pos, gt_pos, faces, face_adj, batch_idx):
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
    B = batch_idx.max().item() + 1
    total_loss = 0.0
    f0, f1 = face_adj[:, 0], face_adj[:, 1]

    for i in range(B):
        mask  = (batch_idx == i)
        p_pos = pred_pos[mask]
        g_pos = gt_pos[mask]

        pred_normals = compute_face_normals(p_pos, faces)
        n1_p, n2_p = pred_normals[f0], pred_normals[f1]

        e_vec_p = p_pos[shared_edges[:, 1]] - p_pos[shared_edges[:, 0]]
        e_dir_p = F.normalize(e_vec_p, dim=1)

        sin_p = (torch.cross(n1_p, n2_p, dim=1) * e_dir_p).sum(dim=1)
        cos_p = (n1_p * n2_p).sum(dim=1)
        theta_pred = torch.atan2(sin_p, cos_p)

        gt_normals = compute_face_normals(g_pos, faces)
        n1_g, n2_g = gt_normals[f0], gt_normals[f1]

        e_vec_g = g_pos[shared_edges[:, 1]] - g_pos[shared_edges[:, 0]]
        e_dir_g = F.normalize(e_vec_g, dim=1)

        sin_g = (torch.cross(n1_g, n2_g, dim=1) * e_dir_g).sum(dim=1)
        cos_g = (n1_g * n2_g).sum(dim=1)
        theta_gt = torch.atan2(sin_g, cos_g)

        total_loss += F.mse_loss(theta_pred, theta_gt)

    return total_loss / B


# ── Loss: Riemannian Conformal Distortion ────────────────────────────────────

def compute_riemannian_loss(pred_pos, gt_pos, faces, batch_idx):
    """
    Conformal distortion energy — measures how much the predicted mesh
    distorts the intrinsic surface metric relative to the GT draped mesh.

    For each triangle, we compute the 2x2 Jacobian of the map from GT
    surface to predicted surface using the local tangent frame. We then
    decompose it into a conformal part (rotation + uniform scale) and a
    non-conformal residual. The loss penalises the non-conformal residual,
    i.e. shear and anisotropic stretch that cannot be explained by a
    uniform scaling of the GT metric.

    Why this fills the gap in v3_2:
      - Drape MSE       : extrinsic position error
      - Asymmetric strain: extrinsic edge length error
      - Normal consistency: surface curvature (angle between adjacent normals)
      - Bending energy  : signed dihedral angle at shared edges
      - Laplacian       : local smoothness of displacement field
      - Collision       : body penetration
      --- none of the above measures in-plane metric distortion ---
      - Riemannian loss : in-plane shear and anisotropic stretch of the
                          surface itself, measured in intrinsic coordinates.
                          A garment can have correct vertex positions and
                          correct fold angles but still have sheared triangles
                          that produce unphysical fabric stress. This catches it.

    Implementation detail — why we use GT as the reference metric:
      We want to penalise deviation from physically correct drape, so GT
      (the simulation output) is the reference Riemannian manifold. The
      template rest pose is NOT used as reference here because the template
      is undeformed — its metric would penalise all realistic draping, not
      just unphysical distortion.

    Args:
        pred_pos  : (N, 3) predicted vertex positions (template + predicted_delta)
        gt_pos    : (N, 3) ground truth vertex positions (template + target_delta)
        faces     : (F, 3) int64 face indices — shared topology, same for pred and gt
        batch_idx : (N,)   batch assignment per vertex

    Returns:
        scalar loss — mean conformal distortion energy across all faces and samples
    """
    B = int(batch_idx.max().item()) + 1
    total_loss = torch.zeros((), device=pred_pos.device)

    for i in range(B):
        mask = (batch_idx == i)
        p = pred_pos[mask]   # (Ni, 3)
        g = gt_pos[mask]     # (Ni, 3)

        # ── Triangle vertices ─────────────────────────────────────────────
        # v0, v1, v2: corners of each face in pred and gt
        p0, p1, p2 = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
        g0, g1, g2 = g[faces[:, 0]], g[faces[:, 1]], g[faces[:, 2]]

        # ── GT local tangent frame ────────────────────────────────────────
        # Build an orthonormal basis {e1, e2} on the GT triangle surface.
        # e1 = normalised edge (g1 - g0)
        # e2 = component of (g2 - g0) orthogonal to e1, then normalised
        # This gives us a 2D coordinate system on the GT triangle.
        gt_e1 = g1 - g0                                         # (F, 3)
        gt_e1_len = torch.norm(gt_e1, dim=1, keepdim=True).clamp(min=1e-8)
        e1 = gt_e1 / gt_e1_len                                  # (F, 3) unit

        gt_e2_raw = g2 - g0
        gt_e2_proj = gt_e2_raw - (gt_e2_raw * e1).sum(dim=1, keepdim=True) * e1
        gt_e2_len = torch.norm(gt_e2_proj, dim=1, keepdim=True).clamp(min=1e-8)
        e2 = gt_e2_proj / gt_e2_len                             # (F, 3) unit, perp to e1

        # ── GT 2D coordinates ─────────────────────────────────────────────
        # Project GT edge vectors into the local frame — gives 2D triangle coords.
        # By construction: gt_u = [[gt_e1_len, 0], [x, gt_e2_len]] (triangular)
        gt_a_u = gt_e1_len.squeeze(1)                           # (F,) = |g1-g0|
        gt_a_v = torch.zeros_like(gt_a_u)                       # by frame construction
        gt_b_u = (gt_e2_raw * e1).sum(dim=1)                    # (F,)
        gt_b_v = gt_e2_len.squeeze(1)                           # (F,) = height

        # ── Pred 2D coordinates ───────────────────────────────────────────
        # Project pred edge vectors into the SAME GT local frame.
        # This gives us 2D coords of the pred triangle in GT tangent space.
        p_e1 = p1 - p0                                          # (F, 3)
        p_e2 = p2 - p0                                          # (F, 3)

        pred_a_u = (p_e1 * e1).sum(dim=1)                       # (F,)
        pred_a_v = (p_e1 * e2).sum(dim=1)                       # (F,)
        pred_b_u = (p_e2 * e1).sum(dim=1)                       # (F,)
        pred_b_v = (p_e2 * e2).sum(dim=1)                       # (F,)

        # ── Jacobian J of map gt_2d -> pred_2d ───────────────────────────
        # We solve: [pred_a_u, pred_b_u]   [gt_a_u, gt_b_u]   [J00, J01]
        #           [pred_a_v, pred_b_v] = [gt_a_v, gt_b_v] * [J10, J11]
        #
        # Since gt frame is triangular: gt_a_v = 0, so:
        #   J00 = pred_a_u / gt_a_u
        #   J10 = pred_a_v / gt_a_u
        #   J01 = (pred_b_u - J00 * gt_b_u) / gt_b_v
        #   J11 = (pred_b_v - J10 * gt_b_u) / gt_b_v

        denom_u = gt_a_u.clamp(min=1e-8)
        denom_v = gt_b_v.clamp(min=1e-8)

        J00 = pred_a_u / denom_u
        J10 = pred_a_v / denom_u
        J01 = (pred_b_u - J00 * gt_b_u) / denom_v
        J11 = (pred_b_v - J10 * gt_b_u) / denom_v

        # ── Conformal distortion: Cauchy-Green -> deviation from identity──
        # The Cauchy-Green deformation tensor C = J^T J measures metric
        # distortion. For a conformal (angle-preserving) map, C = s^2 * I
        # for some scalar s. The conformal energy is:
        #   E_conf = ||C - (tr(C)/2) * I||_F^2 / (2 * det(C) + eps)
        #
        # This is zero iff the map is conformal (pure rotation + uniform
        # scale), and grows with shear and anisotropic stretch.

        # C = J^T J  (2x2, per face)
        C00 = J00 * J00 + J10 * J10   # (F,)
        C01 = J00 * J01 + J10 * J11   # (F,)
        C10 = C01                      # symmetric
        C11 = J01 * J01 + J11 * J11   # (F,)

        trace_C = C00 + C11            # (F,)
        det_C   = (C00 * C11 - C01 * C10).clamp(min=1e-8)

        # Frobenius norm of (C - (tr/2)*I)^2 — the non-conformal residual
        s2 = trace_C / 2.0             # optimal uniform scale^2
        diff00 = C00 - s2
        diff11 = C11 - s2
        frob_sq = diff00 ** 2 + 2 * (C01 ** 2) + diff11 ** 2   # (F,)

        # Normalise by det so thin degenerate triangles don't dominate
        conf_energy = frob_sq / (2.0 * det_C)                   # (F,)

        # Weight by GT triangle area so large faces contribute proportionally
        gt_cross = torch.cross(g1 - g0, g2 - g0, dim=1)        # (F, 3)
        gt_area  = torch.norm(gt_cross, dim=1) * 0.5            # (F,)
        gt_area  = gt_area.clamp(min=1e-8)

        # Area-weighted mean conformal energy for this sample
        total_loss = total_loss + (conf_energy * gt_area).sum() / gt_area.sum().clamp(min=1e-8)

    return total_loss / B


# ── Loss: Collision Penalty (PyKeOps) ─────────────────────────────────────────

def _unpack_min_argmin(result):
    """Robust unpacking across PyKeOps versions.

    PyKeOps has returned `min_argmin` as either:
      - a tuple/list of two tensors (min_values, argmin_indices)
      - a single tensor of shape (..., 2) with [min, argmin] along last dim
    """
    if isinstance(result, (tuple, list)):
        return result[0], result[1]
    return result[..., 0:1], result[..., 1:2]


def compute_collision_penalty(pred_pos, body_pos, body_normals, threshold=0.002):
    """
    Memory-free O(N) collision penalty using PyKeOps.

    Old cdist version was O(N*M) in VRAM, which capped batch size; this
    version symbolically defines the (N, M) distance matrix and only
    materializes the min/argmin reduction.

    PyKeOps requires float32/float64. Under autocast(bf16/fp16) pred_pos
    arrives in a low-precision dtype, so we promote inputs explicitly.
    """
    pred_pos     = pred_pos.float()
    body_pos     = body_pos.float()
    body_normals = body_normals.float()

    x_i = LazyTensor(pred_pos.view(-1, 1, 3))
    y_j = LazyTensor(body_pos.view(1, -1, 3))

    D_ij = ((x_i - y_j) ** 2).sum(-1)
    min_sq_dist, nearest_idx = _unpack_min_argmin(D_ij.min_argmin(dim=1))
    min_sq_dist = min_sq_dist.view(-1)
    nearest_idx = nearest_idx.view(-1).long()
    min_dist = torch.sqrt(min_sq_dist.clamp_min(0.0))

    nearest_normals   = body_normals[nearest_idx]
    direction_vectors = pred_pos - body_pos[nearest_idx]
    inside_mask = (torch.sum(direction_vectors * nearest_normals, dim=1) < 0)

    if not inside_mask.any():
        return pred_pos.sum() * 0.0  # zero with gradient attached

    collision_error = min_dist[inside_mask] + threshold
    return collision_error.mean()


# ── Combined Loss Function ────────────────────────────────────────────────────

def drape_loss(predicted_delta, target_delta, template_pos, edge_index, loss_weight,
               fabric_logits, fabric_labels,
               batch_idx=None, faces=None, face_adj=None, shared_edges=None,
               body_ids=None, get_body_data=None,
               body_pos=None, body_normals=None,
               use_normal_consistency=False, use_bending_energy=False,
               use_laplacian=False, use_riemannian=False,
               cls_weight=0.1, strain_weight=0.1, collision_weight=1.0,
               normal_weight=0.1, bending_weight=0.1, laplacian_weight=0.1,
               riemannian_weight=0.1):
    """
    Combined loss with configurable surface quality terms.

    Collision interface (pick one):
      - body_ids + get_body_data: per-sample collision against each garment's
        specific body, looked up from a cache. Preferred when bodies vary.
      - body_pos + body_normals:   single shared body for the whole batch.
        Legacy fallback.

    When using AutomaticLossWeighter, set the scalar weights to 1.0 and let
    the weighter handle dynamic scaling.

    Returns: total, d_loss, e_loss, col_loss, n_loss, b_loss, lap_loss, r_loss, c_loss
    """

    # 1. Drape (Position MSE)
    sq_err = ((predicted_delta - target_delta) ** 2).sum(dim=-1)
    d_loss = (sq_err * loss_weight).mean()

    pred_pos = template_pos + predicted_delta
    gt_pos   = template_pos + target_delta

    # 2. Asymmetric Edge Strain
    e_loss = compute_edge_strain_loss(pred_pos, gt_pos, edge_index)

    # 3. Collision Penalty
    col_loss = torch.zeros((), device=d_loss.device)
    if body_ids is not None and get_body_data is not None and batch_idx is not None:
        body_ids_cpu = body_ids.tolist()
        B = len(body_ids_cpu)
        batch_col_loss = torch.zeros((), device=d_loss.device)
        for i, b_id in enumerate(body_ids_cpu):
            b_pos, b_norm = get_body_data(b_id, d_loss.device)
            mask = (batch_idx == i)
            batch_col_loss = batch_col_loss + compute_collision_penalty(
                pred_pos[mask], b_pos, b_norm
            )
        col_loss = batch_col_loss / B
    elif body_pos is not None and body_normals is not None:
        col_loss = compute_collision_penalty(pred_pos, body_pos, body_normals)

    # 4. Normal Consistency (toggle via flag)
    n_loss = torch.zeros((), device=d_loss.device)
    if use_normal_consistency and faces is not None and face_adj is not None and batch_idx is not None:
        n_loss = compute_normal_consistency_loss(pred_pos, gt_pos, faces, face_adj, batch_idx)

    # 5. Bending Energy (toggle via flag)
    b_loss = torch.zeros((), device=d_loss.device)
    if use_bending_energy and faces is not None and face_adj is not None and shared_edges is not None and batch_idx is not None:
        b_loss = compute_bending_energy_loss(pred_pos, gt_pos, faces, face_adj, shared_edges, batch_idx)

    # 6. Laplacian Smoothness (toggle via flag)
    lap_loss = torch.zeros((), device=d_loss.device)
    if use_laplacian:
        lap_loss = compute_laplacian_loss(predicted_delta, target_delta, edge_index)

    # 7. Riemannian Conformal Distortion (toggle via flag)
    # Requires faces and batch_idx — same prerequisites as normal consistency.
    r_loss = torch.zeros((), device=d_loss.device)
    if use_riemannian and faces is not None and batch_idx is not None:
        r_loss = compute_riemannian_loss(pred_pos, gt_pos, faces, batch_idx)

    # 8. Aux Classification
    c_loss = F.cross_entropy(fabric_logits, fabric_labels)

    total = (d_loss
             + strain_weight      * e_loss
             + collision_weight   * col_loss
             + normal_weight      * n_loss
             + bending_weight     * b_loss
             + laplacian_weight   * lap_loss
             + riemannian_weight  * r_loss
             + cls_weight         * c_loss)

    return total, d_loss, e_loss, col_loss, n_loss, b_loss, lap_loss, r_loss, c_loss