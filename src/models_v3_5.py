"""
models_v3_5.py (Hierarchical Multi-Scale Architecture) — PATCHED

NonbelieverDrapeModel:
DINOv2 + FiLM-GNN + U-Net Skip Fusion + 2D UV-Refinement + Physics Septet Loss

Patches applied relative to the original v3.5:
  - UVRefinementNet:
      * vectorized scatter-add (no Python batch loop)
      * UVs clamped to [0, 1] before indexing (prevents OOB on seam verts)
      * final Conv2d zero-initialized so the refinement head starts as
        identity; base GNN drives early training before the refiner kicks in
      * default `grid_res=128` as an explicit int (was `grid_res=LATENT_DIM`)
  - compute_collision_penalty:
      * version-robust unpacking of `LazyTensor.min_argmin` (handles both
        tuple and (N, 2) tensor returns across pyKeOps versions)
      * float32 promotion so PyKeOps works correctly under bfloat16 autocast
      * comment clarified: penalty is linear in penetration depth with a
        small threshold floor for near-surface interior points
  - drape_loss:
      * pytorch3d removed entirely — mesh_normal_consistency and
        mesh_laplacian_smoothing replaced with pure PyTorch implementations
      * `body_ids.tolist()` once per call instead of `.item()` per sample
      * single `B = batch_idx.max().item()` sync, reused for both the
        collision loop and the mesh loss construction

Architecture notes (unchanged):
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
   - Position (MVE), Asymmetric Strain, Collision, Normal Consistency,
     Laplacian Smoothness, Aux Classification.
-------------------------------------------------------------------------------
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

UV_GRID_RES = 128  # explicit UV refinement grid resolution (decoupled from LATENT_DIM)


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
    """
    Projects GNN features onto a 2D UV grid, runs a small CNN to paint
    high-frequency displacement detail, and samples back to vertices.

    Notes:
      - UVs are clamped to [0, 1] before indexing to prevent out-of-bounds
        index_add_ on seam vertices or numerically-overshoot UVs.
      - The final Conv2d is zero-initialized so that at step 0 this module
        contributes zero `fine_delta`. The base GNN drives early training
        and the refiner learns residual corrections.
      - Scatter-add is vectorized with a single `index_add_` into a flat
        (B*R*R, C) buffer; no Python loop over batch samples.
    """
    def __init__(self, latent_dim=LATENT_DIM, grid_res=UV_GRID_RES):
        super().__init__()
        self.res = grid_res
        hidden_dim = latent_dim // 2
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, NODE_POS_DIM, kernel_size=1),  # delta x,y,z per pixel
        )
        # Zero-init final conv so the refinement head begins as identity.
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, uvs, batch_idx):
        # x: (N, C), uvs: (N, 2) in [0,1], batch_idx: (N,) long
        B = int(batch_idx.max().item()) + 1
        C = x.size(1)
        R = self.res

        # Clamp UVs to valid range before integer indexing.
        uvs_safe = uvs.clamp(0.0, 1.0)
        coords = (uvs_safe * (R - 1)).long()  # (N, 2) with (u_idx, v_idx)
        u_idx = coords[:, 0]
        v_idx = coords[:, 1]

        # Flat index into a (B*R*R, C) buffer. Layout: (b, v, u) -> b*R*R + v*R + u.
        flat_idx = batch_idx * (R * R) + v_idx * R + u_idx  # (N,)

        flat_grid = torch.zeros((B * R * R, C), device=x.device, dtype=x.dtype)
        flat_grid.index_add_(0, flat_idx, x)  # handles overlapping UVs correctly

        # Reshape to (B, C, R, R): (B, v=H, u=W, C) -> permute channels up.
        grid = flat_grid.view(B, R, R, C).permute(0, 3, 1, 2).contiguous()

        refinement_map = self.net(grid)  # (B, 3, R, R)

        # Sample back to mesh vertices. grid_sample's last-dim convention is
        # (x_W, y_H); we stored u along W and v along H, so (u, v) is correct.
        # Requires equal nodes-per-sample (true for this fixed-topology mesh).
        sample_grid = uvs_safe.view(B, -1, 1, NODE_UV_DIM) * 2.0 - 1.0
        fine_delta = F.grid_sample(
            refinement_map,
            sample_grid,
            align_corners=True,
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
        # With gnn_layers=8, skip indices [3, 7] correspond to layers 4 and 8
        # (~1/2 and end of stack), distinct from the final x.
        self.hierarchical_fuser = nn.Linear(latent_dim * 3, latent_dim)

        # 5. 2D vision-based refinement head (zero-init residual)
        self.uv_refiner = UVRefinementNet(latent_dim)

        # 6. Decoders
        self.decoder = nn.Linear(latent_dim, 3)
        self.fabric_classifier = nn.Linear(embed_dim, NUM_FABRIC_FAMILIES)

    def forward(self, data):
        # A. Vision Feature Extraction
        style_emb = self.vit(data.image)  # (B, 128)

        # B. Context Injection (FiLM)
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
            film_params = self.film_generators[i](global_cond)
            gamma, beta = torch.chunk(film_params[b], 2, dim=-1)
            x, edge_attr = layer(x, data.edge_index, edge_attr, gamma, beta)
            # Snapshot middle-of-stack features for the U-Net skip fusion.
            if i in [3, 7]:
                skip_features.append(x)

        # E. Multi-Scale Fusion
        x = self.hierarchical_fuser(torch.cat([x] + skip_features, dim=-1))

        # F. Dual-Head Prediction
        base_delta = self.decoder(x)

        # G. High-frequency wrinkle detail (starts as zero due to init)
        fine_delta = self.uv_refiner(x, data.uvs, data.batch)

        fabric_logits = self.fabric_classifier(style_emb)

        return base_delta + fine_delta, fabric_logits


# ── Loss: Edge Strain (Asymmetric) ───────────────────────────────────────────

def compute_edge_strain_loss(pred_pos, gt_pos, edge_index, w_ext=5.0, w_comp=0.5):
    row, col = edge_index
    pred_len = torch.norm(pred_pos[row] - pred_pos[col], dim=1)
    gt_len   = torch.norm(gt_pos[row]   - gt_pos[col],   dim=1)

    diff = pred_len - gt_len
    # Penalize extension (stretching) more heavily than compression to
    # encourage realistic fabric buckling.
    strain_sq = torch.where(diff > 0, w_ext * (diff ** 2), w_comp * (diff ** 2))
    return strain_sq.mean()


# ── Loss: Normal Consistency (pure PyTorch) ───────────────────────────────────

def mesh_normal_consistency(pred_pos, faces):
    """
    Vectorized normal consistency loss — pure PyTorch, no pytorch3d required.

    For every pair of faces sharing an edge, penalizes the cosine distance
    between their normals. Encourages smooth, physically-plausible surface
    curvature and correct wrinkle orientation.

    Args:
        pred_pos : (N, 3) vertex positions (flat batch, faces already offset)
        faces    : (F, 3) long — triangle indices into pred_pos
    """
    v0 = pred_pos[faces[:, 0]]
    v1 = pred_pos[faces[:, 1]]
    v2 = pred_pos[faces[:, 2]]
    face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)  # (F, 3)

    F_count = faces.size(0)
    # Each face contributes 3 half-edges; sort each so (a,b)==(b,a).
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], dim=0)  # (3F, 2)
    face_for_edge = torch.arange(F_count, device=faces.device).repeat(3)             # (3F,)
    edge_key = edges.sort(dim=1).values                                               # (3F, 2)

    _, inverse, counts = torch.unique(edge_key, return_inverse=True, return_counts=True, dim=0)

    # Only manifold edges (shared by exactly 2 faces) contribute.
    shared_mask = (counts == 2)[inverse]                    # (3F,) bool
    shared_indices = torch.where(shared_mask)[0]            # indices into 3F
    shared_inverse = inverse[shared_mask]

    # Sort so paired half-edges are adjacent.
    sort_order = shared_inverse.argsort(stable=True)
    sorted_indices = shared_indices[sort_order]

    face_a = face_for_edge[sorted_indices[0::2]]
    face_b = face_for_edge[sorted_indices[1::2]]

    cos_sim = (face_normals[face_a] * face_normals[face_b]).sum(dim=-1)
    return (1.0 - cos_sim).mean()


# ── Loss: Laplacian Smoothing (pure PyTorch) ──────────────────────────────────

def mesh_laplacian_smoothing(pred_pos, faces):
    """
    Vectorized uniform Laplacian smoothing loss — pure PyTorch, no pytorch3d.

    Penalizes each vertex for deviating from the mean of its neighbors,
    suppressing high-frequency 'orange-peel' mesh noise and jittery verts.

    Args:
        pred_pos : (N, 3) vertex positions (flat batch, faces already offset)
        faces    : (F, 3) long — triangle indices into pred_pos
    """
    N = pred_pos.size(0)
    # Collect all directed edges from the face list.
    edges = torch.cat([
        faces[:, [0, 1]], faces[:, [1, 0]],
        faces[:, [1, 2]], faces[:, [2, 1]],
        faces[:, [0, 2]], faces[:, [2, 0]],
    ], dim=0)
    src, dst = edges[:, 0], edges[:, 1]

    neighbor_sum = torch.zeros_like(pred_pos)
    neighbor_sum.index_add_(0, src, pred_pos[dst])

    degree = torch.zeros(N, 1, device=pred_pos.device, dtype=pred_pos.dtype)
    ones   = torch.ones(src.size(0), 1, device=pred_pos.device, dtype=pred_pos.dtype)
    degree.index_add_(0, src, ones)
    degree = degree.clamp_min(1.0)

    laplacian = pred_pos - neighbor_sum / degree
    return (laplacian ** 2).sum(dim=-1).mean()


# ── Loss: Collision Penalty ──────────────────────────────────────────────────

def _unpack_min_argmin(result):
    """Robust unpacking across pyKeOps versions.

    pyKeOps has returned `min_argmin` as either:
      - a tuple/list of two tensors (min_values, argmin_indices)
      - a single tensor of shape (..., 2) with [min, argmin] along last dim
    """
    if isinstance(result, (tuple, list)):
        return result[0], result[1]
    return result[..., 0:1], result[..., 1:2]


def compute_collision_penalty(pred_pos, body_pos, body_normals, threshold=0.002):
    """
    Memory-free collision penalty using PyKeOps. O(N) VRAM.

    For vertices judged to be inside the body (via body-normal dot product),
    the penalty is linear in distance-to-nearest-surface with a small
    threshold floor, so even grazing-interior points incur a non-zero
    gradient that pushes them outward.

    PyKeOps only supports float32/float64 — under autocast (bfloat16 or
    float16) pred_pos arrives in a low-precision dtype, so we promote all
    inputs to float32 before any LazyTensor construction.
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

    min_dist = torch.sqrt(min_sq_dist.clamp_min(0.0))  # numerical guard

    nearest_normals   = body_normals[nearest_idx]
    direction_vectors = pred_pos - body_pos[nearest_idx]

    inside_mask = (torch.sum(direction_vectors * nearest_normals, dim=1) < 0)

    if not inside_mask.any():
        return pred_pos.sum() * 0.0  # zero with gradient attached

    collision_error = min_dist[inside_mask] + threshold
    return collision_error.mean()


# ── Combined Loss Function ───────────────────────────────────────────────────

def drape_loss(predicted_delta, target_delta, template_pos, edge_index, loss_weight,
               fabric_logits, fabric_labels,
               batch_idx=None, faces=None,
               body_ids=None, get_body_data=None,
               use_normal_consistency=False, use_laplacian=False):
    """
    Calculates the raw mathematical losses. Task balancing is handled
    externally by AutomaticLossWeighter.

    Returns: d_loss, e_loss, col_loss, n_loss, lap_loss, c_loss
    """

    # 1. Drape (weighted position MSE)
    sq_err = ((predicted_delta - target_delta) ** 2).sum(dim=-1)
    d_loss = (sq_err * loss_weight).mean()

    pred_pos = template_pos + predicted_delta
    gt_pos   = template_pos + target_delta

    # 2. Edge strain
    e_loss = compute_edge_strain_loss(pred_pos, gt_pos, edge_index)

    # Single sync for batch size / per-sample body IDs, reused below.
    B = None
    body_ids_cpu = None
    if batch_idx is not None:
        if body_ids is not None:
            body_ids_cpu = body_ids.tolist()
            B = len(body_ids_cpu)
        else:
            B = int(batch_idx.max().item()) + 1

    # 3. Collision penalty (per-sample, mean across batch)
    col_loss = torch.zeros((), device=d_loss.device)
    if body_ids_cpu is not None and get_body_data is not None and B is not None:
        batch_col_loss = torch.zeros((), device=d_loss.device)
        for i, b_id in enumerate(body_ids_cpu):
            b_pos, b_norm = get_body_data(b_id, d_loss.device)
            mask = (batch_idx == i)
            batch_col_loss = batch_col_loss + compute_collision_penalty(
                pred_pos[mask], b_pos, b_norm
            )
        col_loss = batch_col_loss / B

    # Build offset face index tensor for batched normal/laplacian losses.
    # Each garment in the batch has the same topology (fixed mesh), so we
    # tile the face list and offset indices by N_per_garment * batch_item.
    faces_batched_flat = None
    if faces is not None and B is not None:
        N_per = pred_pos.size(0) // B
        offsets = torch.arange(B, device=faces.device) * N_per   # (B,)
        # (B, F, 3) + (B, 1, 1) -> (B*F, 3)
        faces_batched_flat = (
            faces.unsqueeze(0) + offsets.view(B, 1, 1)
        ).view(-1, 3)

    # 4. Normal consistency (covers curvature and dihedral angles)
    n_loss = torch.zeros((), device=d_loss.device)
    if use_normal_consistency and faces_batched_flat is not None:
        n_loss = mesh_normal_consistency(pred_pos.float(), faces_batched_flat)

    # 5. Laplacian smoothness
    lap_loss = torch.zeros((), device=d_loss.device)
    if use_laplacian and faces_batched_flat is not None:
        lap_loss = mesh_laplacian_smoothing(pred_pos.float(), faces_batched_flat)

    # 6. Auxiliary classification
    c_loss = F.cross_entropy(fabric_logits, fabric_labels)

    return d_loss, e_loss, col_loss, n_loss, lap_loss, c_loss