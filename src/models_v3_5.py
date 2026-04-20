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
   - Normal Consistency: High-priority cosine similarity to ensure wrinkles 
     face the correct physical direction.
   - Bending Energy: Dihedral angle penalty to maintain soft fabric sweeps.
   - Laplacian Smoothness: MM-space penalty to eliminate 'orange-peel' mesh 
     noise and jittery vertices.
   - Aux Classification: Forces the backbone to distinguish material families.
-------------------------------------------------------------------------------
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
        grid = torch.zeros((B, x.size(1), self.res, self.res), device=x.device)
        
        # Map [0,1] UVs to grid indices
        coords = (uvs * (self.res - 1)).long()
        for i in range(B):
            mask = (batch_idx == i)
            grid[i, :, coords[mask, 1], coords[mask, 0]] = x[mask].t()
            
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


# ── Loss: Edge Strain (Asymmetric) ────────────────────────────────────────────

def compute_edge_strain_loss(pred_pos, gt_pos, edge_index, w_ext=5.0, w_comp=0.5):
    row, col = edge_index
    pred_len = torch.norm(pred_pos[row] - pred_pos[col], dim=1)
    gt_len   = torch.norm(gt_pos[row]   - gt_pos[col],   dim=1)
    
    diff = pred_len - gt_len
    # Penalize extension more heavily than compression
    strain_sq = torch.where(diff > 0, w_ext * (diff**2), w_comp * (diff**2))
    return strain_sq.mean()


# ── Loss: Laplacian Smoothness ────────────────────────────────────────────────

def compute_laplacian_loss(pred_delta, gt_delta, edge_index):
    """
    Penalises differences in the discrete (uniform) Laplacian of the
    displacement field between predicted and GT meshes.

    For each vertex v:
        L(v) = v - mean(neighbours(v))

    L(pred_delta) measures how much each predicted vertex disagrees with the
    average of its neighbours. Matching L(gt_delta) forces the prediction to
    reproduce the same local smoothness/roughness pattern as ground truth.

    Unlike normal consistency (dot-product MSE), this operates directly in mm
    and does NOT suffer the gradient-vanishing problem near cos(θ)≈1 — so it
    actively penalises small-amplitude orange-peel noise in flat regions.

    Operates on deltas: L(pred_pos) - L(gt_pos) = L(pred_delta) - L(gt_delta)
    because the template cancels under the linear Laplacian operator.
    """
    row, col = edge_index                      # (2, E), bidirectional
    N = pred_delta.size(0)
    D = pred_delta.size(1)
    device = pred_delta.device

    # Degree of each vertex (number of neighbours)
    ones = torch.ones(row.size(0), device=device)
    deg = torch.zeros(N, device=device).scatter_add_(0, row, ones).clamp(min=1)

    # Sum of neighbour delta values for each vertex
    index = row.unsqueeze(1).expand(-1, D)     # (E, D)
    pred_sum = torch.zeros_like(pred_delta).scatter_add_(0, index, pred_delta[col])
    gt_sum   = torch.zeros_like(gt_delta  ).scatter_add_(0, index, gt_delta[col])

    # Mean of neighbours
    pred_neigh_mean = pred_sum / deg.unsqueeze(1)
    gt_neigh_mean   = gt_sum   / deg.unsqueeze(1)

    # Laplacian = self - mean(neighbours)
    pred_lap = pred_delta - pred_neigh_mean
    gt_lap   = gt_delta   - gt_neigh_mean

    return F.mse_loss(pred_lap, gt_lap)


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
    
    # Dot product < 0 means the garment is inside the skin
    inside_mask = (torch.sum(direction_vectors * nearest_normals, dim=1) < 0)
    
    if not inside_mask.any():
        return torch.tensor(0.0, device=pred_pos.device)
    
    # Penalize based on how deep it is PLUS the safety threshold
    # Now, deeper penetration = higher loss.
    collision_error = min_dist[inside_mask] + threshold
    return collision_error.mean()


# ── Combined Loss Function ────────────────────────────────────────────────────

def drape_loss(predicted_delta, target_delta, template_pos, edge_index, loss_weight,
               fabric_logits, fabric_labels,
               batch_idx=None, faces=None, face_adj=None, shared_edges=None,
               body_ids=None, get_body_data=None,
               use_normal_consistency=False, use_bending_energy=False, use_laplacian=False):
    """
    Calculates the raw mathematical losses. 
    Task balancing is handled externally by AutomaticLossWeighter.

    Returns: d_loss, e_loss, col_loss, n_loss, b_loss, lap_loss, c_loss
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

    # 4. Normal Consistency (toggle via flag)
    n_loss = torch.tensor(0.0, device=d_loss.device)
    if use_normal_consistency and faces is not None and face_adj is not None and batch_idx is not None:
        n_loss = compute_normal_consistency_loss(pred_pos, gt_pos, faces, face_adj, batch_idx)

    # 5. Bending Energy (toggle via flag)
    b_loss = torch.tensor(0.0, device=d_loss.device)
    if use_bending_energy and faces is not None and face_adj is not None and shared_edges is not None and batch_idx is not None:
        b_loss = compute_bending_energy_loss(pred_pos, gt_pos, faces, face_adj, shared_edges, batch_idx)

    # 6. Laplacian Smoothness (toggle via flag)
    lap_loss = torch.tensor(0.0, device=d_loss.device)
    if use_laplacian:
        lap_loss = compute_laplacian_loss(predicted_delta, target_delta, edge_index)

    # 7. Aux Classification
    c_loss = F.cross_entropy(fabric_logits, fabric_labels)

    return d_loss, e_loss, col_loss, n_loss, b_loss, lap_loss, c_loss