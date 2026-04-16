"""
models_v2.py (Baseline Model)

Architecture:
  StyleViT_B16   : Standard ViT-B/16 (frozen) -> projection head -> 128-dim style embedding
  MeshGraphNet   : encode-process-decode with 10 message passing layers
  Context Inject : Node Concatenation (Baseline Method)
  Losses         : Position MSE + Edge Strain Physics Loss + Aux Classification

Architecture:
  StyleViT_DINO  : DINOv2-Small (frozen) -> projection head -> 128-dim style embedding
  MeshGraphNet   : encode-process-decode with 10 message passing layers
  HybridDrapeModel: wrapper that connects both and defines the training forward pass

Node feature vector (per vertex, 158-dim total):
  pos      (3)   -- template vertex XYZ position
  uvs      (2)   -- UV coordinates (semantic identity on pattern)
  normals  (3)   -- rest-pose surface normal
  style   (128)  -- ViT style embedding broadcast to all nodes
  smpl    (10)   -- SMPL shape betas broadcast to all nodes
  physics (10)   -- log-normalised fabric physics broadcast to all nodes
  size     (2)   -- [width_pct, height_pct] broadcast to all nodes
  ───────────────
  total   (158)

Edge feature vector (per edge, 4-dim):
  dx, dy, dz (3) -- relative position vector between connected verts (rest pose)
  length     (1) -- Euclidean distance between connected verts (rest pose)

Output:
  predicted_delta : (total_nodes, 3) -- delta-v per vertex
  fabric_logits   : (B, 6)           -- fabric family classification (auxiliary)
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


# ── Dimensions (single source of truth) ──────────────────────────────────────

NODE_POS_DIM     = 3
NODE_UV_DIM      = 2
NODE_NORMAL_DIM  = 3
STYLE_DIM        = 128
SMPL_DIM         = 10
PHYSICS_DIM      = 10   # 10 params after removing fFriction and fFurBend
SIZE_DIM         = 2
POSE_DIM         = 72   # included for future multi-pose support

EDGE_IN_DIM  = 4    # [dx, dy, dz, length]
LATENT_DIM   = 128
GNN_LAYERS   = 10

NUM_FABRIC_FAMILIES = 6
NUM_FABRIC_PRESETS  = 12

# Total node input dim -- must match GarmentDataset exactly
NODE_IN_DIM = (NODE_POS_DIM + NODE_UV_DIM + NODE_NORMAL_DIM
               + STYLE_DIM + SMPL_DIM + PHYSICS_DIM + SIZE_DIM)
# = 3 + 2 + 3 + 128 + 10 + 10 + 2 = 158
# Note: pose (72) is NOT in node features for lean-only -- add when scaling to multi-pose


# ── MLP builder ───────────────────────────────────────────────────────────────

def build_mlp(in_dim, out_dim, hidden_dim=256):
    """
    3-layer MLP with LayerNorm and ReLU activations.
    hidden_dim defaults to 256 -- larger than latent_dim to give the network
    enough capacity at the input/output boundaries.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


# ── ViT style encoder (BASELINE: ViT-S/16) ─────────────────────────────────────────────────────────

class StyleViT_S16(nn.Module):
    """
    Frozen standard ViT-S/16 backbone + trainable projection head
    Maps a 224x224 image to a 128-dim style embedding.

    ViT_S_16 outputs a 384-dim CLS token.
    The projection head compresses this to STYLE_DIM (128) dims.

    The backbone is frozen -- only the projection head trains (better for small datasets)
    """

    def __init__(self, embed_dim=STYLE_DIM):
        super().__init__()

        # Load ImageNet pre-trained ViT-Small using timm
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True)

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable projection: 384 -> 256 -> embed_dim
        self.projection_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim),
        )

    def forward(self, images):
        """
        Args:
            images: (B, 3, 224, 224)
        Returns:
            style_code: (B, embed_dim)
        """
        with torch.no_grad():
            features = self.backbone.forward_features(images)  # (B, 197, 384) — patch tokens + CLS
            features = features[:, 0]                          # (B, 384) — CLS token only
        return self.projection_head(features)                  # (B, embed_dim)


# ── Message passing block ─────────────────────────────────────────────────────

class MeshBlock(MessagePassing):
    """
    Single MeshGraphNet message passing step.
    Updates both edge and node representations with residual connections.

    Edge update: f(node_src, node_dst, edge) -> new_edge
    Node update: g(node, sum(incoming_edges)) -> new_node
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__(aggr='sum')   # sum aggregation standard for physics sims

        # Edge MLP: (src, dst, edge) -> edge  -- 3 * latent_dim input
        self.edge_mlp = build_mlp(latent_dim * 3, latent_dim)

        # Node MLP: (node, aggregated_messages) -> node  -- 2 * latent_dim input
        self.node_mlp = build_mlp(latent_dim * 2, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:          (total_nodes, latent_dim)
            edge_index: (2, total_edges)
            edge_attr:  (total_edges, latent_dim)
        Returns:
            x:         (total_nodes, latent_dim)   -- updated nodes
            edge_attr: (total_edges, latent_dim)   -- updated edges
        """
        src, dst = edge_index

        # ── Edge update ───────────────────────────────────────────────────────
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_attr  = edge_attr + self.edge_mlp(edge_input)   # residual

        # ── Node update ───────────────────────────────────────────────────────
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x   = x + self.node_mlp(torch.cat([x, agg], dim=-1))  # residual

        return x, edge_attr

    def message(self, edge_attr):
        # Messages are the updated edge features
        return edge_attr


# ── Hybrid model ─────────────────────────────────────────────────────────────

class HybridDrapeModel(nn.Module):
    """
    Full ViT + GNN drape prediction model.

    Forward pass:
      1. ViT encodes image -> style embedding (B, 128)
      2. Style + physics + SMPL + size broadcast to every node
      3. Node encoder projects 158-dim features to latent_dim
      4. Edge encoder projects 4-dim edge features to latent_dim
      5. GNN_LAYERS MeshBlock layers process the graph
      6. Decoder outputs (total_nodes, 3) delta-v predictions
      7. Auxiliary classifier outputs (B, 6) fabric family logits

    Loss (computed in training loop):
      drape_loss = (MSE(predicted_delta, y) * loss_weight).mean()
      cls_loss   = CrossEntropy(fabric_logits, fabric_family_label)
      total_loss = drape_loss + cls_weight * cls_loss
    """

    def __init__(
        self,
        embed_dim  = STYLE_DIM,
        latent_dim = LATENT_DIM,
        gnn_layers = GNN_LAYERS,
    ):
        super().__init__()

        self.vit = StyleViT_S16(embed_dim=embed_dim)

        # Node encoder: 158 -> latent_dim
        self.node_encoder = build_mlp(NODE_IN_DIM, latent_dim)

        # Edge encoder: 4 -> latent_dim
        self.edge_encoder = build_mlp(EDGE_IN_DIM, latent_dim)

        # Message passing processor
        self.processor = nn.ModuleList([
            MeshBlock(latent_dim) for _ in range(gnn_layers)
        ])

        # Delta-v decoder: latent_dim -> 3
        self.decoder = nn.Linear(latent_dim, 3)

        # Auxiliary fabric family classifier (6 classes)
        # Operates on the style embedding -- gives ViT a direct training signal
        self.fabric_classifier = nn.Linear(embed_dim, NUM_FABRIC_FAMILIES)

    def forward(self, data):
        """
        Args:
            data: PyG Batch object from GarmentDataset containing:
              data.image       (B, 3, 224, 224)
              data.pos         (total_nodes, 3)
              data.uvs         (total_nodes, 2)
              data.normals     (total_nodes, 3)
              data.edge_index  (2, total_edges)
              data.edge_attr   (total_edges, 4)
              data.tgt_smpl    (B, 10)
              data.tgt_physics (B, 10)
              data.tgt_size    (B, 2)
              data.batch       (total_nodes,)  -- node-to-graph mapping

        Returns:
            predicted_delta : (total_nodes, 3)
            fabric_logits   : (B, 6)
        """
        # ── Step 1: ViT encodes image to style embedding ──────────────────────
        style_emb = self.vit(data.image)          # (B, 128)

        # ── ABLATION TEST — comment this out after testing ────────────────────
        # style_emb = torch.zeros_like(style_emb)  # zero out ViT embedding

        # ── Step 2: Broadcast global conditioning to every node ───────────────
        # data.batch maps each node to its graph index in the batch.
        # PyG concatenates per-graph tensors into (B*dim,) during batching
        # so we must reshape to (B, dim) before indexing with data.batch.
        b = data.batch                                      # (total_nodes,)
        B = style_emb.shape[0]

        node_style   = style_emb[b]                         # (total_nodes, 128)
        node_smpl    = data.tgt_smpl.view(B, SMPL_DIM)[b]   # (total_nodes, 10)
        node_physics = data.tgt_physics.view(B, PHYSICS_DIM)[b]  # (total_nodes, 10)
        node_size    = data.tgt_size.view(B, SIZE_DIM)[b]   # (total_nodes, 2)

        # ── Step 3: Build node feature vector ────────────────────────────────
        # [pos | uvs | normals | style | smpl | physics | size] = 158-dim
        x = torch.cat([
            data.pos,       # (total_nodes, 3)
            data.uvs,       # (total_nodes, 2)
            data.normals,   # (total_nodes, 3)
            node_style,     # (total_nodes, 128)
            node_smpl,      # (total_nodes, 10)
            node_physics,   # (total_nodes, 10)
            node_size,      # (total_nodes, 2)
        ], dim=-1)          # (total_nodes, 158)

        # ── Step 4: Encode node and edge features to latent dim ───────────────
        x         = self.node_encoder(x)              # (total_nodes, latent_dim)
        edge_attr = self.edge_encoder(data.edge_attr) # (total_edges, latent_dim)

        # ── Step 5: Message passing ───────────────────────────────────────────
        for layer in self.processor:
            x, edge_attr = layer(x, data.edge_index, edge_attr)

        # ── Step 6: Decode to delta-v ─────────────────────────────────────────
        predicted_delta = self.decoder(x)             # (total_nodes, 3)

        # ── Step 7: Auxiliary fabric classification on style embedding ────────
        fabric_logits = self.fabric_classifier(style_emb)  # (B, 6)

        return predicted_delta, fabric_logits


# ── Loss function ─────────────────────────────────────────────────────────────

def compute_edge_strain_loss(pred_pos, gt_pos, edge_index):
    """
    Calculates the difference in edge lengths between the predicted mesh and ground truth
    This acts as a structural physics penalty to prevent the fabric from melting/stretching
    """
    row, col = edge_index
    
    # Calculate lengths of edges in predicted mesh
    pred_edge_lengths = torch.norm(pred_pos[row] - pred_pos[col], dim=1)
    
    # Calculate lengths of edges in ground truth mesh
    gt_edge_lengths = torch.norm(gt_pos[row] - gt_pos[col], dim=1)
    
    return F.mse_loss(pred_edge_lengths, gt_edge_lengths)

def drape_loss(predicted_delta, target_delta, template_pos, edge_index, loss_weight,
               fabric_logits, fabric_labels, cls_weight=0.1, strain_weight=0.1):
    """
    Combined drape prediction loss + structural strain + auxiliary classification loss.

    Args:
        predicted_delta : (total_nodes, 3)  -- model output (how far each vertex should move from the template)
        target_delta    : (total_nodes, 3)  -- ground truth displacement
        template_pos    : (total_nodes, 3)  -- starting [X, Y, Z] coordinates of the template
        edge_index      : (2, total_edges)  -- tells the strain loss which vertices are physically stitched together
        loss_weight     : (total_nodes,)    -- per-vertex weights
                                               - usually higher for the hem/sleeves (which move a lot) and lower for the collar
        fabric_logits   : (B, 6)            -- fabric family predictions
        fabric_labels   : (B,)              -- ground truth fabric family labels
        cls_weight      : float             -- weight on classification loss (small so it is auxiliary only)
        strain_weight   : float             -- how strict the physics engine should be about fabric stretching

    Returns:
        total_loss  : scalar
        drape_loss  : scalar  (for logging)
        cls_loss    : scalar  (for logging)
        edge_loss   : scalar  (for logging)
    """

    # 1. Absolute Position Loss (MSE)
    # Calculates the physical distance between where the network placed the vertex 
    # and where it belongs, scaled by the vertex's importance weight
    sq_err = ((predicted_delta - target_delta) ** 2).sum(dim=-1)  # (total_nodes,)
    # Weighted mean
    d_loss = (sq_err * loss_weight).mean() # Normalized by vertex count

    # 2. Edge Strain Loss (Physics structure)
    # Reconstruct the absolute XYZ positions to calculate the distance between connected vertices
    # If the predicted distance is too far from the ground truth, the network is punished for "stretching" or "shrinking" the fabric
    pred_pos = template_pos + predicted_delta
    gt_pos = template_pos + target_delta
    e_loss = compute_edge_strain_loss(pred_pos, gt_pos, edge_index)

    # 3. Auxiliary classification loss
    # Punishes the ViT if it fails to recognize the correct fabric type from the image.
    c_loss = F.cross_entropy(fabric_logits, fabric_labels)

    # Combine losses
    total = d_loss + (strain_weight * e_loss) + (cls_weight * c_loss)

    # Return total (for the optimizer) and the individual components (for the logger)
    return total, d_loss, e_loss, c_loss


# ── Parameter count helper ────────────────────────────────────────────────────

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"Parameters:")
    print(f"  Total:     {total:>12,}")
    print(f"  Trainable: {trainable:>12,}")
    print(f"  Frozen:    {frozen:>12,}  (ViT_S_16 backbone)")
    return trainable


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Model smoke test\n")

    # Verify NODE_IN_DIM is correct
    expected = (NODE_POS_DIM + NODE_UV_DIM + NODE_NORMAL_DIM
                + STYLE_DIM + SMPL_DIM + PHYSICS_DIM + SIZE_DIM)
    assert NODE_IN_DIM == expected, \
        f"NODE_IN_DIM mismatch: {NODE_IN_DIM} != {expected}"
    print(f"Node input dim: {NODE_IN_DIM}  (pos={NODE_POS_DIM} + "
          f"uvs={NODE_UV_DIM} + normals={NODE_NORMAL_DIM} + "
          f"style={STYLE_DIM} + smpl={SMPL_DIM} + "
          f"physics={PHYSICS_DIM} + size={SIZE_DIM})")

    # Build model (skip ViT download in smoke test)
    print("\nBuilding model (skipping ViT download)...")

    class DummyViT(nn.Module):
        """Stands in for StyleViT_S16 during smoke test."""
        def forward(self, x):
            return torch.zeros(x.shape[0], STYLE_DIM)

    # Patch HybridDrapeModel to use DummyViT so ViT_S_16 is never downloaded
    class HybridDrapeModelTest(HybridDrapeModel):
        def __init__(self):
            # Call nn.Module.__init__ directly to skip ViT init
            nn.Module.__init__(self)
            self.vit              = DummyViT()
            self.node_encoder     = build_mlp(NODE_IN_DIM, LATENT_DIM)
            self.edge_encoder     = build_mlp(EDGE_IN_DIM, LATENT_DIM)
            self.processor        = nn.ModuleList([MeshBlock(LATENT_DIM) for _ in range(GNN_LAYERS)])
            self.decoder          = nn.Linear(LATENT_DIM, 3)
            self.fabric_classifier = nn.Linear(STYLE_DIM, NUM_FABRIC_FAMILIES)

    model = HybridDrapeModelTest()
    count_parameters(model)

    # Build a fake batch
    B, N, E = 2, 14117, 82988
    from torch_geometric.data import Batch, Data

    graphs = []
    for _ in range(B):
        src = torch.randint(0, N, (E // 2,))
        dst = torch.randint(0, N, (E // 2,))
        ei  = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src])
        ], dim=0)
        graphs.append(Data(
            pos        = torch.randn(N, 3),
            uvs        = torch.randn(N, 2),
            normals    = torch.randn(N, 3),
            edge_index = ei,
            edge_attr  = torch.randn(E, 4),
            y          = torch.randn(N, 3),
            loss_weight= torch.ones(N),
            tgt_smpl   = torch.randn(SMPL_DIM),
            tgt_physics= torch.randn(PHYSICS_DIM),
            tgt_size   = torch.randn(SIZE_DIM),
            image      = torch.randn(3, 224, 224),
            fabric_family_label = torch.tensor(0),
        ))

    batch = Batch.from_data_list(graphs)

    print(f"\nFake batch:")
    print(f"  Graphs:      {B}")
    print(f"  Total nodes: {batch.pos.shape[0]}")
    print(f"  Total edges: {batch.edge_attr.shape[0]}")

    # Manual forward
    style_emb    = torch.zeros(B, STYLE_DIM)
    b            = batch.batch
    node_style   = style_emb[b]
    node_smpl    = batch.tgt_smpl.view(B, SMPL_DIM)[b]
    node_physics = batch.tgt_physics.view(B, PHYSICS_DIM)[b]
    node_size    = batch.tgt_size.view(B, SIZE_DIM)[b]
    x = torch.cat([batch.pos, batch.uvs, batch.normals,
                   node_style, node_smpl, node_physics, node_size], dim=-1)

    print(f"\nNode feature vector: {x.shape}  (expected (total_nodes, {NODE_IN_DIM}))")
    assert x.shape[1] == NODE_IN_DIM, f"Node feature dim mismatch: {x.shape[1]} != {NODE_IN_DIM}"

    x         = model.node_encoder(x)
    edge_attr = model.edge_encoder(batch.edge_attr)
    for layer in model.processor:
        x, edge_attr = layer(x, batch.edge_index, edge_attr)
    delta = model.decoder(x)

    print(f"Output delta-v:      {delta.shape}  (expected (total_nodes, 3))")
    assert delta.shape == (B * N, 3)

    # Loss
    fabric_logits = torch.zeros(B, NUM_FABRIC_FAMILIES)
    fabric_labels = batch.fabric_family_label
    total, d, e, c = drape_loss(delta, batch.y, batch.pos, batch.edge_index,
                                batch.loss_weight, fabric_logits, fabric_labels)
    print(f"\nLoss:  total={total:.4f}  drape={d:.4f}  strain={e:.4f}  cls={c:.4f}")

    print("\nSmoke test passed")
