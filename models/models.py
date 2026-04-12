import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


# ViT model
class StyleViT_DINO(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        
        # Load DINOv2 Small from PyTorch Hub
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Project the embedding down to 128 dims
        # DINOv2-small outputs 384 dims
        self.projection_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, images):
        # DINOv2 returns the CLS token by default
        with torch.no_grad():
            features = self.backbone(images) # Shape: [Batch, 384]
        style_code = self.projection_head(features) # Shape: [Batch, 128]
        return style_code


# Building blocks for GNN
def build_mlp(in_dim, out_dim, hidden_dim=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    )

# GNN model
class MeshBlock(MessagePassing):
    def __init__(self, latent_dim):
        super().__init__(aggr='sum') # Sum aggregation is standard for physics
        self.edge_mlp = build_mlp(latent_dim * 3, latent_dim) # source, target, edge
        self.node_mlp = build_mlp(latent_dim * 2, latent_dim) # node, aggregated_messages

    def forward(self, x, edge_index, edge_attr):
        # Update Edges
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_attr = edge_attr + self.edge_mlp(edge_input) # Residual connection
        
        # Update Nodes
        messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = x + self.node_mlp(torch.cat([x, messages], dim=-1))
        
        return x, edge_attr

    def message(self, edge_attr):
        return edge_attr


# Hybrid Wrapper Model (ViT + GNN)
class HybridDrapeModel(nn.Module):
    def __init__(self, embed_dim=128, smpl_dim=10, mat_dim=4, gnn_layers=10):
        super().__init__()
        self.vit = StyleViT_DINO(embed_dim=embed_dim)
        
        # Node Input: 3 (XYZ pos) + 128 (Style) + 10 (SMPL) + 4 (Material params)
        node_in_dim = 3 + embed_dim + smpl_dim + mat_dim
        # Edge Input: 3 (Relative distance: dx, dy, dz) + 1 (Magnitude)
        edge_in_dim = 4 
        latent_dim = 128
        
        self.node_encoder = build_mlp(node_in_dim, latent_dim)
        self.edge_encoder = build_mlp(edge_in_dim, latent_dim)
        
        self.processor = nn.ModuleList([
            MeshBlock(latent_dim) for _ in range(gnn_layers)
        ])
        
        # Decoder outputs [Delta X, Delta Y, Delta Z]
        self.decoder = nn.Linear(latent_dim, 3)

    def forward(self, data):
        """
        data is a PyTorch Geometric Batch object containing:
        - data.image: [B, 3, 224, 224]
        - data.pos: [Total_Nodes, 3] (Scaled Template Vertices)
        - data.smpl: [B, 10]
        - data.mat: [B, 4]
        - data.edge_index: [2, Total_Edges]
        - data.edge_attr: [Total_Edges, 4]
        - data.batch: [Total_Nodes] (Maps node to its specific graph/image in the batch)
        """
        # Image -> ViT Embedding
        style_emb = self.vit(data.image) # Shape: [B, 128]

        # Broadcast Globals to Nodes
        # PyG flattens all graphs into one giant graph
        # 'data.batch' tells us which node belongs to which image in the batch
        node_style = style_emb[data.batch] # Shape: [Total_Nodes, 128]
        node_smpl = data.smpl[data.batch]  # Shape: [Total_Nodes, 10]
        node_mat = data.mat[data.batch]    # Shape: [Total_Nodes, 4]

        # Concatenate Node Features
        x = torch.cat([data.pos, node_style, node_smpl, node_mat], dim=-1)

        # GNN Forward Pass
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(data.edge_attr)

        for layer in self.processor:
            x, edge_attr = layer(x, data.edge_index, edge_attr)

        # Output Displacements
        predicted_deltas = self.decoder(x)
        return predicted_deltas