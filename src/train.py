import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from models import HybridDrapeModel
from data_loader import CustomGarmentDataset

# -----------------------------------------
# Custom Physics Loss Function
# -----------------------------------------
def compute_edge_strain_loss(pred_pos, gt_pos, edge_index):
    """
    Penalizes the model if the edges (seams/fabric structure) stretch 
    more or less than the ground truth mesh.
    """
    row, col = edge_index
    
    # Calculate lengths of edges in the predicted mesh
    pred_edge_vectors = pred_pos[row] - pred_pos[col]
    pred_edge_lengths = torch.norm(pred_edge_vectors, dim=1)
    
    # Calculate lengths of edges in the ground truth mesh
    gt_edge_vectors = gt_pos[row] - gt_pos[col]
    gt_edge_lengths = torch.norm(gt_edge_vectors, dim=1)
    
    # Mean Squared Error between the edge lengths
    strain_loss = nn.functional.mse_loss(pred_edge_lengths, gt_edge_lengths)
    return strain_loss

# -----------------------------------------
# Main Training Loop
# -----------------------------------------
def train_single_shot(resume_checkpoint=None):
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Ensure checkpoint directory exists
    os.makedirs('checkpoints', exist_ok=True)

    # 2. Load Model
    model = HybridDrapeModel(embed_dim=128, smpl_dim=10, mat_dim=4, gnn_layers=10)
    model.to(device)

    # 3. Setup Optimizer and Hyperparameters
    # We use a lower learning rate for the ViT because it is pretrained
    optimizer = AdamW([
        {'params': model.vit.parameters(), 'lr': 1e-5},  # Finetune slowly
        {'params': model.node_encoder.parameters()},
        {'params': model.edge_encoder.parameters()},
        {'params': model.processor.parameters()},
        {'params': model.decoder.parameters()}
    ], lr=1e-4) # Higher LR for the GNN

    criterion = nn.MSELoss()

    # Hyperparameters
    epochs = 100
    batch_size = 4
    physics_loss_weight = 0.1 # Balances vertex position vs. fabric stretch
    start_epoch = 0 # Default starting point

    # Resume Logic
    if resume_checkpoint is not None:
        if os.path.isfile(resume_checkpoint):
            print(f"Loading checkpoint: '{resume_checkpoint}'")
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1 # Start at the next epoch
            
            print(f"Successfully restored state. Resuming from epoch {start_epoch}.")
        else:
            print(f"Error: Checkpoint '{resume_checkpoint}' not found! Starting from scratch.")

    # 4. Data Loaders
    print("Loading datasets...")
    #=======================================================================================================================
    # UPDATE THESE PATHS to match actual directory structure!!!
    #=======================================================================================================================
    root_data_path = './custom_garment_dataset'
    csv_file_name = 'dataset_index.csv'

    train_dataset = CustomGarmentDataset(root_data_path, csv_file=csv_file_name, split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CustomGarmentDataset(root_data_path, csv_file=csv_file_name, split='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward Pass: End-to-End
            predicted_deltas = model(data)
            
            # The final predicted mesh = Template Pos + Predicted Offsets
            predicted_positions = data.pos + predicted_deltas
            
            # Loss: Mean Squared Error against Ground Truth Vertices
            # Note: data.y contains the ground truth [Total_Nodes, 3] positions
            loss_pos = criterion(predicted_positions, data.y)
            
            # OPTIONAL: Physics Strain Loss (Is the fabric stretched realistically?)
            loss_strain = compute_edge_strain_loss(predicted_positions, data.y, data.edge_index)

            # Combined Loss
            loss = loss_pos + (physics_loss_weight * loss_strain)
            
            # Backward Pass
            loss.backward()
            
            # Gradient Clipping (Crucial for GNN stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.6f}")

        # ------------------
        # VALIDATION (Every 5 epochs)
        # ------------------
        if (epoch + 1) % 5 == 0:
            model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for val_data in val_loader:
                    val_data = val_data.to(device)
                    val_deltas = model(val_data)
                    val_positions = val_data.pos + val_deltas

                    v_loss_pos = criterion(val_positions, val_data.y)
                    v_loss_strain = compute_edge_strain_loss(val_positions, val_data.y, val_data.edge_index)
                    v_loss = v_loss_pos + (physics_loss_weight * v_loss_strain)
                    total_val_loss += v_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"  --> Validation Loss: {avg_val_loss:.6f}")
            
            # Save Checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss
            }
            checkpoint_path = f"checkpoints/hybrid_model_ep{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"  --> Saved full checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid Drape Model")
    parser.add_argument('--resume', type=str, default=None, 
                        help="Path to checkpoint file to resume training (e.g., checkpoints/hybrid_model_ep50.pth)")
    
    args = parser.parse_args()
    
    # Run the training, passing the checkpoint path if provided
    train_single_shot(resume_checkpoint=args.resume)