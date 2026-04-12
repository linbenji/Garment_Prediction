import os
import torch
import pandas as pd
from PIL import Image
from torch_geometric.data import Dataset, Data
from torchvision import transforms

class CustomGarmentDataset(Dataset):
    def __init__(self, root_dir, csv_file, split='train', transform=None):
        """
        root_dir: Base directory of the dataset
        csv_file: The master index file
        split: 'train', 'val', or 'test'
        """
        super().__init__(root_dir, transform)
        self.root_dir = root_dir
        
        # Load the CSV
        full_df = pd.read_csv(os.path.join(root_dir, csv_file))
        
        # Filter by the requested split (train/val/test)
        self.data_frame = full_df[full_df['split_group'] == split].reset_index(drop=True)
        
        # Standard Image Transforms for ViT (224x224, normalized)
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def len(self):
        return len(self.data_frame)

    def get(self, idx):
        # Fetch row data
        row = self.data_frame.iloc[idx]
        
        # ---------------------------
        # A. LOAD IMAGE (ViT Input)
        # ---------------------------
        img_path = os.path.join(self.root_dir, 'images', row['image_name'])
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.img_transform(image)
        
        # ---------------------------
        # B. LOAD SCALARS (Metadata)
        # ---------------------------
        # Assuming SMPL data is saved as small .pt or numpy arrays, or directly in CSV
        target_smpl = torch.tensor(eval(row['target_smpl_array']), dtype=torch.float32)
        
        # Material parameters (e.g., stiffness, friction, density, weight)
        materials = torch.tensor([
            row['mat_stiffness'],
            row['mat_friction'],
            row['mat_density'],
            row['mat_weight']
        ], dtype=torch.float32)
        
        # ---------------------------
        # C. LOAD GRAPHS (GNN Data)
        # ---------------------------
        # 1. The Base Template (e.g., "Large" base mesh)
        template_path = os.path.join(self.root_dir, 'templates', f"template_{row['template_size']}.pt")
        template_data = torch.load(template_path)
        
        template_pos = template_data['pos']             # [Num_Nodes, 3]
        edge_index = template_data['edge_index']        # [2, Num_Edges]
        edge_attr = template_data['edge_attr']          # [Num_Edges, 4]
        
        # 2. The Ground Truth Draped Mesh
        gt_path = os.path.join(self.root_dir, 'ground_truth', row['gt_mesh_name'])
        gt_pos = torch.load(gt_path)                    # [Num_Nodes, 3]
        
        # ---------------------------
        # D. PACKAGE INTO PyG DATA OBJECT
        # ---------------------------
        data = Data(
            image=image_tensor,       # Input to ViT
            pos=template_pos,         # Input to GNN (Starting positions)
            edge_index=edge_index,    # GNN connectivity
            edge_attr=edge_attr,      # GNN edge features (rest lengths)
            smpl=target_smpl,         # GNN Node context
            mat=materials,            # GNN Global context
            y=gt_pos                  # Ground Truth for Loss Calculation
        )
        
        return data