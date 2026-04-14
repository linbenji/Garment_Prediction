import os
import torch
import random
import pandas as pd
from PIL import Image
from torch_geometric.data import Dataset, Data
from torchvision import transforms

class CustomGarmentDataset(Dataset):
    def __init__(self, root_dir, sims_csv, shirts_csv, split='train', transform=None):
        """
        root_dir: Base directory of the dataset
        csv_file: The master index file
        split: 'train', 'val', or 'test'
        """
        super().__init__(root_dir, transform)
        self.root_dir = root_dir
        
        # 1. Load both CSV files into Pandas DataFrames
        sims_df = pd.read_csv(os.path.join(root_dir, sims_csv))
        shirts_df = pd.read_csv(os.path.join(root_dir, shirts_csv))
        
        # 2. Filter simulations by train/val/test split
        sims_df = sims_df[sims_df['split_group'] == split]
        
        # 3. Merge them into one flat table in RAM
        self.df = pd.merge(sims_df, shirts_df, on='shirt_id', how='left')
        
        # Reset the index so PyTorch can count from 0 to N smoothly
        self.df = self.df.reset_index(drop=True)
        
        # Standard Image Transforms for ViT (224x224, normalized)
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        # Define the available camera angles
        self.camera_angles = ['front', 'left_front_45', 'right_back_45', 'back', 'left_back_45', 'right_back_45', 'left', 'right']

    def len(self):
        # The epoch length is exactly the number of target meshes
        return len(self.df)

    def get(self, idx):
        # ---------------------------------------------------
        # STEP 1: DEFINE THE TARGET (What the GNN must predict)
        # ---------------------------------------------------
        row = self.df.iloc[idx]
        
        # Load Target GNN Data
        gt_path = os.path.join(self.root_dir, 'ground_truth', row['gt_mesh_name'])
        gt_pos = torch.load(gt_path) 
        
        template_path = os.path.join(self.root_dir, 'templates', f"template_{row['size']}.pt")
        template_data = torch.load(template_path)
        
        target_smpl = torch.tensor(eval(row['smpl_array']), dtype=torch.float32)
        target_materials = torch.tensor([row['mat_stiff'], row['mat_fric']], dtype=torch.float32)

        # ---------------------------------------------------
        # STEP 2: DYNAMICALLY SAMPLE THE INPUT (What the ViT sees)
        # ---------------------------------------------------
        # Find all rows that are the SAME shirt design & material, 
        # but could be worn by a DIFFERENT body or be a DIFFERENT size.
        valid_inputs = self.df[
            (self.df['shirt_design_id'] == row['shirt_design_id']) &
            (self.df['material_family'] == row['material_family'])
        ]
        
        # Randomly pick one row to act as the ViT's visual input for this specific step
        input_row = valid_inputs.sample(n=1).iloc[0]
        
        # ---------------------------------------------------
        # STEP 3: RANDOM CAMERA ANGLE SELECTION
        # ---------------------------------------------------
        # Pick a random viewing angle for the selected input shirt
        selected_angle = random.choice(self.camera_angles)
        
        # Assuming your images are named like: "shirtA_body1_front.jpg"
        img_name = f"{input_row['base_image_name']}_{selected_angle}.jpg"
        img_path = os.path.join(self.root_dir, 'images', img_name)
        
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.img_transform(image)
        
        # Load Input Metadata for the ViT
        input_smpl = torch.tensor(eval(input_row['smpl_array']), dtype=torch.float32)
        input_materials = torch.tensor([input_row['mat_stiff'], input_row['mat_fric']], dtype=torch.float32)

        # ---------------------------------------------------
        # STEP 4: PACKAGE THE DYNAMIC PAIR
        # ---------------------------------------------------
        data = Data(
            image=image_tensor,       # ViT gets the random Input view
            in_smpl=input_smpl,       # ViT gets the Input SMPL
            in_mat=input_materials,   # ViT gets the Input Material
            
            pos=template_data['pos'], # GNN gets Target Template Geometry
            edge_index=template_data['edge_index'], 
            tgt_smpl=target_smpl,     # GNN gets Target SMPL
            tgt_mat=target_materials, # GNN gets Target Material
            
            y=gt_pos                  # Loss compares against Target GT Mesh
        )
        
        return data