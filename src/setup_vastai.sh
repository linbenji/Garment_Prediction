#!/bin/bash
# setup_vastai.sh
#
# One-shot setup script for a fresh Vast.ai instance.
# Run once after connecting via SSH:
#   bash setup_vastai.sh
#
# Assumes:
#   - dataset.zip and src.zip are already uploaded to /workspace/
#   - Instance is running pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
#     or similar PyTorch 2.6 + CUDA 12.4 image

set -e   # exit on any error

echo "========================================================"
echo "Garment Drape Prediction — Vast.ai Setup"
echo "========================================================"

WORKSPACE=/workspace
DATASET_ZIP=$WORKSPACE/dataset.zip
SRC_ZIP=$WORKSPACE/src.zip

# ── Step 1: Verify CUDA ───────────────────────────────────────────────────────
echo ""
echo "── Step 1: Verify environment ──"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# ── Step 2: Extract dataset ───────────────────────────────────────────────────
echo ""
echo "── Step 2: Extract dataset ──"

if [ -f "$DATASET_ZIP" ]; then
    echo "  Extracting dataset.zip..."
    cd $WORKSPACE
    unzip -q dataset.zip
    echo "  Done. Contents:"
    ls -lh $WORKSPACE/batch_1500_lean/
else
    echo "  WARNING: dataset.zip not found at $DATASET_ZIP"
    echo "  Upload it with: scp dataset.zip user@instance:/workspace/"
fi

# ── Step 3: Extract source files ──────────────────────────────────────────────
echo ""
echo "── Step 3: Extract source files ──"

if [ -f "$SRC_ZIP" ]; then
    echo "  Extracting src.zip..."
    cd $WORKSPACE
    unzip -q src.zip -d src/
    echo "  Done. Contents:"
    ls -lh $WORKSPACE/src/
else
    echo "  WARNING: src.zip not found at $SRC_ZIP"
    echo "  Upload it with: scp src.zip user@instance:/workspace/"
fi

# ── Step 4: Install PyG dependencies ─────────────────────────────────────────
echo ""
echo "── Step 4: Install PyG dependencies ──"

# Detect PyTorch version for correct wheel URL
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_TAG=$(python -c "import torch; v=torch.version.cuda; print('cu'+''.join(v.split('.')))")
echo "  PyTorch: $TORCH_VERSION  CUDA tag: $CUDA_TAG"

pip install torch_geometric --quiet
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html --quiet
echo "  PyG installed"

# ── Step 5: Install remaining dependencies ────────────────────────────────────
echo ""
echo "── Step 5: Install remaining dependencies ──"
pip install pandas pillow wandb tensorboard tqdm matplotlib --quiet
echo "  Dependencies installed"

# ── Step 6: Verify PyG ────────────────────────────────────────────────────────
echo ""
echo "── Step 6: Verify installation ──"
python -c "
import torch
import torch_geometric
import pandas
import PIL
print(f'torch:          {torch.__version__}')
print(f'torch_geometric:{torch_geometric.__version__}')
print(f'CUDA:           {torch.cuda.is_available()}')
print(f'GPU:            {torch.cuda.get_device_name(0)}')
print(f'VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── Step 7: Update paths in train script ─────────────────────────────────────
echo ""
echo "── Step 7: Patching paths for Vast.ai ──"

TRAIN_SCRIPT=$WORKSPACE/src/train_v2.py

if [ -f "$TRAIN_SCRIPT" ]; then
    # Replace Windows paths with Linux paths
    sed -i 's|DATA_ROOT.*=.*r"C:\\.*batch_1500_lean"|DATA_ROOT  = "/workspace/batch_1500_lean"|g' $TRAIN_SCRIPT
    sed -i 's|RUNS_DIR.*=.*r"C:\\.*runs"|RUNS_DIR   = "/workspace/runs"|g' $TRAIN_SCRIPT
    echo "  Paths updated in $TRAIN_SCRIPT"
else
    echo "  WARNING: $TRAIN_SCRIPT not found — update DATA_ROOT and RUNS_DIR manually"
fi

EVAL_SCRIPT=$WORKSPACE/src/eval.py
if [ -f "$EVAL_SCRIPT" ]; then
    sed -i 's|DATA_ROOT.*=.*r"C:\\.*batch_1500_lean"|DATA_ROOT = "/workspace/batch_1500_lean"|g' $EVAL_SCRIPT
    sed -i 's|RUNS_DIR.*=.*r"C:\\.*runs"|RUNS_DIR  = "/workspace/runs"|g' $EVAL_SCRIPT
    echo "  Paths updated in $EVAL_SCRIPT"
fi

# ── Step 8: wandb login ───────────────────────────────────────────────────────
echo ""
echo "── Step 8: wandb login ──"
echo "  Run: wandb login"
echo "  Then paste your API key from https://wandb.ai/authorize"
echo "  (or set WANDB_API_KEY env variable before training)"
echo ""
echo "  To set API key without interactive login:"
echo "  export WANDB_API_KEY=your_key_here"

# ── Step 9: Verify dataset ────────────────────────────────────────────────────
echo ""
echo "── Step 9: Verify dataset ──"
python -c "
import os
root = '/workspace/batch_1500_lean'
checks = [
    ('dataset_index.csv',        os.path.join(root, 'dataset_index.csv')),
    ('per_vertex_std.npy',       os.path.join(root, 'per_vertex_std.npy')),
    ('template/edge_index.npy',  os.path.join(root, 'template', 'edge_index.npy')),
    ('meshes/ (1475 .pt files)', os.path.join(root, 'meshes')),
    ('images/ (72000 .png)',     os.path.join(root, 'images')),
]
all_ok = True
for label, path in checks:
    exists = os.path.exists(path)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path) if not f.startswith('.')])
        status = f'✓  ({count} files)' if exists else '✗  MISSING'
    else:
        status = '✓' if exists else '✗  MISSING'
    if not exists:
        all_ok = False
    print(f'  {status}  {label}')
print()
print('Dataset OK' if all_ok else 'WARNING: some files missing')
"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "Setup complete. To start training:"
echo ""
echo "  cd /workspace/src"
echo "  wandb login   # if not already logged in"
echo "  python train_v2.py --no-debug"
echo ""
echo "Monitor training at: https://wandb.ai"
echo "========================================================"