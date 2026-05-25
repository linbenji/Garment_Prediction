# HybridDrapeNet — Physics-Informed 3D Garment Drape Prediction

> Predicts how a garment physically drapes on a specific body at a specific size — in milliseconds, with no physics simulation at inference.

**Jin Chung · Ben Lin · Antonio Tagliatti** — Northeastern University

---

## Overview

Most virtual try-on systems answer the question *"what would I look like wearing this?"* by editing a 2D image. HybridDrapeNet answers a harder and more commercially valuable question: **"how will this garment actually fit me?"**

The model takes a rendered shirt image, a 3D garment template mesh, SMPL body shape parameters, fabric physics properties, and a garment size encoding — and outputs a per-vertex 3D displacement field. Adding that displacement to the template mesh gives the final draped garment. No physics simulation required at inference time.

This is the physics oracle layer that sits underneath a full virtual try-on pipeline. The predicted mesh is a queryable 3D object: you can compute per-region fabric clearance, identify tight vs. loose zones, derive a fit score, or render it from any angle via a downstream 3DGS or NeRF renderer.

---

## Qualitative Results

**Front view** — ground truth vs. v3.5 vs. v4.5 (post-processed)

![Front view: ground truth, v3.5, v4.5](docs/renders_front.png)

**Back view** — ground truth vs. v3.5 vs. v4.5 (post-processed)

![Back view: ground truth, v3.5, v4.5](docs/renders_back.png)

---

## Results

| Condition | MVE (mm) |
|---|---|
| Train set (seen body + seen material) | ~4.51 |
| Val — unseen material (denim, canvas) | ~4.54 |
| Val — seen body + seen material | ~8.57 |
| Val — unseen body (extreme body types) | ~21.2 |

A shirt is ~700mm tall. The 4.51mm train MVE represents ~0.6% of garment height. The near-identical unseen material score (4.54mm) shows the model learned generalizable physics rather than memorizing fabric texture — a model that memorized training fabrics would fail on held-out denim/canvas.

The unseen body error (~21mm) is the clearest remaining weakness and the primary target for future work.

---

## Architecture

The model chains a vision encoder into a graph neural network, with fabric and body conditioning injected at each GNN layer via FiLM modulation.

```
Input image
    │
    ▼
DINOv2-S/14 (frozen + LoRA rank=8, last 4 blocks)
    │  CLS token + 196 patch tokens (384-dim each)
    │
    ├─── FiLM scale/shift → GNN layer 1
    ├─── FiLM + patch cross-attn → GNN layer 3
    ├─── FiLM scale/shift → GNN layers 2,4,5,6
    ├─── FiLM + patch cross-attn → GNN layer 7
    └─── FiLM scale/shift → GNN layer 8

Template mesh (3D pos + UV + normals)
+ SMPL betas (body shape)
+ Fabric physics (shear stiffness, density, buckling)
+ Size encoding [width_%, height_%]
    │
    ▼
MeshGraphNet — 8 layers, 128-dim latent
    │
    ▼
Vertex displacement field (14117, 3) in mm
    │
    ▼
template_mesh + displacement = draped garment mesh
```

**Model versions:**

| Version | Backbone | Injection | Key change |
|---|---|---|---|
| v2 (baseline) | ViT-S/16, frozen | Node concat | — |
| v3 | DINOv2-S/14, frozen | FiLM per layer | DINOv2 + FiLM |
| v3.5 | DINOv2-S/14, frozen | FiLM + collision loss | Asymmetric edge strain, collision penalty |
| v4 | DINOv2 + LoRA rank=4 | FiLM + CLS cross-attn (layers 3,7) | LoRA fine-tuning |
| v4.5 | DINOv2 + LoRA rank=8 | FiLM + 196 patch cross-attn (layers 3,7) | Full patch attention |

---

## Dataset

4,500 physics simulations generated in **Marvelous Designer**, rendered to 72,000 images via pyrender.

| Axis | Details |
|---|---|
| Bodies | 25 SMPL male bodies, 5'4"–6'4", grid-sampled |
| Fabrics | 12 presets across 6 families (light/medium/heavy × knit/woven) |
| Sizes | S, M, L, XL, XXL |
| Poses | Lean pose (A-pose variant) |
| Rendered images | 8 camera angles × 6 shirt colors = 72K images at 512×512 |

**Two-axis train/val/test split** — held out along both body and material axes independently:

- Held-out material: `heavy_woven` (denim, canvas) — never seen during training
- Held-out bodies: 2 extreme body types (tallest + heaviest)
- Hardest test condition: unseen body + unseen material simultaneously

**Fabric families:**

| Family | Fabrics | Split |
|---|---|---|
| Light Knit | jersey, mesh_tulle | Train |
| Medium Knit | french_terry, pique | Train |
| Heavy Knit | fleece, neoprene_scuba | Train |
| Light Woven | chiffon, organza | Train |
| Medium Woven | satin, flannel | Train |
| Heavy Woven | denim, canvas | **Held out** |

---

## Loss Functions

Six loss terms trained jointly:

| Term | What it measures | Why it exists |
|---|---|---|
| Drape MVE | Mean squared vertex position error | Primary shape signal |
| Edge strain | Change in mesh edge lengths vs. ground truth | Prevents fabric "melting" |
| Normal consistency | Angular difference between neighboring face normals | Forces correct crease patterns |
| Bending energy | Difference in bending angles between adjacent faces | Distinguishes inward vs. outward folds |
| Collision penalty | Vertices that penetrate the body mesh | Prevents body clipping on large sizes |
| Classification CE | Cross-entropy over 6 fabric families (auxiliary) | Forces DINOv2 to encode material texture |

---

## Repo Structure

```
Garment_Prediction/
├── src/                    # Training, model, dataloader, eval code
├── data/                   # Dataset scripts and preprocessing (steps 1–4)
├── models/                 # Saved model checkpoints
├── notebooks/              # Exploratory analysis and visualization
├── results/                # Evaluation outputs, error heatmaps, training curves
└── docs/                   # Presentation slides and write-up
```

---

## Setup

```bash
git clone https://github.com/linbenji/Garment_Prediction.git
cd Garment_Prediction
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.x, PyTorch Geometric, timm (DINOv2), wandb

Supports Mac (MPS), Windows (CUDA), and Linux (CUDA). Training runs were conducted on Vast.ai with an RTX 4090.

---

## Training

```bash
python src/train.py \
  --model_version v4 \
  --batch_size 4 \
  --gnn_layers 8 \
  --epochs 100 \
  --use_normal_consistency \
  --use_bending_energy \
  --use_collision_penalty
```

Metrics logged to W&B. Key hyperparameters: `batch_size=4`, `gnn_layers=8` were the consistent winners across ablation runs.

---

## Evaluation

```bash
python src/eval.py --checkpoint models/v4_best.pt --split test
```

Reports MVE, P90 vertex error, Chamfer distance, Hausdorff distance, voxel IoU, edge strain, normal consistency, and fabric classification accuracy — across all four generalization conditions.

---

## How This Differs From Diffusion-Based Try-On

Current virtual try-on systems (Google VTO, Zara, DiffFit, FIT) produce photorealistic 2D images but have no 3D structure at inference. They cannot answer whether a garment fits — only what it would look like if it did.

HybridDrapeNet outputs an actual 3D mesh. That means:

- **Per-region fit analysis** — compute fabric clearance from body mesh at every vertex
- **Size recommendation** — identify which size minimizes strain on the tightest region
- **Downstream composability** — feed the mesh into a 3DGS renderer for any viewpoint, any lighting
- **Extrapolation** — query body/size combinations not seen in training without requiring new rendered pairs

The natural extension of this work is using the predicted mesh as the geometry input to a diffusion renderer (e.g. via normal maps or depth conditioning), producing both physically correct drape and photorealistic output.

---

## Related Work

- [TailorNet](https://arxiv.org/abs/2003.04583) — pose/shape/style garment prediction
- [HOOD](https://arxiv.org/abs/2212.07242) — physics-based self-supervised cloth simulation
- [FIT](https://arxiv.org/abs/2604.08526) — fit-aware virtual try-on dataset (SIGGRAPH 2026)
- [Fully Convolutional GNN Try-On](https://arxiv.org/abs/2009.04592) — topology-agnostic garment draping
- [GarmentCode](https://github.com/iamittttttt/GarmentCode) — parametric garment generation

---

## Future Work

- **Pose generalization** — extend dataset to multiple SMPL poses; condition GNN on pose parameters
- **Sim-to-real gap** — domain randomization + real-image residual adapter
- **Body generalization** — encode body surface proximity as GNN node features to address ~21mm unseen body error
- **Diffusion rendering layer** — wrap mesh output in a photorealistic image renderer
- **Garment type expansion** — extend beyond shirts to pants, jackets, dresses

---

## License

MIT
