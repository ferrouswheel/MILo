# CLAUDE.md

## Project Overview

MILo (Mesh-In-the-Loop) is a 3D Gaussian Splatting system for surface reconstruction from images. It extracts high-quality meshes by maintaining bidirectional consistency between Gaussian representations and mesh surfaces during optimization.

**Publication**: SIGGRAPH Asia 2025 - Journal Track (TOG)

## Key Commands

```bash
# Training (main entry point)
python train.py -s <colmap_dataset_path> -m <output_dir> --imp_metric outdoor --rasterizer radegs

# High-res mesh training
python train.py -s <path> -m <output> --dense_gaussians --mesh_config highres --decoupled_appearance

# Mesh extraction after training
python mesh_extract_sdf.py -s <data_path> -m <model_path> --rasterizer radegs

# Alternative mesh extraction methods
python mesh_extract_integration.py -s <path> -m <output>
python mesh_extract_regular_tsdf.py -s <path> -m <output>

# Rendering novel views
python render.py -s <data_path> -m <model_path>
```

## Repository Structure

```
milo/
├── arguments/          # CLI argument definitions
├── scene/              # Scene loading, GaussianModel, cameras, mesh rasterization
├── gaussian_renderer/  # Rendering backends (radegs.py, gof.py, pgsr.py)
├── functional/         # Differentiable mesh extraction pipeline
│   ├── pivots.py       # Surface Gaussian sampling
│   ├── delaunay.py     # Delaunay triangulation
│   ├── sdf.py          # SDF computation
│   └── mesh.py         # Marching tetrahedra mesh extraction
├── regularization/     # Training regularization (mesh-in-the-loop, depth-order)
├── configs/            # YAML configs for mesh resolution presets
├── utils/              # Losses, geometry utilities, logging
└── lpipsPyTorch/       # LPIPS loss implementation
```

**Root-level scripts:**
- `train.py` - Main training loop (30K iterations)
- `train_regular_densification.py` - Alternative training with traditional densification
- `mesh_extract_*.py` - Mesh extraction variants
- `render.py` - Novel view rendering
- `convert.py` - COLMAP dataset conversion

## Core Architecture

### Training Pipeline (`train.py`)

1. Load COLMAP dataset and initialize Gaussians from point cloud
2. Iterate 18K steps (default, configurable via `--iterations`):
   - Render Gaussians (depth + normals via RaDe-GS/GOF rasterizer)
   - Compute photometric loss (L1 + SSIM)
   - Apply depth-normal consistency (iter >= 3000)
   - Apply mesh-in-the-loop regularization (iter 8001-18000):
     - Sample surface Gaussians -> Delaunay triangulation -> compute SDF -> extract mesh
     - Render mesh and compute depth/normal consistency losses
   - Densification and pruning (until iter 15000)
3. Extract final mesh

### GaussianModel (`scene/gaussian_model.py`)

Core parameters: means, scales, rotations, opacities, SH features. Supports:
- Learnable SDF values for mesh-in-the-loop
- Optional appearance network for exposure handling
- 3D Mip filtering for quality/efficiency

### Mesh Extraction (`functional/`)

Differentiable pipeline:
1. `sample_gaussians_on_surface()` - Select ~600K surface Gaussians
2. `compute_delaunay_triangulation()` - Tetrahedralize Gaussian pivots
3. `compute_initial_sdf_values()` - Evaluate TSDF at pivots
4. `extract_mesh()` - Marching tetrahedra -> triangle mesh

All operations are differentiable, enabling gradient flow from mesh to Gaussians.

## Important Files

| File | Purpose |
|------|---------|
| `train.py` | Main training entry point |
| `scene/gaussian_model.py` | Gaussian parameters and operations |
| `regularization/regularizer/mesh.py` | Mesh-in-the-loop loss computation |
| `functional/mesh.py` | Differentiable marching tetrahedra |
| `gaussian_renderer/radegs.py` | Main rendering backend |
| `configs/mesh/default.yaml` | Default mesh extraction settings |

## Configuration

Mesh presets in `configs/mesh/`:
- `default.yaml` - ~2M vertices, standard quality
- `highres.yaml` - ~9M vertices, high quality
- `lowres.yaml` - ~500K vertices, smaller output
- `verylowres.yaml` - ~250K vertices, minimal size

Key training arguments:
- `--imp_metric indoor/outdoor` - Scene type for importance sampling
- `--rasterizer radegs/gof` - Rendering backend
- `--dense_gaussians` - More Gaussians, fewer mesh vertices
- `--mesh_config <preset>` - Mesh resolution preset
- `--decoupled_appearance` - Better exposure handling
- `--no_mesh_regularization` - Disable mesh-in-the-loop

## Submodules

Located in `submodules/`:
- `diff-gaussian-rasterization*` - Gaussian rendering (original, MS2, GOF variants)
- `simple-knn` - KNN for densification
- `fused-ssim` - CUDA SSIM kernel
- `tetra_triangulation` - Delaunay + marching tetrahedra (C++ with CGAL)
- `nvdiffrast` - Differentiable mesh rasterization
- `Depth-Anything-V2` - Optional monocular depth estimation

## Build Notes

Requires CUDA. Key dependencies:
- PyTorch 2.3.1+
- Open3D, Trimesh for mesh processing
- CGAL, GMP for tetra_triangulation submodule

Install order: PyTorch -> pip requirements -> submodules (each has setup.py)
