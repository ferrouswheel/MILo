# MILo: Mesh-In-the-Loop Gaussian Splatting

A comprehensive technical overview of the MILo system for detailed and efficient surface reconstruction.

## Table of Contents

1. [Introduction](#introduction)
2. [Repository Layout](#repository-layout)
3. [The Optimization Pipeline](#the-optimization-pipeline)
4. [Core Modules](#core-modules)
5. [Mesh-In-the-Loop Regularization](#mesh-in-the-loop-regularization)
6. [Configuration System](#configuration-system)
7. [Extending MILo](#extending-milo)

---

## Introduction

MILo builds on 3D Gaussian Splatting (3DGS) to produce high-quality surface meshes alongside novel view synthesis. The key innovation is **bidirectional consistency**: during optimization, a differentiable mesh is extracted from the Gaussians and used to regularize them, ensuring Gaussians concentrate on actual surfaces rather than floating in space.

### Key Capabilities

- Photorealistic novel view rendering via Gaussian splatting
- High-quality mesh extraction with significantly fewer vertices than grid-based methods
- Differentiable pipeline allowing gradients to flow from mesh losses to Gaussian parameters
- Scalable to unbounded outdoor scenes (Delaunay-based rather than voxel-based)

---

## Repository Layout

```
MILo/
├── milo/
│   ├── arguments/              # Command-line argument definitions
│   │   └── __init__.py         # ModelParams, PipelineParams, OptimizationParams
│   │
│   ├── scene/                  # Scene representation
│   │   ├── __init__.py         # Scene class (dataset + model container)
│   │   ├── gaussian_model.py   # GaussianModel - core 3DGS parameters
│   │   ├── dataset_readers.py  # COLMAP/Blender dataset loading
│   │   ├── cameras.py          # Camera intrinsics/extrinsics
│   │   ├── mesh.py             # nvdiffrast wrapper for mesh rendering
│   │   └── appearance_network.py  # Optional per-image appearance encoding
│   │
│   ├── gaussian_renderer/      # Gaussian rasterization backends
│   │   ├── radegs.py           # RaDe-GS rasterizer (default, fast)
│   │   ├── gof.py              # Gaussian Opacity Fields rasterizer
│   │   └── pgsr.py             # Additional rendering utilities
│   │
│   ├── functional/             # Differentiable mesh extraction
│   │   ├── pivots.py           # Surface Gaussian sampling
│   │   ├── delaunay.py         # Delaunay triangulation
│   │   ├── sdf.py              # SDF value computation
│   │   ├── mesh.py             # Marching tetrahedra
│   │   └── func_utils.py       # Utility functions
│   │
│   ├── regularization/         # Training regularization
│   │   ├── regularizer/
│   │   │   ├── mesh.py         # Mesh-in-the-loop regularization
│   │   │   └── depth_order.py  # Optional depth-order loss
│   │   ├── sdf/
│   │   │   ├── learnable.py    # Learnable SDF with occupancy
│   │   │   ├── integration.py  # Integrated opacity field SDF
│   │   │   └── depth_fusion.py # TSDF-based SDF
│   │   └── depth/              # Depth-based regularization
│   │
│   ├── configs/                # YAML configuration files
│   │   ├── mesh/               # Mesh resolution presets
│   │   └── depth_order/        # Depth-order regularization configs
│   │
│   ├── utils/                  # Utility modules
│   │   ├── loss_utils.py       # L1, SSIM, combined losses
│   │   ├── geometry_utils.py   # 3D geometry operations
│   │   ├── camera_utils.py     # Camera transformations
│   │   ├── log_utils.py        # Logging and visualization
│   │   └── tetmesh.py          # Marching tetrahedra utilities
│   │
│   ├── lpipsPyTorch/           # LPIPS perceptual loss
│   ├── eval/                   # Evaluation scripts (TNT, DTU)
│   ├── blender/                # Blender addon for editing
│   └── scripts/                # Batch processing scripts
│
├── train.py                    # Main training entry point
├── train_regular_densification.py  # Alternative densification strategy
├── mesh_extract_sdf.py         # Mesh extraction with learnable SDF
├── mesh_extract_integration.py # Mesh extraction via integration
├── mesh_extract_regular_tsdf.py # Traditional TSDF extraction
├── render.py                   # Novel view rendering
├── convert.py                  # Dataset conversion utilities
│
└── submodules/                 # External dependencies (git submodules)
    ├── diff-gaussian-rasterization/      # Original 3DGS rasterizer
    ├── diff-gaussian-rasterization_ms/   # Mini-Splatting2 rasterizer
    ├── diff-gaussian-rasterization_gof/  # GOF rasterizer
    ├── simple-knn/                        # KNN for densification
    ├── fused-ssim/                        # CUDA SSIM kernel
    ├── tetra_triangulation/               # Delaunay + marching tet
    ├── nvdiffrast/                        # Differentiable mesh rasterization
    └── Depth-Anything-V2/                 # Monocular depth (optional)
```

---

## The Optimization Pipeline

### Overview

The training process runs for 18,000 iterations by default (using `configs/fast`), progressively enabling different regularization strategies:

```
Iteration Timeline:
├── 0-3000:     Warm-up, aggressive densification
├── 3000:       Enable depth-normal consistency, simplification
├── 8000:       Second simplification stage
├── 8001:       START mesh-in-the-loop regularization
├── 13001:      Stop SDF value reset
├── 15000:      End densification
├── 18000:      STOP mesh regularization, training complete (default)
```

Note: The base `OptimizationParams` defines 30K iterations, but `configs/fast` (the default) sets 18K.

### Per-Iteration Flow (`train.py`)

```python
for iteration in range(1, opt.iterations + 1):  # 18001 by default
    # 1. Update learning rates
    update_learning_rate(optimizer, iteration)

    # 2. Select random training view
    camera = select_random_camera(training_cameras)

    # 3. Render Gaussians
    render_output = render(
        camera, gaussians, pipe, background,
        return_depth=True, return_normal=True
    )
    image = render_output["render"]
    depth = render_output["depth"]
    normal = render_output["normal"]

    # 4. Compute photometric loss
    gt_image = camera.original_image
    Ll1 = l1_loss(image, gt_image)
    Lssim = ssim_loss(image, gt_image)
    loss = (1 - lambda_dssim) * Ll1 + lambda_dssim * (1 - Lssim)

    # 5. Depth-normal consistency (iter >= 3000)
    if iteration >= 3000:
        loss += depth_normal_consistency_loss(depth, normal)

    # 6. Mesh-in-the-loop regularization (iter 8001-18000)
    if 8001 <= iteration <= 18000:
        mesh_loss = compute_mesh_regularization(
            gaussians, camera, depth, normal, ...
        )
        loss += mesh_loss

    # 7. Backward pass
    loss.backward()

    # 8. Densification (iter < 15000)
    if iteration < 15000:
        densify_and_prune(gaussians, optimizer, ...)

    # 9. Optimizer step
    optimizer.step()
    optimizer.zero_grad()
```

### Gaussian Representation

Each Gaussian is parameterized by:
- **Position** (xyz): 3D mean location
- **Covariance** (scale + rotation): Anisotropic 3D shape
- **Opacity**: Alpha value for blending
- **Spherical Harmonics**: View-dependent color (degree 0-3)

Additional learnable parameters for mesh-in-the-loop:
- **Occupancy/SDF values**: Per-Gaussian surface indicators

### Densification Strategy

Gaussians are adaptively densified based on:
1. **Gradient magnitude**: High view-space gradients indicate under-reconstruction
2. **Scale**: Large Gaussians are split, small ones cloned
3. **Opacity**: Low-opacity Gaussians are pruned

The system uses an importance-based sampling strategy (`--imp_metric indoor/outdoor`) to focus densification on relevant scene regions.

---

## Core Modules

### GaussianModel (`scene/gaussian_model.py`)

The central class managing all Gaussian parameters:

```python
class GaussianModel:
    # Core parameters (nn.Parameter)
    _xyz: Tensor           # [N, 3] positions
    _features_dc: Tensor   # [N, 1, 3] DC spherical harmonics
    _features_rest: Tensor # [N, 15, 3] higher-order SH
    _scaling: Tensor       # [N, 3] log-scale
    _rotation: Tensor      # [N, 4] quaternion
    _opacity: Tensor       # [N, 1] logit-opacity

    # Optional for mesh-in-the-loop
    _occupancy: Tensor     # [N, 1] learnable SDF proxy

    def get_xyz(self) -> Tensor:
        """Returns positions with optional 3D Mip filtering."""

    def get_covariance(self, scaling_modifier=1.0) -> Tensor:
        """Computes 3D covariance matrices from scale and rotation."""

    def densify_and_prune(self, grads, min_opacity, max_screen_size):
        """Adaptive density control."""
```

### Rendering Backends (`gaussian_renderer/`)

Three rasterization backends with different trade-offs:

| Backend | File | Characteristics |
|---------|------|-----------------|
| RaDe-GS | `radegs.py` | Fast, accurate depth/normals, default choice |
| GOF | `gof.py` | Gaussian Opacity Fields, alternative depth computation |
| Original | via submodule | Baseline 3DGS rasterizer |

All backends return:
- `render`: RGB image [3, H, W]
- `depth`: Depth map [1, H, W]
- `normal`: Normal map [3, H, W]
- `alpha`: Accumulated opacity [1, H, W]

### Scene Loading (`scene/dataset_readers.py`)

Supports COLMAP sparse reconstructions:

```python
def readColmapSceneInfo(path, images, eval, llffhold=8):
    """
    Reads COLMAP reconstruction:
    - cameras_text/cameras_bin: Camera intrinsics
    - images_text/images_bin: Camera poses
    - points3D_text/points3D_bin: Initial point cloud

    Returns SceneInfo with cameras, point cloud, and scene bounds.
    """
```

---

## Mesh-In-the-Loop Regularization

The core innovation of MILo is the differentiable mesh extraction and regularization pipeline.

### Pipeline Overview

```
Gaussians → Sample Surface → Delaunay Triangulation → Compute SDF → Extract Mesh → Render → Loss
    ↑                                                                                         |
    └─────────────────────────────────────────── Gradients ──────────────────────────────────┘
```

### Step 1: Sample Surface Gaussians (`functional/pivots.py`)

```python
def sample_gaussians_on_surface(
    gaussians: GaussianModel,
    cameras: List[Camera],
    method: str = "surface",  # or "surface+opacity"
    max_points: int = 600_000
) -> Tensor:
    """
    Identifies Gaussians likely lying on the surface.

    Methods:
    - "surface": Importance-weighted sampling based on visibility
    - "surface+opacity": Additional opacity-based filtering

    Returns indices of selected Gaussians.
    """
```

### Step 2: Delaunay Triangulation (`functional/delaunay.py`)

```python
def compute_delaunay_triangulation(
    xyz: Tensor,           # [N, 3] Gaussian positions
    scales: Tensor,        # [N, 3] Gaussian scales
    max_vertices: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Extracts pivot points from Gaussians and computes Delaunay tetrahedralization.

    Pivots are placed at:
    - Gaussian centers
    - Offset positions along principal axes (based on scale)

    Returns:
    - vertices: [V, 3] pivot positions
    - tetrahedra: [T, 4] tetrahedral connectivity
    - gaussian_indices: mapping from vertices to source Gaussians
    """
```

Uses the `cpp.triangulate` function from the `tetra_triangulation` submodule (CGAL-based).

### Step 3: Compute SDF Values (`functional/sdf.py`)

```python
def compute_initial_sdf_values(
    vertices: Tensor,      # [V, 3] pivot positions
    cameras: List[Camera],
    depth_maps: List[Tensor],
    truncation: float = 0.1
) -> Tensor:
    """
    Computes truncated signed distance at each vertex using depth fusion.

    For each vertex:
    1. Project to visible camera views
    2. Sample rendered depth
    3. Compute signed distance to depth surface
    4. Aggregate across views with truncation

    Returns: [V] SDF values
    """
```

### Step 4: Extract Mesh (`functional/mesh.py`)

```python
def extract_mesh(
    vertices: Tensor,      # [V, 3] Delaunay vertices
    tetrahedra: Tensor,    # [T, 4] tetrahedra
    sdf_values: Tensor,    # [V] signed distance
    vertex_colors: Tensor, # [V, 3] colors from Gaussians
    filter_edges: bool = True
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Differentiable marching tetrahedra.

    For each tetrahedron, identifies surface crossings (sign changes in SDF)
    and interpolates vertex positions to create triangles.

    Returns:
    - mesh_vertices: [M, 3] surface vertices
    - mesh_faces: [F, 3] triangle connectivity
    - mesh_colors: [M, 3] interpolated colors
    """
```

### Step 5: Mesh Regularization Loss (`regularization/regularizer/mesh.py`)

```python
def compute_mesh_regularization(
    gaussians: GaussianModel,
    camera: Camera,
    gaussian_depth: Tensor,
    gaussian_normal: Tensor,
    mesh_state: dict,
    config: dict
) -> Tensor:
    """
    Computes mesh-based regularization losses.

    1. Extract mesh from current Gaussians (if update interval)
    2. Render mesh using nvdiffrast
    3. Compute losses:
       - Depth consistency: |mesh_depth - gaussian_depth|
       - Normal consistency: 1 - dot(mesh_normal, gaussian_normal)
       - Occupancy regularization (optional)

    Returns combined loss for backpropagation.
    """
```

### Learnable SDF (`regularization/sdf/learnable.py`)

The SDF values are optimized during training:

```python
class LearnableSDF:
    """
    Per-vertex learnable SDF values with reset strategies.

    Modes:
    - occupancy_shift: SDF = base_sdf + learnable_offset
    - density_shift: SDF from Gaussian density field

    Reset strategies:
    - ema: Exponential moving average toward TSDF values
    - hard: Direct reset to TSDF
    """
```

### Why This Works

1. **Gradient Flow**: nvdiffrast enables gradients from mesh rendering to flow back through marching tetrahedra to Gaussian parameters

2. **Surface Concentration**: Mesh losses penalize Gaussians that don't lie on coherent surfaces, encouraging them to move toward actual geometry

3. **Detail Preservation**: Delaunay-based extraction captures fine details without voxel resolution limits

4. **Efficiency**: Only surface Gaussians are used for mesh extraction (~600K of potentially millions)

---

## Configuration System

### Mesh Presets (`configs/mesh/`)

| Preset | Max Delaunay Vertices | Typical Output | Use Case |
|--------|----------------------|----------------|----------|
| `verylowres.yaml` | 250K | <20MB mesh | Preview, web |
| `lowres.yaml` | 500K | <50MB mesh | Fast iteration |
| `default.yaml` | 2M | ~100MB mesh | Standard quality |
| `highres.yaml` | 9M | ~500MB mesh | High quality |
| `veryhighres.yaml` | 14M | ~1GB mesh | Maximum detail |

### Key Configuration Parameters

```yaml
# configs/mesh/default.yaml

# When to apply mesh regularization
start_iter: 8001
stop_iter: 18000
mesh_update_interval: 1

# Loss weights
depth_weight: 0.05
normal_weight: 0.05
occupancy_weight: 0.01

# Mesh extraction quality
n_max_points_in_delaunay: 5_400_000
pivot_sampling_method: "surface"

# SDF computation
occupancy_mode: "occupancy_shift"
sdf_reset_interval: 500
learnable_sdf_reset_mode: "ema"

# Edge handling
filter_large_edges: true
collapse_large_edges: false
edge_threshold_multiplier: 2.0

# Rendering
use_scalable_renderer: false  # true for highres+
```

### Training Arguments

```bash
# Scene configuration
-s, --source_path PATH      # COLMAP dataset path
-m, --model_path PATH       # Output directory

# Rendering
--rasterizer {radegs,gof}   # Rendering backend (default: radegs)
--white_background          # Use white background

# Scene type (affects importance sampling)
--imp_metric {indoor,outdoor}

# Gaussian density
--dense_gaussians           # More Gaussians for detail
--sampling_factor FLOAT     # Downsample Gaussians (0-1)

# Mesh configuration
--mesh_config NAME          # Preset from configs/mesh/
--no_mesh_regularization    # Disable mesh-in-the-loop

# Quality options
--decoupled_appearance      # Per-image appearance encoding
--disable_mip_filter        # Disable 3D Mip filtering
--depth_order               # Enable depth-order regularization

# Logging
--log_interval INT          # Log every N iterations
--wandb                     # Enable W&B logging
```

---

## Extending MILo

### Using the Functional API

The `functional/` module provides standalone functions for integration into other projects:

```python
from functional import (
    sample_gaussians_on_surface,
    compute_delaunay_triangulation,
    compute_initial_sdf_values,
    extract_mesh
)

# In your training loop:
surface_indices = sample_gaussians_on_surface(gaussians, cameras)
vertices, tets, colors = compute_delaunay_triangulation(
    gaussians.get_xyz[surface_indices],
    gaussians.get_scaling[surface_indices]
)
sdf = compute_initial_sdf_values(vertices, cameras, depth_maps)
mesh_v, mesh_f, mesh_c = extract_mesh(vertices, tets, sdf, colors)
```

### Adding Custom Regularization

Extend `regularization/regularizer/` with new loss functions:

```python
# regularization/regularizer/custom.py

def compute_custom_regularization(
    gaussians: GaussianModel,
    camera: Camera,
    **kwargs
) -> Tensor:
    """Your custom regularization loss."""
    # Access Gaussian parameters
    xyz = gaussians.get_xyz
    opacity = gaussians.get_opacity

    # Compute your loss
    loss = ...

    return loss
```

Then integrate in `train.py`:

```python
if iteration >= custom_start_iter:
    custom_loss = compute_custom_regularization(gaussians, camera)
    loss += custom_weight * custom_loss
```

### Custom Mesh Extraction

For specialized mesh extraction, subclass or modify `mesh_extract_sdf.py`:

```python
# Key functions to customize:

def initialize_sdf(gaussians, cameras):
    """Initialize SDF values at Delaunay vertices."""

def refine_sdf(gaussians, cameras, sdf_values, iterations=1000):
    """Optimize SDF values with Gaussians frozen."""

def extract_final_mesh(gaussians, sdf_values):
    """Run marching tetrahedra and export mesh."""
```

---

## Performance Considerations

### Memory Usage

- Gaussian count scales with scene complexity (100K-10M typical)
- Delaunay triangulation memory scales with `n_max_points_in_delaunay`
- Mesh rendering via nvdiffrast adds ~2-4GB GPU memory
- Use `--sampling_factor` to reduce Gaussians for memory-constrained systems

### Training Time

Approximate per-iteration costs (RTX 4090):
- Gaussian rendering: ~15ms
- Photometric loss: ~5ms
- Mesh extraction (when active): ~50-100ms
- Mesh rendering: ~20ms

Total training: ~1-2 hours for 18K iterations (default) on typical scenes.

### Quality vs Speed Trade-offs

| Setting | Training Speed | Mesh Quality | Use Case |
|---------|---------------|--------------|----------|
| `--no_mesh_regularization` | Fastest | Requires post-hoc extraction | Novel view only |
| `--mesh_config lowres` | Fast | Good for most uses | Default |
| `--mesh_config highres --dense_gaussians` | Slowest | Best quality | Production |

---

## Evaluation

### Tanks and Temples

```bash
python scripts/evaluate_tnt.py \
    --data_dir ./tnt_colmap \
    --gt_dir ./tnt_gt \
    --output_dir ./results \
    --rasterizer radegs
```

Metrics: F-score against ground truth meshes.

### DTU

```bash
python scripts/evaluate_dtu.py \
    --data_dir ./dtu_colmap \
    --gt_dir ./dtu_gt \
    --output_dir ./results
```

Metrics: Chamfer distance, accuracy, completeness.

### Novel View Synthesis

```bash
python render.py -s ./data -m ./output
# Renders train/test views, computes PSNR/SSIM/LPIPS
```
