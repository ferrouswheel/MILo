# Mesh Extraction Methods

MILo provides three different mesh extraction methods, each with different trade-offs in terms of quality, speed, scalability, and use cases.

## Overview

| Method | Script | Output File | Best For |
|--------|--------|-------------|----------|
| Learnable SDF | `mesh_extract_sdf.py` | `mesh_learnable_sdf.ply` | General use, trained with mesh-in-the-loop |
| Integration | `mesh_extract_integration.py` | `mesh_{mode}_sdf.ply` | Post-hoc extraction, any 3DGS model |
| Regular TSDF | `mesh_extract_regular_tsdf.py` | `mesh_regular_tsdf_res{N}.ply` | Small object-centric scenes (DTU) |

---

## Method 1: Learnable SDF Refinement

**Script**: `mesh_extract_sdf.py`

### How It Works

This is the primary extraction method for models trained with mesh-in-the-loop regularization. It uses the SDF values learned during training and optionally refines them:

1. **Load trained model** with occupancy/SDF values from training
2. **Build Delaunay triangulation** from Gaussian pivot points (center + 8 offset points per Gaussian)
3. **Optionally initialize SDF** using integration or depth fusion (if `--init` is not `learnable`)
4. **Refine SDF values** for N iterations by:
   - Rendering Gaussians for depth/normal reference
   - Extracting mesh via marching tetrahedra
   - Computing depth/normal consistency losses
   - Optimizing the learnable `_occupancy_shift` parameter
5. **Extract final mesh** and compute vertex colors from training views

### Command

```bash
# Default: load from <model_path>/point_cloud/iteration_18000/point_cloud.ply
python mesh_extract_sdf.py \
    -s <source_path> \
    -m <model_path> \
    --rasterizer radegs \
    --config default \
    --refine_iter 1000

# Custom PLY file
python mesh_extract_sdf.py \
    -s <source_path> \
    -m <model_path> \
    --ply_path /path/to/my_gaussians.ply \
    --rasterizer radegs \
    --config default
```

### Parameters

#### Basic Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-s`, `--source_path` | required | Path to COLMAP dataset |
| `-m`, `--model_path` | required | Path to trained model output |
| `--iteration` | 18000 | Training iteration checkpoint to load |
| `--ply_path` | None | Path to custom PLY file. If specified, loads from this path instead of `<model_path>/point_cloud/iteration_<iteration>/point_cloud.ply` |
| `--rasterizer` | `radegs` | Rendering backend (`radegs` or `gof`) |
| `--config` | `default` | Mesh config preset from `configs/mesh/` |

#### Refinement Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--refine_iter` | 1000 | Number of SDF refinement iterations |
| `--refine_lr` | 0.025 | Learning rate for SDF refinement |
| `--init` | `learnable` | SDF initialization method: `learnable`, `integration`, or `depth_fusion` |

#### Mesh Quality Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_delaunay_sites` | -1 | Max Delaunay pivot points (-1 = use config) |
| `--remove_oof_vertices` | false | Remove out-of-field vertices not seen by any camera |
| `--mtet_on_cpu` | false | Run marching tetrahedra on CPU (for memory issues) |

#### SDF Initialization Options (when `--init` is not `learnable`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sdf_default_isosurface` | 0.5 | Isosurface threshold for integration mode |
| `--n_binary_steps_to_reset_sdf` | 8 | Binary search steps for SDF initialization |
| `--sdf_reset_linearization_n_steps` | 20 | Linearization steps for SDF normalization |
| `--transform_initial_sdf_to_linear_space` | false | Transform SDF to linear space |
| `--min_occupancy_value` | 1e-10 | Minimum occupancy value (prevents log(0)) |

### Config File Options (`configs/mesh/*.yaml`)

The config file controls mesh quality and filtering:

```yaml
# Mesh vertex filtering
filter_large_edges: true      # Remove triangles with edges longer than Gaussian scale
collapse_large_edges: false   # Collapse large edges instead of removing

# Face culling (for object-centric captures with inward-facing cameras)
near_plane_cull_distance: -1.0  # Cull faces closer than this (meters). -1.0 to disable.
backface_culling: false         # Cull faces pointing away from camera

# Loss functions for refinement
use_depth_loss: true          # Enable depth consistency loss
depth_weight: 0.05            # Weight for depth loss
use_normal_loss: true         # Enable normal consistency loss
normal_weight: 0.05           # Weight for normal loss
use_depth_normal: true        # Use depth-derived normals (vs rendered normals)

# SDF computation
occupancy_mode: "occupancy_shift"  # or "density_shift"
enforce_occupied_centers: true     # Regularize Gaussian centers to be inside mesh
occupied_centers_weight: 0.005

# Occupancy labels (supervision from rendered mesh)
use_occupancy_labels_loss: true
occupancy_labels_loss_weight: 0.005
reset_occupancy_labels_every: 200

# Delaunay parameters
n_max_points_in_delaunay: 5_400_000  # Max pivot points

# Rendering
use_scalable_renderer: false  # Use scalable renderer for high-res meshes
```

### Config Presets

| Preset | Max Pivots | Scalable Renderer | Use Case |
|--------|-----------|-------------------|----------|
| `verylowres` | 250K | No | Preview, web (<20MB) |
| `lowres` | 500K | No | Fast iteration (<50MB) |
| `default` | 2M | No | Standard quality |
| `highres` | 9M | Yes | High quality |
| `veryhighres` | 14M | Yes | Maximum detail |

---

## Method 2: Integration / Depth Fusion

**Script**: `mesh_extract_integration.py`

### How It Works

This method computes SDF values from scratch by analyzing the trained Gaussian model, making it suitable for any 3DGS model (not just MILo-trained ones):

1. **Load Gaussians** from checkpoint
2. **Build Delaunay triangulation** from Gaussian pivots
3. **Compute SDF values** using one of two sub-methods:
   - **Integration**: Accumulates integrated opacity along rays from all views
   - **Depth Fusion**: Traditional TSDF from rendered depth maps
4. **Run marching tetrahedra** to extract initial mesh
5. **Binary search refinement**: Iteratively refines vertex positions along tetrahedral edges
6. **Compute vertex colors** by projecting to training views

### Command

```bash
# Integration mode (default)
python mesh_extract_integration.py \
    -s <source_path> \
    -m <model_path> \
    --rasterizer gof \
    --sdf_mode integration \
    --isosurface_value 0.5

# Depth fusion mode
python mesh_extract_integration.py \
    -s <source_path> \
    -m <model_path> \
    --rasterizer radegs \
    --sdf_mode depth_fusion \
    --trunc_margin 0.01

# With custom PLY file
python mesh_extract_integration.py \
    -s <source_path> \
    -m <model_path> \
    --ply_path /path/to/my_gaussians.ply \
    --rasterizer gof \
    --sdf_mode integration
```

### Parameters

#### Basic Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-s`, `--source_path` | required | Path to COLMAP dataset |
| `-m`, `--model_path` | required | Path to trained model output |
| `--iteration` | 18000 | Training iteration checkpoint to load |
| `--ply_path` | None | Path to custom PLY file. If specified, loads from this path instead of default |
| `--rasterizer` | `gof` | Rendering backend (`radegs` or `gof`) |

#### SDF Mode Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sdf_mode` | `integration` | SDF computation method: `integration` or `depth_fusion` |

#### Integration Mode Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--isosurface_value` | 0.5 | Opacity threshold for surface (0-1). Lower = larger mesh, higher = tighter mesh |

The integration method computes: `SDF = isosurface_value - min_accumulated_opacity`

A point is inside the surface if its accumulated opacity exceeds `isosurface_value`.

#### Depth Fusion Mode Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trunc_margin` | auto | TSDF truncation distance. Auto-computed as `0.002 * scene_radius` if not specified |

The depth fusion method computes truncated signed distance from rendered depth maps:
- Positive SDF: point is in front of the surface
- Negative SDF: point is behind the surface (truncated to [-1, 1])

#### Quality Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_delaunay_sites` | -1 | Max Delaunay pivot points (-1 = use all Gaussians) |
| `--n_binary_steps` | 8 | Binary search iterations for vertex refinement |
| `--mtet_on_cpu` | false | Run marching tetrahedra on CPU |

#### Gaussian Sampling (when downsampling)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--imp_metric` | `none` | Importance sampling metric (`indoor`, `outdoor`, `none`). Required if `--n_delaunay_sites` is specified |
| `--warn_until_iter` | 3000 | Warm-up iteration for importance sampling |

### Output Files

- `mesh_integration_sdf.ply` - When using `--sdf_mode integration`
- `mesh_depth_fusion_sdf.ply` - When using `--sdf_mode depth_fusion`

---

## Method 3: Regular TSDF (Grid-based)

**Script**: `mesh_extract_regular_tsdf.py`

### How It Works

This is a traditional voxel-based TSDF fusion using Open3D, similar to KinectFusion:

1. **Render depth maps** from all training views
2. **Estimate bounding sphere** from camera positions
3. **Create voxel grid** within the bounding volume
4. **Integrate depth maps** into TSDF volume
5. **Extract mesh** using marching cubes
6. **Post-process** to remove small disconnected components

### Command

```bash
python mesh_extract_regular_tsdf.py \
    -s <source_path> \
    -m <model_path> \
    --rasterizer radegs \
    --mesh_res 1024 \
    --radius_factor 2.0
```

### Parameters

#### Basic Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-s`, `--source_path` | required | Path to COLMAP dataset |
| `-m`, `--model_path` | required | Path to trained model output |
| `--iteration` | -1 | Training iteration (-1 = latest) |
| `--rasterizer` | `radegs` | Rendering backend |

#### Resolution and Scale Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mesh_res` | 1024 | Voxel grid resolution (higher = finer detail, more memory) |
| `--radius_factor` | 2.0 | Multiplier for scene radius to determine depth truncation |
| `--depth_ratio` | 0.6 | Blend between expected (0) and median (1) depth |

#### TSDF Parameters

These are automatically computed from the above but can be overridden:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--voxel_size` | auto | Voxel size (`depth_trunc / mesh_res`) |
| `--depth_trunc` | auto | Max depth range (`radius * radius_factor`) |
| `--sdf_trunc` | auto | SDF truncation (`sdf_trunc_factor * voxel_size`) |
| `--sdf_trunc_factor` | 5.0 | Truncation factor in voxel units |

#### Post-processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_cluster` | 50 | Keep the N largest connected components |

### Output Files

- `mesh_regular_tsdf_res{N}.ply` - Raw extracted mesh
- `mesh_regular_tsdf_res{N}_post.ply` - Post-processed (cleaned) mesh

### Memory Considerations

Memory usage scales cubically with `mesh_res`:
- 512: ~1GB VRAM
- 1024: ~8GB VRAM
- 2048: ~64GB VRAM (may require system RAM)

For large scenes, use Method 1 or 2 instead.

---

## Comparison and Recommendations

### When to Use Each Method

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| MILo-trained model | **Learnable SDF** | Uses learned SDF, best quality |
| Any 3DGS model, unbounded scene | **Integration** | Scalable, no voxel grid |
| Small object (DTU-style) | **Regular TSDF** | Simple, robust for bounded scenes |
| Memory-constrained | **Integration** or **Learnable SDF** | No dense voxel grid |
| Fastest extraction | **Regular TSDF** | No iterative refinement |
| Best surface detail | **Learnable SDF** with `highres` config | Refined SDF + high pivot count |

### Quality vs Speed Trade-offs

```
Quality:  Learnable SDF (highres) > Learnable SDF (default) > Integration > Regular TSDF
Speed:    Regular TSDF > Integration > Learnable SDF (default) > Learnable SDF (highres)
Memory:   Learnable SDF ≈ Integration < Regular TSDF (high res)
```

### Common Tuning Strategies

#### Reduce Mesh Size
```bash
# Method 1: Use lower-res config
python mesh_extract_sdf.py ... --config lowres

# Method 2: Reduce Delaunay sites
python mesh_extract_integration.py ... --n_delaunay_sites 500000

# Method 3: Lower voxel resolution
python mesh_extract_regular_tsdf.py ... --mesh_res 512
```

#### Improve Surface Detail
```bash
# Method 1: Use high-res config + more refinement
python mesh_extract_sdf.py ... --config highres --refine_iter 2000

# Method 2: More binary search steps
python mesh_extract_integration.py ... --n_binary_steps 12

# Method 3: Higher resolution (if memory allows)
python mesh_extract_regular_tsdf.py ... --mesh_res 2048 --sdf_trunc_factor 3.0
```

#### Handle Noisy Surfaces
```bash
# Method 1: Enable edge filtering in config
# In configs/mesh/your_config.yaml:
#   filter_large_edges: true

# Method 2: Use depth fusion with larger truncation
python mesh_extract_integration.py ... --sdf_mode depth_fusion --trunc_margin 0.05

# Method 3: Increase truncation factor
python mesh_extract_regular_tsdf.py ... --sdf_trunc_factor 8.0
```

#### Control Surface Tightness (Integration Mode)
```bash
# Larger mesh (captures more geometry)
python mesh_extract_integration.py ... --isosurface_value 0.3

# Tighter mesh (higher confidence surfaces only)
python mesh_extract_integration.py ... --isosurface_value 0.7
```

---

## Filtering and Post-Processing Options

### Edge Filtering (Methods 1 & 2)

The Delaunay-based methods can filter out triangles with abnormally large edges:

```yaml
# In configs/mesh/*.yaml
filter_large_edges: true   # Remove faces with edges > Gaussian scale sum
collapse_large_edges: false # Alternative: collapse large edges to nearest vertex
```

This removes "floater" triangles that can appear in under-observed regions.

### Near-Plane and Back-Face Culling (Methods 1 & 2)

When cameras are arranged in a ring looking inward (common for object capture), the mesh behind or around the camera can be rendered, causing incorrect depth/normal supervision. Use these options to cull such geometry:

```yaml
# In configs/mesh/*.yaml
near_plane_cull_distance: 0.3  # Cull faces closer than 0.3m to camera. -1.0 to disable.
backface_culling: true          # Cull faces pointing away from camera
```

**Near-plane culling**: Removes faces where any vertex is closer than the specified distance. Useful when geometry surrounds the camera positions.

**Back-face culling**: Removes faces whose normal points away from the camera. This ensures only the front-facing surface is rendered, avoiding issues with inside-out mesh views.

Example usage for object-centric captures:
```bash
# Edit your config or create a custom one
# configs/mesh/object_centric.yaml:
#   near_plane_cull_distance: 0.3
#   backface_culling: true

python train.py -s ./data/object -m ./output --mesh_config object_centric
```

### Out-of-Field Removal (Method 1)

Remove vertices that aren't visible from any training camera:

```bash
python mesh_extract_sdf.py ... --remove_oof_vertices
```

Useful for cleaning up mesh boundaries.

### Connected Component Filtering (Method 3)

Keep only the largest connected components:

```bash
# Keep 50 largest clusters (removes small floaters)
python mesh_extract_regular_tsdf.py ... --num_cluster 50

# Keep only the largest component
python mesh_extract_regular_tsdf.py ... --num_cluster 1
```

---

## Troubleshooting

### Out of Memory

```bash
# Run marching tetrahedra on CPU
python mesh_extract_sdf.py ... --mtet_on_cpu
python mesh_extract_integration.py ... --mtet_on_cpu

# Reduce Delaunay sites
python mesh_extract_sdf.py ... --n_delaunay_sites 1000000

# Use lower-res TSDF
python mesh_extract_regular_tsdf.py ... --mesh_res 512
```

### Holes in Mesh

- Increase truncation margin: `--trunc_margin 0.1`
- Lower isosurface threshold: `--isosurface_value 0.3`
- Disable edge filtering: set `filter_large_edges: false` in config

### Noisy/Bumpy Surface

- Enable edge filtering: `filter_large_edges: true`
- Increase `--sdf_trunc_factor` for TSDF methods
- Use more refinement iterations: `--refine_iter 2000`

### Missing imp_metric Error

When using `--n_delaunay_sites`, you must specify the scene type:

```bash
python mesh_extract_sdf.py ... --n_delaunay_sites 1000000 --imp_metric outdoor
```

### Mesh Depth/Normals Rendered from Inside (Object-Centric Captures)

When cameras point inward at each other (ring arrangement around an object), the mesh behind each camera may be rendered, causing incorrect supervision.

Solution: Enable near-plane culling and/or back-face culling in your mesh config:

```yaml
# configs/mesh/your_config.yaml
near_plane_cull_distance: 0.3  # Cull faces within 0.3m of camera
backface_culling: true          # Only render front-facing surfaces
```

Or create a custom config for object-centric scenes.
