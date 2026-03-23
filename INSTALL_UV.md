# Installation with uv

## 1. Install PyTorch and core dependencies

```bash
uv sync --no-install-project
```

## 2. Install Gaussian Splatting submodules

These need torch available to build, so install them after step 1:

**Note**: The `simple-knn` submodule has been patched for CUDA 12.8 compatibility:
- Added `#include <cfloat>` to `simple_knn.cu`
- Created missing `simple_knn/__init__.py`

```bash
uv pip install --python .venv --no-build-isolation -e submodules/diff-gaussian-rasterization_ms
uv pip install --python .venv --no-build-isolation -e submodules/diff-gaussian-rasterization
uv pip install --python .venv --no-build-isolation -e submodules/diff-gaussian-rasterization_gof
uv pip install --python .venv --no-build-isolation -e submodules/simple-knn
uv pip install --python .venv --no-build-isolation -e submodules/fused-ssim
uv pip install --python .venv --no-build-isolation -e submodules/nvdiffrast
```

## 3. Build tetra_triangulation

Requires system dependencies:

```bash
# Install system deps (one-time)
sudo apt-get install cmake libgmp-dev libcgal-dev  # Ubuntu/Debian
```

**Set CUDA environment variables** (required before building):

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
```

Add these to your `~/.bashrc` or `~/.zshrc` so you don't have to set them every time.

Then build:

```bash
cd submodules/tetra_triangulation
cmake . -DPYTHON_EXECUTABLE=$(uv run which python)
make
uv pip install --python .venv -e .
cd ../..
```

## Optional: Blender addon

```bash
uv pip install --python .venv --no-build-isolation torch-geometric torch-cluster
```

## Usage

```bash
uv run python milo/train.py -s <path> -m <output> --imp_metric indoor --rasterizer radegs
```

Or activate the venv:

```bash
source .venv/bin/activate
python milo/train.py -s <path> -m <output> --imp_metric indoor --rasterizer radegs
```

Done.
