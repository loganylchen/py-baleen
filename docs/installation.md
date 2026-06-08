# Installation

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.9 | 3.9 – 3.11 are tested. |
| [f5c](https://github.com/hasindu2008/f5c) ≥ 1.4 | Must be on `PATH`. Used for nanopore event alignment. |
| CUDA toolkit | **Optional.** Enables GPU-accelerated DTW. Without it, Baleen falls back to a CPU (`tslearn`) backend automatically. |

!!! note "f5c is an external tool"
    Baleen shells out to the `f5c` binary; it is not installed by `pip`.
    Install it separately and make sure `f5c --version` works from your shell
    before running the pipeline.

## Install from source

```bash
git clone https://github.com/loganylchen/py-baleen.git
cd py-baleen

# With CUDA (auto-detected if `nvcc` is available)
pip install .
```

### CPU-only build

Skip CUDA compilation entirely:

```bash
BALEEN_NO_CUDA=1 pip install .
```

### Targeting specific GPU architectures

By default the build compiles for a broad set of compute capabilities. To
restrict (faster builds) or target a specific GPU, set `BALEEN_CUDA_ARCHS` to a
comma-separated list of compute capabilities **without dots**:

```bash
# Ampere (8.6) + Hopper (9.0)
BALEEN_CUDA_ARCHS=86,90 pip install .

# Auto-detect the GPU currently installed
BALEEN_CUDA_ARCHS=native pip install .
```

| Environment variable | Effect |
|----------------------|--------|
| `BALEEN_NO_CUDA=1` | Skip CUDA compilation; CPU backend only. |
| `BALEEN_CUDA_ARCHS=86,90` | Compile only for the listed compute capabilities. |
| `BALEEN_CUDA_ARCHS=native` | Auto-detect and target the installed GPU. |

## Install with extras

```bash
# Test dependencies (pytest)
pip install ".[test]"

# Documentation toolchain (MkDocs Material + mkdocstrings)
pip install ".[docs]"
```

## Docker

Pre-built images are published on Docker Hub:

```bash
# CPU
docker pull loganylchen/py-baleen-cpu:latest

# GPU (requires the NVIDIA Container Toolkit)
docker pull loganylchen/py-baleen-gpu:latest
```

See the [Docker guide](guide/docker.md) for mounting data and running the
pipeline inside a container.

## Verify the installation

```bash
baleen --help
baleen run --help
python -c "import baleen; print('baleen', baleen.__name__, 'import OK')"
```

To confirm which DTW backend was selected:

```python
from baleen._cuda_dtw import backend, is_available
print("DTW backend:", backend())        # "cuda" or "cpu"
print("CUDA available:", is_available())
```
