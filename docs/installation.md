# Installation

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.9 | 3.9 – 3.11 are tested. |
| [krill](https://loganylchen.github.io/krill-dist/) | DTW + eventalign engine. **Required.** Not on PyPI — install from the project index (see below). |
| [slow5tools](https://github.com/hasindu2008/slow5tools) | Must be on `PATH`. Used to index BLOW5 signal files (`slow5tools index`). |
| NVIDIA GPU + driver | **Optional.** The krill cu122 wheel runs DTW on the GPU. Without a GPU, install the plain krill wheel for a CPU backend. |

!!! note "krill is not on PyPI"
    Baleen's DTW and eventalign run through the `krill` package, which is
    published on a project index rather than PyPI. A plain `pip install baleen`
    will **not** pull it — install krill explicitly (below) or use a Docker
    image, which bundles krill and slow5tools for you.

## Install from source

```bash
git clone https://github.com/loganylchen/py-baleen.git
cd py-baleen

# baleen is pure Python — no C extension to build.
pip install .
```

Then install the krill engine from the project index:

```bash
# GPU (CUDA 12.2 wheel) — recommended when a GPU is available
pip install krill --no-deps \
    --index-url https://loganylchen.github.io/krill-dist/cu122/simple/

# CPU-only
pip install krill --no-deps \
    --index-url https://loganylchen.github.io/krill-dist/simple/
```

!!! warning "krill install rules"
    Install krill's runtime deps (`numpy scipy pyslow5 pyfastx`) from PyPI
    first, then install krill itself with `--no-deps` from the project index.
    Do **not** use `krill[...]` extras or `--extra-index-url`.

## Install with extras

```bash
# Test dependencies (pytest)
pip install ".[test]"

# Documentation toolchain (MkDocs Material + mkdocstrings)
pip install ".[docs]"
```

## Docker

Pre-built images bundle baleen + krill + slow5tools and are published on
Docker Hub:

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
from baleen._dtw import backend, is_available
print("DTW backend:", backend())        # "gpu" or "cpu"
print("GPU available:", is_available())
```
