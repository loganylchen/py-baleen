# Docker

Baleen ships two Dockerfiles and a CI workflow that builds and pushes both
images on every push to `main`/`dev`. Both variants live in a **single
repository** `py-baleen`; the variant is a tag **suffix** (`-cpu` / `-gpu`):

| Dockerfile | Tag suffix | Base / f5c build |
|------------|-----------|------------------|
| `Dockerfile.cpu` | `-cpu` | `python:3.11-slim`, **CPU** f5c, CUDA DTW disabled (CPU `tslearn` backend). |
| `Dockerfile.gpu` | `-gpu` | `nvidia/cuda:12.2.2-runtime-ubuntu22.04`, **CUDA** f5c + GPU CUDA DTW. |

Tags follow `<ref>-<variant>`: `latest-*` is published only from `main`;
branch and long-SHA tags are published for every build. Both images bundle
**f5c v1.6** and set `ENTRYPOINT ["baleen"]` with a `/data` working directory.
The GPU image ships f5c's **CUDA build**, which uses the GPU by default
(`--disable-cuda` defaults to `no`) — longer reads go to the GPU and the rest
to CPU automatically, no extra flags needed — and falls back to CPU when no GPU
is visible.

Published to two registries:

- **Docker Hub** — `btrspg/py-baleen`
- **GHCR (public)** — `ghcr.io/loganylchen/py-baleen`

## Pull a published image

```bash
# Docker Hub
docker pull btrspg/py-baleen:latest-cpu
docker pull btrspg/py-baleen:latest-gpu      # requires the NVIDIA Container Toolkit

# GHCR (public)
docker pull ghcr.io/loganylchen/py-baleen:latest-gpu
```

## Build locally

If you prefer to build from source — or are running a fork — build the
Dockerfile directly:

```bash
# CPU
docker build -f Dockerfile.cpu -t py-baleen:cpu .

# GPU (needs nvcc/CUDA toolkit during build)
docker build -f Dockerfile.gpu -t py-baleen:gpu .
```

The GPU build **fails loudly** if the `_cuda_dtw` C extension did not compile, so
a successful image is guaranteed to have a working CUDA backend rather than a
silent CPU fallback.

## Run the pipeline in a container

The entrypoint is `baleen`, so pass sub-command arguments directly. Mount your
data into the container's `/data` working directory:

```bash
# CPU
docker run --rm \
    -v "$PWD":/data \
    py-baleen:cpu run \
        --native-bam native.bam --native-fastq native.fq.gz --native-blow5 native.blow5 \
        --ivt-bam ivt.bam --ivt-fastq ivt.fq.gz --ivt-blow5 ivt.blow5 \
        --ref ref.fa -o results/
```

```bash
# GPU — add --gpus all
docker run --rm --gpus all \
    -v "$PWD":/data \
    py-baleen:gpu run \
        --native-bam native.bam --native-fastq native.fq.gz --native-blow5 native.blow5 \
        --ivt-bam ivt.bam --ivt-fastq ivt.fq.gz --ivt-blow5 ivt.blow5 \
        --ref ref.fa -o results/
```

!!! tip "File ownership"
    Add `-u $(id -u):$(id -g)` so output files under `results/` are owned by your
    host user rather than root.

## Verify the GPU image sees the device

```bash
docker run --rm --gpus all py-baleen:gpu \
    python3 -c "from baleen._cuda_dtw import backend, is_available; \
print('backend:', backend(), 'cuda:', is_available())"
# Expected: backend: cuda cuda: True
```

If it prints `backend: cpu`, the container cannot see the GPU — check the
NVIDIA Container Toolkit installation and that you passed `--gpus all`.
