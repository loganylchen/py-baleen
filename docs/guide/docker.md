# Docker

Baleen ships two Dockerfiles and a CI workflow that builds and pushes both
images to Docker Hub on every push to `main`/`dev`:

| Dockerfile | Image | Base |
|------------|-------|------|
| `Dockerfile.cpu` | `<namespace>/py-baleen-cpu` | CPU-only (`tslearn` DTW backend). |
| `Dockerfile.gpu` | `<namespace>/py-baleen-gpu` | `nvidia/cuda:12.6.3-runtime-ubuntu22.04`, CUDA DTW backend. |

The `latest` tag is published only from `main`; branch and long-SHA tags are
published for every build. Both images bundle **f5c v1.6** and set
`ENTRYPOINT ["baleen"]` with a `/data` working directory.

## Pull a published image

```bash
# CPU
docker pull loganylchen/py-baleen-cpu:latest

# GPU (requires the NVIDIA Container Toolkit)
docker pull loganylchen/py-baleen-gpu:latest
```

## Build locally

If you prefer to build from source — or are running a fork — build the
Dockerfile directly:

```bash
# CPU
docker build -f Dockerfile.cpu -t py-baleen-cpu .

# GPU (needs nvcc/CUDA toolkit during build)
docker build -f Dockerfile.gpu -t py-baleen-gpu .
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
    py-baleen-cpu run \
        --native-bam native.bam --native-fastq native.fq.gz --native-blow5 native.blow5 \
        --ivt-bam ivt.bam --ivt-fastq ivt.fq.gz --ivt-blow5 ivt.blow5 \
        --ref ref.fa -o results/
```

```bash
# GPU — add --gpus all
docker run --rm --gpus all \
    -v "$PWD":/data \
    py-baleen-gpu run \
        --native-bam native.bam --native-fastq native.fq.gz --native-blow5 native.blow5 \
        --ivt-bam ivt.bam --ivt-fastq ivt.fq.gz --ivt-blow5 ivt.blow5 \
        --ref ref.fa -o results/
```

!!! tip "File ownership"
    Add `-u $(id -u):$(id -g)` so output files under `results/` are owned by your
    host user rather than root.

## Verify the GPU image sees the device

```bash
docker run --rm --gpus all py-baleen-gpu \
    python3 -c "from baleen._cuda_dtw import backend, is_available; \
print('backend:', backend(), 'cuda:', is_available())"
# Expected: backend: cuda cuda: True
```

If it prints `backend: cpu`, the container cannot see the GPU — check the
NVIDIA Container Toolkit installation and that you passed `--gpus all`.
