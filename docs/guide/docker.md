# Docker

Baleen ships two Dockerfiles and a CI workflow that builds and pushes both
images to Docker Hub on every push to `main`/`dev`:

| Dockerfile | Image | Base |
|------------|-------|------|
| `Dockerfile.cpu` | `<namespace>/py-baleen-cpu` | `python:3.11-slim`, krill CPU wheel. |
| `Dockerfile.gpu` | `<namespace>/py-baleen-gpu` | `nvidia/cuda:12.2.2-runtime-ubuntu22.04`, krill cu122 GPU wheel. |

The `latest` tag is published only from `main`; branch and long-SHA tags are
published for every build. Both images bundle the **krill** engine and
**slow5tools**, and set `ENTRYPOINT ["baleen"]` with a `/data` working
directory.

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

# GPU
docker build -f Dockerfile.gpu -t py-baleen-gpu .
```

Both builds are pure Python (no C-extension compilation): they `pip install`
baleen, then install the appropriate krill wheel (CPU vs cu122) from the
project index. The GPU image's krill is GPU-capable only at run time when a
device is visible — see the verification step below.

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
docker run --rm --gpus all --entrypoint python3 py-baleen-gpu \
    -c "from baleen._dtw import backend, is_available; \
print('backend:', backend(), 'gpu:', is_available())"
# Expected: backend: gpu gpu: True
```

If it prints `backend: cpu`, the container cannot see the GPU — check the
NVIDIA Container Toolkit installation and that you passed `--gpus all`.
