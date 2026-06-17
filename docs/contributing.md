# Contributing

Contributions are welcome. This page covers the development setup, tests, and
commit conventions.

## Development setup

```bash
git clone https://github.com/loganylchen/py-baleen.git
cd py-baleen

# Editable install with test deps (CPU-only build is fastest for iterating)
BALEEN_NO_CUDA=1 pip install -e ".[test]"
```

For docs work, add the docs extra:

```bash
pip install -e ".[docs]"
mkdocs serve   # live preview at http://127.0.0.1:8000
```

## Running tests

```bash
# Full suite
pytest

# A single file or test
pytest tests/test_dtw.py
pytest tests/test_dtw.py::test_dtw_distance_basic -v
```

CI runs the suite on Python 3.9, 3.10, and 3.11 with a CPU-only build
(`BALEEN_NO_CUDA=1`). Make sure `pytest` passes locally before opening a PR.

## Benchmarks

The benchmark harness requires `testdata/` with mixing stoichiometries:

```bash
python benchmarks/bench.py run --threads 2 --repeat 5
python benchmarks/bench.py compare   # tabulate recent runs
```

!!! note "Per-contig timers"
    `bench.py` per-contig timing breakdowns are only populated with
    `--threads 1`; with more workers the child-process logs are not forwarded to
    the parent. Use `--threads 1` when you need a per-stage breakdown.

## Commit style

Baleen uses [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | For |
|--------|-----|
| `feat:` | New features. |
| `fix:` | Bug fixes. |
| `perf:` | Performance improvements. |
| `build:` | Build system / packaging. |
| `bench:` | Benchmark changes. |
| `ci:` | CI configuration. |
| `refactor:` | Code restructuring without behaviour change. |
| `test:` | Test-only changes. |
| `docs:` | Documentation. |

A `!` after the type (e.g. `feat(filter)!:`) marks a breaking change.

## CUDA notes

The CUDA kernel is **FP32-only** by design (FP16 cripples Pascal consumer GPUs).
If you touch the DTW kernel, verify any "skip work" optimisation actually reduces
thread count or diagonal count — setting cells to infinity in place is pure
overhead. See [Performance & Scaling](guide/performance.md#cuda-kernel-characteristics).

## Project layout

```
baleen/
├── _cuda_dtw/        # CUDA DTW + CPU fallback
└── eventalign/       # pipeline, BAM/signal/f5c IO, hierarchical model, HMM training
tests/                # pytest suite
benchmarks/           # bench.py harness
docs/                 # this site (MkDocs Material)
```

See the [API Reference](api/index.md) for the public Python surface.
