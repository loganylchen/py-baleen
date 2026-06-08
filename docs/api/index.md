# API Reference

The public Python API is re-exported from the top-level `baleen` package and the
`baleen.eventalign` subpackage. Pages here are auto-generated from docstrings.

## Top-level convenience imports

The most common entry points are available directly from `baleen`:

```python
from baleen import (
    run_pipeline_streaming,   # full pipeline (streaming)
    run_pipeline,             # full pipeline (in-memory)
    aggregate_all,            # site aggregation
    write_site_tsv,           # write site_results.tsv
    load_read_results,        # read mod-BAM into a DataFrame
    load_hmm_params,
    save_hmm_params,
)
```

## Reference pages

| Page | Covers |
|------|--------|
| [Pipeline](pipeline.md) | `run_pipeline_streaming`, `run_pipeline`, results I/O, data classes. |
| [Aggregation](aggregation.md) | `SiteResult`, `aggregate_all`, TSV writers. |
| [Modification Probabilities](probability.md) | The three per-read scoring algorithms. |
| [HMM Training](hmm.md) | `HMMParams` and the three training modes. |
| [Read I/O](io.md) | mod-BAM writing and loading. |
| [Read-ID Intersection](read_ids.md) | BAM ∩ FASTQ ∩ BLOW5 enumeration. |

!!! note "Return shape"
    `run_pipeline_streaming` returns a **2-tuple** `(output_paths, metadata)`.
    `output_paths` is a dict with keys `site_tsv`, `read_bam`, `per_contig_dir`,
    `n_total_sites`, and `n_significant`.
