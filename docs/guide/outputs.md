# Outputs

A run produces two primary artifacts in the output directory:

| File | Level | Description |
|------|-------|-------------|
| `site_results.tsv` | per-site | Tab-separated modification calls with statistics. |
| `read_results.bam` | per-read | Standard mod-BAM with `MM`/`ML`/`RG` tags. |

A `per_contig/` directory of intermediate slices is also written during the run
and removed on completion unless you pass `--keep-intermediate`.

## `site_results.tsv` { #site_results-tsv }

One row per tested genomic position. Columns, in order:

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `contig` | str | Reference/transcript name. |
| 2 | `position` | int | 0-based position on the contig. |
| 3 | `kmer` | str | Reference k-mer centred on the position. |
| 4 | `mod_ratio` | float | MAP estimate of modification stoichiometry (Beta-Binomial). |
| 5 | `ci_low` | float | 2.5th percentile of the Beta posterior. |
| 6 | `ci_high` | float | 97.5th percentile of the Beta posterior. |
| 7 | `pvalue` | float | One-sided **Fisher's exact test** on binary modified/unmodified calls (native vs IVT). |
| 8 | `padj` | float | Benjamini-Hochberg FDR-adjusted p-value across all tested sites. |
| 9 | `effect_size` | float | `median(native p_mod_hmm) − median(IVT p_mod_hmm)`. |
| 10 | `n_native` | int | Native reads covering the position. |
| 11 | `n_ivt` | int | IVT reads covering the position. |
| 12 | `mean_p_mod` | float | Mean of native `p_mod_hmm`. |
| 13 | `stoichiometry` | float | Fraction of native reads with `p_mod_hmm > 0.5`. |

!!! info "p-value vs effect size"
    `pvalue`/`padj` answer *“is native different from IVT?”* — at high coverage
    even tiny differences become significant. `effect_size`, `mod_ratio`, and the
    credible interval (`ci_low`, `ci_high`) tell you *how large and how certain*
    the difference is. Use both; see
    [Interpreting & Filtering Results](filtering.md).

### Loading in pandas

```python
import pandas as pd
df = pd.read_csv("results/site_results.tsv", sep="\t")
sig = df[(df.padj < 0.05) & (df.effect_size > 0)]
```

## `read_results.bam` { #read_results-bam }

A standard **mod-BAM**: the original alignments copied through with per-read
modification probabilities encoded in SAM tags. Compatible with
[modkit](https://github.com/nanoporetech/modkit),
[modbamtools](https://github.com/rrwick/modbamtools), and IGV.

| Tag | Type | Meaning |
|-----|------|---------|
| `MM:Z` | string | Delta-encoded modified-base positions. Baleen uses `N+?` — an *unknown* modification on *any* base. |
| `ML:B:C` | uint8 array | Per-position modification probability, quantised to 0–255 (`round(p_mod_hmm × 255)`). |
| `RG:Z` | string | Read group: `native` or `ivt`. |

The `p_mod_hmm` written here is the final per-read probability from the V3
gap-aware HMM stage.

### Reading read-level calls

```bash
# Quick summary
modkit summary results/read_results.bam
```

```python
from baleen import load_read_results
df = load_read_results("results/read_results.bam")
# columns: contig, position, read_name, is_native, p_mod_hmm
print(df.head())
```

`load_read_results_iter` yields the same records one at a time for streaming over
large files. Both reconstruct `p_mod_hmm` by parsing the `MM`/`ML` tags (falling
back to a legacy `MP:f` tag if `MM` is absent).

## Intermediate files (`per_contig/`)

During the streaming run each worker writes `<contig>.tsv` (rows only) and
`<contig>.bam` into `per_contig/`, which are then merged into the two top-level
outputs. A `.run_params.json` fingerprint of the inputs and run-affecting
parameters is written there to support `--resume`. Pass `--keep-intermediate` to
retain this directory after the merge.
