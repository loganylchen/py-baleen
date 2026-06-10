# Interpreting & Filtering Results

`site_results.tsv` gives you several statistics per position. Choosing how to
filter them is the difference between a clean shortlist and a flood of
borderline calls. This page explains what each column means for filtering and
why **p-value alone is not enough** at high coverage.

## Why p-value alone fails at high coverage

The site `pvalue` is a one-sided **Fisher's exact test** asking *“is the
modified fraction higher in native than in IVT?”*. Like any frequentist test,
its power grows with sample size. At deep coverage even a biologically trivial
difference — say 2% vs 1% modified reads — becomes "significant".

So `padj < 0.05` answers *whether* native differs from IVT, but says nothing
about *how much*. Pair it with an effect-size or stoichiometry threshold.

## The columns that matter for filtering

| Column | Use as a filter to… |
|--------|---------------------|
| `padj` | Control the false-discovery rate across all tested sites (BH-corrected). |
| `effect_size` | Require a minimum native−IVT separation: `median(native p_mod_hmm) − median(IVT p_mod_hmm)`. Drops "significant but tiny" sites. |
| `mod_ratio` + `ci_low`/`ci_high` | Require a minimum stoichiometry **with confidence**: filter on `ci_low` to demand the *lower* bound clears a threshold. |
| `stoichiometry` | Fraction of native reads called modified (`p_mod_hmm > 0.5`); a coarse alternative to `mod_ratio`. |
| `n_native`, `n_ivt` | Drop low-coverage sites where any estimate is noisy. |

## Recommended filter

A robust shortlist combines significance **and** effect size **and** coverage:

```bash
awk -F'\t' '
  NR==1 || ($8 < 0.05 && $9 > 0.1 && $10 >= 30 && $11 >= 30)
' results/site_results.tsv > significant_sites.tsv
#            padj<0.05  effect>0.1  n_native>=30  n_ivt>=30
```

Column indices: `$8` = `padj`, `$9` = `effect_size`, `$10` = `n_native`,
`$11` = `n_ivt`. See the [full schema](outputs.md#site_results-tsv).

In pandas:

```python
import pandas as pd
df = pd.read_csv("results/site_results.tsv", sep="\t")
hits = df[
    (df.padj < 0.05)
    & (df.effect_size > 0.1)
    & (df.ci_low > 0.1)        # lower CI bound of mod_ratio clears 10%
    & (df.n_native >= 30)
    & (df.n_ivt >= 30)
]
```

## What you can and cannot filter post-hoc

| Want to change | Possible on the existing TSV? |
|----------------|-------------------------------|
| Stricter `padj`, `effect_size`, `mod_ratio`, `ci_low`, `stoichiometry`, coverage | **Yes** — pure row filtering, no re-run. |
| A different `--mod-threshold` (recounts modified reads → changes `mod_ratio`, `pvalue`, `stoichiometry`) | **No** — re-run `baleen aggregate` with the new threshold. |
| Different HMM parameters | **No** — re-run `baleen aggregate --hmm-params ...`. |
| Different DTW / subsampling / depth | **No** — full `baleen run`. |

The cheap path for threshold sweeps is `baleen aggregate`, which reuses the saved
DTW results and only re-does HMM + aggregation. See the
[CLI Reference](cli.md#baleen-aggregate).

## Benchmarking confidence

At high coverage, controlling false positives needs more than `padj`:

- **Fix the depth.** Use `--subsample-n` so every site is tested at comparable
  read counts; this prevents power inflation from a few ultra-deep positions.
- **Calibrate a threshold against ground truth.** If you have known modified
  sites, sweep `effect_size`/`mod_ratio` cutoffs and pick the one at your target
  precision against a known-mods file.
- **Run an IVT-vs-IVT negative control.** Feed two IVT replicates as
  native/IVT; every call is a false positive, giving an empirical FPR to
  calibrate against.
