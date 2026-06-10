# HMM Training Modes

The V3 stage is a gap-aware Hidden Markov Model that smooths per-read
modification probabilities along each read's trajectory. Its parameters
(`HMMParams`) can come from three modes, trading off how much labeled data you
have.

| Mode | Function | Needs labels? | What it learns |
|------|----------|---------------|----------------|
| **Unsupervised** (default) | `create_unsupervised_params` | No | Nothing — uses hardcoded sensible defaults. |
| **Semi-supervised** | `train_semi_supervised` | ≥ 20 positions (≥ 10 pos / ≥ 10 neg) | Platt-scaling emission calibrator + learned init/transition. |
| **Supervised** | `train_supervised` | Labeled trajectories | MLE transitions + KDE emission model + learned init. |

All three produce an `HMMParams` you serialise to JSON and pass to the pipeline
with `--hmm-params`.

## Topology: `n_states`

`HMMParams.n_states` controls the HMM topology:

- `n_states=2` — **Unmodified / Modified** (original behaviour).
- `n_states=3` (default for unsupervised) — **Unmodified / Flank / Modified**.
  The Flank state uses `Beta(3, 3)` emissions (mean 0.5) to absorb the moderate
  k-NN scores at positions adjacent to a modification.

## Mode A — Unsupervised (default)

No labeled data required. This is what runs if you pass nothing.

```python
from baleen import create_unsupervised_params, save_hmm_params

params = create_unsupervised_params(n_states=3)
save_hmm_params(params, "hmm_unsupervised.json")
```

```bash
baleen run ... --hmm-params hmm_unsupervised.json
# (identical to omitting --hmm-params)
```

## Mode B — Semi-supervised

Fits a Platt-scaling calibrator from a modest set of labeled positions, and
(optionally) learns the per-base stay probability from the labeled trajectories
instead of the hardcoded 0.98.

Requires **≥ 20 labeled positions**, with **≥ 10 modified** and **≥ 10
unmodified**; otherwise it raises `ValueError`.

```python
from baleen import save_hmm_params
from baleen.eventalign import train_semi_supervised

# training_data: {contig_name: ContigModificationResult}  (V1→V2 results)
# labels: {(contig, pipeline_position): is_modified}
params = train_semi_supervised(
    training_data,
    labels,
    learn_transitions=True,
    n_states=2,
)
save_hmm_params(params, "hmm_semi.json")
```

## Mode C — Fully supervised

Learns MLE transition probabilities and a KDE-based emission likelihood model
from labeled read trajectories — the most expressive mode, for when you have
ample ground truth.

```python
from baleen import save_hmm_params
from baleen.eventalign import train_supervised

params = train_supervised(
    training_data,
    labels,
    kde_n_bins=200,        # KDE grid resolution
    kde_bandwidth=None,    # None = Scott's rule
    n_states=2,
)
save_hmm_params(params, "hmm_supervised.json")
```

## Persisting and reusing

`save_hmm_params` / `load_hmm_params` round-trip an `HMMParams` to JSON. Files
without an `n_states` field load as 2-state for backward compatibility.

```python
from baleen import load_hmm_params
params = load_hmm_params("hmm_supervised.json")
print(params.mode, params.n_states)
```

Apply a trained model at inference time on a full run, or cheaply on saved
results without recomputing DTW:

```bash
# Full pipeline with a trained HMM
baleen run ... --hmm-params hmm_supervised.json

# Re-aggregate saved results under a trained HMM (no DTW recompute)
baleen aggregate -i results/pipeline_results.pkl -o sites.tsv \
    --hmm-params hmm_supervised.json
```

Both training functions take `emission_source="p_mod_raw"` by default — train
on the raw per-read probabilities. See the
[API reference](../api/hmm.md) for the full signatures.
