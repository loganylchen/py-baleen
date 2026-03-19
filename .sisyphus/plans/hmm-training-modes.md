# Plan: HMM Training Modes — Unsupervised / Semi-supervised / Fully Supervised

## Context

Current `_hierarchical.py` V3 HMM uses hardcoded parameters:
- `p_stay_per_base = 0.98` (transition)
- `init_prob = [0.5, 0.5]` (initial state)
- Emission = V2 `p_mod_raw` directly as `P(obs|mod)` / `1 - p_mod` as `P(obs|unmod)`
- No Baum-Welch training, no calibration, no learned priors

User has labeled data from multiple species (E. coli, yeast, human) with known modification sites. Goal: allow 3 training modes so user can choose based on data availability and benchmark results.

## Architecture Decision

### New file: `baleen/eventalign/_hmm_training.py`

Separate from `_hierarchical.py` to keep the existing unsupervised pipeline untouched. The training module produces an `HMMParams` object that can be passed into the existing pipeline.

### Data flow

```
Training data (dict[str, ContigModificationResult] + labels)
    │
    ├─ Mode A: no labels → HMMParams with defaults (current behavior)
    ├─ Mode B: labels → Platt-calibrated emissions + learned init_prob
    └─ Mode C: labels → MLE-trained transition + KDE emission model + init
    │
    ▼
HMMParams(p_stay_per_base, init_prob, emission_transform)
    │
    ▼
compute_sequential_modification_probabilities(contig_result, hmm_params=...)
```

### Cross-validation data flow

`cross_validate_hmm` needs both raw pipeline output (`ContigResult`) and V1+V2 results (`ContigModificationResult`). To avoid requiring users to pass both:

```
cross_validate_hmm(
    contig_results: dict[str, ContigResult],         # raw pipeline output
    labels: dict[tuple[str, int], bool],
    mode: ...,
    **hierarchical_kwargs,                           # forwarded to compute_sequential_modification_probabilities
) → CVResult

# Internally per fold:
#   1. Run compute_sequential_modification_probabilities(contig_result, run_hmm=False) on ALL contigs → get V1+V2
#   2. Train HMMParams on train-split ContigModificationResults
#   3. Run compute_sequential_modification_probabilities(contig_result, hmm_params=trained) on test contigs → get V3
#   4. Evaluate p_mod_hmm at labeled positions on test contigs
```

This keeps the CV function self-contained: input is `ContigResult` + labels, output is `CVResult`.

## Deliverables

### 1. `HMMParams` dataclass + `EmissionCalibrator` + `EmissionKDE`

```python
@dataclass
class EmissionCalibrator:
    """Platt-scaling calibrator for V2 → V3 emission mapping (Mode B)."""
    a: float  # slope
    b: float  # intercept

    def transform(self, p_mod_raw: NDArray) -> NDArray:
        """Map raw P(mod) to calibrated P(mod)."""
        return 1.0 / (1.0 + np.exp(-(self.a * p_mod_raw + self.b)))

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> EmissionCalibrator: ...


@dataclass
class EmissionKDE:
    """KDE-based emission likelihood model (Mode C).

    Stores two fitted distributions:
      P(p_mod_raw | unmodified) and P(p_mod_raw | modified)
    Represented as binned histograms (grid + density) for JSON serialization.
    """
    grid: NDArray[np.float64]          # shape (n_bins,), evaluation points
    density_unmod: NDArray[np.float64] # shape (n_bins,), P(x | unmod)
    density_mod: NDArray[np.float64]   # shape (n_bins,), P(x | mod)

    def emission_probs(self, p_mod_raw: NDArray) -> tuple[NDArray, NDArray]:
        """Return (P(obs|unmod), P(obs|mod)) via interpolation on KDE grid."""
        ...

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> EmissionKDE: ...


# EmissionTransform = union type accepted by HMMParams
EmissionTransform = EmissionCalibrator | EmissionKDE | None


@dataclass
class HMMParams:
    """Learned or default HMM parameters for V3."""
    mode: Literal["unsupervised", "semi_supervised", "supervised"]
    p_stay_per_base: float = 0.98
    init_prob: NDArray[np.float64] = field(default_factory=lambda: np.array([0.5, 0.5]))
    # Emission transform (None = use raw p_mod as-is, i.e. unsupervised)
    emission_transform: EmissionTransform = None
    # Metadata
    training_species: list[str] = field(default_factory=list)
    n_training_positions: int = 0
    n_training_reads: int = 0
```

Note: All fields have defaults; `init_prob` uses `field(default_factory=...)` to avoid mutable default issues and field ordering errors.

### 2. Three training functions

#### Mode A: `create_unsupervised_params()` → HMMParams

- Returns current defaults (`p_stay=0.98`, `init=[0.5,0.5]`, `emission_transform=None`)
- No training data needed
- This is what the current code does implicitly

#### Mode B: `train_semi_supervised(training_data, labels)` → HMMParams

Input:
- `training_data`: `dict[str, ContigModificationResult]` keyed by contig name (already ran V1→V2 with `run_hmm=False`)
- `labels`: `dict[tuple[str, int], bool]` mapping `(contig, pipeline_position)` → bool (True = modified)

What it learns:
1. **Emission calibrator** (Platt scaling via `scipy.optimize.minimize`):
   - Collect `(p_mod_raw, is_modified)` pairs from labeled positions
   - For modified positions: native reads → label=1, IVT reads → label=0
   - For unmodified positions: all reads → label=0
   - Fit logistic regression (minimize negative log-likelihood) → `a, b` coefficients
   - Produces `EmissionCalibrator(a, b)`
2. **Learned init_prob**:
   - Compute base rate of modification from training labels: `base_rate = n_mod_positions / n_total_positions`
   - `init_prob = [1 - base_rate, base_rate]`
3. **Transition: keep default `p_stay_per_base = 0.98`** (not enough data to learn reliably)

Minimum requirements: >= 20 labeled positions (10+ modified, 10+ unmodified). Raise `ValueError` if not met.

#### Mode C: `train_supervised(training_data, labels)` → HMMParams

Input: same types as Mode B, but requires more data.

What it learns (everything in Mode B plus):
1. **Transition parameter via MLE on labeled trajectories**:
   - For each read trajectory (from `ContigModificationResult.native_trajectories` and `.ivt_trajectories`) passing through labeled positions:
     - For native trajectories at modified positions: state = 1; at unmodified positions: state = 0
     - For IVT trajectories: state = 0 at all positions (IVT has no modifications)
     - For each consecutive pair `(pos_i, pos_j)` where both have labels:
       - `gap = pos_j - pos_i`
       - `same_state = (state_i == state_j)`
       - Accumulate weighted counts: `same_count += 1/gap`, `diff_count += 1/gap` (gap-normalized)
   - Estimate `p_stay_per_base` via: `p_stay = same_count / (same_count + diff_count)`
   - Clamp to `[0.8, 0.999]` to avoid degenerate values
2. **Emission model via KDE** (using `scipy.stats.gaussian_kde`):
   - Collect `p_mod_raw` values at labeled positions, separated by true label
   - Fit KDE on `p_mod_raw` values for modified reads and unmodified reads separately
   - Evaluate on a fixed grid (e.g., 200 points from 0 to 1) and store as `EmissionKDE`
   - At inference time, `EmissionKDE.emission_probs(p_mod_raw)` returns `(P(obs|unmod), P(obs|mod))` via linear interpolation on the grid
   - Produces `EmissionKDE(grid, density_unmod, density_mod)`

Minimum requirements: >= 50 labeled positions, >= 3 contigs. Raise `ValueError` if not met.

### 3. Labels helper

```python
def labels_from_known_modifications(
    known_mods: dict[tuple[str, int], tuple[str, str]],
    contig_results: dict[str, ContigModificationResult],
    position_offset: int = 3,
    auto_negatives: bool = True,
    min_coverage: int = 5,
) -> dict[tuple[str, int], bool]:
    """Convert known biological modification sites to training labels.

    Parameters
    ----------
    known_mods : {(contig, bio_position): (mod_short, mod_full), ...}
    contig_results : keyed by contig name
    position_offset : bio_pos - offset = pipeline_pos
    auto_negatives : if True, positions with coverage >= min_coverage
        but NOT in known_mods become negative labels
    min_coverage : minimum n_native + n_ivt for auto-negatives

    Returns
    -------
    labels : {(contig, pipeline_position): bool}
    """
```

### 4. Cross-validation utilities

```python
@dataclass
class CVResult:
    """Cross-validation results."""
    per_fold_auroc: list[float]
    per_fold_auprc: list[float]
    mean_auroc: float
    mean_auprc: float
    std_auroc: float
    std_auprc: float
    fold_details: list[dict]  # per-fold metadata (train/test contigs, n_positions)

def cross_validate_hmm(
    contig_results: dict[str, ContigResult],
    labels: dict[tuple[str, int], bool],
    mode: Literal["semi_supervised", "supervised"],
    cv_strategy: Literal["leave_one_contig_out", "kfold"] = "leave_one_contig_out",
    k: int = 5,
    **hierarchical_kwargs,
) -> CVResult:
    """Cross-validate HMM training to detect overfitting.

    Parameters
    ----------
    contig_results : raw pipeline output (ContigResult) per contig
    labels : {(contig, pipeline_position): bool}
    mode : training mode to evaluate
    cv_strategy : leave_one_contig_out or kfold
    hierarchical_kwargs : forwarded to compute_sequential_modification_probabilities

    Internally per fold:
      1. compute_sequential_modification_probabilities(run_hmm=False) on ALL contigs → V1+V2
      2. Train HMMParams on train-split V1+V2 results
      3. compute_sequential_modification_probabilities(hmm_params=trained) on test contigs → V3
      4. Evaluate p_mod_hmm at labeled test positions
    """
```

Note: `cross_validate_hmm` takes `ContigResult` (raw pipeline output), NOT `ContigModificationResult`. It runs V1+V2 internally so it can properly split train/test and apply trained HMM params on the test fold's fresh V3 pass.

Evaluation at each labeled test position:
- `y_true`: 1 for native reads at modified positions, 0 for all other reads
- `y_score`: `position_stats[pos].p_mod_hmm[read_idx]`
- Compute AUROC/AUPRC manually (no sklearn dependency): sort by score, compute TPR/FPR/precision/recall

### 5. Save/load for cross-species transfer

```python
def save_hmm_params(params: HMMParams, path: str | Path) -> None:
    """Serialize trained HMM parameters to JSON.

    EmissionCalibrator: stores {type: "calibrator", a, b}
    EmissionKDE: stores {type: "kde", grid: [...], density_unmod: [...], density_mod: [...]}
    None: stores {type: "none"}
    NDArrays converted to lists for JSON compatibility.
    """

def load_hmm_params(path: str | Path) -> HMMParams:
    """Load previously trained HMM parameters from JSON."""
```

### 6. Modify existing `compute_sequential_modification_probabilities`

In `baleen/eventalign/_hierarchical.py`:

Add optional `hmm_params: HMMParams | None = None` parameter to:
- `compute_sequential_modification_probabilities()` (line ~630)
- `_run_hmm_on_trajectories()` (line ~578)
- `_forward_backward()` (line ~503)

Use `from __future__ import annotations` (already present) + `TYPE_CHECKING` guard for the `HMMParams` import to avoid circular imports:
```python
if TYPE_CHECKING:
    from baleen.eventalign._hmm_training import HMMParams
```

Behavioral changes when `hmm_params` is provided:
- `_run_hmm_on_trajectories`: use `hmm_params.p_stay_per_base` instead of the raw parameter
- `_forward_backward`: use `hmm_params.init_prob` instead of `[0.5, 0.5]`
- Emission building in `_run_hmm_on_trajectories` (around line 598-606):
  - If `emission_transform` is `None`: current behavior (`P(obs|mod) = p_mod_raw`)
  - If `EmissionCalibrator`: `calibrated = transform(p_mod_raw)`, then `emissions = [1-calibrated, calibrated]`
  - If `EmissionKDE`: `(p_unmod, p_mod) = emission_transform.emission_probs(p_mod_raw)`, use directly as emissions

**Backward compatibility**: When `hmm_params=None`, behavior is identical to current code (unsupervised defaults).

### 7. Update `__init__.py` exports

Add to `baleen/eventalign/__init__.py`:
```python
from baleen.eventalign._hmm_training import (
    HMMParams,
    EmissionCalibrator,
    EmissionKDE,
    create_unsupervised_params,
    train_semi_supervised,
    train_supervised,
    labels_from_known_modifications,
    cross_validate_hmm,
    CVResult,
    save_hmm_params,
    load_hmm_params,
)
```

## Atomic commit strategy

Each task below corresponds to one atomic commit. Commit after each task's QA passes.

| Commit | Task | Message | Files |
|--------|------|---------|-------|
| 1 | Task 1 | `feat(hmm): add HMMParams, EmissionCalibrator, EmissionKDE dataclasses and create_unsupervised_params` | `_hmm_training.py` |
| 2 | Task 2 | `feat(hmm): add train_semi_supervised with Platt-scaling emission calibration` | `_hmm_training.py` |
| 3 | Task 3 | `feat(hmm): add train_supervised with MLE transition + KDE emissions` | `_hmm_training.py` |
| 4 | Task 4 | `feat(hmm): add labels_from_known_modifications helper with offset mapping` | `_hmm_training.py` |
| 5 | Task 5 | `feat(hmm): add cross_validate_hmm, CVResult, save/load HMMParams JSON` | `_hmm_training.py` |
| 6 | Task 6 | `feat(hmm): integrate hmm_params into hierarchical pipeline V3` | `_hierarchical.py` |
| 7 | Task 7 | `feat(hmm): export training API from eventalign package` | `__init__.py` |
| 8 | Task 8 | `test(hmm): add 17 tests for HMM training modes` | `test_hmm_training.py` |
| 9 | Task 9 | `feat(hmm): add training workflow demo notebook` | `hmm_training_demo.ipynb` |

## Implementation order with QA

### Task 1: Core dataclasses + Mode A
**File**: `baleen/eventalign/_hmm_training.py`
**Changes**: Create file with `EmissionCalibrator`, `EmissionKDE`, `HMMParams`, `create_unsupervised_params()`
**Commit**: `feat(hmm): add HMMParams, EmissionCalibrator, EmissionKDE dataclasses and create_unsupervised_params`
**QA**:
```bash
python3 -c "
from baleen.eventalign._hmm_training import HMMParams, EmissionCalibrator, EmissionKDE, create_unsupervised_params
import numpy as np
p = create_unsupervised_params()
assert p.mode == 'unsupervised'
assert p.p_stay_per_base == 0.98
assert np.allclose(p.init_prob, [0.5, 0.5])
assert p.emission_transform is None
# Test EmissionCalibrator sigmoid
c = EmissionCalibrator(a=2.0, b=-1.0)
x = np.array([0.0, 0.5, 1.0])
out = c.transform(x)
assert out.shape == (3,)
assert 0 < out[0] < out[1] < out[2] < 1
# Test roundtrip
d = c.to_dict()
c2 = EmissionCalibrator.from_dict(d)
assert np.allclose(c.transform(x), c2.transform(x))
print('Task 1 QA: PASS')
"
```
**Expected**: prints `Task 1 QA: PASS`, exit code 0.

### Task 2: Mode B — `train_semi_supervised()`
**File**: `baleen/eventalign/_hmm_training.py`
**Changes**: Add `train_semi_supervised()` function
**Commit**: `feat(hmm): add train_semi_supervised with Platt-scaling emission calibration`
**QA**:
```bash
python3 -c "
from baleen.eventalign._hmm_training import train_semi_supervised, HMMParams, EmissionCalibrator
import inspect
sig = inspect.signature(train_semi_supervised)
assert 'training_data' in sig.parameters
assert 'labels' in sig.parameters
print('Task 2 QA: PASS (signature check)')
"
```
**Full verification**: deferred to Task 8 (tests with synthetic data). Verifying here that function exists, imports cleanly, and has correct signature.

### Task 3: Mode C — `train_supervised()`
**File**: `baleen/eventalign/_hmm_training.py`
**Changes**: Add `train_supervised()` function with MLE transition learning + KDE emission fitting
**Commit**: `feat(hmm): add train_supervised with MLE transition + KDE emissions`
**QA**:
```bash
python3 -c "
from baleen.eventalign._hmm_training import train_supervised, EmissionKDE
import numpy as np
# Verify EmissionKDE interpolation
grid = np.linspace(0, 1, 100)
d_unmod = np.exp(-((grid - 0.2)**2) / 0.02)
d_mod = np.exp(-((grid - 0.8)**2) / 0.02)
kde = EmissionKDE(grid=grid, density_unmod=d_unmod, density_mod=d_mod)
p_unmod, p_mod = kde.emission_probs(np.array([0.2, 0.8]))
assert p_unmod[0] > p_unmod[1], 'p=0.2 should have higher unmod density'
assert p_mod[1] > p_mod[0], 'p=0.8 should have higher mod density'
# Roundtrip
d = kde.to_dict()
kde2 = EmissionKDE.from_dict(d)
assert np.allclose(kde.grid, kde2.grid)
print('Task 3 QA: PASS')
"
```
**Expected**: prints `Task 3 QA: PASS`, exit code 0.

### Task 4: Labels helper
**File**: `baleen/eventalign/_hmm_training.py`
**Changes**: Add `labels_from_known_modifications()` function
**Commit**: `feat(hmm): add labels_from_known_modifications helper with offset mapping`
**QA**:
```bash
python3 -c "
from baleen.eventalign._hmm_training import labels_from_known_modifications
import inspect
sig = inspect.signature(labels_from_known_modifications)
assert 'known_mods' in sig.parameters
assert 'position_offset' in sig.parameters
assert 'auto_negatives' in sig.parameters
print('Task 4 QA: PASS (signature check)')
"
```
**Full verification**: deferred to Task 8 (test with synthetic ContigModificationResult + known mods + offset).

### Task 5: Cross-validation + save/load
**File**: `baleen/eventalign/_hmm_training.py`
**Changes**: Add `cross_validate_hmm()`, `CVResult`, `save_hmm_params()`, `load_hmm_params()`
**Commit**: `feat(hmm): add cross_validate_hmm, CVResult, save/load HMMParams JSON`
**QA**:
```bash
python3 -c "
from baleen.eventalign._hmm_training import (
    save_hmm_params, load_hmm_params, create_unsupervised_params, CVResult
)
import tempfile, os
# Save/load roundtrip with unsupervised params
p = create_unsupervised_params()
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    path = f.name
try:
    save_hmm_params(p, path)
    p2 = load_hmm_params(path)
    assert p2.mode == p.mode
    assert p2.p_stay_per_base == p.p_stay_per_base
    assert p2.emission_transform is None
    print('Task 5 QA: PASS (save/load roundtrip)')
finally:
    os.unlink(path)
"
```
**Expected**: prints `Task 5 QA: PASS (save/load roundtrip)`, exit code 0.

### Task 6: Integrate `hmm_params` into `_hierarchical.py`
**File**: `baleen/eventalign/_hierarchical.py`
**Changes**: Add `hmm_params` parameter to `compute_sequential_modification_probabilities`, `_run_hmm_on_trajectories`, `_forward_backward`. Apply emission transform and learned init/transition when provided. Use TYPE_CHECKING guard for HMMParams import.
**Commit**: `feat(hmm): integrate hmm_params into hierarchical pipeline V3`
**QA**:
```bash
# Verify backward compatibility: existing tests still pass
python3 -m pytest tests/test_hierarchical.py -q --tb=short
```
**Expected**: all 47 tests pass (no regressions). The existing tests call `compute_sequential_modification_probabilities` without `hmm_params`, verifying backward compat.

Additionally:
```bash
python3 -c "
import inspect
from baleen.eventalign._hierarchical import compute_sequential_modification_probabilities
sig = inspect.signature(compute_sequential_modification_probabilities)
assert 'hmm_params' in sig.parameters
print('Task 6 QA: PASS (hmm_params parameter exists)')
"
```

### Task 7: Update `__init__.py` exports
**File**: `baleen/eventalign/__init__.py`
**Commit**: `feat(hmm): export training API from eventalign package`
**QA**:
```bash
python3 -c "
from baleen.eventalign import (
    HMMParams, EmissionCalibrator, EmissionKDE,
    create_unsupervised_params, train_semi_supervised, train_supervised,
    labels_from_known_modifications, cross_validate_hmm, CVResult,
    save_hmm_params, load_hmm_params,
)
print('Task 7 QA: PASS (all exports importable)')
"
```
**Expected**: prints `Task 7 QA: PASS`, exit code 0.

### Task 8: Comprehensive tests
**File**: `tests/test_hmm_training.py`
**Changes**: Create test file following pattern of `tests/test_hierarchical.py` (synthetic data, no real files needed).
**Commit**: `test(hmm): add 17 tests for HMM training modes`

Test cases (each a separate test function):
1. `test_create_unsupervised_params` — defaults match expected values
2. `test_emission_calibrator_sigmoid` — known input/output pairs
3. `test_emission_calibrator_roundtrip` — to_dict/from_dict preserves values
4. `test_emission_kde_interpolation` — known distributions, verify correct ordering
5. `test_emission_kde_roundtrip` — to_dict/from_dict preserves arrays
6. `test_train_semi_supervised_synthetic` — build synthetic ContigModificationResult with separable modified/unmodified p_mod_raw, verify calibrator `a > 0` and init_prob reflects base rate
7. `test_train_semi_supervised_too_few_labels` — < 20 labels raises ValueError
8. `test_train_supervised_synthetic` — build synthetic data with known transition pattern, verify learned p_stay_per_base is in `[0.8, 0.999]` and EmissionKDE is produced
9. `test_train_supervised_too_few_labels` — < 50 labels raises ValueError
10. `test_train_supervised_too_few_contigs` — < 3 contigs raises ValueError
11. `test_labels_from_known_modifications` — verify offset mapping and auto_negatives
12. `test_labels_from_known_modifications_no_auto_neg` — only positive labels returned
13. `test_save_load_roundtrip_calibrator` — save Mode B params, load, verify equality
14. `test_save_load_roundtrip_kde` — save Mode C params, load, verify equality
15. `test_cross_validate_hmm_smoke` — runs without error on synthetic data, returns CVResult
16. `test_hmm_params_in_pipeline` — run `compute_sequential_modification_probabilities` with custom HMMParams, verify p_mod_hmm differs from unsupervised run
17. `test_backward_compat` — existing pipeline call without hmm_params still works (overlap with Task 6 but explicit)

**QA**:
```bash
python3 -m pytest tests/test_hmm_training.py -v --tb=short
```
**Expected**: all 17 tests pass. Then:
```bash
# Full suite regression check
python3 -m pytest tests/ -q --tb=short
```
**Expected**: 226 + 17 = 243 tests pass, 0 failures.

### Task 9: Training workflow notebook
**File**: `notebooks/hmm_training_demo.ipynb`
**Changes**: Create notebook with sections A–G as described below.
**Commit**: `feat(hmm): add training workflow demo notebook`

Sections:
- **A**: Prepare labels from known E. coli modifications (reuse `KNOWN_MODIFICATIONS` dict from `notebooks/internal_benchmarking.ipynb`)
- **B**: Run V1→V2 on all contigs with `run_hmm=False`, cache results
- **C**: Train Mode B (semi-supervised), print calibrator params, show calibration curve plot
- **D**: Train Mode C (supervised), print learned `p_stay_per_base`, plot KDE emission densities
- **E**: Leave-one-contig-out CV for Mode B and C, compare AUROC/AUPRC vs unsupervised
- **F**: Export trained params to JSON, demonstrate `save_hmm_params` / `load_hmm_params`
- **G**: Placeholder cells for loading yeast/human data and applying trained params

**QA**:
```bash
# Verify valid notebook JSON
python3 -c "import json; nb=json.load(open('notebooks/hmm_training_demo.ipynb')); print(f'Cells: {len(nb[\"cells\"])}'); assert len(nb['cells']) >= 14; print('Task 9 QA: PASS')"
```
**Expected**: prints cell count >= 14 and `Task 9 QA: PASS`. Full runtime test requires user's data/environment.

## Final verification wave

After all 9 tasks:
```bash
# 1. Full test suite
python3 -m pytest tests/ -v --tb=short

# 2. LSP diagnostics
# Run: lsp_diagnostics on _hmm_training.py, _hierarchical.py, __init__.py, test_hmm_training.py
# Expect: 0 new errors on these files

# 3. Import check
python3 -c "from baleen.eventalign import HMMParams, train_semi_supervised, train_supervised, create_unsupervised_params, cross_validate_hmm, save_hmm_params, load_hmm_params, labels_from_known_modifications; print('All imports OK')"

# 4. Backward compat
python3 -m pytest tests/test_hierarchical.py tests/test_probability.py -q --tb=short
# Expect: 47 + 27 = 74 pass, 0 fail
```

## File changes summary

| File | Action | Description |
|------|--------|-------------|
| `baleen/eventalign/_hmm_training.py` | CREATE | ~600 lines: HMMParams, EmissionCalibrator, EmissionKDE, 3 training modes, CV, save/load, labels helper |
| `baleen/eventalign/_hierarchical.py` | MODIFY | Add `hmm_params` parameter to `compute_sequential_modification_probabilities`, `_run_hmm_on_trajectories`, `_forward_backward`; apply emission transform + learned init/transition; TYPE_CHECKING import |
| `baleen/eventalign/__init__.py` | MODIFY | Export new symbols from `_hmm_training` |
| `tests/test_hmm_training.py` | CREATE | ~400 lines: 17 test functions with synthetic data |
| `notebooks/hmm_training_demo.ipynb` | CREATE | Training workflow notebook (sections A–G) |

## Constraints

- MUST NOT break existing unsupervised pipeline (backward compatible — `hmm_params=None` preserves all current behavior)
- MUST NOT require sklearn as hard dependency (use `scipy.optimize.minimize` for logistic regression, `scipy.stats.gaussian_kde` for KDE, manual AUROC/AUPRC computation in CV)
- Position offset (+3) handled in `labels_from_known_modifications()`, NOT in core training code
- Training functions (`train_semi_supervised`, `train_supervised`) work on `ContigModificationResult` (V1+V2 already computed)
- `cross_validate_hmm` works on `ContigResult` (raw pipeline output) — runs V1+V2+V3 internally per fold
- HMMParams serializable to JSON for cross-species portability (NDArrays → lists, EmissionTransform → typed dict)
- `EmissionKDE` stores pre-evaluated grid+densities (not the KDE object itself) for serialization
- Use `TYPE_CHECKING` guard in `_hierarchical.py` for `HMMParams` import to avoid circular imports
