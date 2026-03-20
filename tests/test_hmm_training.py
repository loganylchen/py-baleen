from __future__ import annotations

import importlib
import os
import tempfile

import numpy as np
import pytest
from numpy.typing import NDArray

hmm = importlib.import_module("baleen.eventalign._hmm_training")
hier = importlib.import_module("baleen.eventalign._hierarchical")
pipeline = importlib.import_module("baleen.eventalign._pipeline")

PositionResult = pipeline.PositionResult
ContigResult = pipeline.ContigResult
HMMParams = hmm.HMMParams
EmissionCalibrator = hmm.EmissionCalibrator
EmissionKDE = hmm.EmissionKDE
CVResult = hmm.CVResult


def _make_block_distance_matrix(
    n_native: int,
    n_ivt: int,
    within_native: float = 1.0,
    within_ivt: float = 1.0,
    between: float = 5.0,
    noise: float = 0.1,
    rng: np.random.RandomState | None = None,
) -> NDArray[np.float64]:
    if rng is None:
        rng = np.random.RandomState(42)
    n = n_native + n_ivt
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if i < n_native and j < n_native:
                d = within_native
            elif i >= n_native and j >= n_native:
                d = within_ivt
            else:
                d = between
            d += rng.normal(0, noise)
            d = max(d, 0.0)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _make_homogeneous_matrix(
    n_native: int,
    n_ivt: int,
    base_dist: float = 1.0,
    noise: float = 0.05,
    rng: np.random.RandomState | None = None,
) -> NDArray[np.float64]:
    if rng is None:
        rng = np.random.RandomState(99)
    n = n_native + n_ivt
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = base_dist + rng.normal(0, noise)
            d = max(d, 0.0)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _make_contig_result(
    n_positions: int = 10,
    n_native: int = 15,
    n_ivt: int = 10,
    modified_positions: set[int] | None = None,
    position_start: int = 100,
    position_step: int = 1,
    seed: int = 42,
    contig_name: str = "chr1",
) -> ContigResult:
    if modified_positions is None:
        modified_positions = set()

    rng = np.random.RandomState(seed)
    positions: dict[int, PositionResult] = {}

    native_names = [f"native_{i}" for i in range(n_native)]
    ivt_names = [f"ivt_{i}" for i in range(n_ivt)]

    for idx in range(n_positions):
        pos = position_start + idx * position_step
        if idx in modified_positions:
            dm = _make_block_distance_matrix(
                n_native,
                n_ivt,
                within_native=1.5,
                within_ivt=1.0,
                between=6.0,
                noise=0.15,
                rng=rng,
            )
        else:
            dm = _make_homogeneous_matrix(
                n_native,
                n_ivt,
                base_dist=1.0,
                noise=0.05,
                rng=rng,
            )

        positions[pos] = PositionResult(
            position=pos,
            reference_kmer="AACGT",
            n_native_reads=n_native,
            n_ivt_reads=n_ivt,
            native_read_names=list(native_names),
            ivt_read_names=list(ivt_names),
            distance_matrix=dm,
        )

    return ContigResult(
        contig=contig_name,
        native_depth=float(n_native),
        ivt_depth=float(n_ivt),
        positions=positions,
    )


def _run_v1v2(contig_result, **kwargs):
    compute = hier.compute_sequential_modification_probabilities
    return compute(contig_result, run_hmm=False, **kwargs)


def _build_training_data(
    *,
    n_contigs: int,
    n_positions: int,
    modified_per_contig: int,
    n_native: int = 12,
    n_ivt: int = 10,
    seed_base: int = 100,
) -> tuple[dict[str, object], dict[tuple[str, int], bool], dict[str, ContigResult]]:
    contig_results: dict[str, ContigResult] = {}
    v2_results: dict[str, object] = {}
    labels: dict[tuple[str, int], bool] = {}

    for c_idx in range(n_contigs):
        contig_name = f"chr{c_idx + 1}"
        mod_set = set(range(modified_per_contig))
        cr = _make_contig_result(
            n_positions=n_positions,
            n_native=n_native,
            n_ivt=n_ivt,
            modified_positions=mod_set,
            position_start=1000 + c_idx * 200,
            seed=seed_base + c_idx,
            contig_name=contig_name,
        )
        contig_results[contig_name] = cr
        cmr = _run_v1v2(cr)
        v2_results[contig_name] = cmr

        sorted_pos = sorted(cr.positions.keys())
        for idx, pos in enumerate(sorted_pos):
            labels[(contig_name, pos)] = idx in mod_set

    return v2_results, labels, contig_results


def test_create_unsupervised_params():
    params = hmm.create_unsupervised_params()
    assert params.mode == "unsupervised"
    assert params.n_states == 3
    assert params.p_stay_per_base == 0.98
    np.testing.assert_allclose(params.init_prob, np.array([0.7, 0.2, 0.1]))
    assert params.emission_transform is None


def test_emission_calibrator_sigmoid():
    calibrator = EmissionCalibrator(a=2.0, b=-1.0)
    out = calibrator.transform(np.array([0.0, 0.5, 1.0], dtype=np.float64))
    assert 0 < out[0] < out[1] < out[2] < 1


def test_emission_calibrator_roundtrip():
    calibrator = EmissionCalibrator(a=1.7, b=-0.2)
    x = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    out1 = calibrator.transform(x)
    out2 = EmissionCalibrator.from_dict(calibrator.to_dict()).transform(x)
    np.testing.assert_allclose(out1, out2)


def test_emission_kde_interpolation():
    grid = np.linspace(0.0, 1.0, 201)
    sigma = 0.05
    density_unmod = np.exp(-0.5 * ((grid - 0.2) / sigma) ** 2)
    density_mod = np.exp(-0.5 * ((grid - 0.8) / sigma) ** 2)
    kde = EmissionKDE(grid=grid, density_unmod=density_unmod, density_mod=density_mod)
    p_unmod, p_mod = kde.emission_probs(np.array([0.2, 0.5, 0.8], dtype=np.float64))
    assert p_unmod[0] > p_unmod[1] > p_unmod[2]
    assert p_mod[2] > p_mod[1] > p_mod[0]


def test_emission_kde_roundtrip():
    grid = np.linspace(0.0, 1.0, 101)
    density_unmod = np.linspace(1.0, 2.0, 101)
    density_mod = np.linspace(2.0, 1.0, 101)
    kde = EmissionKDE(grid=grid, density_unmod=density_unmod, density_mod=density_mod)
    loaded = EmissionKDE.from_dict(kde.to_dict())
    np.testing.assert_allclose(loaded.grid, grid)
    np.testing.assert_allclose(loaded.density_unmod, density_unmod)
    np.testing.assert_allclose(loaded.density_mod, density_mod)


def test_train_semi_supervised_synthetic():
    v2_results, labels, _ = _build_training_data(
        n_contigs=3,
        n_positions=10,
        modified_per_contig=5,
        seed_base=200,
    )
    params = hmm.train_semi_supervised(v2_results, labels, species_name="synthetic")
    assert isinstance(params, HMMParams)
    assert params.mode == "semi_supervised"
    assert isinstance(params.emission_transform, EmissionCalibrator)
    assert params.emission_transform.a > 0
    assert params.init_prob.sum() == pytest.approx(1.0)


def test_train_semi_supervised_too_few_labels():
    v2_results, labels, _ = _build_training_data(
        n_contigs=2,
        n_positions=8,
        modified_per_contig=4,
        seed_base=300,
    )
    limited = dict(list(labels.items())[:18])
    with pytest.raises(ValueError):
        hmm.train_semi_supervised(v2_results, limited)


def test_train_supervised_synthetic():
    v2_results, labels, _ = _build_training_data(
        n_contigs=4,
        n_positions=15,
        modified_per_contig=8,
        seed_base=400,
    )
    mixed_labels: dict[tuple[str, int], bool] = {}
    per_contig_counts: dict[str, int] = {}
    for (contig, pos), is_mod in sorted(labels.items()):
        idx = per_contig_counts.get(contig, 0)
        per_contig_counts[contig] = idx + 1
        mixed_labels[(contig, pos)] = is_mod if (idx % 4) else True

    params = hmm.train_supervised(v2_results, mixed_labels, species_name="synthetic")
    assert params.mode == "supervised"
    assert 0.8 <= params.p_stay_per_base <= 0.999
    assert isinstance(params.emission_transform, EmissionKDE)


def test_train_supervised_too_few_labels():
    v2_results, labels, _ = _build_training_data(
        n_contigs=3,
        n_positions=10,
        modified_per_contig=5,
        seed_base=500,
    )
    with pytest.raises(ValueError):
        hmm.train_supervised(v2_results, labels)


def test_train_supervised_too_few_contigs():
    v2_results, labels, _ = _build_training_data(
        n_contigs=2,
        n_positions=30,
        modified_per_contig=15,
        seed_base=600,
    )
    with pytest.raises(ValueError):
        hmm.train_supervised(v2_results, labels)


def test_labels_from_known_modifications():
    cr = _make_contig_result(
        n_positions=10,
        modified_positions={2, 6},
        seed=700,
        contig_name="chr1",
    )
    cmr = _run_v1v2(cr)
    contig_results = {"chr1": cmr}
    positions = sorted(cmr.position_stats.keys())
    known_mods = {
        ("chr1", positions[2] + 3): ("m6A", "N6-methyladenosine"),
        ("chr1", positions[6] + 3): ("m5C", "5-methylcytosine"),
    }
    labels = hmm.labels_from_known_modifications(
        known_mods,
        contig_results,
        position_offset=3,
        auto_negatives=True,
        min_coverage=5,
    )
    assert labels[("chr1", positions[2])]
    assert labels[("chr1", positions[6])]
    negatives = [k for k, v in labels.items() if not v]
    assert len(negatives) > 0


def test_labels_from_known_modifications_no_auto_neg():
    cr = _make_contig_result(
        n_positions=8,
        modified_positions={1, 4},
        seed=701,
        contig_name="chr2",
    )
    cmr = _run_v1v2(cr)
    positions = sorted(cmr.position_stats.keys())
    known_mods = {
        ("chr2", positions[1] + 3): ("m6A", "N6-methyladenosine"),
        ("chr2", positions[4] + 3): ("m5C", "5-methylcytosine"),
    }
    labels = hmm.labels_from_known_modifications(
        known_mods,
        {"chr2": cmr},
        position_offset=3,
        auto_negatives=False,
    )
    assert labels == {
        ("chr2", positions[1]): True,
        ("chr2", positions[4]): True,
    }


def test_save_load_roundtrip_calibrator():
    params = HMMParams(
        mode="semi_supervised",
        n_states=2,
        p_stay_per_base=0.98,
        init_prob=np.array([0.7, 0.3], dtype=np.float64),
        emission_transform=EmissionCalibrator(a=2.3, b=-0.9),
        training_species=["human"],
        n_training_positions=30,
        n_training_reads=660,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    try:
        hmm.save_hmm_params(params, tmp.name)
        loaded = hmm.load_hmm_params(tmp.name)
        assert loaded.mode == params.mode
        assert loaded.p_stay_per_base == params.p_stay_per_base
        np.testing.assert_allclose(loaded.init_prob, params.init_prob)
        assert isinstance(loaded.emission_transform, EmissionCalibrator)
        assert loaded.emission_transform.a == pytest.approx(2.3)
        assert loaded.emission_transform.b == pytest.approx(-0.9)
        assert loaded.training_species == ["human"]
        assert loaded.n_training_positions == 30
        assert loaded.n_training_reads == 660
    finally:
        os.unlink(tmp.name)


def test_save_load_roundtrip_kde():
    grid = np.linspace(0.0, 1.0, 50)
    density_unmod = np.linspace(1.0, 3.0, 50)
    density_mod = np.linspace(3.0, 1.0, 50)
    params = HMMParams(
        mode="supervised",
        n_states=2,
        p_stay_per_base=0.9,
        init_prob=np.array([0.6, 0.4], dtype=np.float64),
        emission_transform=EmissionKDE(
            grid=grid,
            density_unmod=density_unmod,
            density_mod=density_mod,
        ),
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    try:
        hmm.save_hmm_params(params, tmp.name)
        loaded = hmm.load_hmm_params(tmp.name)
        assert loaded.mode == "supervised"
        assert isinstance(loaded.emission_transform, EmissionKDE)
        np.testing.assert_allclose(loaded.emission_transform.grid, grid)
        np.testing.assert_allclose(loaded.emission_transform.density_unmod, density_unmod)
        np.testing.assert_allclose(loaded.emission_transform.density_mod, density_mod)
    finally:
        os.unlink(tmp.name)


def test_cross_validate_hmm_smoke():
    _, labels, raw_contigs = _build_training_data(
        n_contigs=4,
        n_positions=12,
        modified_per_contig=6,
        seed_base=800,
    )
    cv = hmm.cross_validate_hmm(
        raw_contigs,
        labels,
        mode="semi_supervised",
        cv_strategy="leave_one_contig_out",
    )
    assert isinstance(cv, CVResult)
    assert isinstance(cv.mean_auroc, float)
    assert isinstance(cv.mean_auprc, float)
    assert 0.0 <= cv.mean_auroc <= 1.0
    assert 0.0 <= cv.mean_auprc <= 1.0


def test_hmm_params_in_pipeline():
    cr = _make_contig_result(
        n_positions=10,
        n_native=12,
        n_ivt=10,
        modified_positions={2, 5, 8},
        seed=900,
        contig_name="chrX",
    )
    baseline = hier.compute_sequential_modification_probabilities(cr)
    params = HMMParams(
        mode="semi_supervised",
        n_states=2,
        p_stay_per_base=0.98,
        init_prob=np.array([0.5, 0.5], dtype=np.float64),
        emission_transform=EmissionCalibrator(a=2.0, b=-1.0),
    )
    with_params = hier.compute_sequential_modification_probabilities(cr, hmm_params=params)
    diffs = []
    for pos in baseline.position_stats:
        diffs.append(
            np.max(
                np.abs(
                    baseline.position_stats[pos].p_mod_hmm
                    - with_params.position_stats[pos].p_mod_hmm
                )
            )
        )
    assert max(diffs) > 1e-6


def test_backward_compat():
    cr = _make_contig_result(
        n_positions=9,
        n_native=10,
        n_ivt=8,
        modified_positions={3, 7},
        seed=901,
        contig_name="chrY",
    )
    legacy = hier.compute_sequential_modification_probabilities(cr)
    unsup = hier.compute_sequential_modification_probabilities(
        cr,
        hmm_params=hmm.create_unsupervised_params(n_states=2),
    )
    for pos in legacy.position_stats:
        np.testing.assert_allclose(
            legacy.position_stats[pos].p_mod_hmm,
            unsup.position_stats[pos].p_mod_hmm,
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Tests — 3-state unsupervised params
# ---------------------------------------------------------------------------


def test_create_unsupervised_params_3state():
    params = hmm.create_unsupervised_params(n_states=3)
    assert params.mode == "unsupervised"
    assert params.n_states == 3
    assert params.p_stay_per_base == 0.98
    np.testing.assert_allclose(params.init_prob, np.array([0.7, 0.2, 0.1]))
    assert params.emission_transform is None
    assert params.unmod_emission_beta == (2.0, 8.0)
    assert params.flank_emission_beta == (3.0, 3.0)
    assert params.mod_emission_beta == (8.0, 2.0)


def test_create_unsupervised_params_2state():
    params = hmm.create_unsupervised_params(n_states=2)
    assert params.n_states == 2
    np.testing.assert_allclose(params.init_prob, np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# Tests — 3-state serialization
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_3state():
    params = HMMParams(
        mode="unsupervised",
        n_states=3,
        p_stay_per_base=0.97,
        init_prob=np.array([0.7, 0.2, 0.1], dtype=np.float64),
        unmod_emission_beta=(2.0, 8.0),
        flank_emission_beta=(3.0, 3.0),
        mod_emission_beta=(8.0, 2.0),
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    try:
        hmm.save_hmm_params(params, tmp.name)
        loaded = hmm.load_hmm_params(tmp.name)
        assert loaded.n_states == 3
        assert loaded.p_stay_per_base == 0.97
        np.testing.assert_allclose(loaded.init_prob, params.init_prob)
        assert loaded.unmod_emission_beta == (2.0, 8.0)
        assert loaded.flank_emission_beta == (3.0, 3.0)
        assert loaded.mod_emission_beta == (8.0, 2.0)
    finally:
        os.unlink(tmp.name)


def test_train_semi_supervised_3state():
    v2_results, labels, _ = _build_training_data(
        n_contigs=3,
        n_positions=10,
        modified_per_contig=5,
        seed_base=210,
    )
    params = hmm.train_semi_supervised(
        v2_results, labels, species_name="synthetic", n_states=3,
    )
    assert params.mode == "semi_supervised"
    assert params.n_states == 3
    assert params.init_prob.shape == (3,)
    assert params.init_prob.sum() == pytest.approx(1.0)
    assert isinstance(params.emission_transform, EmissionCalibrator)


def test_train_supervised_3state():
    v2_results, labels, _ = _build_training_data(
        n_contigs=4,
        n_positions=15,
        modified_per_contig=8,
        seed_base=410,
    )
    mixed_labels: dict[tuple[str, int], bool] = {}
    per_contig_counts: dict[str, int] = {}
    for (contig, pos), is_mod in sorted(labels.items()):
        idx = per_contig_counts.get(contig, 0)
        per_contig_counts[contig] = idx + 1
        mixed_labels[(contig, pos)] = is_mod if (idx % 4) else True

    params = hmm.train_supervised(
        v2_results, mixed_labels, species_name="synthetic", n_states=3,
    )
    assert params.mode == "supervised"
    assert params.n_states == 3
    assert params.init_prob.shape == (3,)
    assert params.init_prob.sum() == pytest.approx(1.0)
    assert isinstance(params.emission_transform, EmissionKDE)


def test_save_load_roundtrip_3state_calibrator():
    params = HMMParams(
        mode="semi_supervised",
        n_states=3,
        p_stay_per_base=0.96,
        init_prob=np.array([0.5, 0.3, 0.2], dtype=np.float64),
        emission_transform=EmissionCalibrator(a=1.8, b=-0.5),
        training_species=["ecoli"],
        n_training_positions=50,
        n_training_reads=1200,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    try:
        hmm.save_hmm_params(params, tmp.name)
        loaded = hmm.load_hmm_params(tmp.name)
        assert loaded.mode == "semi_supervised"
        assert loaded.n_states == 3
        np.testing.assert_allclose(loaded.init_prob, params.init_prob)
        assert isinstance(loaded.emission_transform, EmissionCalibrator)
        assert loaded.emission_transform.a == pytest.approx(1.8)
        assert loaded.emission_transform.b == pytest.approx(-0.5)
    finally:
        os.unlink(tmp.name)


def test_load_backward_compat_2state():
    """Loading a 2-state JSON (without n_states key) should default to n_states=2."""
    data = {
        "mode": "unsupervised",
        "p_stay_per_base": 0.98,
        "init_prob": [0.5, 0.5],
        "emission_transform": {"type": "none"},
        "training_species": [],
        "n_training_positions": 0,
        "n_training_reads": 0,
    }
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    import json
    json.dump(data, tmp)
    tmp.close()
    try:
        loaded = hmm.load_hmm_params(tmp.name)
        assert loaded.n_states == 2
        np.testing.assert_allclose(loaded.init_prob, np.array([0.5, 0.5]))
    finally:
        os.unlink(tmp.name)
