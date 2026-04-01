#!/usr/bin/env python3
"""Evaluate baleen benchmark: AUPRC/AUROC across stoichiometry levels.

Also exports read-level results and finds the optimal mod_threshold
from the stoich=1.0 dataset.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
STOICH_LEVELS = [f"{x / 10:.1f}" for x in range(11)]  # 0.0 .. 1.0
THRESHOLDS = ["0.9", "0.95", "0.99", "0.999"]
SCORE_COLS = ["mod_ratio", "mean_p_mod", "effect_size", "stoichiometry"]

# Resolved at runtime via CLI arg or default to script directory
TESTDATA: Path = SCRIPT_DIR


def load_ground_truth() -> set[tuple[str, int]]:
    """Load known modification positions as (contig, position) set."""
    # known_modifications.tsv lives next to this script, not necessarily in TESTDATA
    gt_path = TESTDATA / "known_modifications.tsv"
    if not gt_path.exists():
        gt_path = SCRIPT_DIR / "known_modifications.tsv"
    gt = pd.read_csv(gt_path, sep="\t")
    return set(zip(gt["contig"], gt["position"]))


def _output_dir(stoich: str, mode: str, threshold: str) -> Path:
    """Return the output directory for a given stoich/mode/threshold combo."""
    if mode == "legacy":
        return TESTDATA / stoich / f"output_legacy_t{threshold}"
    return TESTDATA / stoich / f"output_t{threshold}"


def load_results(stoich: str, mode: str, threshold: str = "0.9") -> pd.DataFrame | None:
    """Load site_results.tsv for a given stoichiometry, scoring mode, and threshold."""
    path = _output_dir(stoich, mode, threshold) / "site_results.tsv"
    if not path.exists():
        # Fall back to old directory layout (output/ or output_legacy/)
        subdir = "output" if mode == "new" else "output_legacy"
        path = TESTDATA / stoich / subdir / "site_results.tsv"
        if not path.exists():
            return None
    return pd.read_csv(path, sep="\t")


def evaluate(df: pd.DataFrame, gt: set[tuple[str, int]]) -> dict:
    """Compute AUPRC and AUROC for each score column."""
    labels = np.array(
        [(row.contig, row.position) in gt for row in df.itertuples()], dtype=int
    )
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {}

    results = {}
    for col in SCORE_COLS:
        if col not in df.columns:
            continue
        scores = df[col].to_numpy(dtype=float)
        mask = np.isfinite(scores)
        if mask.sum() < 10:
            continue
        y, s = labels[mask], scores[mask]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        results[f"auprc_{col}"] = average_precision_score(y, s)
        results[f"auroc_{col}"] = roc_auc_score(y, s)
    return results


# ---------------------------------------------------------------------------
# Read-level export from intermediate .pkl files
# ---------------------------------------------------------------------------


def export_read_level(stoich: str, mode: str = "new", threshold: str = "0.9") -> pd.DataFrame | None:
    """Export read-level p_mod values from intermediate .pkl files.

    Returns a DataFrame with columns:
        contig, position, kmer, read_name, condition (native/ivt),
        p_mod_raw, p_mod_knn, p_mod_hmm
    """
    import pickle

    out_dir = _output_dir(stoich, mode, threshold)
    if not out_dir.exists():
        # Fall back to old layout
        subdir = "output" if mode == "new" else "output_legacy"
        out_dir = TESTDATA / stoich / subdir
    intermediate_dir = out_dir / "intermediate"

    # Try loading per-contig .pkl files first
    pkl_files = sorted(intermediate_dir.glob("*.pkl")) if intermediate_dir.exists() else []

    # Fall back to pipeline_results.pkl
    if not pkl_files:
        pipeline_pkl = out_dir / "pipeline_results.pkl"
        if not pipeline_pkl.exists():
            return None
        with pipeline_pkl.open("rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, tuple):
            contig_results = loaded[0]
        else:
            contig_results = loaded

        # Need to run HMM to get per-read scores
        from baleen.eventalign._hierarchical import (
            compute_sequential_modification_probabilities,
        )

        hmm_results = {}
        for contig, cr in contig_results.items():
            hmm_results[contig] = compute_sequential_modification_probabilities(cr)
    else:
        # Load intermediate ContigResult files and run HMM
        import pickle

        from baleen.eventalign._hierarchical import (
            compute_sequential_modification_probabilities,
        )

        hmm_results = {}
        for pkl_file in pkl_files:
            with pkl_file.open("rb") as f:
                cr = pickle.load(f)
            contig = cr.contig
            hmm_results[contig] = compute_sequential_modification_probabilities(cr)

    # Extract read-level data
    rows = []
    for contig, cmr in sorted(hmm_results.items()):
        for pos, ps in sorted(cmr.position_stats.items()):
            for i, name in enumerate(ps.native_read_names):
                rows.append({
                    "contig": contig,
                    "position": pos,
                    "kmer": ps.reference_kmer,
                    "read_name": name,
                    "condition": "native",
                    "p_mod_raw": float(ps.p_mod_raw[i]),
                    "p_mod_knn": float(ps.p_mod_knn[i]),
                    "p_mod_hmm": float(ps.p_mod_hmm[i]),
                })
            for j, name in enumerate(ps.ivt_read_names):
                idx = ps.n_native + j
                rows.append({
                    "contig": contig,
                    "position": pos,
                    "kmer": ps.reference_kmer,
                    "read_name": name,
                    "condition": "ivt",
                    "p_mod_raw": float(ps.p_mod_raw[idx]),
                    "p_mod_knn": float(ps.p_mod_knn[idx]),
                    "p_mod_hmm": float(ps.p_mod_hmm[idx]),
                })

    if not rows:
        return None
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold optimization from stoich=1.0
# ---------------------------------------------------------------------------


def find_optimal_threshold(
    read_df: pd.DataFrame,
    gt: set[tuple[str, int]],
    score_col: str = "p_mod_hmm",
    thresholds: np.ndarray | None = None,
) -> dict:
    """Find optimal per-read threshold using stoich=1.0 data.

    At stoich=1.0, all known modification sites should be fully modified
    in the native sample. We evaluate thresholds by:
    1. For each threshold, compute site-level mod_ratio (fraction of native
       reads above threshold)
    2. Classify sites as positive if mod_ratio > 0 (or use a site-level cutoff)
    3. Compute AUPRC on true positive sites

    Returns dict with threshold, metrics, and recommendation.
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 1.001, 0.01)

    native_df = read_df[read_df["condition"] == "native"].copy()
    ivt_df = read_df[read_df["condition"] == "ivt"].copy()

    # For each threshold, compute per-site mod_ratio
    results = []
    for thresh in thresholds:
        site_rows = []
        for (contig, pos), grp in native_df.groupby(["contig", "position"]):
            scores = grp[score_col].values
            n_mod = np.sum(scores > thresh)
            n_total = len(scores)
            mod_ratio = n_mod / n_total if n_total > 0 else 0.0

            # Also get IVT mod_ratio for Fisher-like comparison
            ivt_grp = ivt_df[(ivt_df["contig"] == contig) & (ivt_df["position"] == pos)]
            ivt_scores = ivt_grp[score_col].values if len(ivt_grp) > 0 else np.array([])
            ivt_mod = np.sum(ivt_scores > thresh) if len(ivt_scores) > 0 else 0
            ivt_total = len(ivt_scores)
            ivt_ratio = ivt_mod / ivt_total if ivt_total > 0 else 0.0

            is_true = (contig, pos) in gt
            site_rows.append({
                "contig": contig,
                "position": pos,
                "mod_ratio": mod_ratio,
                "ivt_ratio": ivt_ratio,
                "is_modified": is_true,
                "n_native": n_total,
                "n_native_mod": int(n_mod),
            })

        site_df = pd.DataFrame(site_rows)
        labels = site_df["is_modified"].astype(int).values
        scores = site_df["mod_ratio"].values

        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        auprc = average_precision_score(labels, scores)
        auroc = roc_auc_score(labels, scores)

        # FP rate at stoich=0 proxy: IVT mod_ratio on unmodified sites
        unmod = site_df[~site_df["is_modified"]]
        fp_rate = (unmod["mod_ratio"] > 0).mean() if len(unmod) > 0 else 0.0

        # Sensitivity: fraction of true positive sites with mod_ratio > 0
        mod_sites = site_df[site_df["is_modified"]]
        sensitivity = (mod_sites["mod_ratio"] > 0).mean() if len(mod_sites) > 0 else 0.0

        # Mean mod_ratio at true sites
        mean_mod_ratio_true = mod_sites["mod_ratio"].mean() if len(mod_sites) > 0 else 0.0

        results.append({
            "threshold": float(thresh),
            "auprc": auprc,
            "auroc": auroc,
            "fp_rate": fp_rate,
            "sensitivity": sensitivity,
            "mean_mod_ratio_true": mean_mod_ratio_true,
        })

    if not results:
        return {"error": "No valid thresholds found"}

    results_df = pd.DataFrame(results)

    # Best threshold: maximize AUPRC with penalty for FP
    # Score = AUPRC - 0.5 * fp_rate
    results_df["score"] = results_df["auprc"] - 0.5 * results_df["fp_rate"]
    best_idx = results_df["score"].idxmax()
    best = results_df.loc[best_idx]

    return {
        "best_threshold": best["threshold"],
        "best_auprc": best["auprc"],
        "best_auroc": best["auroc"],
        "best_fp_rate": best["fp_rate"],
        "best_sensitivity": best["sensitivity"],
        "all_results": results_df,
    }


def main():
    global TESTDATA
    if len(sys.argv) > 1:
        TESTDATA = Path(sys.argv[1]).resolve()
        print(f"Using data directory: {TESTDATA}")
    else:
        print(f"Using default data directory: {TESTDATA}")
        print("  (pass a path as argument to override, e.g. python evaluate_benchmark.py /SSD/testdata)")

    gt = load_ground_truth()

    # ---- Discover available thresholds ----
    # Check which thresholds have results (support both new and old layout)
    available_thresholds = set()
    for stoich in STOICH_LEVELS:
        stoich_dir = TESTDATA / stoich
        if not stoich_dir.exists():
            continue
        for d in stoich_dir.iterdir():
            if d.is_dir() and d.name.startswith("output"):
                # Extract threshold from dir name like output_t0.9 or output_legacy_t0.95
                if "_t" in d.name:
                    t = d.name.rsplit("_t", 1)[1]
                    available_thresholds.add(t)
        # Check old layout (output/ output_legacy/ without threshold suffix)
        if (stoich_dir / "output" / "site_results.tsv").exists():
            available_thresholds.add("default")
    if not available_thresholds:
        available_thresholds = set(THRESHOLDS)
    available_thresholds = sorted(available_thresholds)
    print(f"Thresholds found: {available_thresholds}")

    # ---- Standard site-level evaluation ----
    rows = []
    for stoich in STOICH_LEVELS:
        for thresh in available_thresholds:
            for mode in ("new", "legacy"):
                df = load_results(stoich, mode, thresh)
                if df is None:
                    continue
                metrics = evaluate(df, gt)
                if not metrics:
                    continue
                rows.append({
                    "stoichiometry_level": stoich,
                    "mode": mode,
                    "threshold": thresh,
                    **metrics,
                })

    if not rows:
        print("No results found. Run run_benchmark.sh first.")
        sys.exit(1)

    summary = pd.DataFrame(rows)
    print("\n=== Summary Table ===")
    print(summary.to_string(index=False, float_format="%.4f"))
    summary.to_csv(TESTDATA / "benchmark_summary.csv", index=False)
    print(f"\nSaved to {TESTDATA / 'benchmark_summary.csv'}")

    # ---- Read-level export ----
    print("\n=== Exporting read-level results ===")
    for stoich in STOICH_LEVELS:
        for thresh in available_thresholds:
            read_df = export_read_level(stoich, "new", thresh)
            if read_df is None:
                continue
            out_dir = _output_dir(stoich, "new", thresh)
            if not out_dir.exists():
                out_dir = TESTDATA / stoich / "output"
            out_path = out_dir / "read_results.tsv"
            read_df.to_csv(out_path, sep="\t", index=False)
            n_reads = len(read_df)
            n_sites = read_df.groupby(["contig", "position"]).ngroups
            print(f"  {stoich} new t={thresh}: {n_reads} reads, {n_sites} sites -> {out_path}")

    # ---- Threshold optimization from stoich=1.0 ----
    print("\n=== Threshold Optimization (stoich=1.0) ===")
    # Use the first available threshold's read-level data for optimization
    read_1_0 = None
    for thresh in available_thresholds:
        read_1_0 = export_read_level("1.0", "new", thresh)
        if read_1_0 is not None:
            break
    if read_1_0 is not None:
        opt = find_optimal_threshold(read_1_0, gt)
        if "error" not in opt:
            print(f"\n  Recommended threshold:  {opt['best_threshold']:.2f}")
            print(f"  AUPRC at best:         {opt['best_auprc']:.4f}")
            print(f"  AUROC at best:         {opt['best_auroc']:.4f}")
            print(f"  FP rate (unmod sites): {opt['best_fp_rate']:.4f}")
            print(f"  Sensitivity (TP > 0):  {opt['best_sensitivity']:.4f}")

            # Save full threshold scan
            opt_df = opt["all_results"]
            opt_path = TESTDATA / "threshold_optimization.csv"
            opt_df.to_csv(opt_path, index=False)
            print(f"\n  Full scan saved to {opt_path}")

            # Print top-10 thresholds
            print("\n  Top 10 thresholds by score (AUPRC - 0.5*FP_rate):")
            top = opt_df.nlargest(10, "score")
            print(top.to_string(index=False, float_format="%.4f"))

            # Plot threshold scan
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            ax = axes[0]
            ax.plot(opt_df["threshold"], opt_df["auprc"], "-o", markersize=3, label="AUPRC")
            ax.plot(opt_df["threshold"], opt_df["auroc"], "-s", markersize=3, label="AUROC")
            ax.axvline(opt["best_threshold"], color="red", ls="--", alpha=0.7,
                       label=f"best={opt['best_threshold']:.2f}")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Score")
            ax.set_title("Site-level classification (stoich=1.0)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            ax.plot(opt_df["threshold"], opt_df["sensitivity"], "-o", markersize=3, label="Sensitivity")
            ax.plot(opt_df["threshold"], 1 - opt_df["fp_rate"], "-s", markersize=3, label="1 - FP rate")
            ax.axvline(opt["best_threshold"], color="red", ls="--", alpha=0.7)
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Rate")
            ax.set_title("Sensitivity vs Specificity")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[2]
            ax.plot(opt_df["threshold"], opt_df["mean_mod_ratio_true"], "-o", markersize=3)
            ax.axvline(opt["best_threshold"], color="red", ls="--", alpha=0.7)
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Mean mod_ratio at true sites")
            ax.set_title("Signal strength at known sites")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = TESTDATA / "threshold_optimization.png"
            plt.savefig(plot_path, dpi=150)
            print(f"\n  Plot saved to {plot_path}")
        else:
            print(f"  Error: {opt['error']}")
    else:
        print("  No read-level data for stoich=1.0. Run with --keep-intermediate first.")

    # ---- Plot AUPRC vs stoichiometry (per threshold x scoring mode) ----
    score_col = None
    for candidate in ["auprc_mean_p_mod", "auprc_mod_ratio", "auprc_effect_size"]:
        if candidate in summary.columns:
            score_col = candidate
            break

    if score_col is None:
        print("No AUPRC columns available for plotting.")
        return

    unique_thresholds = sorted(summary["threshold"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    styles = {
        ("new",): "-",
        ("legacy",): "--",
    }
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_thresholds), 1)))

    for ax, metric_prefix, ylabel in [
        (axes[0], "auprc", "AUPRC"),
        (axes[1], "auroc", "AUROC"),
    ]:
        col = score_col.replace("auprc_", f"{metric_prefix}_")
        if col not in summary.columns:
            continue
        for i, thresh in enumerate(unique_thresholds):
            for mode, ls in [("new", "-"), ("legacy", "--")]:
                sub = summary[
                    (summary["mode"] == mode) & (summary["threshold"] == thresh)
                ].copy()
                if sub.empty:
                    continue
                sub["x"] = sub["stoichiometry_level"].astype(float)
                sub = sub.sort_values("x")
                ax.plot(
                    sub["x"], sub[col], ls,
                    color=colors[i], marker="o", markersize=4,
                    label=f"{mode} t={thresh}",
                )
        ax.set_xlabel("Stoichiometry level")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Stoichiometry ({col.split('_', 1)[1]})")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plot_path = TESTDATA / "benchmark_auprc.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
