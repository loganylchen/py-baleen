#!/usr/bin/env python3
"""Evaluate baleen benchmark: AUPRC/AUROC across stoichiometry levels."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TESTDATA = Path(__file__).resolve().parent
STOICH_LEVELS = [f"{x / 10:.1f}" for x in range(11)]  # 0.0 .. 1.0
SCORE_COLS = ["mod_ratio", "mean_p_mod", "effect_size", "stoichiometry"]


def load_ground_truth() -> set[tuple[str, int]]:
    """Load known modification positions as (contig, position) set."""
    gt = pd.read_csv(TESTDATA / "known_modifications.tsv", sep="\t")
    return set(zip(gt["contig"], gt["position"]))


def load_results(stoich: str, mode: str) -> pd.DataFrame | None:
    """Load site_results.tsv for a given stoichiometry and scoring mode."""
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


def main():
    gt = load_ground_truth()
    rows = []

    for stoich in STOICH_LEVELS:
        for mode in ("new", "legacy"):
            df = load_results(stoich, mode)
            if df is None:
                print(f"  MISSING: {stoich} {mode}")
                continue
            metrics = evaluate(df, gt)
            if not metrics:
                print(f"  NO VALID METRICS: {stoich} {mode}")
                continue
            rows.append({"stoichiometry_level": stoich, "mode": mode, **metrics})

    if not rows:
        print("No results found. Run run_benchmark.sh first.")
        sys.exit(1)

    summary = pd.DataFrame(rows)
    print("\n=== Summary Table ===")
    print(summary.to_string(index=False, float_format="%.4f"))
    summary.to_csv(TESTDATA / "benchmark_summary.csv", index=False)
    print(f"\nSaved to {TESTDATA / 'benchmark_summary.csv'}")

    # --- Plot AUPRC vs stoichiometry (new vs legacy) ---
    # Use best available score column
    score_col = None
    for candidate in ["auprc_mean_p_mod", "auprc_mod_ratio", "auprc_effect_size"]:
        if candidate in summary.columns:
            score_col = candidate
            break

    if score_col is None:
        print("No AUPRC columns available for plotting.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric_prefix, ylabel in [
        (axes[0], "auprc", "AUPRC"),
        (axes[1], "auroc", "AUROC"),
    ]:
        col = score_col.replace("auprc_", f"{metric_prefix}_")
        if col not in summary.columns:
            continue
        for mode, style in [("new", "-o"), ("legacy", "--s")]:
            sub = summary[summary["mode"] == mode].copy()
            sub["x"] = sub["stoichiometry_level"].astype(float)
            sub = sub.sort_values("x")
            ax.plot(sub["x"], sub[col], style, label=mode, markersize=6)
        ax.set_xlabel("Stoichiometry level")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Stoichiometry ({col.split('_', 1)[1]})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plot_path = TESTDATA / "benchmark_auprc.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
