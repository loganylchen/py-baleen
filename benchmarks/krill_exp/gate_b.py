"""Gate B: end-to-end AUROC/AUPRC for the current eventalign engine.

Runs the production pipeline on testdata for the requested stoichiometry
levels (fixed params: padding=1, subsample_n=300, run_hmm=True) and scores
site calls against testdata/known_modifications.tsv.  Emits one JSON line per
stoich to a results file so two engines (f5c vs krill) can be compared:

    AUROC_krill >= AUROC_f5c - 0.02   at every level.

Run inside the GPU image (krill GPU DTW is identical to the legacy _cuda_dtw
GPU kernel, so this isolates the eventalign engine):

    docker run --rm --gpus all -v "$PWD":/work -w /work \
        --entrypoint python3 py-baleen-krill:gpu \
        benchmarks/krill_exp/gate_b.py --label krill --stoich 0.5,1.0 \
        --out benchmarks/krill_exp/gate_b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TESTDATA = REPO / "testdata"

# Ensure the mounted repo source wins over any installed baleen, and that
# bench.py (one level up) is importable for its scoring helpers.  Without the
# REPO insert, `python3 benchmarks/krill_exp/gate_b.py` puts the *script dir*
# on sys.path[0] and `import baleen` silently resolves to site-packages.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))
from bench import _compute_accuracy, _load_known_mods  # noqa: E402


def run_one(stoich: str) -> tuple[list, float]:
    import tempfile

    from baleen.eventalign._pipeline import run_pipeline_streaming

    d = TESTDATA / stoich / "data"
    nat = d / "native_1"
    ctl = d / "control_1"
    with tempfile.TemporaryDirectory(prefix="gateb_") as tmp:
        t0 = time.perf_counter()
        paths, _meta = run_pipeline_streaming(
            native_bam=nat / "native_1.bam",
            native_fastq=nat / "fastq" / "pass.fq.gz",
            native_blow5=nat / "blow5" / "nanopore.blow5",
            ivt_bam=ctl / "control_1.bam",
            ivt_fastq=ctl / "fastq" / "pass.fq.gz",
            ivt_blow5=ctl / "blow5" / "nanopore.blow5",
            ref_fasta=TESTDATA / "ref.fa",
            min_depth=10,
            padding=1,
            subsample_n=300,
            run_hmm=True,
            use_cuda=None,
            threads=1,
            output_dir=tmp,
            write_bam=False,
            cleanup_temp=True,
        )
        wall = time.perf_counter() - t0
        sites = _load_sites(Path(paths["site_tsv"]))
    return sites, wall


def _load_sites(tsv: Path) -> list:
    import csv
    from types import SimpleNamespace
    rows = []
    with tsv.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(SimpleNamespace(
                contig=row["contig"], position=int(row["position"]),
                mod_ratio=float(row["mod_ratio"]),
                mean_p_mod=float(row["mean_p_mod"]),
                effect_size=float(row["effect_size"]),
                stoichiometry=float(row["stoichiometry"]),
                pvalue=float(row["pvalue"]), padj=float(row["padj"]),
            ))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="engine label, e.g. f5c or krill")
    ap.add_argument("--stoich", default="0.5,1.0")
    ap.add_argument("--out", default="benchmarks/krill_exp/gate_b.jsonl")
    args = ap.parse_args()

    known = _load_known_mods()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for stoich in [s.strip() for s in args.stoich.split(",")]:
        sites, wall = run_one(stoich)
        acc = _compute_accuracy(sites, known)
        rec = {"label": args.label, "stoich": stoich, "wall_s": round(wall, 1),
               "n_sites": len(sites), "accuracy": acc}
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        def _g(k):
            v = acc.get(k)
            return f"{v:.4f}" if v is not None else "n/a"
        print(f"[{args.label}] stoich={stoich} sites={len(sites)} "
              f"AUROC(mod_ratio)={_g('auroc_mod_ratio')} "
              f"AUPRC(mod_ratio)={_g('auprc_mod_ratio')} "
              f"AUROC(-log10p)={_g('auroc_nlog10_pvalue')} ({wall:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
