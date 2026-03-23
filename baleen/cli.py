"""Command-line interface for the baleen modification detection pipeline.

Usage::

    # Full pipeline: DTW + HMM + site-level aggregation
    baleen run \\
        --native-bam native.bam \\
        --native-fastq native.fq.gz \\
        --native-blow5 native.blow5 \\
        --ivt-bam ivt.bam \\
        --ivt-fastq ivt.fq.gz \\
        --ivt-blow5 ivt.blow5 \\
        --ref ref.fa \\
        -o results/

    # Site-level aggregation only (from saved results)
    baleen aggregate -i results/pipeline_results.pkl -o results/sites.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from baleen.eventalign import (
    aggregate_all,
    compute_sequential_modification_probabilities,
    load_hmm_params,
    load_results,
    run_pipeline,
    save_results,
    write_site_tsv,
)
from baleen.eventalign._read_bam import write_read_bam


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``run`` sub-command."""
    # Required inputs
    req = parser.add_argument_group("required inputs")
    req.add_argument("--native-bam", required=True, help="Native BAM file")
    req.add_argument("--native-fastq", required=True, help="Native FASTQ file")
    req.add_argument("--native-blow5", required=True, help="Native BLOW5 file")
    req.add_argument("--ivt-bam", required=True, help="IVT control BAM file")
    req.add_argument("--ivt-fastq", required=True, help="IVT control FASTQ file")
    req.add_argument("--ivt-blow5", required=True, help="IVT control BLOW5 file")
    req.add_argument("--ref", required=True, help="Reference FASTA file")

    # Output
    out = parser.add_argument_group("output")
    out.add_argument(
        "-o", "--output-dir", default="baleen_output",
        help="Output directory (default: baleen_output)",
    )

    # Pipeline parameters
    pipe = parser.add_argument_group("pipeline parameters")
    pipe.add_argument(
        "--padding", type=int, default=0,
        help="Flanking positions for DTW signal concatenation (default: 0)",
    )
    pipe.add_argument(
        "--min-depth", type=int, default=15,
        help="Minimum read depth per contig (default: 15)",
    )
    pipe.add_argument(
        "--min-mapq", type=int, default=0,
        help="Minimum mapping quality (default: 0)",
    )
    pipe.add_argument(
        "--threads", type=int, default=1,
        help="Number of parallel workers for contig processing (default: 1)",
    )

    # DTW options
    dtw = parser.add_argument_group("DTW options")
    cuda_group = dtw.add_mutually_exclusive_group()
    cuda_group.add_argument(
        "--cuda", action="store_true", default=False,
        help="Force CUDA for DTW computation",
    )
    cuda_group.add_argument(
        "--no-cuda", action="store_true", default=False,
        help="Force CPU for DTW computation",
    )
    dtw.add_argument(
        "--open-start", action="store_true", default=False,
        help="Allow open-start DTW alignment",
    )
    dtw.add_argument(
        "--open-end", action="store_true", default=False,
        help="Allow open-end DTW alignment",
    )

    # HMM options
    hmm = parser.add_argument_group("HMM options")
    hmm.add_argument(
        "--hmm-params", type=str, default=None,
        help="Path to trained HMM parameters JSON (default: 3-state unsupervised)",
    )
    hmm.add_argument(
        "--no-hmm", action="store_true", default=False,
        help="Skip HMM smoothing (output V2 scores only)",
    )

    # f5c options
    f5c = parser.add_argument_group("f5c options")
    f5c.add_argument(
        "--no-rna", action="store_true", default=False,
        help="Disable RNA mode for f5c eventalign",
    )
    f5c.add_argument(
        "--kmer-model", type=str, default=None,
        help="Custom kmer model for f5c eventalign",
    )

    # Misc
    misc = parser.add_argument_group("miscellaneous")
    misc.add_argument(
        "--no-primary-only", action="store_true", default=False,
        help="Include secondary/supplementary alignments",
    )
    misc.add_argument(
        "--keep-temp", action="store_true", default=False,
        help="Do not clean up temporary files",
    )
    misc.add_argument(
        "--no-read-bam", action="store_true", default=False,
        help="Skip writing per-read BAM output (read_results.bam)",
    )


def _add_aggregate_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``aggregate`` sub-command."""
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to saved pipeline results (.pkl)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output TSV file path",
    )
    parser.add_argument(
        "--score-field", default="p_mod_hmm",
        choices=["p_mod_hmm", "p_mod_knn", "p_mod_raw"],
        help="Per-read score field to aggregate (default: p_mod_hmm)",
    )
    parser.add_argument(
        "--hmm-params", type=str, default=None,
        help="Path to trained HMM parameters JSON (re-run HMM before aggregation)",
    )
    parser.add_argument(
        "--no-read-bam", action="store_true", default=False,
        help="Skip writing per-read BAM output",
    )
    parser.add_argument(
        "--ref", type=str, default=None,
        help="Reference FASTA (required for per-read BAM output)",
    )


def _validate_input_files(args: argparse.Namespace) -> None:
    """Validate that all required input files exist."""
    files_to_check = [
        ("native_bam", "--native-bam"),
        ("native_fastq", "--native-fastq"),
        ("native_blow5", "--native-blow5"),
        ("ivt_bam", "--ivt-bam"),
        ("ivt_fastq", "--ivt-fastq"),
        ("ivt_blow5", "--ivt-blow5"),
        ("ref", "--ref"),
    ]
    for attr, arg_name in files_to_check:
        path = Path(getattr(args, attr))
        if not path.exists():
            raise SystemExit(f"error: {arg_name} file not found: {path}")


def _cmd_run(args: argparse.Namespace) -> None:
    """Execute the ``run`` sub-command."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input files exist
    _validate_input_files(args)

    # Resolve CUDA option
    use_cuda: bool | None = None
    if args.cuda:
        use_cuda = True
    elif args.no_cuda:
        use_cuda = False

    # Step 1: Run DTW pipeline
    logger = logging.getLogger("baleen")
    logger.info("Step 1/4: Running DTW pipeline...")
    t0 = time.perf_counter()

    contig_results, metadata = run_pipeline(
        native_bam=args.native_bam,
        native_fastq=args.native_fastq,
        native_blow5=args.native_blow5,
        ivt_bam=args.ivt_bam,
        ivt_fastq=args.ivt_fastq,
        ivt_blow5=args.ivt_blow5,
        ref_fasta=args.ref,
        min_depth=args.min_depth,
        use_cuda=use_cuda,
        use_open_start=args.open_start,
        use_open_end=args.open_end,
        padding=args.padding,
        output_dir=output_dir,
        cleanup_temp=not args.keep_temp,
        rna=not args.no_rna,
        kmer_model=args.kmer_model,
        min_mapq=args.min_mapq,
        primary_only=not args.no_primary_only,
        threads=args.threads,
    )

    dtw_time = time.perf_counter() - t0
    logger.info("DTW pipeline done in %.1fs", dtw_time)

    # Save raw results
    pkl_path = output_dir / "pipeline_results.pkl"
    save_results(contig_results, metadata, pkl_path)

    # Step 2: HMM smoothing
    if args.no_hmm:
        logger.info("Step 2/4: HMM skipped (--no-hmm)")
        hmm_results = {}
        for contig, cr in contig_results.items():
            hmm_results[contig] = compute_sequential_modification_probabilities(
                cr, run_hmm=False,
            )
    else:
        logger.info("Step 2/4: Running HMM pipeline...")
        t0 = time.perf_counter()

        hmm_params = None
        if args.hmm_params:
            hmm_params = load_hmm_params(args.hmm_params)
            logger.info("  Loaded HMM params: %s (%d-state %s)",
                        args.hmm_params, hmm_params.n_states, hmm_params.mode)

        hmm_results = {}
        for contig, cr in contig_results.items():
            hmm_results[contig] = compute_sequential_modification_probabilities(
                cr, hmm_params=hmm_params,
            )

        hmm_time = time.perf_counter() - t0
        logger.info("HMM pipeline done in %.1fs", hmm_time)

    # Step 3: Site-level aggregation
    logger.info("Step 3/4: Aggregating site-level results...")
    t0 = time.perf_counter()

    sites = aggregate_all(hmm_results)
    tsv_path = output_dir / "site_results.tsv"
    write_site_tsv(sites, tsv_path)

    agg_time = time.perf_counter() - t0
    logger.info("Aggregation done in %.1fs", agg_time)

    # Step 4: Write per-read BAM
    if not args.no_read_bam:
        logger.info("Step 4/4: Writing per-read BAM...")
        t0 = time.perf_counter()
        bam_path = output_dir / "read_results.bam"
        write_read_bam(hmm_results, contig_results, args.ref, bam_path)
        bam_time = time.perf_counter() - t0
        logger.info("Per-read BAM done in %.1fs", bam_time)
    else:
        bam_path = None

    # Summary
    n_sig = sum(1 for s in sites if s.padj < 0.05)
    n_total = len(sites)
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("  Total sites:      %d", n_total)
    logger.info("  Significant sites: %d (padj < 0.05)", n_sig)
    logger.info("  Output directory:  %s", output_dir)
    logger.info("  Site results:      %s", tsv_path)
    if bam_path:
        logger.info("  Read results:      %s", bam_path)
    logger.info("  Raw results:       %s", pkl_path)
    logger.info("=" * 60)


def _cmd_aggregate(args: argparse.Namespace) -> None:
    """Execute the ``aggregate`` sub-command."""
    logger = logging.getLogger("baleen")

    logger.info("Loading pipeline results from %s ...", args.input)
    loaded = load_results(args.input)
    if isinstance(loaded, tuple):
        contig_results, metadata = loaded
    else:
        contig_results = loaded
    logger.info("Loaded %d contigs", len(contig_results))

    # Run HMM
    hmm_params = None
    if args.hmm_params:
        hmm_params = load_hmm_params(args.hmm_params)
        logger.info("Loaded HMM params: %s (%d-state %s)",
                    args.hmm_params, hmm_params.n_states, hmm_params.mode)

    logger.info("Running HMM pipeline...")
    hmm_results = {}
    for contig, cr in contig_results.items():
        hmm_results[contig] = compute_sequential_modification_probabilities(
            cr, hmm_params=hmm_params,
        )

    # Aggregate
    logger.info("Aggregating site-level results...")
    sites = aggregate_all(hmm_results, score_field=args.score_field)
    write_site_tsv(sites, args.output)

    if not args.no_read_bam:
        if args.ref is None:
            logger.warning("Skipping read-level BAM: --ref not provided")
        else:
            bam_path = Path(args.output).parent / "read_results.bam"
            write_read_bam(hmm_results, contig_results, args.ref, bam_path)
            logger.info("Wrote per-read BAM to %s", bam_path)

    n_sig = sum(1 for s in sites if s.padj < 0.05)
    logger.info("Wrote %d sites (%d significant) to %s", len(sites), n_sig, args.output)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the baleen CLI."""
    parser = argparse.ArgumentParser(
        prog="baleen",
        description="Baleen: nanopore RNA modification detection via DTW + HMM",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress all output except errors",
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # run
    run_parser = subparsers.add_parser(
        "run",
        help="Run full pipeline: DTW + HMM + site-level aggregation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run the full baleen pipeline from raw inputs to site-level\n"
            "modification calls.\n\n"
            "Example:\n"
            "  baleen run \\\n"
            "    --native-bam native.bam \\\n"
            "    --native-fastq native.fq.gz \\\n"
            "    --native-blow5 native.blow5 \\\n"
            "    --ivt-bam ivt.bam \\\n"
            "    --ivt-fastq ivt.fq.gz \\\n"
            "    --ivt-blow5 ivt.blow5 \\\n"
            "    --ref ref.fa \\\n"
            "    -o results/"
        ),
    )
    _add_run_args(run_parser)

    # aggregate
    agg_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate saved results into site-level TSV",
        description=(
            "Re-run HMM and/or aggregate previously saved pipeline results\n"
            "into a site-level TSV. Useful for trying different HMM parameters\n"
            "without re-computing DTW distances.\n\n"
            "Example:\n"
            "  baleen aggregate -i results/pipeline_results.pkl -o sites.tsv\n"
            "  baleen aggregate -i results/pipeline_results.pkl -o sites.tsv \\\n"
            "    --hmm-params trained_hmm.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_aggregate_args(agg_parser)

    args = parser.parse_args(argv)

    # Configure logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    if args.command is None:
        parser.print_help()
        sys.exit(1)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "aggregate":
        _cmd_aggregate(args)


if __name__ == "__main__":
    main()
