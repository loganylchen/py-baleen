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
    run_pipeline_streaming,
    save_results,
    write_site_tsv,
)
from baleen.eventalign._read_bam import write_mod_bam


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
        "--padding", type=int, default=1,
        help="Flanking positions for DTW signal concatenation (default: 1)",
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
        "--threads", type=int, default=8,
        help="Number of parallel workers for contig processing (default: 8)",
    )

    pipe.add_argument(
        "--target", type=str, default=None,
        help="Target contig(s): a contig name, comma-separated list, "
             "or path to file with one contig per line",
    )
    pipe.add_argument(
        "--keep-intermediate", action="store_true", default=False,
        help="Save per-contig DTW intermediate results (ContigResult files)",
    )
    pipe.add_argument(
        "--subsample", action="store_true", default=False,
        help="Subsample reads per condition per contig to reduce memory usage",
    )
    pipe.add_argument(
        "--subsample-n", type=int, default=300,
        help="Max reads per condition per contig when --subsample is enabled (default: 300)",
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
    dtw.add_argument(
        "--gpu-memory-limit", type=int, default=None, metavar="BYTES",
        help="GPU memory budget in bytes for concurrent DTW workers (default: auto-detect)",
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
        "--f5c-threads", type=int, default=None,
        help="CPU threads per f5c eventalign call (default: auto = total_cores / threads)",
    )
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
        help="Skip writing mod-BAM output (read_results.bam with MM/ML tags)",
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
        help="Skip writing mod-BAM output",
    )
    parser.add_argument(
        "--ref", type=str, default=None,
        help="Reference FASTA (required for mod-BAM output)",
    )
    parser.add_argument(
        "--native-bam", type=str, default=None,
        help="Native BAM file (required for mod-BAM output)",
    )
    parser.add_argument(
        "--ivt-bam", type=str, default=None,
        help="IVT BAM file (required for mod-BAM output)",
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

    # Parse --target
    target_contigs = None
    if args.target:
        target_path = Path(args.target)
        if target_path.is_file():
            target_contigs = [l.strip() for l in target_path.read_text().splitlines() if l.strip()]
        else:
            target_contigs = [c.strip() for c in args.target.split(',')]

    # Load HMM params if provided
    logger = logging.getLogger("baleen")
    hmm_params = None
    if args.hmm_params:
        hmm_params = load_hmm_params(args.hmm_params)
        logger.info("Loaded HMM params: %s (%d-state %s)",
                    args.hmm_params, hmm_params.n_states, hmm_params.mode)

    # Auto-compute f5c threads: total_cores / pipeline_workers, clamped to [2, 16]
    import os
    f5c_threads = args.f5c_threads
    if f5c_threads is None:
        total_cores = os.cpu_count() or 4
        f5c_threads = max(2, min(16, total_cores // max(args.threads, 1)))
    # Inject into extra_f5c_args (f5c uses -t for threads, --iop for I/O threads)
    extra_f5c_args = []
    if hasattr(args, 'extra_f5c_args') and args.extra_f5c_args:
        extra_f5c_args = list(args.extra_f5c_args)
    if '-t' not in extra_f5c_args:
        extra_f5c_args.extend(['-t', str(f5c_threads)])
    iop_threads = max(1, f5c_threads // 2)
    if '--iop' not in extra_f5c_args:
        extra_f5c_args.extend(['--iop', str(iop_threads)])
    logger.info("f5c threads: -t %d --iop %d (pipeline workers: %d)",
                f5c_threads, iop_threads, args.threads)

    # Run streaming pipeline (DTW → HMM → aggregation fused per contig)
    hmm_results, sites, metadata = run_pipeline_streaming(
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
        extra_f5c_args=extra_f5c_args,
        min_mapq=args.min_mapq,
        primary_only=not args.no_primary_only,
        threads=args.threads,
        run_hmm=not args.no_hmm,
        hmm_params=hmm_params,
        target_contigs=target_contigs,
        keep_intermediate=args.keep_intermediate,
        gpu_memory_limit=args.gpu_memory_limit,
        subsample=args.subsample,
        subsample_n=args.subsample_n,
    )

    # Write outputs
    tsv_path = output_dir / "site_results.tsv"
    write_site_tsv(sites, tsv_path)

    bam_path = None
    if not args.no_read_bam:
        bam_path = output_dir / "read_results.bam"
        t0 = time.perf_counter()
        write_mod_bam(hmm_results, args.native_bam, args.ivt_bam, args.ref, bam_path)
        logger.info("mod-BAM done in %.1fs", time.perf_counter() - t0)

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
    if args.keep_intermediate:
        logger.info("  Intermediate:      %s", output_dir / "intermediate")
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
        missing = []
        if args.ref is None:
            missing.append("--ref")
        if args.native_bam is None:
            missing.append("--native-bam")
        if args.ivt_bam is None:
            missing.append("--ivt-bam")
        if missing:
            logger.warning(
                "Skipping mod-BAM output: %s not provided", ", ".join(missing),
            )
        else:
            output_parent = Path(args.output).parent.resolve()
            bam_path = output_parent / "read_results.bam"
            write_mod_bam(hmm_results, args.native_bam, args.ivt_bam, args.ref, bam_path)
            logger.info("Wrote mod-BAM to %s", bam_path)

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
            '''                                                                                                    
                                                                                                    
           *#M###***                                                                                
         *  #  /  8.  **+                                                                           
      &w                 p *o                                                                       
      #[a                   m   *                                          *#                  oa   
     #                          :  #*                                     MB  ****%    M***o*    *  
     M                             `  ook                                 M    ^ ] # *    |      #  
     M                                 \  M*#                             #         * x       `to   
    *   8%                                 '   o                           #               C O o    
    *                                            a                           oo         "`#**       
    #                                               koo                          o0   *   .         
    M                                                }   {**                   * .   *              
     *                                                       Moo    #       WM       *              
       #*****?* <# #M#                                         M     *****.          *              
           f **#**#*#*# #MMM            #M                             &+           zC              
                 MM#***#***#|*#         ]                                         v.*               
                     il##********#\                                               *.                
                     L **###**M#***   0*                                          *                 
              & o#*#o*;#*0##. ..         #M                                      *                  
            #                        o         &                           ^  ^.*                   
             ,&#**& ^ #MxJ   MM-    o    M#    aw                     t     dh(                     
                  ^|uvx# & :W#   QaqO   *o     **         #              % *d                       
                        MMWx{i fxMM   `'#     #***          #O*#M''      #*                         
                              M##J{   ***8    n**#I:        *   8p  #oa                             
                                  M######^     #*o**         l*##*                                  
                                  '###****#    **oo**M '       o                                    
                                    **##*#             #          o                                 
                                       *#*               joo | ; .@#                 '''
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
