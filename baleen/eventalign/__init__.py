"""Eventalign pipeline for nanopore signal analysis.

This package provides an end-to-end pipeline for comparing native and IVT
nanopore signals using f5c eventalign and DTW distance computation.

Public API
----------
run_pipeline
    Run the full eventalign pipeline from BAM/FASTQ/BLOW5 inputs.
save_results
    Persist pipeline results to disk (pickle format).
load_results
    Load previously saved pipeline results.

Data classes
------------
PositionResult
    Per-position DTW distance matrix and metadata.
ContigResult
    Per-contig results containing all position results.
PipelineMetadata
    Pipeline run metadata (f5c version, filtering stats, etc.).
ContigStats
    Per-contig mapping statistics from BAM.
ContigFilterResult
    Outcome of depth-based contig filtering.
FilterReason
    Enum describing why a contig passed or failed filtering.
PositionSignals
    Grouped eventalign signals for a single genomic position.
"""

from baleen.eventalign._bam import (
    ContigFilterResult,
    ContigStats,
    FilterReason,
)
from baleen.eventalign._pipeline import (
    ContigResult,
    PipelineMetadata,
    PositionResult,
    load_results,
    run_pipeline,
    save_results,
)
from baleen.eventalign._signal import PositionSignals

__all__ = [
    "run_pipeline",
    "save_results",
    "load_results",
    "PositionResult",
    "ContigResult",
    "PipelineMetadata",
    "ContigStats",
    "ContigFilterResult",
    "FilterReason",
    "PositionSignals",
]
