# Read-ID Intersection

Enumeration of read IDs from each input file type and the per-condition
three-way intersection `reads(BAM) ∩ reads(FASTQ) ∩ reads(BLOW5)`. For the
rationale see [Inputs › Read-ID intersection](../guide/inputs.md#read-id-intersection).

!!! note "Internal module"
    These live in `baleen.eventalign._read_ids`. They are not part of the stable
    top-level API but are documented here because the intersection is a core
    pipeline guarantee.

## Enumeration

::: baleen.eventalign._read_ids.read_ids_from_bam

::: baleen.eventalign._read_ids.read_ids_from_fastq

::: baleen.eventalign._read_ids.read_ids_from_blow5

## Intersection & persistence

::: baleen.eventalign._read_ids.compute_condition_intersection

::: baleen.eventalign._read_ids.write_read_ids

::: baleen.eventalign._read_ids.load_read_ids
