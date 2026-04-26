# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""etlantis.transform — Polars-native data transformation primitives.

Subsystems (Phase 1):

    consolidate       concat_frames() vertically combines a list of
                      DataFrames with optional dedupe + sort + column-name
                      normalization. Generic — apps assemble bucket
                      routing themselves. Lifted from cleanroom
                      P0_consolidate_vectorized.py.

    output            write_parquet() persists a DataFrame as Parquet,
                      single-file (atomic via .part rename) or hive-
                      partitioned. Lifted from cleanroom L0_output_parquet.py.

Planned (Phase 2+):

    schema_map        Wraps clio.extract.schema_map for manifest-driven
                      column rename + type coercion across heterogeneous
                      sources.

    drift             Wraps clio.track.drift for per-column drift
                      detection between today's snapshot and yesterday's.
"""

from etlantis.transform.consolidate import ConsolidationResult, concat_frames
from etlantis.transform.output import Compression, WriteResult, write_parquet

__all__ = [
    "concat_frames",
    "ConsolidationResult",
    "write_parquet",
    "WriteResult",
    "Compression",
]
