# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""etlantis.transform.output — write Polars DataFrames to Parquet.

Generalized from cleanroom `L0_output_parquet.py`. The cleanroom version
was pandas-based and DBPR-specific (county-partitioned by default).
This version is:

  * Polars-native — writes via `pl.DataFrame.write_parquet()` which
    delegates to PyArrow.
  * Generic — partition columns are caller-specified, no defaults baked
    in. Single-file write when `partition_cols` is empty/None.
  * Atomic for single-file writes — writes to a `.part` then renames so
    a crash mid-write doesn't leave a half-written parquet on disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

Compression = Literal["snappy", "gzip", "lz4", "zstd", "uncompressed"]


@dataclass(frozen=True)
class WriteResult:
    """Outcome of a `write_parquet()` call."""

    path: Path
    """Where the file (or partitioned directory root) landed."""

    rows: int
    """Row count written."""

    files_written: int
    """For non-partitioned writes: 1. For partitioned writes: count of
    `.parquet` files under `path` (one per partition combination)."""

    bytes_written: int
    """Total parquet bytes on disk under `path`."""


def write_parquet(
    df: pl.DataFrame,
    path: Path | str,
    *,
    partition_cols: list[str] | None = None,
    compression: Compression = "snappy",
    overwrite: bool = False,
) -> WriteResult:
    """Write a Polars DataFrame to Parquet.

    Behavior:
      - When `partition_cols` is empty or None: single-file write to
        `path`. Atomic via `<path>.part` + rename. An existing file at
        `path` is overwritten (the rename is atomic).
      - When `partition_cols` is non-empty: hive-partitioned write under
        `path` (treated as a directory). Polars delegates the partition
        layout to PyArrow via `write_parquet(... use_pyarrow=True,
        pyarrow_options={"partition_cols": ...})`. PyArrow APPENDS to
        existing partitions rather than replacing them, so writing into
        a non-empty `path` would silently mix old and new data and the
        returned `files_written`/`bytes_written` would count stale files
        too. To prevent that we refuse a non-empty `path` unless
        `overwrite=True` is passed (in which case the directory tree is
        cleared first). Atomic semantics for partitioned writes are NOT
        provided — a crash mid-partition leaves a partial tree; callers
        wanting full atomicity should write to a staging dir and rename
        the directory.

    Args:
        df: DataFrame to persist.
        path: Output file path (or directory root for partitioned).
            Parent directories are created.
        partition_cols: Columns to partition by. Empty / None → single
            file. The columns are encoded into the directory layout
            (e.g. `path/county=DADE/...`) and removed from the parquet
            payload (standard hive convention).
        compression: Parquet compression codec. Default snappy.
        overwrite: When True, allow writing into a non-empty
            partitioned-output directory by clearing it first. Has no
            effect on single-file writes (rename is atomic). Default
            False so callers must opt in to destructive behavior.

    Returns:
        WriteResult with paths + sizes.

    Raises:
        ValueError: If `df` is empty (zero rows); if a partition column
            isn't in the DataFrame; or if `path` is a non-empty directory
            and `overwrite=False` for a partitioned write.
        OSError: Any underlying file-system error during write or rename.
    """
    if df.height == 0:
        raise ValueError("refusing to write empty DataFrame to parquet")

    path = Path(path)

    if partition_cols:
        return _write_partitioned(df, path, partition_cols, compression, overwrite)
    return _write_single(df, path, compression)


def _write_single(df: pl.DataFrame, path: Path, compression: Compression) -> WriteResult:
    """Single-file write with `.part` rename for atomicity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    try:
        df.write_parquet(tmp_path, compression=compression)
    except Exception:
        # If the write blew up, there's potentially a partial .part on
        # disk. Try to clean up so the next call doesn't see it.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as exc:
                logger.warning("[output] could not remove partial %s: %s", tmp_path, exc)
        raise

    tmp_path.replace(path)
    bytes_written = path.stat().st_size
    logger.info("[output] wrote %s (%d rows, %d bytes)", path, df.height, bytes_written)
    return WriteResult(path=path, rows=df.height, files_written=1, bytes_written=bytes_written)


def _write_partitioned(
    df: pl.DataFrame,
    path: Path,
    partition_cols: list[str],
    compression: Compression,
    overwrite: bool,
) -> WriteResult:
    """Hive-partitioned write via PyArrow underneath Polars.

    Polars exposes partitioning through `write_parquet(..., use_pyarrow=True,
    pyarrow_options={"partition_cols": [...]})`. Validates partition columns
    exist in the frame before delegating, since PyArrow's error message for
    a missing column is opaque.

    Refuse-to-clobber: if `path` exists and contains files, abort unless
    `overwrite=True`. PyArrow appends rather than replaces, so writing
    into a non-empty dataset root would silently mix new partitions with
    stale ones from a prior run.
    """
    missing = [c for c in partition_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"partition columns not present in DataFrame: {missing}; "
            f"available columns: {df.columns}"
        )

    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise ValueError(
                f"partitioned output path {path} is not empty; "
                f"pass overwrite=True to clear it before writing, or "
                f"choose a fresh path. PyArrow appends rather than "
                f"replaces, so writing into a non-empty dataset would "
                f"mix old and new data."
            )
        # Clear the existing tree before writing.
        import shutil

        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    df.write_parquet(
        path,
        compression=compression,
        use_pyarrow=True,
        pyarrow_options={"partition_cols": partition_cols},
    )
    files = sorted(path.rglob("*.parquet"))
    bytes_written = sum(f.stat().st_size for f in files)
    logger.info(
        "[output] wrote %d partition files under %s (%d rows, %d bytes)",
        len(files),
        path,
        df.height,
        bytes_written,
    )
    return WriteResult(
        path=path,
        rows=df.height,
        files_written=len(files),
        bytes_written=bytes_written,
    )


__all__ = ["write_parquet", "WriteResult", "Compression"]
