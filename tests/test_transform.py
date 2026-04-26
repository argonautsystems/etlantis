# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""Tests for etlantis.transform — consolidate + output."""

from __future__ import annotations

import polars as pl
import pytest

from etlantis.transform import (
    ConsolidationResult,
    WriteResult,
    concat_frames,
    write_parquet,
)
from etlantis.transform.consolidate import _normalize_columns

# ============================================================================
# concat_frames — empty / passthrough
# ============================================================================


def test_concat_empty_list_returns_empty_result():
    result = concat_frames([])
    assert isinstance(result, ConsolidationResult)
    assert result.df.height == 0
    assert result.input_frames == 0
    assert result.rows_in == 0
    assert result.rows_out == 0
    assert result.duplicates_removed == 0
    assert result.column_renames == ()


def test_concat_filters_out_zero_height_frames():
    """Empty frames in the list mustn't break concat or count toward
    input_frames."""
    a = pl.DataFrame({"x": [1, 2]})
    b = pl.DataFrame({"x": []})  # zero rows
    c = pl.DataFrame({"x": [3]})
    result = concat_frames([a, b, c])
    assert result.input_frames == 2
    assert result.rows_in == 3
    assert result.rows_out == 3


def test_concat_single_frame_passthrough():
    df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    result = concat_frames([df])
    assert result.df.equals(df)
    assert result.duplicates_removed == 0


# ============================================================================
# concat_frames — vertical concat
# ============================================================================


def test_concat_vertical_concat():
    a = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    b = pl.DataFrame({"x": [3, 4], "y": ["c", "d"]})
    result = concat_frames([a, b])
    assert result.df.height == 4
    assert result.df["x"].to_list() == [1, 2, 3, 4]
    assert result.input_frames == 2


def test_concat_diagonal_relaxed_handles_disjoint_columns():
    """Diagonal-relaxed concat fills missing columns with null instead
    of failing — useful for merging week-1 vs week-2 schema drift."""
    a = pl.DataFrame({"x": [1], "y": [10]})
    b = pl.DataFrame({"x": [2], "z": [20]})
    result = concat_frames([a, b])
    # Both 'y' and 'z' present; one row null for each
    assert "y" in result.df.columns
    assert "z" in result.df.columns
    assert result.df.height == 2


# ============================================================================
# concat_frames — dedupe
# ============================================================================


def test_concat_dedupe_subset_drops_duplicates():
    a = pl.DataFrame({"id": [1, 2], "v": ["a", "b"]})
    b = pl.DataFrame({"id": [2, 3], "v": ["b2", "c"]})
    result = concat_frames([a, b], dedupe_subset=["id"])
    # First-occurrence wins: id=2 keeps "b" not "b2"
    assert result.df.height == 3
    assert result.duplicates_removed == 1


def test_concat_dedupe_no_subset_skips_dedupe():
    a = pl.DataFrame({"id": [1, 1]})
    result = concat_frames([a])  # dedupe_subset=None
    assert result.df.height == 2  # preserved
    assert result.duplicates_removed == 0


def test_concat_dedupe_empty_subset_dedupes_all_columns():
    """dedupe_subset=[] drops fully-duplicate rows."""
    a = pl.DataFrame({"x": [1, 1, 2], "y": ["a", "a", "b"]})
    result = concat_frames([a], dedupe_subset=[])
    assert result.df.height == 2  # one (1,a) row dropped
    assert result.duplicates_removed == 1


def test_concat_dedupe_rejects_missing_key_in_any_frame():
    """If dedupe_subset names a column missing from any input frame,
    diagonal_relaxed would null-fill it and unique() would collapse all
    null-keyed rows into one — silent severe data loss. Catch upfront."""
    a = pl.DataFrame({"id": [1, 2], "v": ["a", "b"]})
    b = pl.DataFrame({"v": ["c", "d"]})  # no 'id'
    with pytest.raises(ValueError, match="dedupe_subset.*missing"):
        concat_frames([a, b], dedupe_subset=["id"])


# ============================================================================
# concat_frames — sort
# ============================================================================


def test_concat_sort_ascending():
    a = pl.DataFrame({"x": [3, 1, 2]})
    result = concat_frames([a], sort_by=["x"])
    assert result.df["x"].to_list() == [1, 2, 3]


def test_concat_sort_descending():
    a = pl.DataFrame({"x": [3, 1, 2]})
    result = concat_frames([a], sort_by=["x"], sort_descending=True)
    assert result.df["x"].to_list() == [3, 2, 1]


# ============================================================================
# concat_frames — column normalization
# ============================================================================


def test_concat_strips_column_whitespace():
    a = pl.DataFrame({" name": ["x"], "value": [1]})
    result = concat_frames([a])
    assert "name" in result.df.columns
    assert " name" not in result.df.columns
    # Per-frame rename map: index 0 → its own dict
    assert result.column_renames == ({" name": "name"},)


def test_concat_disambiguates_collisions_clean_owns_canonical():
    """Two columns that strip to the same name: the already-clean version
    keeps the canonical (unsuffixed) slot. The dirty version takes _1.
    This protects manifest-driven downstream code that expects the clean
    column to BE the clean column."""
    a = pl.DataFrame({" name": ["x"], "name": ["y"]})
    result = concat_frames([a])
    # ' name' → 'name_1' (collision-suffixed because 'name' is already clean)
    # 'name'  → 'name'   (canonical stays canonical)
    assert "name" in result.df.columns
    assert "name_1" in result.df.columns
    frame_renames = result.column_renames[0]
    assert frame_renames[" name"] == "name_1"
    # 'name' was unchanged — not in the rename map at all
    assert "name" not in frame_renames


def test_concat_normalize_off_preserves_names():
    a = pl.DataFrame({" name": ["x"]})
    result = concat_frames([a], normalize_column_names=False)
    assert " name" in result.df.columns
    # Rename map: one frame, empty dict (no normalization)
    assert result.column_renames == ({},)


def test_concat_per_frame_renames_are_independent():
    """Two frames with different rename needs should produce two
    independent rename dicts in the result, not a flat map that loses
    cross-frame distinctions."""
    a = pl.DataFrame({" name": ["x"]})  # ' name' → 'name'
    b = pl.DataFrame({"name ": ["y"]})  # 'name ' → 'name'
    result = concat_frames([a, b])
    assert len(result.column_renames) == 2
    assert result.column_renames[0] == {" name": "name"}
    assert result.column_renames[1] == {"name ": "name"}


# ============================================================================
# _normalize_columns helper
# ============================================================================


def test_normalize_columns_no_changes_when_clean():
    a = pl.DataFrame({"name": ["x"], "id": [1]})
    out, renames = _normalize_columns([a])
    assert renames == ({},)
    assert out[0].columns == ["name", "id"]


def test_normalize_columns_preserves_order():
    """Renames should not reshuffle column order."""
    a = pl.DataFrame({"c": [1], " a": [2], "b ": [3]})
    out, _ = _normalize_columns([a])
    assert out[0].columns == ["c", "a", "b"]


# ============================================================================
# write_parquet — single file
# ============================================================================


def test_write_parquet_single_file_round_trip(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    path = tmp_path / "data.parquet"
    result = write_parquet(df, path)
    assert isinstance(result, WriteResult)
    assert result.rows == 3
    assert result.files_written == 1
    assert result.bytes_written > 0
    # Round-trip
    re_read = pl.read_parquet(path)
    assert re_read.equals(df)


def test_write_parquet_creates_parent_dirs(tmp_path):
    df = pl.DataFrame({"x": [1]})
    path = tmp_path / "deep" / "nested" / "data.parquet"
    write_parquet(df, path)
    assert path.exists()


def test_write_parquet_atomic_no_part_left_on_success(tmp_path):
    df = pl.DataFrame({"x": [1]})
    path = tmp_path / "data.parquet"
    write_parquet(df, path)
    # .part file must be renamed away on success
    assert not path.with_suffix(path.suffix + ".part").exists()


def test_write_parquet_refuses_empty_frame(tmp_path):
    df = pl.DataFrame({"x": []})
    with pytest.raises(ValueError, match="empty"):
        write_parquet(df, tmp_path / "empty.parquet")
    # And nothing was written
    assert not (tmp_path / "empty.parquet").exists()


# ============================================================================
# write_parquet — partitioned
# ============================================================================


def test_write_parquet_partitioned_creates_subdirs(tmp_path):
    df = pl.DataFrame(
        {
            "county": ["DADE", "DADE", "BROWARD"],
            "value": [1, 2, 3],
        }
    )
    path = tmp_path / "out"
    result = write_parquet(df, path, partition_cols=["county"])
    assert result.files_written >= 2
    # Hive layout: county=DADE/, county=BROWARD/
    assert (path / "county=DADE").exists()
    assert (path / "county=BROWARD").exists()


def test_write_parquet_partitioned_rejects_missing_column(tmp_path):
    df = pl.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="partition columns not present"):
        write_parquet(df, tmp_path / "out", partition_cols=["county"])


def test_write_parquet_partitioned_refuses_non_empty_path(tmp_path):
    """PyArrow appends rather than replaces, so a second write into the
    same dataset root would mix old and new data. Default behavior:
    refuse non-empty path."""
    df = pl.DataFrame({"county": ["A"], "value": [1]})
    path = tmp_path / "out"
    write_parquet(df, path, partition_cols=["county"])

    df2 = pl.DataFrame({"county": ["B"], "value": [2]})
    with pytest.raises(ValueError, match="not empty"):
        write_parquet(df2, path, partition_cols=["county"])


def test_write_parquet_partitioned_overwrite_clears_first(tmp_path):
    """overwrite=True should clear the existing dataset root before
    writing — no stale partitions, no mixed-data results."""
    df = pl.DataFrame({"county": ["A"], "value": [1]})
    path = tmp_path / "out"
    write_parquet(df, path, partition_cols=["county"])

    df2 = pl.DataFrame({"county": ["B"], "value": [2]})
    result = write_parquet(df2, path, partition_cols=["county"], overwrite=True)

    # Old partition gone; only the new one remains
    assert (path / "county=B").exists()
    assert not (path / "county=A").exists()
    # files_written counts only the new partition's files
    assert result.files_written >= 1


def test_write_parquet_partitioned_round_trip(tmp_path):
    """Partitioned write + read returns the same logical frame.

    Hive partition columns are encoded in the directory name (e.g.
    `county=DADE/`), not in the parquet file payload. Reading them back
    requires `hive_partitioning=True` on `pl.scan_parquet`. Without it,
    the partition column won't be reconstructed and we'd see only the
    `value` column.
    """
    df = pl.DataFrame(
        {
            "county": ["A", "A", "B"],
            "value": [1, 2, 3],
        }
    ).sort("county", "value")
    path = tmp_path / "out"
    write_parquet(df, path, partition_cols=["county"])
    re_read = (
        pl.scan_parquet(path / "**/*.parquet", hive_partitioning=True)
        .sort("county", "value")
        .collect()
    )
    assert sorted(re_read["value"].to_list()) == [1, 2, 3]
    assert sorted(re_read["county"].to_list()) == ["A", "A", "B"]
