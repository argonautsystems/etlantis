# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""Tests for etlantis.ingest.reader — read_table + read_many."""

from __future__ import annotations

import polars as pl
import pytest

from etlantis.ingest.reader import (
    ReadResult,
    _null_fraction,
    read_many,
    read_table,
)

# ============================================================================
# Format dispatch
# ============================================================================


def test_read_table_dispatches_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    result = read_table(path)
    assert result.df is not None
    assert result.df.height == 2
    assert result.encoding == "utf-8"
    assert result.rows == 2


def test_read_table_dispatches_tsv(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    result = read_table(path, csv_kwargs={"separator": "\t"})
    assert result.df is not None
    assert result.df.height == 1


def test_read_table_dispatches_parquet(tmp_path):
    df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    path = tmp_path / "data.parquet"
    df.write_parquet(path)
    result = read_table(path)
    assert result.df is not None
    assert result.df.height == 3
    assert result.encoding is None  # parquet has no encoding


def test_read_table_unsupported_extension(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"\x00\x01\x02")
    result = read_table(path)
    assert result.df is None
    assert "unsupported extension" in result.error


def test_read_table_missing_file(tmp_path):
    result = read_table(tmp_path / "nope.csv")
    assert result.df is None
    assert "file not found" in result.error


# ============================================================================
# CSV encoding fallback
# ============================================================================


def test_csv_utf8_happy_path(tmp_path):
    path = tmp_path / "ascii.csv"
    path.write_text("name,value\nfoo,1\nbar,2\n", encoding="utf-8")
    result = read_table(path)
    assert result.df is not None
    assert result.encoding == "utf-8"


def test_csv_falls_back_to_cp1252_for_smart_quotes(tmp_path):
    """Windows-1252 has \x93 / \x94 for curly quotes — invalid in UTF-8.
    The fallback chain should pick cp1252 ahead of latin-1 (cleaner
    decode for Windows-shaped sources)."""
    path = tmp_path / "smart.csv"
    # \x93 is opening curly quote in cp1252; \x94 is closing.
    path.write_bytes(b"name,value\n\x93cafe\x94,1\n\x93brun\xe9\x94,2\n")
    result = read_table(path)
    assert result.df is not None
    assert result.encoding in {"cp1252", "latin-1"}
    # cp1252 should win because it's earlier in the chain
    assert result.encoding == "cp1252"


def test_csv_falls_back_to_latin1_when_others_fail(tmp_path):
    """Custom encoding chain that excludes cp1252 — confirms the
    fallback walks every entry. latin-1 is universal so it always
    'succeeds' at decoding."""
    path = tmp_path / "latin.csv"
    path.write_bytes(b"name,value\ncaf\xe9,1\n")
    # Force a chain where utf-8 fails and only latin-1 remains
    result = read_table(path, encodings=("utf-8", "latin-1"))
    assert result.df is not None
    assert result.encoding == "latin-1"


def test_csv_uses_custom_encoding_chain(tmp_path):
    """Explicit encodings tuple overrides defaults."""
    path = tmp_path / "x.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    result = read_table(path, encodings=("utf-8",))
    assert result.df is not None
    assert result.encoding == "utf-8"


def test_csv_warns_on_high_null_fraction(tmp_path):
    """When delimiter is wrong, polars often parses one giant column.
    Heuristic: high null fraction → warn."""
    path = tmp_path / "wrong.csv"
    # Provide three columns but only one has data — null fraction = 2/3 = 0.667
    path.write_text("a,b,c\n1,,\n2,,\n", encoding="utf-8")
    result = read_table(path, null_warn_threshold=0.5)
    assert result.df is not None
    assert any("null cells" in w for w in result.warnings)


def test_csv_no_warn_when_below_threshold(tmp_path):
    path = tmp_path / "fine.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    result = read_table(path)
    # Healthy CSV — no warnings expected
    assert result.warnings == ()


def test_csv_default_is_strict(tmp_path):
    """A ragged CSV (extra columns on some rows) must NOT silently parse
    away — the default reader is strict, callers opt into lenient mode."""
    path = tmp_path / "ragged.csv"
    path.write_text("a,b\n1,2,3,4\n", encoding="utf-8")  # extra fields on row 1
    result = read_table(path)
    # Either the read returns an error OR (depending on Polars version) a
    # frame missing the dropped fields. The contract: we must NOT have
    # silently dropped data with no signal.
    assert result.df is None or result.warnings != ()


def test_csv_lenient_via_csv_kwargs(tmp_path):
    """Caller can opt in to lenient parsing for known-messy gov data."""
    path = tmp_path / "ragged.csv"
    path.write_text("a,b\n1,2,3,4\n", encoding="utf-8")
    result = read_table(
        path,
        csv_kwargs={"truncate_ragged_lines": True, "ignore_errors": True},
    )
    assert result.df is not None  # parses through the bad row


def test_csv_kwargs_encoding_is_ignored_with_warning(tmp_path, caplog):
    """If the caller passes csv_kwargs={'encoding': '...'}, we must NOT
    crash with 'multiple values for keyword argument' — the encoding chain
    is ours to control. The override is dropped with a warning."""
    import logging

    path = tmp_path / "ascii.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        result = read_table(path, csv_kwargs={"encoding": "ascii"})
    assert result.df is not None
    assert any("encoding" in r.message for r in caplog.records)


def test_csv_warns_on_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("a,b,c\n", encoding="utf-8")  # header only
    result = read_table(path)
    assert result.df is not None
    assert result.df.height == 0
    assert any("empty" in w for w in result.warnings)


# ============================================================================
# Parquet
# ============================================================================


def test_parquet_round_trip(tmp_path):
    src = pl.DataFrame({"id": [1, 2], "name": ["alpha", "beta"]})
    path = tmp_path / "x.parquet"
    src.write_parquet(path)
    result = read_table(path)
    assert result.df is not None
    assert result.df.equals(src)


def test_parquet_corrupt_file_returns_error(tmp_path):
    path = tmp_path / "fake.parquet"
    path.write_bytes(b"not actually parquet")
    result = read_table(path)
    assert result.df is None
    assert "parquet parse failed" in result.error


# ============================================================================
# read_many
# ============================================================================


def test_read_many_handles_mixed_outcomes(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("a,b\n1,2\n", encoding="utf-8")
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\x00")
    missing = tmp_path / "nope.csv"

    results = read_many([good, bad, missing])
    assert len(results) == 3
    assert results[0].df is not None
    assert results[1].df is None
    assert results[2].df is None


def test_read_many_preserves_order(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"f{i}.csv"
        p.write_text(f"a\n{i}\n", encoding="utf-8")
        paths.append(p)
    results = read_many(paths)
    assert [r.path for r in results] == paths


# ============================================================================
# Internal helpers
# ============================================================================


def test_null_fraction_empty_frame():
    df = pl.DataFrame({"a": []})
    assert _null_fraction(df) == 0.0


def test_null_fraction_no_nulls():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert _null_fraction(df) == 0.0


def test_null_fraction_half_nulls():
    df = pl.DataFrame({"a": [1, None], "b": [None, 2]})
    assert _null_fraction(df) == pytest.approx(0.5)


def test_read_result_warnings_is_immutable_tuple(tmp_path):
    """warnings is a tuple — frozen dataclass should be transitively
    immutable through attributes too, not just at the top level."""
    r = ReadResult(path=tmp_path / "x", df=None)
    assert r.warnings == ()
    assert isinstance(r.warnings, tuple)
    # Frozen dataclass: re-assignment of the field is rejected
    with pytest.raises((AttributeError, TypeError)):
        r.warnings = ("test",)  # type: ignore[misc]
