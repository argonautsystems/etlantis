# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""Tests for etlantis.score.weighted — WeightedScorer + ScoreBand."""

from __future__ import annotations

import polars as pl
import pytest

from etlantis.score import ScoreBand, WeightedScorer

# ============================================================================
# ScoreBand invariants
# ============================================================================


def test_score_band_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        ScoreBand(name="", min_score=0.0, max_score=10.0)


def test_score_band_rejects_max_le_min():
    with pytest.raises(ValueError, match="strictly greater"):
        ScoreBand(name="bad", min_score=10.0, max_score=10.0)
    with pytest.raises(ValueError, match="strictly greater"):
        ScoreBand(name="bad", min_score=10.0, max_score=5.0)


def test_score_band_accepts_unbounded_endpoints():
    """None on either bound is valid — used for the bottom and top
    bands of a partition (e.g. low: max=10 with min=None)."""
    ScoreBand(name="low", min_score=None, max_score=10.0)
    ScoreBand(name="high", min_score=20.0, max_score=None)
    ScoreBand(name="all", min_score=None, max_score=None)


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, True),
        (5.0, True),
        (10.0, False),  # exclusive max
        (-0.1, False),
    ],
)
def test_score_band_contains_inclusive_min_exclusive_max(score, expected):
    band = ScoreBand(name="low", min_score=0.0, max_score=10.0)
    assert band.contains(score) is expected


def test_score_band_contains_unbounded():
    """No-bound bands accept everything on the unbounded side."""
    no_min = ScoreBand(name="below_high", min_score=None, max_score=10.0)
    assert no_min.contains(-9999.0)
    no_max = ScoreBand(name="above_low", min_score=20.0, max_score=None)
    assert no_max.contains(1e9)


# ============================================================================
# WeightedScorer init
# ============================================================================


def test_weighted_scorer_rejects_empty_weights():
    """Empty weights would silently produce all-zero scores —
    classic footgun. Reject at construction."""
    with pytest.raises(ValueError, match="at least one weighted column"):
        WeightedScorer(weights={})


def test_weighted_scorer_validates_band_overlap():
    """Two overlapping bands should be caught at init, not silently
    masking each other at score-time."""
    overlapping = [
        ScoreBand(name="a", min_score=0.0, max_score=15.0),
        ScoreBand(name="b", min_score=10.0, max_score=20.0),
    ]
    with pytest.raises(ValueError, match="overlap"):
        WeightedScorer(weights={"x": 1.0}, bands=overlapping)


def test_weighted_scorer_accepts_adjacent_bands():
    """Adjacent bands (low.max == high.min) are NOT an overlap because
    max is exclusive. Should pass validation."""
    adjacent = [
        ScoreBand(name="low", min_score=0.0, max_score=10.0),
        ScoreBand(name="high", min_score=10.0, max_score=20.0),
    ]
    WeightedScorer(weights={"x": 1.0}, bands=adjacent)  # should not raise


def test_weighted_scorer_accepts_unsorted_bands():
    """Bands in input order shouldn't matter; the scorer sorts
    internally for overlap-validation and chain-building."""
    unsorted = [
        ScoreBand(name="high", min_score=20.0, max_score=None),
        ScoreBand(name="low", min_score=None, max_score=10.0),
        ScoreBand(name="medium", min_score=10.0, max_score=20.0),
    ]
    scorer = WeightedScorer(weights={"x": 1.0}, bands=unsorted)
    df = pl.DataFrame({"x": [5.0, 15.0, 25.0]})
    result = scorer.score(df)
    assert result["_band"].to_list() == ["low", "medium", "high"]


# ============================================================================
# WeightedScorer.score — happy path
# ============================================================================


def test_score_simple_weighted_sum():
    scorer = WeightedScorer(weights={"a": 1.0, "b": 2.0})
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    result = scorer.score(df)
    # a*1 + b*2 → [21, 42, 63]
    assert result["_score"].to_list() == [21.0, 42.0, 63.0]


def test_score_negative_weights():
    """Negative weights subtract from the score — used for credits or
    risk-reduction factors."""
    scorer = WeightedScorer(weights={"violations": 5.0, "credit": -3.0})
    df = pl.DataFrame({"violations": [4, 4], "credit": [0, 1]})
    result = scorer.score(df)
    # 4*5 - 0*3 = 20; 4*5 - 1*3 = 17
    assert result["_score"].to_list() == [20.0, 17.0]


def test_score_passes_through_other_columns():
    """The score column gets added; existing columns survive untouched
    in their original order."""
    scorer = WeightedScorer(weights={"x": 1.0})
    df = pl.DataFrame({"id": [1, 2], "x": [10, 20], "name": ["a", "b"]})
    result = scorer.score(df)
    assert "id" in result.columns
    assert "name" in result.columns
    assert "_score" in result.columns
    assert result["id"].to_list() == [1, 2]


def test_score_does_not_mutate_input():
    """Polars frames are immutable but defensive sanity check —
    `df` should be unchanged after scoring."""
    scorer = WeightedScorer(weights={"x": 1.0})
    df = pl.DataFrame({"x": [1, 2, 3]})
    cols_before = df.columns
    _ = scorer.score(df)
    assert df.columns == cols_before
    assert "_score" not in df.columns


def test_score_treats_null_as_zero():
    """Public-records data is rife with sparse columns; treating null
    as 0 lets a row with one missing field still produce a coherent
    score from its other weighted columns."""
    scorer = WeightedScorer(weights={"a": 1.0, "b": 2.0})
    df = pl.DataFrame({"a": [1, None, 3], "b": [10, 20, None]})
    result = scorer.score(df)
    # 1*1 + 10*2 = 21; 0*1 + 20*2 = 40; 3*1 + 0*2 = 3
    assert result["_score"].to_list() == [21.0, 40.0, 3.0]


# ============================================================================
# WeightedScorer.score — schema-drift defense
# ============================================================================


def test_score_raises_on_missing_weighted_column():
    """Treating a missing column as zero would silently let a renamed
    column tank scores. Hard error instead — flag the schema drift."""
    scorer = WeightedScorer(weights={"missing_col": 1.0})
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing from DataFrame"):
        scorer.score(df)


def test_score_error_lists_available_columns():
    """The error message must surface what columns DO exist so callers
    can spot the rename without round-tripping through a print."""
    scorer = WeightedScorer(weights={"score_field": 1.0})
    df = pl.DataFrame({"x": [1], "y": [2]})
    with pytest.raises(ValueError) as excinfo:
        scorer.score(df)
    assert "x" in str(excinfo.value)
    assert "y" in str(excinfo.value)


# ============================================================================
# WeightedScorer + bands
# ============================================================================


def test_score_with_bands_assigns_categorical():
    scorer = WeightedScorer(
        weights={"violations": 5.0},
        bands=[
            ScoreBand(name="low", min_score=None, max_score=10.0),
            ScoreBand(name="medium", min_score=10.0, max_score=20.0),
            ScoreBand(name="high", min_score=20.0, max_score=None),
        ],
    )
    # violations * 5 → 5, 15, 25 → low, medium, high
    df = pl.DataFrame({"violations": [1, 3, 5]})
    result = scorer.score(df)
    assert result["_band"].to_list() == ["low", "medium", "high"]


def test_score_with_bands_exclusive_max_at_boundary():
    """A score at the exact max boundary should fall into the NEXT
    band, not the band whose max it equals."""
    scorer = WeightedScorer(
        weights={"x": 1.0},
        bands=[
            ScoreBand(name="low", min_score=0.0, max_score=10.0),
            ScoreBand(name="high", min_score=10.0, max_score=20.0),
        ],
    )
    df = pl.DataFrame({"x": [9.999, 10.0, 10.001]})
    result = scorer.score(df)
    # 9.999 → low; 10.0 → high; 10.001 → high
    assert result["_band"].to_list() == ["low", "high", "high"]


def test_score_with_bands_uncovered_score_yields_null():
    """A score that falls outside every band gets None — apps decide
    whether that's a bug or expected (e.g. the bands intentionally
    only cover the "interesting" range)."""
    scorer = WeightedScorer(
        weights={"x": 1.0},
        bands=[ScoreBand(name="middle", min_score=10.0, max_score=20.0)],
    )
    df = pl.DataFrame({"x": [5.0, 15.0, 25.0]})
    result = scorer.score(df)
    assert result["_band"].to_list() == [None, "middle", None]


def test_score_no_bands_omits_band_column():
    """When bands is None, the band column shouldn't appear at all
    (caller didn't ask for it)."""
    scorer = WeightedScorer(weights={"x": 1.0})
    df = pl.DataFrame({"x": [1, 2]})
    result = scorer.score(df)
    assert "_band" not in result.columns


def test_score_custom_column_names():
    """Apps shipping side-by-side scores need to override the default
    `_score` / `_band` names."""
    scorer = WeightedScorer(
        weights={"x": 1.0},
        bands=[ScoreBand(name="any", min_score=None, max_score=None)],
        score_column="risk_score",
        band_column="risk_band",
    )
    df = pl.DataFrame({"x": [10]})
    result = scorer.score(df)
    assert "risk_score" in result.columns
    assert "risk_band" in result.columns
    assert "_score" not in result.columns
    assert "_band" not in result.columns
