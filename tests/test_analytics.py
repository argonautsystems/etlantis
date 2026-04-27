# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""Tests for etlantis.analytics — velocity helpers + TrajectoryClassifier."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from etlantis.analytics import (
    TrajectoryClassifier,
    age_in_days,
    days_between,
    days_since,
    days_since_last,
)

# ============================================================================
# velocity.days_since / age_in_days / days_between
# ============================================================================


def test_days_since_with_explicit_anchor():
    df = pl.DataFrame({"d": [date(2024, 1, 1), date(2025, 1, 1)]})
    result = df.with_columns(days_since("d", anchor=date(2026, 1, 1)).alias("days"))
    # 2026-01-01 minus 2024-01-01 = 731 (leap year), minus 2025-01-01 = 365
    assert result["days"].to_list() == [731, 365]


def test_days_since_default_anchor_uses_today():
    """Default anchor is `date.today()`. Compute against a fixed past
    date and verify the result is non-negative + roughly today-ish."""
    df = pl.DataFrame({"d": [date(2024, 1, 1)]})
    result = df.with_columns(days_since("d").alias("days"))
    days = result["days"].item()
    # As of any time after 2024-01-01, should be at least 1 day.
    assert days >= 1


def test_days_since_negative_for_future_dates():
    df = pl.DataFrame({"d": [date(2030, 1, 1)]})
    result = df.with_columns(days_since("d", anchor=date(2026, 1, 1)).alias("days"))
    # 2026-01-01 minus 2030-01-01 = -1461 (4 years, includes one leap)
    assert result["days"].item() < 0


def test_days_since_propagates_null():
    df = pl.DataFrame({"d": [date(2024, 1, 1), None]}, schema={"d": pl.Date})
    result = df.with_columns(days_since("d", anchor=date(2026, 1, 1)).alias("days"))
    days = result["days"].to_list()
    assert days[0] == 731
    assert days[1] is None


def test_days_since_handles_datetime_anchor():
    """Passing a datetime as anchor should work — it's truncated to date."""
    df = pl.DataFrame({"d": [date(2024, 1, 1)]})
    result = df.with_columns(days_since("d", anchor=datetime(2026, 1, 1, 12, 30)).alias("days"))
    # Datetime anchor at 2026-01-01 12:30 truncates to date 2026-01-01.
    assert result["days"].item() == 731


def test_age_in_days_alias_for_days_since():
    """age_in_days is the same calculation; only the call-site name
    differs for readability."""
    df = pl.DataFrame({"open_date": [date(2020, 1, 1)]})
    a = df.with_columns(age_in_days("open_date", anchor=date(2026, 1, 1)).alias("age"))
    b = df.with_columns(days_since("open_date", anchor=date(2026, 1, 1)).alias("age"))
    assert a["age"].item() == b["age"].item()


def test_days_between_basic():
    df = pl.DataFrame(
        {
            "start": [date(2024, 1, 1)],
            "end": [date(2024, 12, 31)],
        }
    )
    result = df.with_columns(days_between("start", "end").alias("span"))
    # 2024 is a leap year → 365 days from Jan 1 to Dec 31 is 365 days.
    assert result["span"].item() == 365


def test_days_between_negative_when_end_before_start():
    df = pl.DataFrame(
        {
            "start": [date(2024, 6, 1)],
            "end": [date(2024, 1, 1)],
        }
    )
    result = df.with_columns(days_between("start", "end").alias("span"))
    # 2024-01-01 - 2024-06-01 = -152 days (Jan + Feb 29 + Mar + Apr + May)
    assert result["span"].item() == -152


# ============================================================================
# velocity.days_since_last (per-entity window)
# ============================================================================


def test_days_since_last_per_entity():
    """For each row, returns days from THIS entity's most-recent date
    to anchor. Multiple rows for the same entity all see the same
    days_since_last value."""
    df = pl.DataFrame(
        {
            "license": ["A", "A", "B"],
            "date": [date(2024, 1, 1), date(2025, 6, 1), date(2024, 6, 1)],
        }
    )
    result = df.with_columns(
        days_since_last("date", "license", anchor=date(2026, 1, 1)).alias("days")
    )
    # Both A rows: days from 2025-06-01 (A's max) to 2026-01-01 = 214
    # B row: days from 2024-06-01 to 2026-01-01 = 579
    by_license = dict(zip(result["license"].to_list(), result["days"].to_list(), strict=True))
    # A appears twice with same value
    a_rows = [
        d
        for lic, d in zip(result["license"].to_list(), result["days"].to_list(), strict=True)
        if lic == "A"
    ]
    assert len(a_rows) == 2
    assert a_rows[0] == a_rows[1] == 214
    assert by_license["B"] == 579


def test_days_since_last_isolates_entities():
    """A's max date should NOT bleed into B's calculation, even when
    B has earlier dates than A."""
    df = pl.DataFrame(
        {
            "license": ["A", "B"],
            "date": [date(2024, 1, 1), date(2024, 6, 1)],
        }
    )
    result = df.with_columns(
        days_since_last("date", "license", anchor=date(2026, 1, 1)).alias("days")
    )
    # A: 2026-01-01 - 2024-01-01 = 731
    # B: 2026-01-01 - 2024-06-01 = 579
    pairs = dict(zip(result["license"].to_list(), result["days"].to_list(), strict=True))
    assert pairs["A"] == 731
    assert pairs["B"] == 579


# ============================================================================
# TrajectoryClassifier
# ============================================================================


def test_trajectory_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="must be <="):
        TrajectoryClassifier(
            entity_column="e",
            date_column="d",
            score_column="s",
            improving_threshold=1.0,
            declining_threshold=-1.0,
        )


def test_trajectory_first_observation_is_new():
    """Each entity's first observation has no prior to compare to —
    must be classified 'new', NOT 'stable' or 'declining'."""
    df = pl.DataFrame(
        {
            "e": ["A"],
            "d": [date(2024, 1, 1)],
            "s": [10.0],
        }
    )
    classifier = TrajectoryClassifier(entity_column="e", date_column="d", score_column="s")
    result = classifier.classify(df)
    assert result["_trajectory"].to_list() == ["new"]
    assert result["_previous_score"].to_list() == [None]
    assert result["_score_change"].to_list() == [None]


def test_trajectory_improving_when_score_dropped():
    """Lower score = better in the cleanroom convention. A negative
    change <= improving_threshold (default -0.5) classifies as
    'improving'."""
    df = pl.DataFrame(
        {
            "e": ["A", "A"],
            "d": [date(2024, 1, 1), date(2024, 6, 1)],
            "s": [10.0, 5.0],
        }
    )
    classifier = TrajectoryClassifier(entity_column="e", date_column="d", score_column="s")
    result = classifier.classify(df).sort("d")
    # Second row: change = 5.0 - 10.0 = -5.0 → <= -0.5 → improving
    assert result["_trajectory"].to_list() == ["new", "improving"]


def test_trajectory_declining_when_score_rose():
    df = pl.DataFrame(
        {
            "e": ["A", "A"],
            "d": [date(2024, 1, 1), date(2024, 6, 1)],
            "s": [5.0, 12.0],
        }
    )
    classifier = TrajectoryClassifier(entity_column="e", date_column="d", score_column="s")
    result = classifier.classify(df).sort("d")
    # Second row: change = 12 - 5 = +7 → >= +0.5 → declining
    assert result["_trajectory"].to_list() == ["new", "declining"]


def test_trajectory_stable_when_change_within_thresholds():
    """A change within ±0.5 (the default stable band) classifies as
    'stable', not improving/declining."""
    df = pl.DataFrame(
        {
            "e": ["A", "A"],
            "d": [date(2024, 1, 1), date(2024, 6, 1)],
            "s": [10.0, 10.2],
        }
    )
    classifier = TrajectoryClassifier(entity_column="e", date_column="d", score_column="s")
    result = classifier.classify(df).sort("d")
    # Change = 0.2 → within (-0.5, +0.5) → stable
    assert result["_trajectory"].to_list() == ["new", "stable"]


def test_trajectory_uses_chronological_prior_not_input_order():
    """If the input is shuffled out of order, classify() should still
    use the chronologically-prior observation per entity, not the
    previous row in input order."""
    df = pl.DataFrame(
        {
            "e": ["A", "A", "A"],
            # Out-of-order input: middle date first
            "d": [date(2024, 6, 1), date(2024, 1, 1), date(2024, 12, 1)],
            "s": [5.0, 10.0, 1.0],
        }
    )
    classifier = TrajectoryClassifier(entity_column="e", date_column="d", score_column="s")
    result = classifier.classify(df).sort("d")
    # Sorted by date:
    #   2024-01-01: s=10  → new
    #   2024-06-01: s=5   → change=-5  → improving
    #   2024-12-01: s=1   → change=-4  → improving
    assert result["_trajectory"].to_list() == ["new", "improving", "improving"]


def test_trajectory_isolates_entities():
    """One entity's score history should not contaminate another's
    classification."""
    df = pl.DataFrame(
        {
            "e": ["A", "B", "A", "B"],
            "d": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 6, 1),
                date(2024, 6, 1),
            ],
            "s": [10.0, 1.0, 10.5, 1.5],
        }
    )
    classifier = TrajectoryClassifier(entity_column="e", date_column="d", score_column="s")
    result = classifier.classify(df)
    # Each entity's second observation should be 'stable' (change 0.5 is
    # right at declining_threshold, so >= 0.5 → declining).
    by_entity = (
        result.sort("e", "d")
        .group_by("e", maintain_order=True)
        .agg(pl.col("_trajectory").alias("traj"))
    )
    for row in by_entity.to_dicts():
        # Each entity has exactly two rows: [new, declining]
        assert row["traj"] == ["new", "declining"]


def test_trajectory_custom_thresholds():
    """Custom thresholds let apps tune what counts as
    improving/declining for their domain."""
    classifier = TrajectoryClassifier(
        entity_column="e",
        date_column="d",
        score_column="s",
        improving_threshold=-2.0,
        declining_threshold=2.0,
    )
    df = pl.DataFrame(
        {
            "e": ["A", "A"],
            "d": [date(2024, 1, 1), date(2024, 6, 1)],
            "s": [10.0, 11.0],  # change=+1, between -2 and +2
        }
    )
    result = classifier.classify(df).sort("d")
    # Default declining_threshold=0.5 would classify this as declining;
    # widened to 2.0, it's stable.
    assert result["_trajectory"].to_list() == ["new", "stable"]


def test_trajectory_custom_column_names():
    classifier = TrajectoryClassifier(
        entity_column="e",
        date_column="d",
        score_column="s",
        previous_score_column="prev",
        score_change_column="delta",
        trajectory_column="trend",
    )
    df = pl.DataFrame({"e": ["A"], "d": [date(2024, 1, 1)], "s": [10.0]})
    result = classifier.classify(df)
    assert {"prev", "delta", "trend"} <= set(result.columns)
    assert "_trajectory" not in result.columns


def test_trajectory_raises_on_missing_columns():
    classifier = TrajectoryClassifier(entity_column="e", date_column="d", score_column="s")
    df = pl.DataFrame({"e": ["A"]})  # missing d, s
    with pytest.raises(ValueError, match="required columns missing"):
        classifier.classify(df)


def test_trajectory_threshold_inclusivity():
    """Boundary check: change EXACTLY at threshold should classify
    as the relevant transition (not stable). Inclusive on both ends."""
    df = pl.DataFrame(
        {
            "e": ["A", "A", "B", "B"],
            "d": [
                date(2024, 1, 1),
                date(2024, 6, 1),
                date(2024, 1, 1),
                date(2024, 6, 1),
            ],
            "s": [
                10.0,
                9.5,  # exactly improving_threshold
                10.0,
                10.5,  # exactly declining_threshold
            ],
        }
    )
    classifier = TrajectoryClassifier(entity_column="e", date_column="d", score_column="s")
    result = classifier.classify(df).sort("e", "d")
    # A row 2: change=-0.5 → exactly improving_threshold → improving
    # B row 2: change=+0.5 → exactly declining_threshold → declining
    trajs = result["_trajectory"].to_list()
    assert trajs == ["new", "improving", "new", "declining"]
