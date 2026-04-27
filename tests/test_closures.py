# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""Tests for etlantis.closures — TransitionExtractor + SupertransitionDetector."""

from __future__ import annotations

import re
from datetime import date

import polars as pl
import pytest

from etlantis.closures import (
    SupertransitionDetector,
    TransitionExtractor,
    TransitionRule,
)

# ============================================================================
# TransitionRule invariants
# ============================================================================


def test_transition_rule_rejects_empty_event_type():
    with pytest.raises(ValueError, match="non-empty"):
        TransitionRule(event_type="", pattern="anything")


def test_transition_rule_compile_substring_default():
    """Default pattern shape is substring + case-insensitive."""
    rule = TransitionRule(event_type="X", pattern="hello")
    compiled = rule.compile()
    assert compiled.search("Hello world") is not None
    assert compiled.search("HELLO") is not None


def test_transition_rule_compile_case_sensitive():
    rule = TransitionRule(event_type="X", pattern="hello", case_sensitive=True)
    compiled = rule.compile()
    assert compiled.search("hello") is not None
    assert compiled.search("HELLO") is None


def test_transition_rule_compile_regex_mode():
    rule = TransitionRule(event_type="X", pattern=r"\d{3}", is_regex=True)
    compiled = rule.compile()
    assert compiled.search("abc 123 xyz") is not None
    assert compiled.search("no digits here") is None


def test_transition_rule_compile_substring_escapes_metachars():
    """Substring mode (default) should treat regex metacharacters
    literally — a pattern with ``.`` shouldn't match arbitrary chars."""
    rule = TransitionRule(event_type="X", pattern="a.b")
    compiled = rule.compile()
    assert compiled.search("a.b") is not None
    assert compiled.search("axb") is None  # literal dot, not regex


def test_transition_rule_accepts_precompiled_pattern():
    """Pre-compiled patterns should pass through unchanged, including
    their flags."""
    pre = re.compile(r"^Emergency", re.MULTILINE)
    rule = TransitionRule(event_type="X", pattern=pre)
    assert rule.compile() is pre


# ============================================================================
# TransitionExtractor
# ============================================================================


def test_extractor_rejects_empty_rules():
    with pytest.raises(ValueError, match="at least one rule"):
        TransitionExtractor(status_column="status", rules=[])


def test_extractor_classifies_by_first_match_wins():
    """When multiple rules could match, the first listed wins."""
    rules = [
        TransitionRule(event_type="EMERGENCY", pattern="Emergency"),
        TransitionRule(event_type="ANY", pattern=""),  # would match everything
    ]
    extractor = TransitionExtractor(status_column="status", rules=rules)
    df = pl.DataFrame({"status": ["Emergency order", "routine"]})
    result = extractor.extract(df)
    # Both rows match the second rule (empty substring matches anything),
    # but the first should still win for the Emergency row.
    assert result["_event_type"].to_list() == ["EMERGENCY", "ANY"]


def test_extractor_unmatched_rows_get_unmatched_event():
    rules = [TransitionRule(event_type="X", pattern="needle")]
    extractor = TransitionExtractor(status_column="status", rules=rules, unmatched_event="OTHER")
    df = pl.DataFrame({"status": ["with needle here", "haystack only"]})
    result = extractor.extract(df)
    assert result["_event_type"].to_list() == ["X", "OTHER"]


def test_extractor_unmatched_event_default_is_none():
    rules = [TransitionRule(event_type="X", pattern="needle")]
    extractor = TransitionExtractor(status_column="status", rules=rules)
    df = pl.DataFrame({"status": ["with needle", "without"]})
    result = extractor.extract(df)
    assert result["_event_type"].to_list() == ["X", None]


def test_extractor_handles_null_status():
    """Null in the status column should classify as unmatched, not error."""
    rules = [TransitionRule(event_type="X", pattern="needle")]
    extractor = TransitionExtractor(status_column="status", rules=rules, unmatched_event="OTHER")
    df = pl.DataFrame({"status": ["with needle", None]})
    result = extractor.extract(df)
    # Null status → no rule matches → otherwise branch (UNMATCHED)
    assert result["_event_type"].to_list() == ["X", "OTHER"]


def test_extractor_raises_on_missing_status_column():
    rules = [TransitionRule(event_type="X", pattern="x")]
    extractor = TransitionExtractor(status_column="missing", rules=rules)
    df = pl.DataFrame({"status": ["x"]})
    with pytest.raises(ValueError, match="not found in DataFrame"):
        extractor.extract(df)


def test_extractor_custom_event_column_name():
    rules = [TransitionRule(event_type="X", pattern="x")]
    extractor = TransitionExtractor(status_column="status", rules=rules, event_column="phase")
    df = pl.DataFrame({"status": ["x"]})
    result = extractor.extract(df)
    assert "phase" in result.columns
    assert "_event_type" not in result.columns


def test_extractor_filter_to_matched_only():
    """`filter_to(None)` keeps only rows that matched some rule."""
    rules = [TransitionRule(event_type="X", pattern="needle")]
    extractor = TransitionExtractor(status_column="status", rules=rules)
    df = pl.DataFrame({"status": ["needle 1", "haystack", "needle 2"]})
    result = extractor.filter_to(df)
    assert result.height == 2
    assert result["_event_type"].to_list() == ["X", "X"]


def test_extractor_filter_to_event_whitelist():
    rules = [
        TransitionRule(event_type="A", pattern="alpha"),
        TransitionRule(event_type="B", pattern="beta"),
    ]
    extractor = TransitionExtractor(status_column="status", rules=rules)
    df = pl.DataFrame({"status": ["alpha", "beta", "gamma"]})
    result = extractor.filter_to(df, event_types=["A"])
    assert result.height == 1
    assert result["_event_type"].to_list() == ["A"]


def test_extractor_dbpr_realistic_scenario():
    """End-to-end scenario mirroring the cleanroom DBPR use case."""
    rules = [
        TransitionRule(
            event_type="CLOSURE",
            pattern="Emergency order recommended|Emergency Order Callback Not Complied",
            is_regex=True,
        ),
        TransitionRule(
            event_type="REOPENING",
            pattern="Emergency Order Callback Complied",
        ),
    ]
    extractor = TransitionExtractor(status_column="Inspection Disposition", rules=rules)
    df = pl.DataFrame(
        {
            "Inspection Disposition": [
                "Emergency order recommended",
                "Emergency Order Callback Complied",
                "Routine inspection",
                "Emergency Order Callback Not Complied",
            ]
        }
    )
    result = extractor.extract(df)
    assert result["_event_type"].to_list() == [
        "CLOSURE",
        "REOPENING",
        None,
        "CLOSURE",
    ]


# ============================================================================
# SupertransitionDetector
# ============================================================================


def _events_df():
    """Build a fixture event frame.

    license A: 4 events (over-threshold for default min_count=3)
    license B: 2 events (under-threshold)
    license C: 3 events (exactly at threshold) but spans 10 years
    """
    return pl.DataFrame(
        {
            "license": ["A", "A", "A", "A", "B", "B", "C", "C", "C"],
            "event_date": [
                date(2024, 1, 1),
                date(2024, 6, 1),
                date(2025, 1, 1),
                date(2025, 6, 1),
                date(2025, 1, 1),
                date(2025, 6, 1),
                date(2015, 1, 1),
                date(2020, 1, 1),
                date(2025, 1, 1),
            ],
            "_event_type": [
                "CLOSURE",
                "REOPENING",
                "CLOSURE",
                "REOPENING",
                "CLOSURE",
                "REOPENING",
                "CLOSURE",
                "CLOSURE",
                "CLOSURE",
            ],
        }
    )


def test_supertransition_min_count_threshold():
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    result = detector.detect(_events_df(), min_count=3)
    # A (4 events) and C (3 events) qualify. B (2) does not.
    assert sorted(result["entity"].to_list()) == ["A", "C"]


def test_supertransition_min_count_filters_lower_thresholds():
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    # Default min_count=3 — B (2 events) excluded
    result = detector.detect(_events_df())
    assert "B" not in result["entity"].to_list()


def test_supertransition_event_type_whitelist():
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    # Only count CLOSUREs — A has 2 closures (under 3), C has 3 (qualifies)
    result = detector.detect(_events_df(), event_types=["CLOSURE"], min_count=3)
    assert result["entity"].to_list() == ["C"]


def test_supertransition_window_filter():
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    # 5-year window from 2026-01-01 — C's 2015 event drops out.
    # C now has only 2 in-window events (2020, 2025) → under threshold.
    result = detector.detect(
        _events_df(),
        min_count=3,
        window_days=365 * 5,
        anchor_date=date(2026, 1, 1),
    )
    assert sorted(result["entity"].to_list()) == ["A"]


def test_supertransition_summary_columns():
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    result = detector.detect(_events_df(), min_count=3)
    expected_cols = {
        "entity",
        "event_count",
        "first_event_date",
        "last_event_date",
        "days_span",
        "event_types",
        "current_event_type",
    }
    assert set(result.columns) == expected_cols


def test_supertransition_event_count_correct():
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    result = detector.detect(_events_df(), min_count=3)
    by_entity = {row["entity"]: row["event_count"] for row in result.to_dicts()}
    assert by_entity["A"] == 4
    assert by_entity["C"] == 3


def test_supertransition_chronological_event_types():
    """`event_types` is a list[str] in chronological order so apps can
    inspect the sequence of changes (e.g. CLOSURE→REOPENING→CLOSURE)."""
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    result = detector.detect(_events_df(), min_count=3)
    a_row = result.filter(pl.col("entity") == "A").to_dicts()[0]
    assert a_row["event_types"] == ["CLOSURE", "REOPENING", "CLOSURE", "REOPENING"]


def test_supertransition_current_event_type_is_last():
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    result = detector.detect(_events_df(), min_count=3)
    a_row = result.filter(pl.col("entity") == "A").to_dicts()[0]
    # A's latest event (2025-06-01) is REOPENING
    assert a_row["current_event_type"] == "REOPENING"


def test_supertransition_days_span():
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    result = detector.detect(_events_df(), min_count=3)
    a_row = result.filter(pl.col("entity") == "A").to_dicts()[0]
    # 2024-01-01 to 2025-06-01 → 517 days
    assert a_row["days_span"] == (date(2025, 6, 1) - date(2024, 1, 1)).days


def test_supertransition_skips_unclassified_events():
    """Rows with null event_type should be ignored entirely (typical
    when the upstream TransitionExtractor used unmatched_event=None)."""
    df = pl.DataFrame(
        {
            "license": ["A", "A", "A", "A"],
            "event_date": [
                date(2024, 1, 1),
                date(2024, 6, 1),
                date(2025, 1, 1),
                date(2025, 6, 1),
            ],
            "_event_type": ["CLOSURE", None, None, "CLOSURE"],
        }
    )
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    # Only 2 classified events — under threshold of 3.
    result = detector.detect(df, min_count=3)
    assert result.height == 0


def test_supertransition_empty_input_returns_empty_typed_frame():
    """An empty input must produce an empty output frame with the
    documented schema, not a generic pl.DataFrame()."""
    df = pl.DataFrame(
        schema={
            "license": pl.Utf8,
            "event_date": pl.Date,
            "_event_type": pl.Utf8,
        }
    )
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    result = detector.detect(df, min_count=3)
    assert result.height == 0
    assert "entity" in result.columns
    assert "event_count" in result.columns


def test_supertransition_raises_on_missing_columns():
    df = pl.DataFrame({"license": ["A"]})
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    with pytest.raises(ValueError, match="required columns missing"):
        detector.detect(df)


def test_supertransition_rejects_min_count_zero():
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    df = pl.DataFrame(
        {
            "license": ["A"],
            "event_date": [date(2025, 1, 1)],
            "_event_type": ["X"],
        }
    )
    with pytest.raises(ValueError, match="min_count must be >= 1"):
        detector.detect(df, min_count=0)


def test_supertransition_custom_event_column():
    """`event_column` config flows through both null-filtering and
    whitelist application."""
    df = pl.DataFrame(
        {
            "license": ["A", "A", "A"],
            "event_date": [date(2024, 1, 1), date(2024, 6, 1), date(2025, 1, 1)],
            "phase": ["X", "Y", "X"],
        }
    )
    detector = SupertransitionDetector(
        entity_column="license",
        date_column="event_date",
        event_column="phase",
    )
    result = detector.detect(df, min_count=3)
    assert result.height == 1
    assert result["entity"].to_list() == ["A"]


# ============================================================================
# Codex review #18 regressions
# ============================================================================


def test_supertransition_window_caps_at_anchor_date():
    """P1 from review #18: future-dated events must NOT be counted in a
    window that's supposed to describe the past. The upper bound is
    `anchor_date` (inclusive); events after that date drop out."""
    df = pl.DataFrame(
        {
            "license": ["A"] * 4,
            "event_date": [
                date(2024, 1, 1),
                date(2024, 6, 1),
                date(2025, 1, 1),
                date(2030, 1, 1),  # future-dated event (anchor is 2026)
            ],
            "_event_type": ["X"] * 4,
        }
    )
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    result = detector.detect(
        df,
        min_count=3,
        window_days=365 * 5,  # 5-year lookback
        anchor_date=date(2026, 1, 1),
    )
    # 3 events in window (2024-01, 2024-06, 2025-01); 1 future event excluded.
    # Qualifies at min_count=3 — but if we had counted the 2030 event too,
    # the test would still pass for the wrong reason. So also assert
    # last_event_date is the in-window max, not the future date.
    assert result["entity"].to_list() == ["A"]
    a = result.to_dicts()[0]
    assert a["event_count"] == 3
    assert a["last_event_date"] == date(2025, 1, 1)


def test_supertransition_filters_null_dates():
    """P1 from review #18: null dates pollute first/last extraction
    and event_count if not filtered. Null-dated rows must drop out
    entirely — not contribute to the qualifying count."""
    df = pl.DataFrame(
        {
            "license": ["A"] * 4,
            "event_date": [
                date(2024, 1, 1),
                None,  # null date
                None,  # null date
                date(2025, 1, 1),
            ],
            "_event_type": ["X"] * 4,
        }
    )
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    # Two null-dated events would push count to 4 (over min_count=3); the
    # fix excludes them so count is 2 and the entity does NOT qualify.
    result = detector.detect(df, min_count=3)
    assert result.height == 0


def test_supertransition_empty_frame_preserves_input_dtypes():
    """P2 from review #18: empty result frame should match the input's
    entity + date dtypes, not hardcode Utf8/Date."""
    df = pl.DataFrame(
        schema={
            "license_id": pl.Int64,  # numeric entity, not string
            "event_ts": pl.Datetime,  # Datetime, not Date
            "_event_type": pl.Utf8,
        }
    )
    detector = SupertransitionDetector(entity_column="license_id", date_column="event_ts")
    result = detector.detect(df, min_count=3)
    assert result.height == 0
    assert result.schema["entity"] == pl.Int64
    assert result.schema["first_event_date"] == pl.Datetime
    assert result.schema["last_event_date"] == pl.Datetime


def test_supertransition_rejects_non_temporal_date_column():
    """P2 from review #18: string or integer date columns should fail
    with a clear error early, not produce nonsense output later."""
    df = pl.DataFrame(
        {
            "license": ["A", "A", "A"],
            "event_date": ["2024-01-01", "2024-06-01", "2025-01-01"],
            "_event_type": ["X", "X", "X"],
        }
    )
    detector = SupertransitionDetector(entity_column="license", date_column="event_date")
    with pytest.raises(ValueError, match="must be Polars Date or Datetime"):
        detector.detect(df, min_count=3)


def test_extractor_preserves_multiline_flag_from_precompiled_pattern():
    """P2 from review #18: pre-compiled patterns with MULTILINE / DOTALL
    flags should preserve those flags when carried into Polars'
    str.contains. Without the inline-flag fix, MULTILINE patterns
    would behave as if the flag was unset."""
    # Pattern with MULTILINE — `^Emergency` matches lines starting
    # with "Emergency" anywhere in the string. Without MULTILINE, only
    # matches at the very start of the string.
    pre = re.compile(r"^Emergency", re.MULTILINE)
    rules = [TransitionRule(event_type="X", pattern=pre)]
    extractor = TransitionExtractor(status_column="status", rules=rules)
    # Multi-line strings — only one starts with "Emergency" at position 0,
    # the other has "Emergency" at the start of a non-first line.
    df = pl.DataFrame(
        {
            "status": [
                "prefix line\nEmergency happens here",  # MULTILINE → match
                "no Emergency at line start",  # never matches
            ]
        }
    )
    result = extractor.extract(df)
    # With MULTILINE preserved, first row matches via `^Emergency` on
    # the second line. Without the fix this would be [None, None].
    assert result["_event_type"].to_list() == ["X", None]


def test_extractor_preserves_dotall_flag_from_precompiled_pattern():
    """P2 from review #18 follow-up: DOTALL allows `.` to match newlines
    too. Pattern without DOTALL: `a.b` doesn't match `a\\nb`. With
    DOTALL: it does."""
    pre = re.compile(r"a.b", re.DOTALL)
    rules = [TransitionRule(event_type="X", pattern=pre)]
    extractor = TransitionExtractor(status_column="status", rules=rules)
    df = pl.DataFrame({"status": ["a\nb", "no match"]})
    result = extractor.extract(df)
    assert result["_event_type"].to_list() == ["X", None]
