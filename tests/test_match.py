# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""Tests for etlantis.match — exact + fuzzy matchers and the shared base."""

from __future__ import annotations

import pytest

from etlantis.match import (
    ExactMatcher,
    FuzzyMatcher,
    Matcher,
    default_normalize,
)
from etlantis.match.base import NormalizeFn

# ============================================================================
# default_normalize
# ============================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("hello", "HELLO"),
        ("  hello  ", "HELLO"),
        ("HeLLo", "HELLO"),
        ("", ""),
        (None, ""),
    ],
)
def test_default_normalize(raw, expected):
    assert default_normalize(raw) == expected


# ============================================================================
# Protocol conformance
# ============================================================================


def test_exact_matcher_is_a_matcher():
    """ExactMatcher must satisfy the Matcher Protocol — apps that take a
    Matcher Protocol arg must accept ExactMatcher without TypeError."""
    matcher = ExactMatcher()
    assert isinstance(matcher, Matcher)


def test_fuzzy_matcher_is_a_matcher():
    matcher = FuzzyMatcher()
    assert isinstance(matcher, Matcher)


# ============================================================================
# ExactMatcher — happy path
# ============================================================================


def test_exact_match_basic_case_insensitive():
    matcher = ExactMatcher()
    left = ["ABC LLC", "XYZ Corp", "no match"]
    right = ["abc llc", "XYZ CORP", "DEF Inc"]
    assert matcher.match(left, right) == {0: 0, 1: 1}


def test_exact_match_handles_whitespace():
    matcher = ExactMatcher()
    left = ["  spaced  "]
    right = ["spaced"]
    assert matcher.match(left, right) == {0: 0}


def test_exact_match_no_match_returns_empty():
    matcher = ExactMatcher()
    assert matcher.match(["a"], ["b"]) == {}


def test_exact_match_empty_inputs():
    matcher = ExactMatcher()
    assert matcher.match([], []) == {}
    assert matcher.match(["a"], []) == {}
    assert matcher.match([], ["a"]) == {}


def test_exact_match_skips_empty_left_values():
    """Empty left values normalize to "" and must not match anything,
    even if the right side has an empty entry."""
    matcher = ExactMatcher()
    left = ["", None, "valid"]
    right = ["valid", ""]
    matches = matcher.match(left, right)
    # Only "valid" → "valid" (idx 2 → idx 0). Empty left values absent.
    assert matches == {2: 0}


def test_exact_match_skips_empty_right_values():
    matcher = ExactMatcher()
    left = ["valid"]
    right = [None, "", "valid"]
    assert matcher.match(left, right) == {0: 2}


def test_exact_match_first_occurrence_wins_in_right():
    """If two right rows normalize to the same key, only the FIRST
    is registered. Defensive behavior for noisy enrichment sources
    where a license number appears twice."""
    matcher = ExactMatcher()
    left = ["acme"]
    right = ["ACME", "Acme", "ACME LLC"]
    matches = matcher.match(left, right)
    # First "ACME" wins — not the second "Acme" or third entry
    assert matches == {0: 0}


def test_exact_match_custom_normalizer_drops_dashes():
    """Domain example: license-number normalizer that also strips
    dashes and inner whitespace. The matcher must honor it on BOTH
    sides of the comparison."""

    def license_normalize(s: str | None) -> str:
        if not s:
            return ""
        return str(s).strip().upper().replace("-", "").replace(" ", "")

    matcher = ExactMatcher(normalization_fn=license_normalize)
    left = ["AB-12 34"]
    right = ["ab1234"]
    assert matcher.match(left, right) == {0: 0}


def test_exact_match_threshold_is_ignored():
    """ExactMatcher's threshold arg is Protocol-parity ballast — it
    should not affect the result regardless of value."""
    matcher = ExactMatcher()
    for thresh in (0.0, 0.5, 1.0, 99.0):
        assert matcher.match(["a"], ["A"], threshold=thresh) == {0: 0}


def test_exact_match_rate():
    matcher = ExactMatcher()
    assert matcher.match_rate(["a", "b", "c"], ["a", "c"]) == pytest.approx(2 / 3)
    assert matcher.match_rate([], ["a"]) == 0.0
    assert matcher.match_rate(["a"], []) == 0.0


# ============================================================================
# FuzzyMatcher — happy path
# ============================================================================


def test_fuzzy_match_typo_tolerance():
    matcher = FuzzyMatcher()
    left = ["ABC RESTAURANT"]
    right = ["ABC RESTRAUNT"]  # transposition typo
    assert matcher.match(left, right, threshold=0.85) == {0: 0}


def test_fuzzy_match_below_threshold_excluded():
    matcher = FuzzyMatcher()
    left = ["ABC RESTAURANT"]
    right = ["XYZ COMPLETELY UNRELATED PLACE"]
    assert matcher.match(left, right, threshold=0.85) == {}


def test_fuzzy_match_picks_best_among_candidates():
    """When multiple right rows score above threshold, the highest one
    wins (rapidfuzz's process.extractOne semantics)."""
    matcher = FuzzyMatcher()
    left = ["SUBWAY SANDWICHES"]
    right = ["SUBWAY", "SUBWAY SANDWICHE", "BURGER KING"]
    matches = matcher.match(left, right, threshold=0.65)
    assert matches[0] == 1  # closer to "SUBWAY SANDWICHE" than "SUBWAY"


def test_fuzzy_match_handles_token_reorder():
    """WRatio scorer (default) handles token reorderings — useful for
    business-name workloads where word order varies."""
    matcher = FuzzyMatcher()
    left = ["ACME RESTAURANT LLC"]
    right = ["LLC ACME RESTAURANT"]
    assert matcher.match(left, right, threshold=0.85) == {0: 0}


def test_fuzzy_match_empty_inputs():
    matcher = FuzzyMatcher()
    assert matcher.match([], []) == {}
    assert matcher.match(["a"], []) == {}
    assert matcher.match([], ["a"]) == {}


def test_fuzzy_match_skips_empty_left():
    matcher = FuzzyMatcher()
    left = ["", None, "ACME"]
    right = ["ACME"]
    assert matcher.match(left, right, threshold=0.85) == {2: 0}


def test_fuzzy_match_threshold_endpoints():
    """threshold=0.0 should match anything non-empty against the first
    candidate. threshold=1.0 collapses to exact-after-normalize."""
    matcher = FuzzyMatcher()
    # Lenient — anything matches
    matches = matcher.match(["xyz"], ["abc"], threshold=0.0)
    assert matches == {0: 0}
    # Strict — only exact normalized matches survive
    assert matcher.match(["abc"], ["ABC"], threshold=1.0) == {0: 0}
    assert matcher.match(["abc"], ["abd"], threshold=1.0) == {}


def test_fuzzy_match_custom_normalizer():
    """Custom normalizer (e.g. drop corporate suffixes) should improve
    recall — without it, "ACME LLC" vs "ACME" might fail at strict
    thresholds; with it, both normalize to "ACME"."""

    def drop_suffix(s: str | None) -> str:
        if not s:
            return ""
        return s.strip().upper().replace(" LLC", "").replace(" INC", "")

    matcher = FuzzyMatcher(normalization_fn=drop_suffix)
    assert matcher.match(["ACME LLC"], ["ACME INC"], threshold=1.0) == {0: 0}


def test_fuzzy_get_all_candidates_returns_top_n_sorted():
    matcher = FuzzyMatcher()
    left = ["SUBWAY"]
    right = ["SUBWAY SANDWICHES", "SUB WAY", "TOTALLY DIFFERENT", "SUBWAY"]
    candidates = matcher.get_all_candidates(left, right, threshold=0.5, top_n=2)
    assert 0 in candidates
    rows = candidates[0]
    assert len(rows) <= 2
    # Sorted by score descending
    scores = [score for _idx, score in rows]
    assert scores == sorted(scores, reverse=True)
    # Best candidate is exact-match "SUBWAY" (idx 3)
    best_idx, best_score = rows[0]
    assert best_idx == 3
    assert best_score == pytest.approx(1.0)


def test_fuzzy_get_all_candidates_empty_when_below_threshold():
    matcher = FuzzyMatcher()
    candidates = matcher.get_all_candidates(["ACME"], ["VERY DIFFERENT"], threshold=0.95)
    assert candidates == {}


# ============================================================================
# Cross-matcher: a FuzzyMatcher used at threshold=1.0 vs ExactMatcher
# ============================================================================


def test_fuzzy_at_unity_threshold_equivalent_to_exact():
    """At threshold=1.0 + default normalizer, fuzzy should produce the
    same matches as exact for clean data. Useful invariant for callers
    who want one matcher that escalates from strict to lenient by
    parameter rather than by class swap."""
    left = ["ABC", "XYZ", "DEF"]
    right = ["abc", "XYZ", "QQQ"]
    expected = {0: 0, 1: 1}
    assert ExactMatcher().match(left, right) == expected
    assert FuzzyMatcher().match(left, right, threshold=1.0) == expected


# ============================================================================
# NormalizeFn type alias is callable-shaped
# ============================================================================


def test_normalize_fn_alias_is_a_callable():
    """Just a smoke check that the public NormalizeFn alias matches
    `Callable[[str | None], str]` — guards against accidental
    re-typing in future refactors."""
    fn: NormalizeFn = default_normalize
    assert callable(fn)
    assert fn("hi ") == "HI"


# ============================================================================
# Threshold validation (codex review #15 P2)
# ============================================================================


@pytest.mark.parametrize("bad", [-0.1, -1.0, 1.0001, 1.5, 100.0])
def test_fuzzy_match_rejects_out_of_range_threshold(bad):
    """Threshold contract is [0.0, 1.0] inclusive. Values outside that
    range should raise immediately, not silently transform into
    rapidfuzz cutoffs that produce surprising results."""
    matcher = FuzzyMatcher()
    with pytest.raises(ValueError, match="threshold must be in"):
        matcher.match(["a"], ["b"], threshold=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_fuzzy_get_all_candidates_rejects_out_of_range_threshold(bad):
    matcher = FuzzyMatcher()
    with pytest.raises(ValueError, match="threshold must be in"):
        matcher.get_all_candidates(["a"], ["b"], threshold=bad)


@pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
def test_fuzzy_match_accepts_endpoint_thresholds(good):
    """Both 0.0 and 1.0 must remain valid — they're documented edges
    of the contract."""
    matcher = FuzzyMatcher()
    matcher.match(["a"], ["a"], threshold=good)  # should not raise


# ============================================================================
# Empty-right-row regression (codex review #15 P2)
# ============================================================================


def test_fuzzy_empty_right_row_doesnt_mask_valid_later_candidate():
    """Regression: at threshold=0.0 the prior implementation kept blank
    normalized right rows in the choice list. A blank could tie at
    score 0, win the slot, and the `if not right_indexed[right_idx]`
    guard would skip the row entirely — but the SAME left value
    against a non-blank right candidate would never be evaluated.
    The fix is to filter empties out of rapidfuzz's choice list and
    map the chosen index back to the original right_values index."""
    matcher = FuzzyMatcher()
    # Right list: one blank row at index 0, one valid match at index 1.
    # At threshold=0.0 the matcher MUST find idx 1, not skip the row.
    result = matcher.match(["acme"], ["", "ACME"], threshold=0.0)
    assert result == {0: 1}


def test_fuzzy_empty_right_row_handled_at_normal_threshold_too():
    """Same regression at the standard threshold — extra defensive
    coverage."""
    matcher = FuzzyMatcher()
    result = matcher.match(["ACME RESTAURANT"], ["", "ACME RESTRAUNT"], threshold=0.85)
    assert result == {0: 1}


# ============================================================================
# get_all_candidates threshold re-check (codex review #15 P2)
# ============================================================================


def test_get_all_candidates_filters_borderline_below_threshold():
    """rapidfuzz's score_cutoff uses (threshold - 0.001) * 100 to let
    borderline matches survive the C-side prune. match() then re-checks
    the public threshold before accepting. get_all_candidates must do
    the SAME re-check or callers see candidates that match() would
    reject — confusing for threshold tuning.
    """
    # Use the deterministic ratio scorer (not WRatio) so we can predict
    # the exact borderline score without depending on WRatio's composite
    # heuristics.
    from rapidfuzz import fuzz

    matcher = FuzzyMatcher(scorer=fuzz.ratio)
    # "abcd" vs "abce" gives ratio = 0.75 (3 of 4 chars match * 2 / 8 = 0.75)
    candidates = matcher.get_all_candidates(["abcd"], ["abce"], threshold=0.751)
    # Score is 0.75 — below threshold of 0.751 — so result must be empty.
    assert candidates == {}


# ============================================================================
# ExactMatcher duplicate-key logging (codex review #15 P3)
# ============================================================================


def test_exact_match_logs_dropped_duplicate_keys(caplog):
    """ExactMatcher's docstring says duplicate normalized keys are logged.
    Verify it actually does — silent suppression would mask
    data-quality bugs in the right-side enrichment source."""
    import logging

    matcher = ExactMatcher()
    with caplog.at_level(logging.DEBUG, logger="etlantis.match.exact"):
        matcher.match(["foo"], ["FOO", "foo", "Foo"])
    # All three normalize to "FOO" — first wins, second + third are
    # dropped with a debug log each.
    log_messages = [r.message for r in caplog.records]
    assert any("duplicate normalized key" in m for m in log_messages)
    # Two duplicates should be logged.
    drop_logs = [m for m in log_messages if "duplicate normalized key" in m]
    assert len(drop_logs) == 2
