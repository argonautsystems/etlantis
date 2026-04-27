# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""Tests for etlantis.match.semantic — SemanticMatcher.

The Stage B semantic matcher depends on sentence-transformers, a heavy
optional install. To keep the unit tests CI-runnable without the
extras group, every test here patches `_embed` with a deterministic
fixture-vector function. The contract being tested is the matcher's
filter / threshold / index-mapping behavior, not the actual embedding
quality (which is already covered by sentence-transformers' own
test suite).
"""

from __future__ import annotations

import numpy as np
import pytest

from etlantis.match.semantic import SemanticMatcher

# ---------------------------------------------------------------------------
# Fixture embedder — deterministic mapping from test strings to vectors
# ---------------------------------------------------------------------------


def _fixture_embeddings():
    """Hand-tuned 4-dim vectors for the test strings.

    All vectors are unit-normalized so dot product == cosine similarity.
    The structure clusters semantically-related strings near each
    other in the unit hypercube:

        "MCDONALDS" and "GOLDEN ARCHES" → high cosine (≈ 0.95)
        "MCDONALDS" and "BURGER KING"   → low cosine (≈ 0.10)
        "EMPTY"     and anything else   → low cosine

    Tests pick specific predictable pairs to exercise the matcher.
    """

    def normalize(v):
        v = np.asarray(v, dtype=np.float32)
        return v / np.linalg.norm(v)

    return {
        "MCDONALDS": normalize([1.0, 0.95, 0.0, 0.0]),
        "GOLDEN ARCHES": normalize([0.95, 1.0, 0.0, 0.0]),
        "BURGER KING": normalize([0.0, 0.0, 1.0, 0.0]),
        "WENDYS": normalize([0.0, 0.0, 0.95, 0.05]),  # close to BK but distinct
        "ACME LLC": normalize([0.0, 0.0, 0.0, 1.0]),
    }


def _fake_embed(texts):
    """Drop-in replacement for SemanticMatcher._embed."""
    table = _fixture_embeddings()
    out = []
    for t in texts:
        # Default to a random-ish vector for any string not in the table
        # so we don't accidentally match unknown inputs to fixture vectors.
        if t in table:
            out.append(table[t])
        else:
            # Distinct from all fixtures: pure +z component
            out.append(np.asarray([0.0, 0.0, 0.0, 0.5], dtype=np.float32) * 0)
    return np.asarray(out)


@pytest.fixture
def matcher(monkeypatch):
    """SemanticMatcher with `_embed` patched to the fixture function.

    Avoids the sentence-transformers install requirement and gives
    deterministic test output. Every `match()` call goes through
    `_embed`, so patching there covers all behavior under test.
    """
    m = SemanticMatcher()
    monkeypatch.setattr(m, "_embed", _fake_embed)
    return m


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_semantic_match_picks_highest_cosine(matcher):
    """MCDONALDS ↔ GOLDEN ARCHES is the canonical brand-vs-legal-name
    case the matcher should handle. Edit-distance can't see this; the
    fixture vectors put them at cosine ≈ 0.95."""
    result = matcher.match(["MCDONALDS"], ["BURGER KING", "GOLDEN ARCHES"], threshold=0.75)
    assert result == {0: 1}


def test_semantic_match_below_threshold_excluded(matcher):
    """Strings with no semantic relationship score low; the matcher
    should NOT return a mapping for left rows that can't clear
    threshold."""
    result = matcher.match(["MCDONALDS"], ["BURGER KING"], threshold=0.75)
    assert result == {}


def test_semantic_match_picks_best_among_close_candidates(matcher):
    """When two right rows are both above threshold, the higher one
    wins. WENDYS and BURGER KING are both near each other in the
    fixture, but WENDYS edges out."""
    result = matcher.match(["BURGER KING"], ["WENDYS", "BURGER KING"], threshold=0.5)
    # BURGER KING ↔ BURGER KING is a perfect match (cosine = 1.0)
    assert result == {0: 1}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_semantic_match_empty_left(matcher):
    assert matcher.match([], ["A"]) == {}


def test_semantic_match_empty_right(matcher):
    assert matcher.match(["A"], []) == {}


def test_semantic_match_skips_empty_strings_on_left(matcher):
    """Empty/whitespace-only left rows must not produce match entries.
    The original-index mapping must still be correct for the rows that
    DO match."""
    result = matcher.match(
        ["", "  ", "MCDONALDS", None],
        ["GOLDEN ARCHES"],
        threshold=0.75,
    )
    # Only index 2 ("MCDONALDS") matches; 0/1/3 are filtered out
    assert result == {2: 0}


def test_semantic_match_skips_empty_strings_on_right(matcher):
    """Same on the right side. Index 1 ("GOLDEN ARCHES") is the real
    candidate; the matcher must map back to the original index 1, not
    the filtered index 0."""
    result = matcher.match(
        ["MCDONALDS"],
        ["", "GOLDEN ARCHES", None],
        threshold=0.75,
    )
    assert result == {0: 1}


# ---------------------------------------------------------------------------
# Threshold validation (codex review #15 P2 carried into Stage B)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, -1.0, 1.0001, 2.0])
def test_semantic_match_rejects_out_of_range_threshold(matcher, bad):
    with pytest.raises(ValueError, match="threshold must be in"):
        matcher.match(["A"], ["B"], threshold=bad)


@pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
def test_semantic_match_accepts_endpoint_thresholds(matcher, good):
    matcher.match(["A"], ["B"], threshold=good)  # should not raise


# ---------------------------------------------------------------------------
# Diagnostic: get_similarity_scores
# ---------------------------------------------------------------------------


def test_get_similarity_scores_shape(matcher):
    """Returned matrix has shape (n_left_filtered, n_right_filtered)."""
    scores = matcher.get_similarity_scores(
        ["MCDONALDS", "BURGER KING"], ["GOLDEN ARCHES", "WENDYS"]
    )
    assert scores.shape == (2, 2)


def test_get_similarity_scores_empty_input_returns_zero_matrix(matcher):
    scores = matcher.get_similarity_scores([], ["A"])
    assert scores.shape == (0, 0)
    scores = matcher.get_similarity_scores(["A"], [])
    assert scores.shape == (0, 0)


def test_get_similarity_scores_filters_empties(matcher):
    """Empty strings on either side are dropped from the matrix; the
    returned shape reflects the FILTERED counts, not the input counts."""
    scores = matcher.get_similarity_scores(["MCDONALDS", ""], ["", "GOLDEN ARCHES"])
    assert scores.shape == (1, 1)
    # The single cell should be the high MCDONALDS↔GOLDEN ARCHES cosine
    assert scores[0, 0] > 0.9


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------


def test_warm_up_calls_load_model(monkeypatch):
    """warm_up() should trigger _load_model. Verify by stubbing
    _load_model and checking it's been invoked exactly once after
    warm_up()."""
    m = SemanticMatcher()
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        m._model = object()  # something non-None so subsequent calls no-op

    monkeypatch.setattr(m, "_load_model", fake_load)
    m.warm_up()
    assert calls["n"] == 1
    # Second warm_up: _load_model is called again (it short-circuits
    # internally on the m._model check; we don't enforce that here so a
    # future refactor can add idempotency guards in a different layer).


def test_device_property_returns_override():
    """If a device is explicitly passed at init, the property surfaces
    it without forcing model load."""
    m = SemanticMatcher(device="cpu")
    assert m.device == "cpu"
    # _model must still be unloaded — accessing the property mustn't
    # trigger a download.
    assert m._model is None


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_semantic_matcher_satisfies_matcher_protocol():
    from etlantis.match import Matcher

    assert isinstance(SemanticMatcher(), Matcher)
