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

"""etlantis.match.fuzzy — edit-distance string matching via rapidfuzz.

Lifted from `etlantis-prototype-2026-03/etlantis/stages/matchers/
fuzzy.py` (rvmaps lineage), with the algorithm swapped from `difflib.
SequenceMatcher` to `rapidfuzz`. Same matcher contract, ~10–100×
faster, and rapidfuzz's `process.extractOne` collapses the inner loop
to a single C call per left row.

Why rapidfuzz over difflib:

  * Speed. The cleanroom hit a wall at ~50K×50K matches with difflib;
    rapidfuzz keeps the same workload in single-digit seconds.
  * Better default similarity metric. `WRatio` is rapidfuzz's
    composite scorer that handles abbreviations, partial matches, and
    token reorderings — exactly the failure modes that bit the
    cleanroom (e.g. ``"MCDONALDS RESTAURANT"`` vs ``"McDonald's"``).
  * Already an etlantis dep (`rapidfuzz>=3.0.0` in pyproject.toml).

Performance: O(n × m) in the worst case but the inner loop is C, and
rapidfuzz's batch helper amortizes the cost of normalization. For
10K × 10K matches: ~2–5s on commodity CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

from rapidfuzz import fuzz, process

from etlantis.match.base import NormalizeFn, _validate_threshold, default_normalize

# Default scorer. WRatio (weighted ratio) is rapidfuzz's composite
# scorer — it handles partials, token reorderings, and abbreviations.
# Returns 0–100 (we divide by 100 inside the matcher to keep the public
# threshold semantics in [0.0, 1.0]).
_DEFAULT_SCORER = fuzz.WRatio


class FuzzyMatcher:
    """Fuzzy string matcher using rapidfuzz scorers.

    For each left value, finds the right value with the highest
    similarity score that meets or exceeds `threshold`. Ties are
    broken by first-occurrence in `right_values` (rapidfuzz's stable
    behavior).

    Args:
        normalization_fn: Callable applied to each value before scoring.
            Default: strip + uppercase. Pass a domain-aware normalizer
            (e.g. a business-name normalizer that drops "LLC" / "INC"
            suffixes) for better recall on noisy enrichment sources.
        scorer: rapidfuzz scoring callable. Default `fuzz.WRatio` —
            composite of token_ratio + partial_ratio handling
            abbreviations and token reorderings well. Other reasonable
            choices: `fuzz.ratio` (simple Levenshtein), `fuzz.token_set_ratio`
            (order-insensitive). The scorer must return a float in [0, 100].
        score_cutoff_buffer: Internal cutoff floor below which rapidfuzz
            short-circuits the comparison. We default it to
            `(threshold - 0.001) * 100` so the cutoff lives just below
            our public threshold and rapidfuzz can prune aggressively
            without dropping borderline matches.

    Threshold semantics: a float in [0.0, 1.0] (we convert to rapidfuzz's
    0–100 internally). Typical values:
        * 0.85 strict — names that should be near-identical
        * 0.75 moderate — most business-name workloads
        * 0.65 lenient — when recall matters more than precision
    """

    def __init__(
        self,
        normalization_fn: NormalizeFn | None = None,
        scorer=_DEFAULT_SCORER,
    ):
        self.normalize = normalization_fn or default_normalize
        self.scorer = scorer

    def match(
        self,
        left_values: Sequence[str | None],
        right_values: Sequence[str | None],
        threshold: float = 0.85,
    ) -> dict[int, int]:
        """Best-match each left value against the right list.

        Args:
            left_values: Primary list.
            right_values: Enrichment list.
            threshold: Minimum similarity in [0.0, 1.0]. Out-of-range
                values raise ValueError. Right rows with score below
                threshold are skipped; left rows with no accepting
                right match are absent from the result.

        Returns:
            ``{left_idx: right_idx}`` — only left rows with an
            accepting match appear.

        Edge cases:
            * Empty `left_values` or `right_values` → empty dict.
            * `normalize(value)` returning ``""`` → that row is
              skipped on whichever side it falls (won't match anything).
            * `threshold=1.0` collapses to exact-after-normalization;
              for that workload, prefer ExactMatcher (O(n+m) vs O(n×m)).
        """
        _validate_threshold(threshold)
        if not left_values or not right_values:
            return {}

        # Build a "choice list" of NON-EMPTY normalized right values plus
        # a parallel list of original right indices. We must NOT pass
        # blanked-out slots to rapidfuzz: at threshold=0.0 a blank row
        # can tie at score 0 and win the slot, masking a valid later
        # candidate. Filtering preserves "first non-empty wins" semantics.
        choices: list[str] = []
        choice_to_orig_idx: list[int] = []
        for orig_idx, raw in enumerate(right_values):
            normalized = self.normalize(raw)
            if normalized:
                choices.append(normalized)
                choice_to_orig_idx.append(orig_idx)
        if not choices:
            return {}

        # rapidfuzz wants 0–100 cutoffs.
        score_cutoff = max(0.0, (threshold - 0.001) * 100.0)

        matches: dict[int, int] = {}
        for left_idx, raw in enumerate(left_values):
            normalized = self.normalize(raw)
            if not normalized:
                continue
            # process.extractOne searches `choices` and returns
            # (best_match_string, score, best_match_index_into_choices)
            # or None when no candidate clears the cutoff. Map the
            # choice index back to the original right_values index.
            best = process.extractOne(
                normalized,
                choices,
                scorer=self.scorer,
                score_cutoff=score_cutoff,
            )
            if best is None:
                continue
            _matched_str, score, choice_idx = best
            # rapidfuzz returns 0–100; our public threshold is 0–1.
            if score / 100.0 >= threshold:
                matches[left_idx] = choice_to_orig_idx[choice_idx]

        return matches

    def get_all_candidates(
        self,
        left_values: Sequence[str | None],
        right_values: Sequence[str | None],
        threshold: float = 0.85,
        top_n: int = 3,
    ) -> dict[int, list[tuple[int, float]]]:
        """Return top-N scored candidates per left row.

        Useful for debugging / tuning thresholds: when the live match
        rate looks too low, this surface tells you whether candidates
        existed just below threshold (in which case lower the threshold)
        or whether nothing scored at all (in which case the normalizer
        or scorer is the wrong fit).

        Args:
            left_values: Primary list.
            right_values: Enrichment list.
            threshold: Minimum similarity in [0.0, 1.0]. Out-of-range
                values raise ValueError.
            top_n: Cap on candidates per left row.

        Returns:
            ``{left_idx: [(right_idx, score_0_to_1), …]}`` sorted by
            score descending. Only candidates with score >= threshold
            are included — borderline rows in the rapidfuzz prune
            buffer are filtered out before return. Empty list (and
            absent key) for left rows with no candidate clearing the
            threshold.
        """
        _validate_threshold(threshold)
        if not left_values or not right_values:
            return {}

        # Same filter-and-remap as match() — never pass empty rows to
        # rapidfuzz, never let an empty slot tie at score 0 and win.
        choices: list[str] = []
        choice_to_orig_idx: list[int] = []
        for orig_idx, raw in enumerate(right_values):
            normalized = self.normalize(raw)
            if normalized:
                choices.append(normalized)
                choice_to_orig_idx.append(orig_idx)
        if not choices:
            return {}

        score_cutoff = max(0.0, (threshold - 0.001) * 100.0)

        candidates: dict[int, list[tuple[int, float]]] = {}
        for left_idx, raw in enumerate(left_values):
            normalized = self.normalize(raw)
            if not normalized:
                continue
            results = process.extract(
                normalized,
                choices,
                scorer=self.scorer,
                limit=top_n,
                score_cutoff=score_cutoff,
            )
            row: list[tuple[int, float]] = []
            for _matched_str, score, choice_idx in results:
                normalized_score = score / 100.0
                # Re-check the public threshold — rapidfuzz's score_cutoff
                # uses the epsilon-lowered version (so it doesn't drop
                # borderline matches), so the kept candidates may include
                # a sliver of below-threshold rows. Filter them here.
                if normalized_score >= threshold:
                    row.append((choice_to_orig_idx[choice_idx], normalized_score))
            if row:
                candidates[left_idx] = row
        return candidates


__all__ = ["FuzzyMatcher"]
