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

"""etlantis.match.exact — case-insensitive exact-match via hash lookup.

Generalized from `etlantis-prototype-2026-03/etlantis/stages/matchers/
exact.py` (rvmaps lineage). Same algorithm, just slotted under the
substrate Matcher Protocol with no `etlantis.stages.base` dependency.

Performance: O(n + m) — hash-table lookup. ~1M comparisons/second on
commodity hardware. The right-side list is indexed once; every left
value is then a single dict lookup. For exact ID/license-number style
matching where typos and case are the only variability, this is the
right tool.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from etlantis.match.base import NormalizeFn, default_normalize

logger = logging.getLogger(__name__)


class ExactMatcher:
    """Case-insensitive exact-match.

    The match is exact AFTER normalization. The default normalizer
    strips + uppercases, so ``" Acme Inc "`` and ``"ACME INC"`` collapse
    to the same key. Callers needing stricter or domain-aware behavior
    pass their own `normalization_fn` (e.g. license-number normalizer
    that also strips dashes).

    Args:
        normalization_fn: Callable applied to each value before lookup.
            Empty / falsy outputs are excluded from both the lookup
            table and the match attempt. Default:
            `etlantis.match.base.default_normalize` (strip + upper).

    Threshold semantics: ignored. Exact match is binary — either both
    sides normalize to the same key (score 1.0) or they don't (no row
    in the result). The argument is accepted for Protocol parity with
    the fuzzy and semantic matchers.

    Tie-breaking: first occurrence in `right_values` wins. If two right
    rows normalize to the same key (e.g. ``"acme inc"`` and ``"ACME INC"``
    both normalize to ``"ACME INC"``), only the first is registered and
    a debug-level log entry is emitted naming the dropped index. This
    is intentional — duplicate keys in the right side typically indicate
    a data-quality bug, but they're not fatal; the matcher records the
    issue and continues rather than silently picking a non-deterministic
    one.
    """

    def __init__(self, normalization_fn: NormalizeFn | None = None):
        self.normalize = normalization_fn or default_normalize

    def match(
        self,
        left_values: Sequence[str | None],
        right_values: Sequence[str | None],
        threshold: float = 1.0,  # noqa: ARG002 — Protocol parity
    ) -> dict[int, int]:
        """Return ``{left_idx: right_idx}`` for every exact match.

        Args:
            left_values: Primary list. Each entry is normalized and
                looked up in the right-side index.
            right_values: Enrichment list. Indexed by normalized key
                with first-occurrence-wins tie-breaking.
            threshold: Ignored — kept for Protocol parity.

        Returns:
            ``{left_idx: right_idx}`` mapping for every left value whose
            normalized form is also present on the right side.
        """
        right_lookup: dict[str, int] = {}
        for right_idx, value in enumerate(right_values):
            normalized = self.normalize(value)
            if not normalized:
                continue
            if normalized in right_lookup:
                logger.debug(
                    "ExactMatcher: dropping duplicate normalized key %r "
                    "at right index %d (first occurrence at index %d wins)",
                    normalized,
                    right_idx,
                    right_lookup[normalized],
                )
                continue
            right_lookup[normalized] = right_idx

        matches: dict[int, int] = {}
        for left_idx, value in enumerate(left_values):
            normalized = self.normalize(value)
            if normalized:
                right_idx = right_lookup.get(normalized)
                if right_idx is not None:
                    matches[left_idx] = right_idx
        return matches

    def match_rate(
        self, left_values: Sequence[str | None], right_values: Sequence[str | None]
    ) -> float:
        """Hit-rate diagnostic: what fraction of `left_values` matched.

        Useful for ad-hoc data-quality reporting — pipelines that expect
        a 75-95% hit rate from a clean license-number join can flag
        runs that drop below threshold.
        """
        if not left_values:
            return 0.0
        return len(self.match(left_values, right_values)) / len(left_values)


__all__ = ["ExactMatcher"]
