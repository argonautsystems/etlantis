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

"""etlantis.match.base — common Matcher Protocol + Match dataclass.

All matchers share the same shape:

    matcher.match(left, right, threshold=…) -> dict[int, int]

mapping `left_index -> right_index` for every left value that found an
acceptable right side. Unmatched left values are absent from the result.

This file pins that contract. Concrete matchers (exact, fuzzy, semantic)
implement it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable


def default_normalize(value: str | None) -> str:
    """Strip + uppercase. Default normalization for every matcher.

    Returns ``""`` for ``None`` or empty input so downstream "treat empties
    as non-matches" logic can be a single ``if not normalized:`` check
    instead of a None guard plus an emptiness guard.
    """
    if not value:
        return ""
    return str(value).strip().upper()


@runtime_checkable
class Matcher(Protocol):
    """The matching surface every concrete matcher exposes.

    `match` is the workhorse: takes two parallel string lists and returns
    a left-index → right-index mapping for every left value that found a
    sufficiently-good right side. Unmatched left values are simply absent
    from the result dict — no sentinel values, no "matched but bad score"
    rows. Callers that want telemetry on unmatched values can take the
    set difference of `range(len(left))` and `result.keys()`.

    Threshold semantics differ by matcher (1.0 for exact, 0.85 default
    for fuzzy, 0.75 default for semantic), but all are upper-bounded at
    1.0 (perfect match) and lower-bounded at 0.0.
    """

    def match(
        self,
        left_values: Sequence[str | None],
        right_values: Sequence[str | None],
        threshold: float = ...,
    ) -> dict[int, int]:
        """Return ``{left_idx: right_idx}`` for matched pairs."""
        ...


# A re-export for app code that wants the shared default normalizer
# without depending on the private name.
NormalizeFn = Callable[[str | None], str]


def _validate_threshold(value: float) -> None:
    """Reject thresholds outside [0.0, 1.0].

    Centralized so every matcher's `match()` (and any threshold-typed
    diagnostic helper) gives the same error wording. The bound is
    inclusive on both sides — 0.0 means "anything matches" and 1.0
    means "exact-after-normalize", both of which are valid edge cases.
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"threshold must be in [0.0, 1.0]; got {value!r}. "
            "Use 0.0 to accept anything, 1.0 for exact-after-normalize."
        )


__all__ = ["Matcher", "NormalizeFn", "default_normalize"]
