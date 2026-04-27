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

"""etlantis.closures.transitions — status-transition extraction primitives.

The cleanroom's ``C0_closures.py`` baked Florida-DBPR-specific status text
into the substrate: ``"Emergency order recommended"``,
``"Emergency Order Callback Complied"``, hardcoded "Inspection
Disposition" column variants. None of that domain coupling belongs in
a generic library.

This module ships only the SHAPE: declarative ``TransitionRule`` mapping
status text to event type, and ``TransitionExtractor`` that walks rules
in order and assigns the first matching event_type per row. App-
specific patterns and column names live in the consuming app's
manifest.

Example usage:

    from etlantis.closures import TransitionRule, TransitionExtractor

    rules = [
        TransitionRule(event_type="CLOSURE",
                       pattern="Emergency order recommended"),
        TransitionRule(event_type="REOPENING",
                       pattern="Emergency Order Callback Complied"),
    ]
    extractor = TransitionExtractor(
        status_column="Inspection Disposition",
        rules=rules,
    )
    classified = extractor.extract(inspections_df)
    # classified now has '_event_type' column populated with CLOSURE,
    # REOPENING, or None for non-matching rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern

import polars as pl


@dataclass(frozen=True)
class TransitionRule:
    """A single status→event mapping rule.

    Args:
        event_type: The categorical value emitted when this rule
            matches. Convention is uppercase ("CLOSURE", "REOPENING",
            "OWNERSHIP_CHANGE"). Empty string is rejected.
        pattern: Either a string (treated as a substring/regex search
            depending on `is_regex`) or a pre-compiled `re.Pattern`.
            String patterns are used with `re.search`, NOT exact
            equality — so a rule with pattern ``"Emergency order"``
            matches both ``"Emergency order recommended"`` and
            ``"Emergency order issued"``.
        is_regex: When True, treat the pattern string as a regex.
            When False (default), match-anywhere substring (still via
            re.search but with the pattern escaped). Pre-compiled
            Pattern objects bypass this flag entirely.
        case_sensitive: When False (default), match case-insensitively.
            Ignored for pre-compiled Pattern objects (they carry
            their own flags).
    """

    event_type: str
    pattern: str | Pattern[str]
    is_regex: bool = False
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("TransitionRule.event_type must be non-empty")

    def compile(self) -> Pattern[str]:
        """Return a compiled regex matching this rule's predicate.

        Pre-compiled patterns pass through unchanged. String patterns
        are escaped (when `is_regex=False`) and compiled with
        IGNORECASE (when `case_sensitive=False`).
        """
        if isinstance(self.pattern, Pattern):
            return self.pattern
        flags = 0 if self.case_sensitive else re.IGNORECASE
        if self.is_regex:
            return re.compile(self.pattern, flags)
        return re.compile(re.escape(self.pattern), flags)


class TransitionExtractor:
    """Classify rows into event types based on a status column.

    Walks `rules` in order; the FIRST rule whose pattern matches the
    row's status value wins. Rows that match no rule are assigned
    `unmatched_event` (default ``None``).

    Args:
        status_column: Name of the column containing the status text.
            Must be present in any DataFrame passed to `extract`. Null
            values in this column are treated as no-match (don't error).
        rules: List of TransitionRule. Order matters — first match
            wins. Apps with overlapping patterns should order
            most-specific-first.
        unmatched_event: Value to assign to rows that match no rule.
            Default ``None`` so apps can use `.filter(pl.col(event)
            .is_not_null())` to keep only classified rows.
        event_column: Output column name. Default ``"_event_type"``.

    Polars-native: the per-row regex check is implemented via
    `pl.col(status_column).str.contains(pattern, literal=...)` for
    each rule, then chained `when().then()...otherwise()`. Polars
    selects the FIRST true branch's then-value for the output, but
    is free to evaluate all branch predicates in parallel rather
    than short-circuit at the engine level — that's a performance
    detail, not a correctness one. Apps with very expensive regexes
    should put the cheap rules first anyway, since the engine may
    still need to compile every rule.
    """

    def __init__(
        self,
        status_column: str,
        rules: list[TransitionRule],
        unmatched_event: str | None = None,
        event_column: str = "_event_type",
    ):
        if not rules:
            raise ValueError(
                "TransitionExtractor requires at least one rule; "
                "an empty rules list would classify every row as unmatched"
            )
        self.status_column = status_column
        self.rules = list(rules)
        self.unmatched_event = unmatched_event
        self.event_column = event_column

    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return `df` with `event_column` added.

        Raises:
            ValueError: when `status_column` is not present in `df`.
        """
        if self.status_column not in df.columns:
            raise ValueError(
                f"TransitionExtractor: status column {self.status_column!r} "
                f"not found in DataFrame. Available columns: {df.columns}"
            )
        return df.with_columns(self._classify_expr().alias(self.event_column))

    def filter_to(self, df: pl.DataFrame, event_types: list[str] | None = None) -> pl.DataFrame:
        """Run `extract` then keep only rows matching the event_types.

        Args:
            df: Input frame.
            event_types: Whitelist of event types to keep. ``None``
                keeps every row that matched ANY rule (i.e. drops only
                the unmatched ones).

        Returns:
            New DataFrame with `event_column` populated and unmatched/
            non-listed rows filtered out.
        """
        classified = self.extract(df)
        col = pl.col(self.event_column)
        if event_types is None:
            return classified.filter(col.is_not_null())
        return classified.filter(col.is_in(event_types))

    def _classify_expr(self) -> pl.Expr:
        """Build the chained when/then/otherwise expression."""
        status = pl.col(self.status_column)
        # Use `pl.lit(None)`-handling: `str.contains` on null returns
        # null, which propagates through the when/then chain as no-match.
        chain: pl.Expr | None = None
        for rule in self.rules:
            compiled = rule.compile()
            # Polars' str.contains takes a regex string; flags must be
            # baked in inline via the (?ims...) prefix because Polars
            # doesn't expose a flags arg. _polars_inline_flags translates
            # the Python re flags (IGNORECASE, MULTILINE, DOTALL) into
            # the inline-flag prefix so pre-compiled patterns with
            # MULTILINE / DOTALL don't silently drop their flags when
            # carried into the Polars expression.
            pattern_str = _with_inline_flags(compiled.pattern, compiled.flags)
            cond = status.str.contains(pattern_str, literal=False)
            if chain is None:
                chain = pl.when(cond).then(pl.lit(rule.event_type))
            else:
                chain = chain.when(cond).then(pl.lit(rule.event_type))

        assert chain is not None  # __init__ guards empty rules
        # Default for unmatched: if unmatched_event is None we need an
        # explicit Utf8 null so the column dtype stays string.
        if self.unmatched_event is None:
            otherwise = pl.lit(None, dtype=pl.Utf8)
        else:
            otherwise = pl.lit(self.unmatched_event)
        return chain.otherwise(otherwise)


def _with_inline_flags(pattern: str, flags: int) -> str:
    """Bake re-module flags into a regex string via the (?ims...) prefix.

    Polars' `str.contains` accepts a regex string but no flags arg, so
    pre-compiled patterns with MULTILINE / DOTALL / IGNORECASE flags
    would silently lose those flags when only the pattern text is
    handed to Polars. This helper adds an inline-flag prefix that
    Rust's regex crate (which Polars uses) understands.

    Translation:
      re.IGNORECASE → 'i'
      re.MULTILINE  → 'm'
      re.DOTALL     → 's'

    Other Python re flags (re.VERBOSE, re.UNICODE, re.ASCII, re.DEBUG,
    re.LOCALE) don't have direct inline equivalents in Rust regex
    syntax — UNICODE is the default, the others are silently dropped
    with no equivalent. Apps relying on those flags for an
    InOut-of-Polars match should pre-classify into a string column
    via Python and pass that into TransitionExtractor.
    """
    if not flags:
        return pattern
    inline = ""
    if flags & re.IGNORECASE:
        inline += "i"
    if flags & re.MULTILINE:
        inline += "m"
    if flags & re.DOTALL:
        inline += "s"
    if not inline:
        return pattern
    return f"(?{inline}){pattern}"


__all__ = ["TransitionRule", "TransitionExtractor"]
