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

"""etlantis.closures.supertransition — N-events-in-Y-window detection.

The cleanroom's ``detect_supertransitions`` function detected "Florida
restaurant addresses with 3+ ownership changes within 5 years" — useful
domain logic, but the substrate shape underneath is generic: given a
table of events tagged by entity + date + type, find the entities with
N+ events of the configured types within the window.

This module ships only that generic primitive. Apps decide what the
"entity" is (license number, address, license + zip), what events
qualify (any, just CLOSURE, just OWNERSHIP_CHANGE), and the (N,
window_days) thresholds.

Output is a one-row-per-qualifying-entity summary frame with:

    * `entity` (from the named column)
    * `event_count` (total events that qualified)
    * `first_event_date`, `last_event_date`
    * `days_span` (last - first, in days)
    * `event_types` (list[str] in chronological order)
    * `current_event_type` (the event type of the most recent event)

Apps wanting the per-event detail rows for a qualifying entity simply
join the output back into their original event table on the entity
column.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl


@dataclass(frozen=True)
class SupertransitionResult:
    """Summary stats for a single qualifying entity.

    `events_frame` is also exposed as a Polars frame on
    `SupertransitionDetector.detect()`; this dataclass exists so apps
    can iterate `.row()` outputs with named-attribute access in domain
    code that prefers Python objects to Polars expressions.
    """

    entity: str
    event_count: int
    first_event_date: date | datetime
    last_event_date: date | datetime
    days_span: int
    event_types: list[str]
    current_event_type: str


class SupertransitionDetector:
    """Detect entities with N+ qualifying events in a rolling window.

    Args:
        entity_column: Name of the column identifying the entity
            (license number, address, etc.).
        date_column: Name of the column containing event dates. Must
            be a Polars Date or Datetime column. Apps with string
            dates should `.str.to_date(...)` before passing.
        event_column: Name of the column containing the classified
            event type (typically the output of TransitionExtractor).
            Default ``"_event_type"``.

    Polars-native: groupby + agg + filter, no per-row Python loop. For
    100K-row event tables, runs in well under a second on a single
    core. Apps that need per-entity event-detail rows can re-join the
    detector output back to the input frame on `entity_column`.
    """

    def __init__(
        self,
        entity_column: str,
        date_column: str,
        event_column: str = "_event_type",
    ):
        self.entity_column = entity_column
        self.date_column = date_column
        self.event_column = event_column

    def detect(
        self,
        df: pl.DataFrame,
        event_types: list[str] | None = None,
        min_count: int = 3,
        window_days: int | None = None,
        anchor_date: date | datetime | None = None,
    ) -> pl.DataFrame:
        """Return one row per qualifying entity.

        Args:
            df: Input event frame. Must contain `entity_column`,
                `date_column`, and `event_column`.
            event_types: Whitelist of event types to count. ``None``
                counts every event regardless of type (still excludes
                rows where event_column is null).
            min_count: Minimum number of qualifying events for an
                entity to appear in the output. Default 3 — conservative
                for cleanroom-style hot-spot detection. Must be >= 1.
            window_days: When set, only events with date in
                ``[anchor_date - window_days, anchor_date]`` are
                counted. ``None`` counts all events regardless of age.
            anchor_date: The "today" against which `window_days` is
                measured. Default `date.today()`. Useful for
                deterministic backtesting (pin the anchor and the
                detector becomes reproducible).

        Returns:
            DataFrame with one row per qualifying entity and the
            columns documented in the module docstring. Empty frame
            if no entity qualifies.

        Raises:
            ValueError: when required columns are missing or
                `min_count < 1`.
        """
        self._validate_columns(df)
        self._validate_date_dtype(df)
        if min_count < 1:
            raise ValueError(f"min_count must be >= 1; got {min_count}")

        # Filter step 1: drop rows with null event_type or null date.
        # Null event types are unclassified (TransitionExtractor's
        # default for non-matching rows); null dates can't be ordered,
        # would skew first/last extraction, and would still contribute
        # to event_count if not excluded — silent contamination of
        # the chronology. Both must be non-null to participate.
        scoped = df.filter(
            pl.col(self.event_column).is_not_null() & pl.col(self.date_column).is_not_null()
        )

        # Filter step 2: optional event-type allowlist.
        if event_types is not None:
            scoped = scoped.filter(pl.col(self.event_column).is_in(event_types))

        # Filter step 3: optional date-window. Both bounds applied —
        # the lower bound (`anchor - window_days`) is the lookback
        # horizon; the upper bound (`anchor`) caps at "today" so
        # future-dated events (rare but possible — typo'd inspection
        # records, anchored backtests against historical data) don't
        # leak into a window meant to describe past activity.
        if window_days is not None:
            anchor = anchor_date or date.today()
            window_start = anchor - timedelta(days=window_days)
            scoped = scoped.filter(
                (pl.col(self.date_column) >= window_start) & (pl.col(self.date_column) <= anchor)
            )

        if scoped.height == 0:
            return self._empty_result_frame(df)

        # Sort once so the per-group aggregations naturally see events in
        # chronological order; then group by entity.
        sorted_df = scoped.sort(self.date_column)

        agg = sorted_df.group_by(self.entity_column).agg(
            [
                pl.len().alias("event_count"),
                pl.col(self.date_column).min().alias("first_event_date"),
                pl.col(self.date_column).max().alias("last_event_date"),
                pl.col(self.event_column).alias("event_types"),
                pl.col(self.event_column).last().alias("current_event_type"),
            ]
        )

        qualified = agg.filter(pl.col("event_count") >= min_count)
        if qualified.height == 0:
            return self._empty_result_frame(df)

        # Compute days_span = (last - first).days. Polars Date/Datetime
        # subtraction yields a duration; .dt.total_days() is the
        # canonical extraction.
        return qualified.with_columns(
            (pl.col("last_event_date") - pl.col("first_event_date"))
            .dt.total_days()
            .cast(pl.Int64)
            .alias("days_span")
        ).select(
            [
                pl.col(self.entity_column).alias("entity"),
                "event_count",
                "first_event_date",
                "last_event_date",
                "days_span",
                "event_types",
                "current_event_type",
            ]
        )

    def _validate_columns(self, df: pl.DataFrame) -> None:
        required = [self.entity_column, self.date_column, self.event_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"SupertransitionDetector: required columns missing: "
                f"{missing}. Available columns: {df.columns}"
            )

    def _validate_date_dtype(self, df: pl.DataFrame) -> None:
        """Ensure the date column is a temporal Polars dtype.

        Polars' subtraction returns sensible Duration values for Date
        and Datetime; for other types (Utf8 strings, integer offsets,
        etc.) the subtraction either errors with a low-quality message
        or produces nonsense. Surface a clear error here instead.
        """
        dtype = df.schema[self.date_column]
        if dtype not in (pl.Date, pl.Datetime):
            raise ValueError(
                f"SupertransitionDetector: date column "
                f"{self.date_column!r} must be Polars Date or Datetime; "
                f"got {dtype!r}. Use `df.with_columns(pl.col(...).str.to_date(...))` "
                f"to convert string dates before passing."
            )

    def _empty_result_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        """A correctly-typed empty frame derived from the input schema.

        Returning a frame with the right columns and types is friendlier
        than returning ``pl.DataFrame()`` — downstream code can safely
        iterate / select / cast without conditional branches. The
        entity and date dtypes mirror the input columns so callers
        joining the empty frame back to their event tables don't see
        a dtype mismatch.
        """
        entity_dtype = df.schema[self.entity_column]
        date_dtype = df.schema[self.date_column]
        return pl.DataFrame(
            schema={
                "entity": entity_dtype,
                "event_count": pl.UInt32,
                "first_event_date": date_dtype,
                "last_event_date": date_dtype,
                "days_span": pl.Int64,
                "event_types": pl.List(pl.Utf8),
                "current_event_type": pl.Utf8,
            }
        )


__all__ = ["SupertransitionDetector", "SupertransitionResult"]
