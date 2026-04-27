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

"""etlantis.analytics.velocity — date-math expression builders.

Thin Polars-expression helpers for the velocity computations that
public-records pipelines do over and over: how old is this license,
how many days since this entity's last event, how long has this been
open. The cleanroom did each of these inline with hand-rolled
``(today - df['date']).dt.days`` calls; lifting them into named
expression builders gives apps a single place to shift "today" for
testing, plus consistent semantics across pipelines.

Every helper returns a `pl.Expr`. Callers compose them into
`with_columns` / `select` calls:

    from etlantis.analytics.velocity import days_since, age_in_days

    df = df.with_columns(
        days_since("last_inspection_date").alias("days_since_inspection"),
        age_in_days("license_open_date").alias("license_age"),
    )

The default anchor is `date.today()`; pass `anchor=date(2024, 1, 1)`
for deterministic backtests.
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl


def days_since(
    date_column: str,
    anchor: date | datetime | None = None,
) -> pl.Expr:
    """Days from ``date_column`` to ``anchor`` (default: today).

    Returns a positive integer for past dates, negative for future
    dates. Null in `date_column` propagates to null. Polars handles
    Date and Datetime sources transparently — Datetime values are
    truncated to the underlying calendar day for the subtraction.

    Args:
        date_column: Column name to subtract from anchor.
        anchor: The "now" against which days are measured. Default
            ``date.today()`` at expression-evaluation time. Pass an
            explicit date for deterministic backtesting.

    Returns:
        Polars expression yielding `Int64` days. Null on null input.
    """
    anchor_date = _resolve_anchor(anchor)
    return (pl.lit(anchor_date) - pl.col(date_column)).dt.total_days().cast(pl.Int64)


def age_in_days(
    date_column: str,
    anchor: date | datetime | None = None,
) -> pl.Expr:
    """Alias for `days_since` with semantic clarity for ages.

    Reads better at the call site for license-age / business-age
    calculations:

        df.with_columns(age_in_days("license_open_date").alias("license_age"))
    """
    return days_since(date_column, anchor=anchor)


def days_between(start_column: str, end_column: str) -> pl.Expr:
    """Days from `start_column` to `end_column`. Negative if reversed.

    Useful for closure-duration calculations (closure_open_date →
    closure_close_date) and for measuring inspection-cycle length.
    """
    return (pl.col(end_column) - pl.col(start_column)).dt.total_days().cast(pl.Int64)


def days_since_last(
    date_column: str,
    entity_column: str,
    anchor: date | datetime | None = None,
) -> pl.Expr:
    """Days since the latest `date_column` value PER entity.

    Group semantics: for every row, returns the days from THIS row's
    entity's most-recent `date_column` value to anchor. Equivalent to
    `df.group_by(entity).agg(date.max())` joined back to the original
    frame, expressed in one Polars window expression.

    Args:
        date_column: Per-event date.
        entity_column: Column identifying the entity (license number,
            address, etc.). Each entity's days_since is computed
            against its own most-recent observation.
        anchor: Override "now" for deterministic backtests.

    Returns:
        Polars expression yielding Int64 days, one value per row in
        the input frame (window function — same row count as input).
    """
    anchor_date = _resolve_anchor(anchor)
    # `over(entity)` is Polars' window-function syntax: compute the
    # max within each entity group, broadcast back to the row count.
    return (
        (pl.lit(anchor_date) - pl.col(date_column).max().over(entity_column))
        .dt.total_days()
        .cast(pl.Int64)
    )


def _resolve_anchor(anchor: date | datetime | None) -> date:
    """Materialize the anchor date.

    None → today (resolved at expression-evaluation time, not module-
    import time, so long-running processes don't drift). Datetime
    → date (we work in calendar days; sub-day precision adds nothing
    for the public-records workloads etlantis serves).
    """
    if anchor is None:
        return date.today()
    if isinstance(anchor, datetime):
        return anchor.date()
    return anchor


__all__ = [
    "days_since",
    "age_in_days",
    "days_between",
    "days_since_last",
]
