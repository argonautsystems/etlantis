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

"""etlantis.analytics.trajectory — per-entity score-trajectory classifier.

Generalized from the cleanroom's ``add_trending_metrics.py`` (which
operated inline on the enforcement_landscape parquet via pandas). This
substrate version is configurable, Polars-native, and substrate-clean:
no hardcoded column names, no hardcoded thresholds.

Concept: given a frame of (entity, date, score) observations, classify
each row's *trajectory* by comparing the row's score to the entity's
PREVIOUS observation's score. Four labels:

    "new"        — first observation for this entity (no prior score)
    "improving"  — score change <= improving_threshold (lower is better)
    "declining"  — score change >= declining_threshold (higher is worse)
    "stable"     — score change is between the two thresholds

The "lower is better" convention matches the cleanroom risk-score
posture: higher scores = more violations = worse. Apps with the
opposite convention (higher = better) just flip the sign of their
thresholds.

Output columns added (all configurable names):
    `_previous_score`     — the prior observation's score, or null
    `_score_change`       — current - previous, or null
    `_trajectory`         — categorical label
"""

from __future__ import annotations

import polars as pl


class TrajectoryClassifier:
    """Per-entity rolling-window trend classifier.

    Args:
        entity_column: Column identifying the entity.
        date_column: Column ordering observations within an entity.
            Polars Date or Datetime. Must be sortable.
        score_column: Column containing the numeric score whose
            change is being classified.
        improving_threshold: Score deltas <= this value are
            "improving". Default ``-0.5`` — a score that DROPPED by
            half a point or more. Negative because lower scores =
            better in the cleanroom convention.
        declining_threshold: Score deltas >= this value are
            "declining". Default ``0.5``.
        previous_score_column: Output column for the prior score.
            Default ``"_previous_score"``.
        score_change_column: Output column for current - previous.
            Default ``"_score_change"``.
        trajectory_column: Output column for the categorical label.
            Default ``"_trajectory"``.

    Threshold semantics:
        * `improving_threshold` must be <= `declining_threshold` so
          there's a stable band between them. Equal is allowed (no
          stable band — every change is improving or declining).
        * Both are inclusive on their respective sides:
            change <= improving_threshold  → improving
            change >= declining_threshold  → declining
            otherwise                      → stable
        * `change is null` (i.e. first observation) → "new"

    Polars-native: uses `shift().over(entity)` window expressions so
    classification fits in a single `with_columns` call.
    """

    def __init__(
        self,
        entity_column: str,
        date_column: str,
        score_column: str,
        improving_threshold: float = -0.5,
        declining_threshold: float = 0.5,
        previous_score_column: str = "_previous_score",
        score_change_column: str = "_score_change",
        trajectory_column: str = "_trajectory",
    ):
        if improving_threshold > declining_threshold:
            raise ValueError(
                f"improving_threshold ({improving_threshold}) must be "
                f"<= declining_threshold ({declining_threshold}); "
                f"otherwise no score change can be classified as 'stable'"
            )
        self.entity_column = entity_column
        self.date_column = date_column
        self.score_column = score_column
        self.improving_threshold = improving_threshold
        self.declining_threshold = declining_threshold
        self.previous_score_column = previous_score_column
        self.score_change_column = score_change_column
        self.trajectory_column = trajectory_column

    def classify(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add trajectory columns to `df`.

        Sorts by (entity, date) internally so the prior-score lookup
        is deterministic regardless of input row order. The original
        column order is preserved in the output (sort is internal).

        Args:
            df: Input frame containing entity, date, and score columns.

        Returns:
            New DataFrame with three added columns:
                `previous_score_column` — the row's entity's prior
                    score, or null for that entity's first observation.
                `score_change_column`  — current - previous, or null.
                `trajectory_column`    — "new" | "improving" |
                    "stable" | "declining".

        Raises:
            ValueError: when any required column is missing.
        """
        self._validate_columns(df)

        # Sort by (entity, date) so shift().over(entity) gives the
        # CHRONOLOGICALLY prior row, not just the previous row in input
        # order. We don't restore the original order — Polars frames
        # are unordered by spec, and downstream consumers should
        # explicitly sort if they need a specific layout.
        sorted_df = df.sort(self.entity_column, self.date_column)

        prev_score_expr = pl.col(self.score_column).shift(1).over(self.entity_column)
        change_expr = pl.col(self.score_column) - prev_score_expr

        # Trajectory chain. Order matters because we want "new" to win
        # for the first observation per entity (where prev_score is
        # null and so `change_expr` is null).
        trajectory_expr = (
            pl.when(prev_score_expr.is_null())
            .then(pl.lit("new"))
            .when(change_expr <= self.improving_threshold)
            .then(pl.lit("improving"))
            .when(change_expr >= self.declining_threshold)
            .then(pl.lit("declining"))
            .otherwise(pl.lit("stable"))
        )

        return sorted_df.with_columns(
            prev_score_expr.alias(self.previous_score_column),
            change_expr.alias(self.score_change_column),
            trajectory_expr.alias(self.trajectory_column),
        )

    def _validate_columns(self, df: pl.DataFrame) -> None:
        required = [self.entity_column, self.date_column, self.score_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"TrajectoryClassifier: required columns missing: "
                f"{missing}. Available columns: {df.columns}"
            )


__all__ = ["TrajectoryClassifier"]
