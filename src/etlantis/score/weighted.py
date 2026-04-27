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

"""etlantis.score.weighted — Polars-native weighted scoring primitive.

The cleanroom's ``P0_scorer.py`` baked DBPR-specific column names
("Number of High Priority Violations", RDAR fine columns, pest-penalty
flags) and DBPR-specific bands (high≥20, medium 10-20, low<10) directly
into the substrate. None of that belongs in a generic library.

This module ships only the SHAPE: a configurable weighted-sum scorer
and an optional band classifier. App-specific column choices, weight
values, and band thresholds live in the consuming app's manifest.

Example usage:

    from etlantis.score import WeightedScorer, ScoreBand

    scorer = WeightedScorer(
        weights={
            "high_priority_violations": 5.0,
            "intermediate_violations": 2.0,
            "basic_violations": 0.5,
            "pest_penalty": 10.0,
        },
        bands=[
            ScoreBand(name="high", min_score=20.0, max_score=None),
            ScoreBand(name="medium", min_score=10.0, max_score=20.0),
            ScoreBand(name="low", min_score=None, max_score=10.0),
        ],
    )

    scored = scorer.score(df)
    # Adds two columns: '_score' (weighted sum) and '_band' (str)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import polars as pl


@dataclass(frozen=True)
class ScoreBand:
    """Inclusive-min, exclusive-max numerical band with a categorical name.

    A band classifies a numerical score into a named category. Multiple
    bands form a partition of the score range; pass them to
    `WeightedScorer(bands=[...])` for automatic banding.

    Args:
        name: The categorical value emitted for scores in this band.
            Anything stringy. App convention: lowercase ("high",
            "medium", "low"). Empty string is rejected.
        min_score: Inclusive lower bound. ``None`` means "no lower
            bound" (used for the bottom band).
        max_score: Exclusive upper bound. ``None`` means "no upper
            bound" (used for the top band). Must be strictly greater
            than ``min_score`` when both are non-None.

    The exclusive-max convention means a 10/20 split classifies 10.0 as
    "medium" and 19.999 as "medium" but 20.0 as "high" — this matches
    histogram-bin semantics and avoids edge-case ambiguity.
    """

    name: str
    min_score: float | None
    max_score: float | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ScoreBand.name must be non-empty")
        if (
            self.min_score is not None
            and self.max_score is not None
            and self.max_score <= self.min_score
        ):
            raise ValueError(
                f"ScoreBand {self.name!r}: max_score ({self.max_score}) must be "
                f"strictly greater than min_score ({self.min_score})"
            )

    def contains(self, score: float) -> bool:
        """Cheap bool predicate for unit-testing band-assignment logic."""
        if self.min_score is not None and score < self.min_score:
            return False
        if self.max_score is not None and score >= self.max_score:
            return False
        return True


class WeightedScorer:
    """Compute a weighted sum across configured DataFrame columns.

    Args:
        weights: ``{column_name: weight}``. The score for each row is
            ``sum(row[col] * weight for col, weight in weights.items())``.
            Missing columns at score-time raise ValueError; missing
            values within a present column are treated as 0 (Polars'
            standard `.fill_null(0)` posture).
        bands: Optional list of ScoreBand instances. When present, an
            additional categorical column is added to the output with
            the matching band name per row. Bands are validated at
            construction for non-overlap; rows that don't fall into any
            band get ``None`` in the band column (caller decides
            whether to treat that as data-quality bug or expected).
        score_column: Name of the output numerical-score column.
            Default ``"_score"`` (matches the cleanroom convention so
            adapters porting from rvmaps don't need column renames).
        band_column: Name of the output band column. Ignored if
            `bands` is None.

    Threshold / weight semantics:
        * Weights are floats, can be positive or negative. Negative
          weights subtract from the score.
        * Empty `weights` dict raises ValueError — silent zero-output
          would be a footgun.
        * Weights for columns containing string / categorical types
          raise at score-time (Polars' multiplication semantics will
          object); this is by design — callers should normalize
          categoricals to numerical proxies before scoring.
    """

    def __init__(
        self,
        weights: dict[str, float],
        bands: list[ScoreBand] | None = None,
        score_column: str = "_score",
        band_column: str = "_band",
    ):
        if not weights:
            raise ValueError(
                "WeightedScorer requires at least one weighted column; got empty weights dict"
            )
        self.weights = dict(weights)
        self.bands = list(bands) if bands else None
        self.score_column = score_column
        self.band_column = band_column

        if self.bands is not None:
            self._validate_bands(self.bands)

    @staticmethod
    def _validate_bands(bands: list[ScoreBand]) -> None:
        """Check that bands form a non-overlapping partition.

        Two bands overlap if any score `x` would fall in both. Detection:
        sort by `min_score` (None = -inf) and verify each band's
        `min_score` is >= the previous band's `max_score`. Equal allowed
        (exclusive-max convention).
        """
        if not bands:
            return  # empty bands list is allowed (degenerate but valid)

        def lower(b: ScoreBand) -> float:
            return float("-inf") if b.min_score is None else b.min_score

        def upper(b: ScoreBand) -> float:
            return float("inf") if b.max_score is None else b.max_score

        sorted_bands = sorted(bands, key=lower)
        for prev, curr in pairwise(sorted_bands):
            if lower(curr) < upper(prev):
                raise ValueError(
                    f"ScoreBand overlap: {prev.name!r} ends at {upper(prev)} "
                    f"but {curr.name!r} starts at {lower(curr)}"
                )

    def score(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute the weighted sum and (optionally) the band classification.

        Args:
            df: Input frame. Must contain every column named in
                `self.weights`. Extra columns are passed through
                untouched.

        Returns:
            A new DataFrame (Polars semantics — `df` is unmodified) with
            two added columns:
              * ``self.score_column`` (Float64): the weighted sum
              * ``self.band_column`` (Utf8 / Categorical): present only
                when `self.bands` is set; ``None`` for rows that don't
                fall into any band.

        Raises:
            ValueError: when a weighted column is missing from `df`.
                This is a hard error — silently treating a missing
                column as zero would let a renamed input column produce
                misleadingly low scores without surfacing the schema
                drift.
        """
        missing = [col for col in self.weights if col not in df.columns]
        if missing:
            raise ValueError(
                f"WeightedScorer: weighted columns missing from DataFrame: "
                f"{missing}. Available columns: {df.columns}"
            )

        # Build the weighted-sum expression. fill_null(0) so a row with
        # a missing value in one column still gets a coherent score from
        # the other weighted columns; treating a missing as 0 is the
        # standard public-records-data posture (the field wasn't
        # observed, score-as-baseline rather than dropping the row).
        score_expr = pl.lit(0.0)
        for col, weight in self.weights.items():
            score_expr = score_expr + pl.col(col).fill_null(0) * weight

        scored = df.with_columns(score_expr.alias(self.score_column))

        if self.bands is None:
            return scored

        # Band classification via chained when/then expressions — one
        # branch per band, ordered to match the band's bounds. Falls
        # through to None for rows that miss every band.
        return scored.with_columns(self._band_expr().alias(self.band_column))

    def _band_expr(self) -> pl.Expr:
        """Polars expression assigning a band name per row.

        Builds a `when().then().when().then()...otherwise(None)` chain.
        The chain is evaluated top-to-bottom so band order doesn't
        matter functionally, but for clarity we walk in min-score-asc
        order.
        """
        assert self.bands is not None  # caller guards
        score = pl.col(self.score_column)

        def lower(b: ScoreBand) -> float:
            return float("-inf") if b.min_score is None else b.min_score

        sorted_bands = sorted(self.bands, key=lower)
        # Start with the most-restrictive guard (which means the chain
        # short-circuits at the first matching band; we walk asc so
        # rows in the lowest band get classified by the lowest band's
        # condition).
        chain: pl.Expr | None = None
        for band in sorted_bands:
            cond = pl.lit(True)
            if band.min_score is not None:
                cond = cond & (score >= band.min_score)
            if band.max_score is not None:
                cond = cond & (score < band.max_score)
            if chain is None:
                chain = pl.when(cond).then(pl.lit(band.name))
            else:
                chain = chain.when(cond).then(pl.lit(band.name))
        # `chain` is non-None because `bands` was non-empty (validated
        # in __init__'s `_validate_bands` early-return on empty list +
        # `score()`'s `if self.bands is None` guard).
        assert chain is not None
        return chain.otherwise(pl.lit(None, dtype=pl.Utf8))


__all__ = ["WeightedScorer", "ScoreBand"]
