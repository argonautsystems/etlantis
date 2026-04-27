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

"""etlantis.geo.region — hierarchical region classification.

The cleanroom's county_utils.py hardcoded a 67-county Florida-to-metro
mapping ("Dade → Miami Metro", "Broward → Miami Metro", etc.). For
the etlantis substrate the right shape is generic: a configurable
mapping table that classifies rows from one geographic granularity
to another.

This module is pure-Python (no Shapely required) — the optional
`etlantis[geo]` extras only matter for `point_in_polygon` and other
Shapely-backed helpers in `geometry.py`.
"""

from __future__ import annotations

import polars as pl


class RegionClassifier:
    """Map values from one geographic granularity to another.

    Built around a single configurable lookup dict — apps pass their
    own (e.g. county→metro, zip→neighborhood, state→region). Polars-
    native: the lookup is implemented as a `pl.col(input).replace`
    expression that runs on the Polars engine.

    Args:
        mapping: ``{source_value: target_value}``. Empty dict raises
            ValueError — silent all-defaults output is a footgun.
            Keys are matched case-sensitively against the input column;
            apps that need case-insensitive matching should normalize
            with `pl.col(c).str.to_uppercase()` before classifying.
        default: Value to assign when the input doesn't appear in
            `mapping`. Default ``None`` so unmapped rows get a clean
            null. Apps with their own "unknown" sentinel can pass
            e.g. ``"OTHER"``.
        case_insensitive: When True, normalize both lookup keys and
            input values via `.str.to_uppercase()` before matching.
            Default False — explicit > implicit. Useful for messy
            public-records data where county names arrive as
            "DADE", "Dade", "dade" interchangeably.

    Output column type matches `mapping`'s value type (str when
    values are strings, etc.). Returns the input frame with one
    added column (custom name configurable).
    """

    def __init__(
        self,
        mapping: dict[str, str],
        default: str | None = None,
        case_insensitive: bool = False,
    ):
        if not mapping:
            raise ValueError(
                "RegionClassifier requires at least one entry in `mapping`; "
                "got empty dict (every row would resolve to `default`)"
            )
        self.case_insensitive = case_insensitive
        if case_insensitive:
            self.mapping = {k.upper(): v for k, v in mapping.items()}
        else:
            self.mapping = dict(mapping)
        self.default = default

    def classify(
        self,
        df: pl.DataFrame,
        source_column: str,
        target_column: str,
    ) -> pl.DataFrame:
        """Add `target_column` to `df` with the classified values.

        Args:
            df: Input frame.
            source_column: Column whose values are looked up in `mapping`.
            target_column: Output column name.

        Returns:
            New DataFrame with `target_column` added.

        Raises:
            ValueError: when `source_column` is missing from `df`.
        """
        if source_column not in df.columns:
            raise ValueError(
                f"RegionClassifier: source column {source_column!r} not "
                f"found in DataFrame. Available columns: {df.columns}"
            )

        # Build the lookup expression. Polars' `replace_strict` handles
        # the "what to do when a value isn't in the mapping" question
        # via the `default` parameter; we use plain `replace` instead so
        # both the mapped and default outputs share the same dtype.
        source_expr = pl.col(source_column)
        if self.case_insensitive:
            source_expr = source_expr.str.to_uppercase()

        # `replace_strict` is Polars 1.0+'s preferred form: substitutes
        # mapping hits and falls back to `default` for misses (vs plain
        # `replace` which would keep the original value for misses).
        # Polars deprecated the `default=` arg on `replace` in 1.0.0
        # in favor of this stricter variant.
        return df.with_columns(
            source_expr.replace_strict(self.mapping, default=self.default).alias(target_column)
        )


__all__ = ["RegionClassifier"]
