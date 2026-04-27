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

"""etlantis.geo.distance — haversine distance and proximity helpers.

Haversine is the standard great-circle distance for lat/lng pairs on
a sphere — accurate to ~0.5% for terrestrial distances. Pure-Python
math (no Shapely / pyproj required), so this module is always
available regardless of `etlantis[geo]` extras install state.

For sub-meter accuracy or projected-CRS work, use `etlantis.geo.geometry`
instead (Shapely + pyproj, requires the [geo] extras group).
"""

from __future__ import annotations

import math

import polars as pl

# Earth's mean radius in meters. WGS84 mean radius (6371008.8) rounded
# to the nearest meter; the haversine approximation's intrinsic error
# (~0.5%) dwarfs any sub-meter precision in this constant.
EARTH_RADIUS_METERS = 6_371_008


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in meters between two (lat, lng) points.

    Standard haversine formula. Inputs in decimal degrees. Returns
    meters as a float; for kilometers divide by 1000, for miles
    multiply by 0.000621371.

    Edge cases:
        * Antipodal points (e.g. 0,0 and 0,180) return half Earth's
          circumference (~20015086 m), as expected.
        * Identical points return 0.0.
        * Out-of-range inputs (lat outside [-90, 90], lng outside
          [-180, 180]) are NOT validated — the formula degrades
          gracefully but produces nonsense for clearly-invalid data.
          Apps validating user input should clamp upstream.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_METERS * c


def haversine_expr(lat1_col: str, lng1_col: str, lat2_col: str, lng2_col: str) -> pl.Expr:
    """Polars expression: haversine distance between two pairs of columns.

    Vectorized — runs the formula across the full frame in one Polars
    expression. Useful for proximity-join pipelines:

        df.with_columns(
            haversine_expr("lat", "lng", "ref_lat", "ref_lng")
            .alias("distance_m")
        )

    Args:
        lat1_col, lng1_col: First point's columns.
        lat2_col, lng2_col: Second point's columns.

    Returns:
        Polars Expr yielding Float64 meters per row. Null on null
        input in any of the four columns.
    """
    # Convert each column to radians via degree → radian (* π/180).
    deg_to_rad = math.pi / 180.0
    phi1 = pl.col(lat1_col) * deg_to_rad
    phi2 = pl.col(lat2_col) * deg_to_rad
    delta_phi = (pl.col(lat2_col) - pl.col(lat1_col)) * deg_to_rad
    delta_lambda = (pl.col(lng2_col) - pl.col(lng1_col)) * deg_to_rad

    # haversine: a = sin²(Δφ/2) + cos(φ1)·cos(φ2)·sin²(Δλ/2)
    half_phi = delta_phi / 2
    half_lambda = delta_lambda / 2
    a = half_phi.sin() ** 2 + phi1.cos() * phi2.cos() * half_lambda.sin() ** 2
    # c = 2·atan2(√a, √(1-a)). Polars exposes atan2 as a top-level
    # function `pl.arctan2(y, x)` (no Expr method form).
    c = 2 * pl.arctan2(a.sqrt(), (1 - a).sqrt())
    return EARTH_RADIUS_METERS * c


__all__ = ["haversine_distance", "haversine_expr", "EARTH_RADIUS_METERS"]
