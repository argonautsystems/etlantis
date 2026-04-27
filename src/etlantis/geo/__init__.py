# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""etlantis.geo — geographic primitives.

Subsystems (v0.2.0+):

    region          RegionClassifier — hierarchical lookup mapping
                    one geographic granularity to another (e.g.
                    county → metro area). Pure-Python, always
                    available regardless of extras install.

    distance        Haversine great-circle distance in meters. Both
                    a scalar Python helper and a Polars expression
                    builder for vectorized application across a
                    frame. Pure-Python math (no Shapely required),
                    accurate to ~0.5% for terrestrial distances.

Planned (v0.2.1+, requires `pip install etlantis[geo]`):

    geometry        Point-in-polygon and other Shapely-backed
                    primitives. Will require the [geo] extras
                    (shapely + pyproj + geopandas).

    proximity_dedup Cross-source proximity dedup: given two frames
                    with lat/lng, dedupe rows within X meters of
                    each other. Built on haversine + spatial index.
"""

from etlantis.geo.distance import (
    EARTH_RADIUS_METERS,
    haversine_distance,
    haversine_expr,
)
from etlantis.geo.region import RegionClassifier

__all__ = [
    "RegionClassifier",
    "haversine_distance",
    "haversine_expr",
    "EARTH_RADIUS_METERS",
]
