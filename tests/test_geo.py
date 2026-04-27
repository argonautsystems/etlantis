# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""Tests for etlantis.geo — RegionClassifier + haversine distance."""

from __future__ import annotations

import math

import polars as pl
import pytest

from etlantis.geo import (
    EARTH_RADIUS_METERS,
    RegionClassifier,
    haversine_distance,
    haversine_expr,
)

# ============================================================================
# RegionClassifier
# ============================================================================


def test_region_classifier_rejects_empty_mapping():
    with pytest.raises(ValueError, match="at least one entry"):
        RegionClassifier(mapping={})


def test_region_classify_basic():
    mapping = {"DADE": "Miami Metro", "BROWARD": "Miami Metro", "DUVAL": "Jacksonville"}
    classifier = RegionClassifier(mapping=mapping)
    df = pl.DataFrame({"county": ["DADE", "BROWARD", "DUVAL", "ORANGE"]})
    result = classifier.classify(df, source_column="county", target_column="metro")
    assert result["metro"].to_list() == [
        "Miami Metro",
        "Miami Metro",
        "Jacksonville",
        None,
    ]


def test_region_classify_custom_default():
    mapping = {"DADE": "Miami Metro"}
    classifier = RegionClassifier(mapping=mapping, default="OTHER")
    df = pl.DataFrame({"county": ["DADE", "ORANGE"]})
    result = classifier.classify(df, source_column="county", target_column="metro")
    assert result["metro"].to_list() == ["Miami Metro", "OTHER"]


def test_region_classify_case_insensitive():
    """Real public-records data has 'DADE', 'Dade', 'dade' interchangeably.
    case_insensitive=True normalizes both sides via .upper() before
    matching."""
    mapping = {"DADE": "Miami Metro"}
    classifier = RegionClassifier(mapping=mapping, case_insensitive=True)
    df = pl.DataFrame({"county": ["dade", "Dade", "DADE"]})
    result = classifier.classify(df, source_column="county", target_column="metro")
    assert result["metro"].to_list() == ["Miami Metro", "Miami Metro", "Miami Metro"]


def test_region_classify_case_sensitive_default():
    """Default is case-sensitive: 'dade' shouldn't match 'DADE' in mapping."""
    mapping = {"DADE": "Miami Metro"}
    classifier = RegionClassifier(mapping=mapping)  # case_insensitive=False
    df = pl.DataFrame({"county": ["dade"]})
    result = classifier.classify(df, source_column="county", target_column="metro")
    # 'dade' doesn't match 'DADE' — falls to default (None)
    assert result["metro"].to_list() == [None]


def test_region_classify_raises_on_missing_source_column():
    classifier = RegionClassifier(mapping={"X": "Y"})
    df = pl.DataFrame({"other_col": ["X"]})
    with pytest.raises(ValueError, match="not found in DataFrame"):
        classifier.classify(df, source_column="missing", target_column="out")


def test_region_classify_handles_null_input():
    """Nulls in the source column should pass through to default
    (None) — not error or match an empty-string key."""
    mapping = {"X": "Y"}
    classifier = RegionClassifier(mapping=mapping)
    df = pl.DataFrame({"col": ["X", None]})
    result = classifier.classify(df, source_column="col", target_column="out")
    assert result["out"].to_list() == ["Y", None]


# ============================================================================
# haversine_distance (scalar)
# ============================================================================


def test_haversine_distance_zero_for_identical_points():
    assert haversine_distance(40.0, -74.0, 40.0, -74.0) == 0.0


def test_haversine_distance_known_value_nyc_to_la():
    """NYC (40.7128, -74.0060) to LA (34.0522, -118.2437) is ~3935 km
    by great-circle distance. Allow 1% tolerance for the haversine
    approximation."""
    distance = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    expected_km = 3935  # canonical great-circle reference
    actual_km = distance / 1000
    assert abs(actual_km - expected_km) / expected_km < 0.01


def test_haversine_distance_antipodal_is_half_circumference():
    """Antipodal points should be ~half Earth's circumference."""
    # (0, 0) and (0, 180) are antipodal
    distance = haversine_distance(0.0, 0.0, 0.0, 180.0)
    expected = math.pi * EARTH_RADIUS_METERS  # half circumference
    assert abs(distance - expected) / expected < 0.001


def test_haversine_distance_one_degree_latitude():
    """One degree of latitude is ~111 km anywhere on Earth."""
    # Lat 40.0 vs 41.0 at the same longitude
    distance = haversine_distance(40.0, -74.0, 41.0, -74.0)
    expected_km = 111.2  # canonical latitude-degree reference
    actual_km = distance / 1000
    assert abs(actual_km - expected_km) / expected_km < 0.01


def test_haversine_distance_symmetric():
    """Order of arguments shouldn't change the result."""
    a_to_b = haversine_distance(40.0, -74.0, 34.0, -118.0)
    b_to_a = haversine_distance(34.0, -118.0, 40.0, -74.0)
    assert a_to_b == pytest.approx(b_to_a)


# ============================================================================
# haversine_expr (Polars)
# ============================================================================


def test_haversine_expr_matches_scalar():
    """Vectorized expression should produce the same result as the
    scalar function for the same inputs."""
    df = pl.DataFrame(
        {
            "lat1": [40.7128],
            "lng1": [-74.0060],
            "lat2": [34.0522],
            "lng2": [-118.2437],
        }
    )
    result = df.with_columns(haversine_expr("lat1", "lng1", "lat2", "lng2").alias("dist_m"))
    scalar = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    # Allow a tiny float tolerance; both should produce ~3.94M meters
    assert abs(result["dist_m"].item() - scalar) < 1.0  # within 1 meter


def test_haversine_expr_handles_multiple_rows():
    df = pl.DataFrame(
        {
            "lat1": [40.0, 0.0],
            "lng1": [-74.0, 0.0],
            "lat2": [40.0, 0.0],  # same as lat1 → 0
            "lng2": [-74.0, 180.0],  # antipodal in row 2
        }
    )
    result = df.with_columns(haversine_expr("lat1", "lng1", "lat2", "lng2").alias("dist_m"))
    distances = result["dist_m"].to_list()
    # Row 0: identical points → 0
    assert distances[0] == pytest.approx(0.0)
    # Row 1: antipodal → ~half circumference
    expected_half_circ = math.pi * EARTH_RADIUS_METERS
    assert abs(distances[1] - expected_half_circ) / expected_half_circ < 0.001


def test_haversine_expr_propagates_null():
    df = pl.DataFrame(
        {
            "lat1": [40.0, None],
            "lng1": [-74.0, -74.0],
            "lat2": [41.0, 41.0],
            "lng2": [-74.0, -74.0],
        },
        schema={
            "lat1": pl.Float64,
            "lng1": pl.Float64,
            "lat2": pl.Float64,
            "lng2": pl.Float64,
        },
    )
    result = df.with_columns(haversine_expr("lat1", "lng1", "lat2", "lng2").alias("dist_m"))
    distances = result["dist_m"].to_list()
    # Row 0: ~111 km (one degree of latitude)
    assert abs(distances[0] - 111_200) / 111_200 < 0.01
    # Row 1: null in lat1 → null result
    assert distances[1] is None
