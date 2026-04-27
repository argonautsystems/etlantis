# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""etlantis.analytics — Polars-native trajectory + velocity primitives.

Subsystems (v0.2.0+):

    velocity        Date-math expression builders. Polars `pl.Expr`
                    helpers for the date arithmetic public-records
                    pipelines do over and over: `days_since`,
                    `age_in_days`, `days_between`, `days_since_last`
                    (per-entity windowed). Apps compose them into
                    `with_columns` / `select`.

    trajectory      TrajectoryClassifier — per-entity rolling-window
                    trend detection. Given (entity, date, score),
                    classifies each observation as new / improving /
                    stable / declining via shift().over(entity)
                    window expressions. Lifted from cleanroom
                    add_trending_metrics with all DBPR-coupling
                    removed (column names + thresholds are now
                    configurable).
"""

from etlantis.analytics.trajectory import TrajectoryClassifier
from etlantis.analytics.velocity import (
    age_in_days,
    days_between,
    days_since,
    days_since_last,
)

__all__ = [
    "TrajectoryClassifier",
    "days_since",
    "age_in_days",
    "days_between",
    "days_since_last",
]
