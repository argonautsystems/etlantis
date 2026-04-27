# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""etlantis.score — Polars-native weighted scoring primitives.

Subsystems (v0.2.0+):

    weighted        WeightedScorer + ScoreBand. Generic weighted-sum
                    scorer with optional categorical banding. Apps
                    configure column→weight maps and band thresholds
                    via their manifest; the substrate stays generic.

The cleanroom's P0_scorer.py was DBPR-coupled (hardcoded HP/INT/Basic
column names, hardcoded high≥20 / medium 10-20 / low<10 bands). This
substrate ships only the SHAPE; domain-specific values move into the
consuming app's manifest.
"""

from etlantis.score.weighted import ScoreBand, WeightedScorer

__all__ = ["WeightedScorer", "ScoreBand"]
