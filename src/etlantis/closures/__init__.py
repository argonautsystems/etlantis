# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""etlantis.closures — status-transition extraction + supertransition detection.

Subsystems (v0.2.0+):

    transitions       TransitionRule (status-text → event_type mapping)
                      + TransitionExtractor (Polars expression that
                      classifies rows into events using rules in
                      first-match-wins order). Lifted from cleanroom
                      C0_closures.py with all DBPR-specific patterns
                      moved out of the substrate.

    supertransition   SupertransitionDetector — entities with N+
                      qualifying events in a rolling window. Polars-
                      native group_by + agg with optional event-type
                      whitelist and date-window filters.

Phase 2+ scope notes: cross-source permanent-state inference (combine
EOS closure data + license-status data + inspection callback data to
infer permanent-vs-temporary closure) is a Stage 2 follow-up; the
underlying primitives (extract events, detect supertransitions) ship
in this v0.2.0.
"""

from etlantis.closures.supertransition import (
    SupertransitionDetector,
    SupertransitionResult,
)
from etlantis.closures.transitions import TransitionExtractor, TransitionRule

__all__ = [
    "TransitionRule",
    "TransitionExtractor",
    "SupertransitionDetector",
    "SupertransitionResult",
]
