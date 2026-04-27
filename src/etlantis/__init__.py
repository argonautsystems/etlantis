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

"""etlantis — manifest-driven Polars-native ETL substrate for messy public-records data.

Built on clio (https://gitlab.com/perlowja/clio) for AI-driven extraction
primitives. Honors the Fremen Protocol for respectful data acquisition.

Subsystems:

    config        Manifest loader + pydantic schemas — IMPLEMENTED v0.1.0.

    ingest        HTTPClient (Fremen Protocol), HTMLDiscoverer, Archiver,
                  Polars-native reader with UTF-8 → cp1252 → latin-1
                  encoding fallback — IMPLEMENTED v0.1.0.

    transform     concat_frames + write_parquet (atomic single-file or
                  hive-partitioned, refuse-or-overwrite) — IMPLEMENTED v0.1.0.

    match         ExactMatcher + FuzzyMatcher (rapidfuzz-backed) +
                  SemanticMatcher (sentence-transformers, optional install
                  via etlantis[semantic]). Common Matcher Protocol —
                  IMPLEMENTED v0.2.0.

    score         WeightedScorer + ScoreBand. Configurable weighted-sum
                  scoring with optional categorical banding —
                  IMPLEMENTED v0.2.0.

    closures      TransitionExtractor (declarative status→event mapping)
                  + SupertransitionDetector (N events in Y days per
                  entity) — IMPLEMENTED v0.2.0.

    analytics     TrajectoryClassifier (per-entity rolling-window trend
                  detection) + velocity helpers (days_since,
                  age_in_days, days_between, days_since_last) —
                  IMPLEMENTED v0.2.0.

    geo           RegionClassifier (hierarchical region lookup) +
                  haversine_distance / haversine_expr. Shapely-backed
                  point-in-polygon and proximity-dedup are Phase 3 —
                  IMPLEMENTED (Stage 1) v0.2.0.

Planned (Phase 3+):

    pipeline      Stage runner driven by manifest-declared stage
                  taxonomy, resume-from-stage checkpointing, workload
                  routing across fleet hosts.

    enforcement   4-point enforcement-landscape join (inspections + EOS
                  closures + master license + unlicensed enforcement).

Status: alpha. v0.2.0 ships the full Phase 1 + Phase 2 substrate
surface (config + ingest + transform + match + score + closures +
analytics + geo). pipeline + enforcement remain Phase 3+ stubs.
"""

__version__ = "0.2.0"
