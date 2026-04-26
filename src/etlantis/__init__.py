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

Subsystems (see docs/PHASE_3_ETLANTIS_DESIGN.md for the full design):

    config        Manifest loader + pydantic schemas — IMPLEMENTED in v0.1.0.

    ingest        HTTPClient (Fremen Protocol), HTMLDiscoverer, Archiver
                  (dated SHA256-fingerprinted snapshots), Polars-native
                  reader with UTF-8 → cp1252 → latin-1 encoding fallback —
                  IMPLEMENTED in v0.1.0. JS-render and Bright Data tiers
                  remain stubs for Phase 2+.

    transform     concat_frames (vertical concat with dedupe + sort +
                  column-name normalization) and write_parquet (single-file
                  atomic + hive-partitioned, with overwrite-or-refuse
                  contract) — IMPLEMENTED in v0.1.0. Schema unification via
                  clio.extract.schema_map is Phase 2+.

    match         Two-stage exact-then-fuzzy matcher (lifted from the GOLD_MASTER
                  Sunbiz pipeline; ~75-95% expected hit rate). Incremental
                  delta-only mode for daily 120x speedups. STUB — Phase 2+.

    score         Configurable weighted scoring with bands. Spec-formula default;
                  apps override weights at construction. STUB — Phase 2+.

    closures      Status-transition extraction (closures, openings, ownership
                  changes), supertransition detection (N changes in Y window),
                  cross-source permanent-state inference, closure-merge
                  orchestration. STUB — Phase 2+.

    analytics     Rolling-window trajectory classifier, velocity primitives
                  (days_since / age / churn_rate / license_age), operator
                  rollup, geographic context (per-capita + deviation),
                  statewide metrics, severity-distribution histograms.
                  STUB — Phase 2+.

    geo           Hierarchical region classification, Shapely / pyproj /
                  Polars-geo primitives, cross-source proximity dedup. Optional
                  dep group: pip install etlantis[geo]. STUB — Phase 2+.

    pipeline      Stage runner driven by manifest-declared stage taxonomy,
                  resume-from-stage checkpointing, workload routing across
                  fleet hosts. STUB — Phase 2+.

    enforcement   4-point enforcement-landscape join (inspections + EOS
                  closures + master license + unlicensed enforcement).
                  STUB — Phase 2+.

Status: alpha. v0.1.0 ships the Phase 1 substrates listed above (config +
ingest + transform). Phase 2+ subsystems are scaffolded with docstring
stubs only; the design-doc roadmap details the phase-by-phase plan.
"""

__version__ = "0.1.0"
