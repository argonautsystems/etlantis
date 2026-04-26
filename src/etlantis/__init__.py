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

    config        Manifest loader + pydantic schemas. Generalized from the
                  rvmaps prototype manifest_loader.py.

    ingest        Plain-HTTP downloader (Fremen Protocol), JS-render fallback
                  (Playwright, optional), Bright Data paid-proxy tiers,
                  parallel CSV/Excel/parquet reader, file archival, URL+filename
                  discovery via HTML scraping.

    transform     Schema unification (driven by clio.extract.schema_map),
                  multi-file consolidation + dedup + sort, partitioned parquet
                  output writer.

    match         Two-stage exact-then-fuzzy matcher (lifted from the GOLD_MASTER
                  Sunbiz pipeline; ~75-95% expected hit rate). Incremental
                  delta-only mode for daily 120x speedups.

    score         Configurable weighted scoring with bands. Spec-formula default;
                  apps override weights at construction.

    closures      Status-transition extraction (closures, openings, ownership
                  changes), supertransition detection (N changes in Y window),
                  cross-source permanent-state inference, closure-merge
                  orchestration.

    analytics     Rolling-window trajectory classifier, velocity primitives
                  (days_since / age / churn_rate / license_age), operator
                  rollup, geographic context (per-capita + deviation),
                  statewide metrics, severity-distribution histograms.

    geo           Hierarchical region classification, Shapely / pyproj /
                  Polars-geo primitives, cross-source proximity dedup. Optional
                  dep group: pip install etlantis[geo].

    pipeline      Stage runner driven by manifest-declared stage taxonomy,
                  resume-from-stage checkpointing, workload routing across
                  fleet hosts.

    enforcement   4-point enforcement-landscape join (inspections + EOS
                  closures + master license + unlicensed enforcement).

Status: bootstrap (v0.0.1-rc1-bootstrap). All subsystems are scaffolded with
docstring stubs; implementations land Phase 1+. See the design doc for the
phase-by-phase roadmap.
"""

__version__ = "0.0.1-rc1"
