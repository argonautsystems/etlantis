# etlantis

Manifest-driven Polars-native ETL substrate for messy public-records data. Built on [clio](https://gitlab.com/perlowja/clio) for AI-driven extraction primitives. Honors the **Fremen Protocol** for respectful data acquisition.

> **Status**: alpha — bootstrap (`v0.0.1-rc1-bootstrap`). Phase 1 (manifest spine + ingest + transform + output) lands next.

---

## What this is

A **substrate** for the kind of pipeline that public-records work demands:

- **Heterogeneous sources**: federal APIs (RIDB, BLM Geocommunicator), state portals (DBPR, OMMU), county GIS layers, retail menus, and everything in between. Each with its own auth model, rate limit, schema, encoding, and update cadence.
- **Schema drift over years**: a CSV schema in 2016 isn't the same one in 2025. Column names rename, dtypes shift, fields get added or removed. Pipelines must survive without a fresh manual remap each time.
- **Geospatial primary keys**: campsites at lat/lng. Restaurants at street-address-plus-county. Cross-source dedup needs proximity + name fuzzy in addition to exact join keys.
- **Cross-jurisdictional reconciliation**: federal + state + county records describing the same entity, each with their own naming convention.
- **Ethical extraction**: government and small-publisher sites deserve respectful traffic patterns. Rate limits, polite User-Agents, exponential backoff, fallback to representative data on failure. The Fremen Protocol.

etlantis is the engine. Apps like RiskyEats (Florida restaurant-inspection data journalism), rvmaps (free RV parking on government land), weederboard (medical-cannabis price comparison) sit on top with their own configs and templates.

---

## Three-layer architecture

```
                                Adapter
   ┌───────────────────────────────────────────────────────────────┐
   │  RiskyEats   rvmaps   weederboard   InvestorClaw   future N   │
   │   (apps consume substrates; ship branded UX/agent surface)    │
   └───────────────────────────────────────────────────────────────┘
                                  │
              imports             ▼            imports
   ┌────────────────────┐    ┌────────────────────┐
   │  ic-engine         │    │  etlantis          │   Domain substrate
   │  (financial math)  │    │  (public-records   │
   │                    │    │   ETL pipelines)   │
   └────────────────────┘    └────────────────────┘
                                  │
                                  ▼
                          ┌──────────────────┐
                          │  clio            │       Foundation
                          │  (AI extraction  │
                          │   + tracking +   │
                          │   drift)         │
                          └──────────────────┘
```

etlantis is a **domain substrate**: provides ETL-pipeline primitives generic to any public-records-style data, without baking in any specific data domain. It depends on **clio** (the foundation library) for AI-driven extraction (`clio.extract.schema_map`, `clio.extract.normalize`, `clio.track`, `clio.drift`).

clio and etlantis stay independent of MNEMOS (the fleet's agentic-memory infrastructure); cross-references between them are by metadata convention, not by hard dependency. The architectural decision is documented in ADR-0001 (substrate library independence).

---

## The Fremen Protocol

> *"Walk without rhythm. It won't attract the worm."*

A documented respectful-extraction philosophy for fleet apps to honor when pulling from external sources:

1. **Polite User-Agents**: every request identifies the project + a contact URL. Operators of source systems should be able to reach a human if they need to.
2. **Rate limiting**: configurable delay between requests; default 2-5 seconds with jitter. Never burst.
3. **Exponential backoff**: 3 retries with 5s base + jitter. Don't hammer.
4. **Graceful fallback**: when a source is unreachable, fall back to representative cached data rather than failing the pipeline. The operator decides when stale data is unacceptable.
5. **Logging**: every request gets a timestamp + status. Operators of source systems can identify and contact us if our traffic patterns become a problem.
6. **No bypass**: WAFs and rate-limiters exist for good reasons. If the polite path is blocked, the answer is a manual conversation with the source operator, not a stronger anti-detect tool.

The protocol applies to everything in `etlantis.ingest.*`. Apps that opt out are not honoring etlantis discipline and shouldn't claim to.

---

## Manifest-driven pipelines

etlantis pipelines are declared in JSON / YAML manifests, not in Python code. A manifest defines:

- **Sources**: per-source endpoint, auth, rate limit, fallback, output path
- **Stages**: app-declared pipeline taxonomy (rvmaps uses `E0 → E1 → P → T → L`; RiskyEats uses `E → P → T → C → L`; etlantis enforces no specific shape)
- **Routing**: which stages run on which fleet hosts (CPU-only vs GPU-equipped)
- **Workload**: which datasets to process, how to partition

The Python layer is a runner: it loads manifests, validates them via pydantic schemas, executes stages, and emits parquet. Apps customize by editing manifests, not by editing etlantis.

This pattern was lifted from the rvmaps prototype `manifest_loader.py` (battle-tested across a 50-state extraction pipeline against BLM, USFS, NPS, state-park, and county GIS sources). etlantis generalizes it across the fleet.

---

## Install

> *Not yet on PyPI.* During the bootstrap phase, install from git:

```bash
uv pip install "git+https://gitlab.com/perlowja/etlantis.git"
```

For development:

```bash
git clone https://gitlab.com/perlowja/etlantis.git
cd etlantis
uv sync
PYTHONPATH=src python3 -m pytest
```

Optional dep groups:

```bash
uv pip install "etlantis[geo]"      # Shapely / pyproj / geopandas for geospatial apps
uv pip install "etlantis[scrape]"   # Playwright for SPA-rendered scrape targets
uv pip install "etlantis[dev]"      # pytest + ruff
```

---

## Layout

```
etlantis/
├── pyproject.toml
├── LICENSE                           # Apache-2.0
├── README.md                         # this file
├── .gitlab-ci.yml                    # uv sync + pytest gate
├── docs/
│   └── PHASE_3_ETLANTIS_DESIGN.md    # comprehensive design doc + Phase 1+ roadmap
├── src/etlantis/
│   ├── __init__.py
│   ├── config/                       # manifest_loader + pydantic schemas
│   ├── ingest/                       # download, scrape, archive, parallel reader
│   ├── transform/                    # schema-unify, consolidate, output
│   ├── match/                        # two-stage exact + fuzzy + cross-source dedup
│   ├── score/                        # weighted scoring with configurable weights
│   ├── closures/                     # status-transition detection
│   ├── analytics/                    # statewide / regional metrics, trends, velocity
│   ├── geo/                          # geospatial primitives (geo extra)
│   └── pipeline/                     # stage runner + manifest orchestration
├── tests/                            # pytest suite (lands Phase 1)
└── examples/                         # tutorial scripts (lands Phase 1)
```

---

## Authorization & licensing context

etlantis is Apache 2.0 single-licensed. The fleet uses Apache 2.0 single across substrate libraries (MNEMOS, clio, etlantis); commercial differentiation lives in apps that consume them where appropriate.

All commits are authored as `Jason Perlow <jperlow@gmail.com>` per fleet discipline.

---

## Contributing

Open issues at [gitlab.com/perlowja/etlantis/-/issues](https://gitlab.com/perlowja/etlantis/-/issues). PRs welcome once the v0.1.0 release stabilizes the public API. Until then, the design doc is the source of truth — read `docs/PHASE_3_ETLANTIS_DESIGN.md` before opening any non-trivial change.

Pre-commit checks:

```bash
uvx ruff check --fix src/ tests/
uvx ruff format src/ tests/
PYTHONPATH=src python3 -m pytest
```

---

## See also

- **[clio](https://gitlab.com/perlowja/clio)** — foundation library: AI-driven extraction primitives. etlantis depends on clio.
- **[ic-engine](https://gitlab.com/perlowja/ic-engine)** — peer domain substrate: deterministic financial-analysis math.
- **[InvestorClaw](https://gitlab.com/perlowja/InvestorClaw)** — claws-family adapter consuming ic-engine.
- **RiskyEats** — first reference implementation of an etlantis-consuming adapter (FL DBPR public records). Currently in cleanroom; will move to `github.com/perlowja/riskyeats` when the etlantis migration completes.
- **rvmaps** — geospatial reference implementation (federal + state + county RV-camping data).
