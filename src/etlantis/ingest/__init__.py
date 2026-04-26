# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""etlantis.ingest — Fremen Protocol data acquisition.

Lands Phase 1. The Fremen Protocol is etlantis's documented respectful-
extraction discipline: polite User-Agents, rate limiting with jitter,
exponential backoff, graceful fallback to representative cached data,
no anti-detect bypass tooling. See README §"Fremen Protocol" for the
full protocol text.

Planned modules:

    http_client       Plain HTTP client with HTML-disguise detection (so a
                      WordPress 200 redirected over a missing CSV doesn't
                      get processed as data). Lifted from cleanroom
                      download_dbpr.py.

    reader            Parallel CSV/Excel/parquet reader with multi-encoding
                      fallback (utf-8 -> latin-1 -> cp1252). Lifted from
                      cleanroom E1_ingest_vectorized.py.

    archive           Config-driven file archival to dated subdirectories.
                      Lifted from cleanroom A0_archive.py.

    discovery         URL + filename discovery via HTML scraping of source
                      portals. Lifted from cleanroom E0_discovery.py.

    scrape/           JS render (Playwright, optional) and Bright Data paid
                      proxy tiers for sites that need more than plain HTTP.
                      bright_data/ contains proxies / scraping_browser /
                      web_unlocker submodules.
"""
