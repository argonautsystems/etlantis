# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""etlantis.ingest — Fremen Protocol data acquisition.

The Fremen Protocol is etlantis's documented respectful-extraction
discipline: polite User-Agents identifying project + contact URL, rate
limiting with jitter, exponential backoff on transient failures, graceful
fallback to representative cached data, NO anti-detect bypass tooling.
See README §"Fremen Protocol" for the full protocol text and
`etlantis.config.schema.FremenProtocol` for the per-pipeline configuration
model.

Subsystems (Phase 1):

    http_client       HTTPClient honoring the Fremen Protocol. Atomic
                      .part-then-rename writes, HTML-disguise detection
                      (so a WordPress 200-on-missing-CSV doesn't get
                      processed as data), retry classification
                      (5xx/429/408/network -> retry, 4xx-permanent ->
                      fail). Lifted from cleanroom download_dbpr.py.

    discovery         HTMLDiscoverer for index-page link extraction.
                      Generic file-extension allowlist + optional regex
                      filters on href and text. Delegates HTTP through
                      HTTPClient so cookies, proxies, retry, and
                      rate-limit are honored consistently. Lifted from
                      cleanroom E0_discovery.py.

Planned (later in Phase 1):

    archive           Config-driven file archival to dated subdirectories.
                      Lifted from cleanroom A0_archive.py.

    reader            Parallel CSV/Excel/parquet reader with multi-encoding
                      fallback (utf-8 -> latin-1 -> cp1252). Lifted from
                      cleanroom E1_ingest_vectorized.py.

    scrape/           JS render (Playwright, optional) and Bright Data paid
                      proxy tiers for sites that need more than plain HTTP.
"""

from etlantis.ingest.discovery import DiscoveredLink, HTMLDiscoverer
from etlantis.ingest.http_client import (
    DownloadResult,
    DownloadStatus,
    HTTPClient,
)

__all__ = [
    "HTTPClient",
    "DownloadResult",
    "DownloadStatus",
    "HTMLDiscoverer",
    "DiscoveredLink",
]
