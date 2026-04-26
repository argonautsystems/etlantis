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

    archive           Archiver for dated snapshots of source files with
                      SHA256 fingerprints and a per-snapshot manifest.
                      Idempotent within a calendar day so partial-failure
                      reruns pick up where they left off without burning
                      rate-limit budget. Lifted from cleanroom A0_archive.py.

    reader            Polars-native CSV/Excel/Parquet reader with UTF-8 →
                      cp1252 → latin-1 encoding fallback for CSV. Returns
                      ReadResult so parse failures are inspectable rather
                      than thrown. Lifted from cleanroom
                      E1_ingest_vectorized.py.

Planned (later in Phase 1):

    scrape/           JS render (Playwright, optional) and Bright Data paid
                      proxy tiers for sites that need more than plain HTTP.
"""

from etlantis.ingest.archive import (
    Archiver,
    CapturedFile,
    CaptureResult,
    FailedFile,
)
from etlantis.ingest.discovery import DiscoveredLink, HTMLDiscoverer
from etlantis.ingest.http_client import (
    DownloadResult,
    DownloadStatus,
    HTTPClient,
)
from etlantis.ingest.reader import ReadResult, read_many, read_table

__all__ = [
    "HTTPClient",
    "DownloadResult",
    "DownloadStatus",
    "HTMLDiscoverer",
    "DiscoveredLink",
    "Archiver",
    "CaptureResult",
    "CapturedFile",
    "FailedFile",
    "ReadResult",
    "read_table",
    "read_many",
]
