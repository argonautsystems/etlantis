#!/usr/bin/env python3
# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""End-to-end ingest example wiring every Phase 1 substrate together.

What this script demonstrates:

    1. Spin up a local HTTP server hosting a fixture HTML index page that
       links to a few CSVs (the shape every public-records portal we
       care about exposes).
    2. Load a manifest via etlantis.config — pydantic validation +
       ${VAR} env substitution.
    3. Use etlantis.ingest.HTMLDiscoverer (which rides on top of the
       Fremen-Protocol-honoring HTTPClient) to walk the index and find
       links matching a file-extension allowlist.
    4. Use etlantis.ingest.Archiver to download the discovered files
       into a dated snapshot directory with SHA256 fingerprints and a
       per-snapshot manifest.
    5. Use etlantis.ingest.read_many to load each archived CSV into a
       Polars DataFrame, with UTF-8 → cp1252 → latin-1 encoding fallback.
    6. Use etlantis.transform.concat_frames to vertically combine the
       per-district frames, normalizing column names and deduping rows.
    7. Use etlantis.transform.write_parquet to persist the consolidated
       frame as a hive-partitioned dataset (partitioned by `Status`).

What this script is NOT:

    - A live integration test against a real public-records site. The
      fixture server keeps the example deterministic and offline-safe.
      Real apps point HTTPClient at real URLs.
    - A benchmark. The fixture data is tiny (~30 rows total).

Run from the repo root:

    uv run python examples/01_ingest_csv_to_parquet.py

Outputs:

    /tmp/etlantis-example-out/archives/<source>/<YYYY-MM-DD>/*.csv +
        _snapshot.json (per-source SHA256-keyed inventory)
    /tmp/etlantis-example-out/parquet/inspections/Status=*/...parquet
"""

from __future__ import annotations

import contextlib
import http.server
import os
import shutil
import socket
import socketserver
import sys
import threading
from datetime import datetime
from pathlib import Path

# Resolve the repo root so this script works whether invoked from the repo
# root or from inside examples/. We don't want to require an install for
# the tutorial.
EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from etlantis.config import ManifestLoader  # noqa: E402
from etlantis.ingest import (  # noqa: E402
    Archiver,
    HTMLDiscoverer,
    HTTPClient,
    read_many,
)
from etlantis.transform import concat_frames, write_parquet  # noqa: E402

FIXTURE_DIR = EXAMPLE_DIR / "fixtures"
MANIFEST_DIR = EXAMPLE_DIR / "manifests"
OUTPUT_DIR = Path("/tmp/etlantis-example-out")


# ---------------------------------------------------------------------------
# Fixture server: serve examples/fixtures/ over loopback so HTTPClient gets
# to exercise the real network path.
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    """Bind a socket to port 0 to ask the OS for a free ephemeral port,
    then close it and return the port number. There's a small race window
    between close() and the server's reuse — fine for a tutorial that
    runs once per invocation, not fine for production code."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def fixture_server(directory: Path):
    """Spin up a loopback HTTP server rooted at `directory` and yield the
    base URL. The server runs in a background thread; the context-manager
    exit shuts it down cleanly."""
    port = _pick_free_port()

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            # Suppress per-request access-log spam — the example's own
            # logging is louder and more useful.
            return

    def handler_factory(*a, **k):
        return QuietHandler(*a, directory=str(directory), **k)

    server = socketserver.TCPServer(("127.0.0.1", port), handler_factory)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Pipeline driver.
# ---------------------------------------------------------------------------


def main() -> int:
    print("[etlantis-example] starting end-to-end Phase-1 substrate smoke...\n")

    # Reset the output dir so re-runs are deterministic.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["EXAMPLE_OUTPUT_DIR"] = str(OUTPUT_DIR)

    with fixture_server(FIXTURE_DIR) as base_url:
        os.environ["EXAMPLE_BASE_URL"] = base_url
        print(f"[1/6] fixture server up at {base_url}")

        # ---- 2. Load + validate the manifest. -----------------------------
        # ManifestLoader handles JSON/YAML auto-detection, ${VAR}
        # substitution, pydantic validation, and per-path caching.
        loader = ManifestLoader(MANIFEST_DIR)
        manifest = loader.load("example_pipeline.yaml")
        print(
            f"[2/6] manifest validated: app={manifest.metadata.app!r} "
            f"sources={[s.source_id for s in manifest.sources]} "
            f"stages={[s.name for s in manifest.stages]}"
        )

        # ---- 3. Discover downloadable links. ------------------------------
        # HTTPClient honors the Fremen Protocol (rate-limit + jitter,
        # exponential backoff, HTML-disguise detection, atomic writes).
        # HTMLDiscoverer reuses it so cookies/proxies/retries apply
        # consistently to discovery and download.
        client = HTTPClient(manifest.global_settings.fremen_protocol)
        discoverer = HTMLDiscoverer(client)
        source = manifest.sources[0]
        # v0.1.0 schema convention: for a discovery-driven source, the first
        # entry of endpoint.paths is the index URL the discoverer crawls.
        # Phase 2 will add a first-class index_path field.
        index_url = source.endpoint.base_url + source.endpoint.paths[0]
        # Discovery filter lives on the `discover` stage's config dict, not
        # on the source itself — the same source could feed multiple
        # discovery stages with different extension allowlists.
        discover_stage = next(s for s in manifest.stages if s.name == "discover")
        extensions = set(discover_stage.config.get("extensions") or [])
        links = discoverer.discover(url=index_url, extensions=extensions)
        print(f"[3/6] discovery: found {len(links)} CSV link(s) at {index_url}")
        for link in links:
            print(f"          - {link.filename}  ({link.text!r})")

        if not links:
            print("ERROR: discovery returned zero links; fixture is broken.")
            return 1

        # ---- 4. Archive the discovered links. -----------------------------
        # Archiver.capture writes into archive_root/<source>/<YYYY-MM-DD>/
        # with SHA256 fingerprints and a per-snapshot _snapshot.json.
        archive_root = loader.get_directory(manifest, "archive_root")
        if archive_root is None:
            print("ERROR: archive_root not configured in manifest.")
            return 1
        archiver = Archiver(client=client, archive_root=archive_root)
        snapshot = archiver.capture(
            source_id=source.source_id,
            urls=[(link.url, link.filename) for link in links],
            snapshot_date=datetime.now().date(),
        )
        print(
            f"[4/6] archive: captured {len(snapshot.captured)} files "
            f"({snapshot.total_bytes} bytes, "
            f"{len(snapshot.failed)} failed, "
            f"{len(snapshot.skipped)} skipped) → {snapshot.snapshot_dir}"
        )

        # ---- 5. Load the snapshot into Polars frames. ---------------------
        snapshot_files = sorted(snapshot.snapshot_dir.glob("*.csv"))
        results = read_many(snapshot_files)
        frames = []
        for r in results:
            if r.df is None:
                print(f"          read failed: {r.path.name} — {r.error}")
                continue
            for w in r.warnings:
                print(f"          warn {r.path.name}: {w}")
            frames.append(r.df)
        print(f"[5/6] read: parsed {len(frames)} CSV(s) into Polars frames")

        # ---- 6. Consolidate + persist as parquet. -------------------------
        consolidated = concat_frames(
            frames,
            dedupe_subset=["License Number"],
            sort_by=["License Number"],
            normalize_column_names=True,
        )
        print(
            f"          consolidated: {consolidated.input_frames} frames → "
            f"{consolidated.rows_in} rows in, "
            f"{consolidated.rows_out} rows out, "
            f"{consolidated.duplicates_removed} duplicate(s) removed"
        )

        parquet_root = loader.get_directory(manifest, "parquet_out")
        if parquet_root is None:
            print("ERROR: parquet_out not configured in manifest.")
            return 1
        write_result = write_parquet(
            consolidated.df,
            parquet_root / "inspections",
            partition_cols=["Status"],
            overwrite=True,
        )
        print(
            f"[6/6] write_parquet: {write_result.files_written} partition file(s), "
            f"{write_result.bytes_written} bytes → {write_result.path}"
        )

    print("\n[etlantis-example] DONE — all six Phase-1 substrates exercised.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
