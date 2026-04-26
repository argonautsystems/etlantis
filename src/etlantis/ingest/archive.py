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

"""etlantis.ingest.archive — dated snapshots of source files with fingerprints.

Generalized from the RiskyEats cleanroom `A0_archive.py`. The cleanroom
version was DBPR-specific (hardcoded DBPR_DATA_SOURCES, MD5 fingerprints,
free-form state-metadata.json). This version is:

  * Manifest-driven — caller passes a list of (url, basename) tuples; the
    Source config lives in `etlantis.config.schema.Source`.
  * Content-addressable — SHA256 fingerprints (clio convention), not MD5.
  * Snapshot-manifest-based — each capture writes a small JSON describing
    what landed where, instead of a single mutable state file.

Lifecycle:

    archiver = Archiver(client=HTTPClient(), archive_root=Path("archives"))
    result = archiver.capture(
        source_id="dbpr_inspections",
        urls=[
            ("https://example.gov/data/1.csv", "1.csv"),
            ("https://example.gov/data/2.csv", "2.csv"),
        ],
    )
    # result.snapshot_dir == archives/dbpr_inspections/2026-04-26/
    # result.snapshot_dir/_snapshot.json describes captured + failed + skipped
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path

from etlantis.ingest.http_client import DownloadStatus, HTTPClient

logger = logging.getLogger(__name__)

_HASH_CHUNK = 65536
_SNAPSHOT_FILENAME = "_snapshot.json"

# Tight allowlists for path components. Conservative on purpose: we allow
# only the characters that survive cleanly across POSIX and Windows
# filesystems. source_id is a single identifier; basename allows dots
# (file extensions) but not leading dots (no hidden files) and not
# trailing dots/spaces (Windows trims them silently).
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_BASENAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*[A-Za-z0-9_-]$|^[A-Za-z0-9_]$")


@dataclass(frozen=True)
class CapturedFile:
    """A file that landed in the snapshot directory."""

    filename: str
    """Basename inside the snapshot directory (e.g. 'inspections_1.csv')."""

    url: str
    """The fully-qualified URL the bytes came from."""

    size: int
    """Bytes on disk."""

    sha256: str
    """Hex-encoded SHA256 of the file contents — content-addressable id."""

    content_type: str | None = None
    """Server-reported Content-Type, if any."""


@dataclass(frozen=True)
class FailedFile:
    """A URL that failed to capture in this snapshot run."""

    filename: str
    url: str
    reason: str
    """One of {'failed', 'html_disguise', 'hash_failed'}."""


@dataclass
class CaptureResult:
    """Outcome of a single Archiver.capture() run.

    `mutable` so we can build it up file-by-file. Caller should treat the
    object as immutable once `capture()` returns.
    """

    source_id: str
    snapshot_date: str
    """ISO-8601 date string (YYYY-MM-DD) of the snapshot subdirectory."""

    snapshot_dir: Path
    captured: list[CapturedFile] = field(default_factory=list)
    failed: list[FailedFile] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    """Filenames already present in the snapshot dir, not re-fetched."""

    @property
    def total_bytes(self) -> int:
        """Sum of bytes captured in this run (excludes skipped)."""
        return sum(c.size for c in self.captured)


def _sha256_file(path: Path) -> str:
    """Stream-hash a file as SHA256. Suitable for arbitrarily large CSVs."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_unlink(path: Path) -> None:
    """Best-effort unlink. Logs and swallows OSError so cleanup can never
    mask the underlying error in the caller.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("[archive] could not remove %s: %s", path, exc)


class Archiver:
    """Snapshot source files into dated subdirectories with content fingerprints.

    Each `capture()` call writes into
    `archive_root/<source_id>/<YYYY-MM-DD>/`. Files already present in that
    directory are NOT re-fetched (idempotent within a single calendar day),
    so a re-run after a partial failure picks up where it left off without
    burning the source's rate-limit budget.

    A `_snapshot.json` is written at the end with the captured / failed /
    skipped breakdown plus per-file SHA256s. Downstream change-detection
    diffs today's manifest against yesterday's.

    Args:
        client: HTTPClient honoring the Fremen Protocol. The archiver
            reuses the client's rate-limit window across all files in a
            capture run, so the configured polite-delay is preserved.
        archive_root: Base directory for snapshots. Created on first use.
    """

    def __init__(self, client: HTTPClient, archive_root: Path | str):
        self.client = client
        self.archive_root = Path(archive_root)

    def capture(
        self,
        source_id: str,
        urls: list[tuple[str, str]],
        *,
        snapshot_date: date_cls | None = None,
    ) -> CaptureResult:
        """Capture a list of URLs into a dated snapshot directory.

        Args:
            source_id: Logical identifier for this source. Used as the
                first-level subdirectory under `archive_root`. Allowlist:
                ``^[A-Za-z0-9][A-Za-z0-9_-]*$`` — no slashes, no dots, no
                whitespace, no Unicode lookalikes.
            urls: List of (url, basename) tuples. Basenames must be unique
                within a single capture call; duplicates raise ValueError
                up front (rather than being silently masked by the
                same-day skip path). `basename` is the filename inside the
                snapshot directory; it MAY differ from the URL's basename
                when the source uses query strings or weird path segments.
            snapshot_date: Date for the snapshot subdirectory. Defaults to
                today (UTC-naive local date — sufficient resolution for
                daily public-records pulls).

        Returns:
            CaptureResult with `captured` describing the COMPLETE inventory
            of files in the snapshot directory at the end of the run
            (including pre-existing files, re-hashed for change detection),
            plus `failed` for URLs that couldn't be captured this run.
            `_snapshot.json` is written into the snapshot directory.
        """
        self._validate_source_id(source_id)
        self._validate_unique_basenames(urls)

        snapshot_date = snapshot_date or datetime.now().date()
        date_str = snapshot_date.isoformat()
        snapshot_dir = self.archive_root / source_id / date_str
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        result = CaptureResult(
            source_id=source_id,
            snapshot_date=date_str,
            snapshot_dir=snapshot_dir,
        )

        for url, basename in urls:
            self._validate_basename(basename)
            dest = snapshot_dir / basename

            if dest.exists():
                # Re-hash the pre-existing file so the snapshot manifest
                # always has a complete inventory. If hashing fails, treat
                # it as a poisoned file: unlink and re-fetch (don't let a
                # broken byte sequence on disk wedge subsequent reruns).
                inventory_entry = self._inventory_existing(basename, url, dest)
                if inventory_entry is not None:
                    result.captured.append(inventory_entry)
                    result.skipped.append(basename)
                    logger.debug("[archive] %s already present — re-hashed", dest)
                    continue
                logger.warning(
                    "[archive] pre-existing %s could not be hashed; will re-fetch",
                    dest,
                )

            download = self.client.download_one(url, dest)
            if download.status == DownloadStatus.OK:
                try:
                    digest = _sha256_file(dest)
                except OSError as exc:
                    # Hash failed after a successful download. The bytes
                    # are unverifiable, so unlink the file: leaving it on
                    # disk would cause the same-day rerun path above to
                    # skip it forever and never recover.
                    logger.warning(
                        "[archive] could not hash %s: %s — unlinking and recording as failed",
                        dest,
                        exc,
                    )
                    _safe_unlink(dest)
                    result.failed.append(
                        FailedFile(filename=basename, url=url, reason="hash_failed")
                    )
                    continue
                result.captured.append(
                    CapturedFile(
                        filename=basename,
                        url=url,
                        size=download.size,
                        sha256=digest,
                        content_type=download.content_type,
                    )
                )
                logger.info("[archive] captured %s (%d bytes)", basename, download.size)
            else:
                reason = (
                    "html_disguise" if download.status == DownloadStatus.HTML_DISGUISE else "failed"
                )
                result.failed.append(FailedFile(filename=basename, url=url, reason=reason))
                logger.warning("[archive] %s: %s", basename, reason)

        self._write_snapshot_manifest(result)
        return result

    def _inventory_existing(self, basename: str, url: str, dest: Path) -> CapturedFile | None:
        """Hash a pre-existing file in the snapshot dir.

        Returns a CapturedFile entry on success, None if hashing fails
        (caller is expected to unlink and re-fetch in that case). Size is
        read from the filesystem; content_type is unknown for files that
        weren't fetched in this run, so it's left as None.
        """
        try:
            digest = _sha256_file(dest)
            size = dest.stat().st_size
        except OSError as exc:
            logger.warning("[archive] hash/stat failed for %s: %s", dest, exc)
            _safe_unlink(dest)
            return None
        return CapturedFile(
            filename=basename,
            url=url,
            size=size,
            sha256=digest,
            content_type=None,
        )

    def _write_snapshot_manifest(self, result: CaptureResult) -> None:
        """Serialize CaptureResult to <snapshot_dir>/_snapshot.json.

        `captured` is the COMPLETE inventory of files in the snapshot
        directory at end-of-run (including pre-existing files that this
        run skipped over but re-hashed). `skipped` is the subset of
        basenames in `captured` that were not freshly downloaded — purely
        informational, useful for telemetry. Downstream change detection
        should diff `captured` between snapshots, not `skipped`.
        """
        manifest_path = result.snapshot_dir / _SNAPSHOT_FILENAME
        payload = {
            "source_id": result.source_id,
            "snapshot_date": result.snapshot_date,
            "captured": [asdict(c) for c in result.captured],
            "failed": [asdict(f) for f in result.failed],
            "skipped": list(result.skipped),
            "total_bytes": result.total_bytes,
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug("[archive] wrote snapshot manifest %s", manifest_path)

    @staticmethod
    def _validate_source_id(source_id: str) -> None:
        """Reject source_ids that would escape archive_root or alias on
        Windows. Allowlist: starts alphanumeric, then alphanumerics +
        underscore + dash. Excludes dots, slashes, whitespace, and
        non-ASCII Unicode that could lookalike-collide on case-folding
        filesystems.
        """
        if not _SOURCE_ID_RE.match(source_id):
            raise ValueError(
                f"invalid source_id {source_id!r}; must match ^[A-Za-z0-9][A-Za-z0-9_-]*$"
            )

    @staticmethod
    def _validate_basename(basename: str) -> None:
        """Reject basenames containing path separators, traversal tokens,
        leading dots, trailing dots/spaces, or non-allowlist characters.
        Allows internal dots so file extensions work (e.g. 'data.csv').
        """
        if not _BASENAME_RE.match(basename):
            raise ValueError(
                f"invalid basename {basename!r}; "
                "must match ^[A-Za-z0-9_][A-Za-z0-9._-]*[A-Za-z0-9_-]$ "
                "(no leading dot, no trailing dot/space, no separators)"
            )

    @staticmethod
    def _validate_unique_basenames(urls: list[tuple[str, str]]) -> None:
        """Catch duplicate basenames before any HTTP work.

        Two entries with the same basename would resolve to the same dest
        path; the second download would overwrite the first (or be silently
        masked by the same-day skip path), and the snapshot manifest would
        lose track of one of the URLs. Surface that as a config error
        rather than a silent data loss.
        """
        seen: dict[str, str] = {}
        for url, basename in urls:
            if basename in seen:
                raise ValueError(
                    f"duplicate basename {basename!r} in capture; "
                    f"first url={seen[basename]!r}, second url={url!r}. "
                    "Each capture call must use unique basenames."
                )
            seen[basename] = url


__all__ = [
    "Archiver",
    "CaptureResult",
    "CapturedFile",
    "FailedFile",
]
