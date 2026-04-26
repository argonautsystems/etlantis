# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""Tests for etlantis.ingest.archive — Archiver + CaptureResult."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from etlantis.ingest.archive import (
    Archiver,
    CapturedFile,
    CaptureResult,
    FailedFile,
)
from etlantis.ingest.http_client import DownloadResult, DownloadStatus

# ============================================================================
# CaptureResult invariants
# ============================================================================


def test_capture_result_total_bytes_excludes_failed(tmp_path):
    result = CaptureResult(
        source_id="s1",
        snapshot_date="2026-04-26",
        snapshot_dir=tmp_path,
        captured=[
            CapturedFile(filename="a.csv", url="u1", size=100, sha256="h1"),
            CapturedFile(filename="b.csv", url="u2", size=200, sha256="h2"),
        ],
        failed=[FailedFile(filename="c.csv", url="u3", reason="failed")],
    )
    assert result.total_bytes == 300


def test_capture_result_total_bytes_zero_on_empty(tmp_path):
    result = CaptureResult(
        source_id="s1", snapshot_date="2026-04-26", snapshot_dir=tmp_path
    )
    assert result.total_bytes == 0


# ============================================================================
# Archiver._validate_source_id / _validate_basename
# ============================================================================


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "/abs",
        "a/b",
        "a\\b",
        ".",
        "..",
        "_leadingunderscore",  # rule requires alphanumeric start
        "-leadingdash",
        "with space",
        "with.dot",
        "trailing-",  # rule allows internal underscores/dashes; trailing dash ok
        # Wait — rule is ^[A-Za-z0-9][A-Za-z0-9_-]*$ which DOES allow trailing dash.
        # The bad case is dot/whitespace/non-ASCII.
        "with\ttab",
    ],
)
def test_validate_source_id_rejects_unsafe(bad):
    if bad == "trailing-":
        # actually accepted by the regex; not a real bad case
        Archiver._validate_source_id(bad)
        return
    with pytest.raises(ValueError):
        Archiver._validate_source_id(bad)


@pytest.mark.parametrize(
    "good",
    [
        "dbpr_inspections",
        "florida-dbpr",
        "Source_42",
        "a",  # single char allowed
        "a-b-c-d",
    ],
)
def test_validate_source_id_accepts_safe(good):
    Archiver._validate_source_id(good)  # should not raise


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "../etc/passwd",
        "a/b.csv",
        "a\\b.csv",
        ".",
        "..",
        ".hidden",  # leading dot rejected
        "trailing.",  # trailing dot rejected (Windows trims it)
        "with space.csv",
    ],
)
def test_validate_basename_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        Archiver._validate_basename(bad)


@pytest.mark.parametrize(
    "good",
    [
        "data.csv",
        "rdar0825.csv",
        "report_2026.xlsx",
        "a",  # single-char ok
        "_underscore.csv",  # leading underscore allowed (not dot)
    ],
)
def test_validate_basename_accepts_safe(good):
    Archiver._validate_basename(good)  # should not raise


# ============================================================================
# Archiver.capture — happy path
# ============================================================================


def _make_client_returning(*download_results: DownloadResult) -> MagicMock:
    """Create a mock HTTPClient whose download_one() yields the given results in order.

    Side-effect: writes plausible bytes to each `dest_path` so SHA256 hashing
    doesn't fail.
    """
    client = MagicMock()
    iterator = iter(download_results)

    def fake_download(url, dest_path):
        result = next(iterator)
        if result.status == DownloadStatus.OK:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"col1,col2\n1,2\n")
        return result

    client.download_one.side_effect = fake_download
    return client


def test_capture_writes_snapshot_directory(tmp_path):
    client = _make_client_returning(
        DownloadResult(
            url="https://e.com/a.csv",
            dest_path=tmp_path / "a.csv",
            status=DownloadStatus.OK,
            size=14,
            attempts=1,
            content_type="text/csv",
        ),
    )
    archiver = Archiver(client, tmp_path / "archives")
    snapshot_date = date(2026, 4, 26)

    result = archiver.capture(
        source_id="dbpr",
        urls=[("https://e.com/a.csv", "a.csv")],
        snapshot_date=snapshot_date,
    )

    assert result.snapshot_dir == tmp_path / "archives" / "dbpr" / "2026-04-26"
    assert result.snapshot_dir.exists()
    assert (result.snapshot_dir / "a.csv").exists()
    assert len(result.captured) == 1
    assert result.captured[0].sha256  # non-empty hex
    assert len(result.captured[0].sha256) == 64  # SHA256 hex length


def test_capture_writes_snapshot_manifest(tmp_path):
    client = _make_client_returning(
        DownloadResult(
            url="https://e.com/a.csv",
            dest_path=tmp_path / "a.csv",
            status=DownloadStatus.OK,
            size=14,
            attempts=1,
        ),
    )
    archiver = Archiver(client, tmp_path / "archives")
    archiver.capture(
        source_id="dbpr",
        urls=[("https://e.com/a.csv", "a.csv")],
        snapshot_date=date(2026, 4, 26),
    )

    manifest_path = tmp_path / "archives" / "dbpr" / "2026-04-26" / "_snapshot.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["source_id"] == "dbpr"
    assert payload["snapshot_date"] == "2026-04-26"
    assert len(payload["captured"]) == 1
    assert payload["captured"][0]["filename"] == "a.csv"
    assert payload["total_bytes"] == 14


def test_capture_skips_already_present_files(tmp_path):
    """A pre-existing file is not re-downloaded but IS re-hashed and
    appears in the captured inventory so the snapshot manifest stays
    complete across reruns."""
    archive_root = tmp_path / "archives"
    snapshot_dir = archive_root / "dbpr" / "2026-04-26"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "a.csv").write_bytes(b"existing-bytes")

    client = MagicMock()
    archiver = Archiver(client, archive_root)
    result = archiver.capture(
        source_id="dbpr",
        urls=[("https://e.com/a.csv", "a.csv")],
        snapshot_date=date(2026, 4, 26),
    )

    assert "a.csv" in result.skipped
    assert client.download_one.call_count == 0
    # Re-hashed entry must appear in captured inventory
    assert len(result.captured) == 1
    assert result.captured[0].filename == "a.csv"
    assert result.captured[0].size == len(b"existing-bytes")
    assert len(result.captured[0].sha256) == 64


def test_capture_records_html_disguise_as_failure(tmp_path):
    client = MagicMock()
    client.download_one.return_value = DownloadResult(
        url="https://e.com/a.csv",
        dest_path=tmp_path / "a.csv",
        status=DownloadStatus.HTML_DISGUISE,
        size=0,
        attempts=1,
        content_type="text/html",
    )
    archiver = Archiver(client, tmp_path / "archives")
    result = archiver.capture(
        source_id="dbpr",
        urls=[("https://e.com/a.csv", "a.csv")],
        snapshot_date=date(2026, 4, 26),
    )

    assert len(result.failed) == 1
    assert result.failed[0].reason == "html_disguise"


def test_capture_records_failed_status(tmp_path):
    client = MagicMock()
    client.download_one.return_value = DownloadResult(
        url="https://e.com/a.csv",
        dest_path=tmp_path / "a.csv",
        status=DownloadStatus.FAILED,
        size=0,
        attempts=4,
    )
    archiver = Archiver(client, tmp_path / "archives")
    result = archiver.capture(
        source_id="dbpr",
        urls=[("https://e.com/a.csv", "a.csv")],
        snapshot_date=date(2026, 4, 26),
    )

    assert len(result.failed) == 1
    assert result.failed[0].reason == "failed"


def test_capture_uses_today_when_date_omitted(tmp_path):
    client = _make_client_returning(
        DownloadResult(
            url="https://e.com/a.csv",
            dest_path=tmp_path / "a.csv",
            status=DownloadStatus.OK,
            size=14,
            attempts=1,
        ),
    )
    archiver = Archiver(client, tmp_path / "archives")
    result = archiver.capture(
        source_id="dbpr",
        urls=[("https://e.com/a.csv", "a.csv")],
    )

    today = date.today().isoformat()
    assert result.snapshot_date == today
    assert (tmp_path / "archives" / "dbpr" / today).exists()


def test_capture_rejects_traversal_in_source_id(tmp_path):
    client = MagicMock()
    archiver = Archiver(client, tmp_path / "archives")
    with pytest.raises(ValueError):
        archiver.capture(
            source_id="../escape",
            urls=[("https://e.com/a.csv", "a.csv")],
        )


def test_capture_rejects_traversal_in_basename(tmp_path):
    client = _make_client_returning(
        DownloadResult(
            url="https://e.com/x",
            dest_path=tmp_path / "x",
            status=DownloadStatus.OK,
            size=1,
            attempts=1,
        ),
    )
    archiver = Archiver(client, tmp_path / "archives")
    with pytest.raises(ValueError):
        archiver.capture(
            source_id="dbpr",
            urls=[("https://e.com/x", "../escape.csv")],
        )


def test_capture_rejects_duplicate_basenames(tmp_path):
    """Two URLs with the same basename must raise — silent overwrite would
    lose track of one of the URLs in the snapshot manifest."""
    client = MagicMock()
    archiver = Archiver(client, tmp_path / "archives")
    with pytest.raises(ValueError, match="duplicate basename"):
        archiver.capture(
            source_id="dbpr",
            urls=[
                ("https://e.com/a.csv", "data.csv"),
                ("https://other.com/b.csv", "data.csv"),
            ],
            snapshot_date=date(2026, 4, 26),
        )
    # Crucially: validation happened BEFORE any HTTP work
    assert client.download_one.call_count == 0


def test_capture_unlinks_file_on_hash_failure(tmp_path, monkeypatch):
    """If hashing the freshly-downloaded file raises OSError, the file
    must be removed so the next same-day rerun can recover."""
    client = _make_client_returning(
        DownloadResult(
            url="https://e.com/a.csv",
            dest_path=tmp_path / "a.csv",
            status=DownloadStatus.OK,
            size=14,
            attempts=1,
        ),
    )
    archiver = Archiver(client, tmp_path / "archives")

    # Force _sha256_file to raise on the first call (post-download path)
    from etlantis.ingest import archive as archive_mod

    real_hash = archive_mod._sha256_file
    call_count = {"n": 0}

    def flaky_hash(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated hash failure")
        return real_hash(path)

    monkeypatch.setattr(archive_mod, "_sha256_file", flaky_hash)

    result = archiver.capture(
        source_id="dbpr",
        urls=[("https://e.com/a.csv", "a.csv")],
        snapshot_date=date(2026, 4, 26),
    )

    # Recorded as failed, NOT captured
    assert len(result.captured) == 0
    assert len(result.failed) == 1
    assert result.failed[0].reason == "hash_failed"
    # And the poisoned file is gone — next run won't skip it
    assert not (result.snapshot_dir / "a.csv").exists()


def test_capture_rerun_preserves_inventory(tmp_path):
    """Same-day rerun must produce a snapshot manifest with the COMPLETE
    inventory of files in the snapshot dir — not just the empty 'skipped'
    list with everything else dropped."""
    client = _make_client_returning(
        DownloadResult(
            url="https://e.com/a.csv",
            dest_path=tmp_path / "a.csv",
            status=DownloadStatus.OK,
            size=14,
            attempts=1,
        ),
    )
    archiver = Archiver(client, tmp_path / "archives")

    # First run: file is freshly captured.
    archiver.capture(
        source_id="dbpr",
        urls=[("https://e.com/a.csv", "a.csv")],
        snapshot_date=date(2026, 4, 26),
    )

    # Second run: file already exists; same-day skip path triggers.
    second = archiver.capture(
        source_id="dbpr",
        urls=[("https://e.com/a.csv", "a.csv")],
        snapshot_date=date(2026, 4, 26),
    )

    # Manifest still describes the file fully
    manifest_path = second.snapshot_dir / "_snapshot.json"
    payload = json.loads(manifest_path.read_text())
    assert len(payload["captured"]) == 1
    assert payload["captured"][0]["filename"] == "a.csv"
    assert payload["captured"][0]["size"] == 14
    assert len(payload["captured"][0]["sha256"]) == 64
    assert "a.csv" in payload["skipped"]


def test_capture_rerun_recovers_from_unhashable_pre_existing(tmp_path, monkeypatch):
    """If a pre-existing file in the snapshot dir cannot be hashed (e.g.
    permission glitch), the archiver must unlink it and re-fetch from the
    network rather than wedging forever on a poisoned byte sequence."""
    archive_root = tmp_path / "archives"
    snapshot_dir = archive_root / "dbpr" / "2026-04-26"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "a.csv").write_bytes(b"poisoned-existing")

    # download_one will be called as a recovery; arrange a successful response.
    client = _make_client_returning(
        DownloadResult(
            url="https://e.com/a.csv",
            dest_path=snapshot_dir / "a.csv",
            status=DownloadStatus.OK,
            size=14,
            attempts=1,
        ),
    )
    archiver = Archiver(client, archive_root)

    from etlantis.ingest import archive as archive_mod

    real_hash = archive_mod._sha256_file
    call_count = {"n": 0}

    def flaky_hash(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated read failure on pre-existing")
        return real_hash(path)

    monkeypatch.setattr(archive_mod, "_sha256_file", flaky_hash)

    result = archiver.capture(
        source_id="dbpr",
        urls=[("https://e.com/a.csv", "a.csv")],
        snapshot_date=date(2026, 4, 26),
    )

    # Re-fetched into captured; not in skipped (since it's a fresh fetch)
    assert len(result.captured) == 1
    assert "a.csv" not in result.skipped
    assert client.download_one.call_count == 1


def test_capture_sha256_matches_file_contents(tmp_path):
    """Sanity check: the SHA256 we record really does correspond to the
    bytes on disk, so downstream change-detection can rely on it."""
    import hashlib

    expected_bytes = b"col1,col2\n1,2\n"
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()

    client = _make_client_returning(
        DownloadResult(
            url="https://e.com/a.csv",
            dest_path=tmp_path / "a.csv",
            status=DownloadStatus.OK,
            size=len(expected_bytes),
            attempts=1,
        ),
    )
    archiver = Archiver(client, tmp_path / "archives")
    result = archiver.capture(
        source_id="dbpr",
        urls=[("https://e.com/a.csv", "a.csv")],
        snapshot_date=date(2026, 4, 26),
    )
    assert result.captured[0].sha256 == expected_hash
