# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""Tests for etlantis.ingest — http_client + discovery."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from etlantis.config.schema import FremenProtocol
from etlantis.ingest.discovery import (
    DiscoveredLink,
    HTMLDiscoverer,
    _extract_links,
    _filename_of,
    _normalize_extensions,
)
from etlantis.ingest.http_client import (
    DownloadResult,
    DownloadStatus,
    HTTPClient,
    _is_html_disguise,
    _is_transient_error,
)

# ============================================================================
# _is_html_disguise
# ============================================================================


def test_html_disguise_via_content_type():
    assert _is_html_disguise("text/html; charset=utf-8", b"") is True


def test_html_disguise_via_doctype_body():
    assert _is_html_disguise(None, b"<!DOCTYPE html><html>") is True


def test_html_disguise_via_html_body():
    assert _is_html_disguise("application/octet-stream", b"<html><head>") is True


def test_html_disguise_csv_body_returns_false():
    assert _is_html_disguise("text/csv", b"col1,col2,col3\n") is False


def test_html_disguise_handles_leading_whitespace():
    assert _is_html_disguise(None, b"  \n  <!doctype html") is True


# ============================================================================
# _is_transient_error — retry classification
# ============================================================================


def _http_error(status_code: int) -> requests.exceptions.HTTPError:
    response = MagicMock()
    response.status_code = status_code
    return requests.exceptions.HTTPError(f"{status_code} error", response=response)


def test_transient_500_is_retryable():
    assert _is_transient_error(_http_error(500)) is True


def test_transient_502_is_retryable():
    assert _is_transient_error(_http_error(502)) is True


def test_transient_429_is_retryable():
    assert _is_transient_error(_http_error(429)) is True


def test_transient_408_is_retryable():
    assert _is_transient_error(_http_error(408)) is True


def test_permanent_404_is_not_retryable():
    assert _is_transient_error(_http_error(404)) is False


def test_permanent_410_is_not_retryable():
    assert _is_transient_error(_http_error(410)) is False


def test_permanent_401_is_not_retryable():
    assert _is_transient_error(_http_error(401)) is False


def test_permanent_403_is_not_retryable():
    assert _is_transient_error(_http_error(403)) is False


def test_connection_error_is_transient():
    assert _is_transient_error(requests.exceptions.ConnectionError()) is True


def test_timeout_is_transient():
    assert _is_transient_error(requests.exceptions.Timeout()) is True


def test_http_error_without_response_is_transient():
    """HTTPError with no response object — defensive: assume retryable."""
    exc = requests.exceptions.HTTPError("something")
    assert exc.response is None
    assert _is_transient_error(exc) is True


# ============================================================================
# HTTPClient — wiring (no real network calls)
# ============================================================================


def test_http_client_default_fremen():
    client = HTTPClient()
    assert client.fremen.rate_limit_delay_seconds == 2.5
    assert client.fremen.retry_attempts == 3


def test_http_client_custom_fremen():
    fp = FremenProtocol(rate_limit_delay_seconds=1.0, retry_attempts=0)
    client = HTTPClient(fp, jitter=0.0)
    assert client.fremen.retry_attempts == 0
    assert client.jitter == 0.0


def test_http_client_jitter_clamped_to_zero():
    client = HTTPClient(jitter=-1.0)
    assert client.jitter == 0.0


def test_http_client_get_text_returns_none_on_404(monkeypatch, tmp_path):
    """A 404 should not retry; should return None on first attempt."""
    fp = FremenProtocol(retry_attempts=3, rate_limit_delay_seconds=0.0)
    client = HTTPClient(fp, jitter=0.0)

    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        response = MagicMock()
        response.status_code = 404
        response.raise_for_status.side_effect = _http_error(404)
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    result = client.get_text("https://example.com/missing.html")
    assert result is None
    assert call_count["n"] == 1  # not retried


def test_http_client_get_text_retries_500(monkeypatch):
    """A 500 should retry up to retry_attempts+1 times."""
    fp = FremenProtocol(retry_attempts=2, rate_limit_delay_seconds=0.0, retry_backoff_seconds=0.0)
    client = HTTPClient(fp, jitter=0.0)

    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        response = MagicMock()
        response.status_code = 500
        response.raise_for_status.side_effect = _http_error(500)
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    result = client.get_text("https://example.com/server-error")
    assert result is None
    assert call_count["n"] == 3  # initial + 2 retries


def test_http_client_get_text_returns_text_on_success(monkeypatch):
    fp = FremenProtocol(rate_limit_delay_seconds=0.0)
    client = HTTPClient(fp, jitter=0.0)

    def fake_get(*args, **kwargs):
        response = MagicMock()
        response.status_code = 200
        response.text = "<html><body>hi</body></html>"
        response.raise_for_status.return_value = None
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    result = client.get_text("https://example.com/")
    assert result == "<html><body>hi</body></html>"


def test_http_client_get_text_marks_request_after_failure(monkeypatch):
    """After a final failure, _last_request_at must be set so the next call
    still respects the rate-limit window."""
    fp = FremenProtocol(retry_attempts=0, rate_limit_delay_seconds=0.0)
    client = HTTPClient(fp, jitter=0.0)
    assert client._last_request_at is None

    def fake_get(*args, **kwargs):
        response = MagicMock()
        response.status_code = 404
        response.raise_for_status.side_effect = _http_error(404)
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    client.get_text("https://example.com/missing")
    assert client._last_request_at is not None  # marked despite failure


def test_http_client_get_text_uses_session_when_provided(monkeypatch):
    """A session passed at construction must be used for HTTP, not the
    module-level requests.get — otherwise cookies/proxies/test-doubles
    aren't honored."""
    fp = FremenProtocol(rate_limit_delay_seconds=0.0)
    fake_session = MagicMock()
    fake_response = MagicMock()
    fake_response.text = "ok"
    fake_response.raise_for_status.return_value = None
    fake_session.get.return_value = fake_response

    client = HTTPClient(fp, jitter=0.0, session=fake_session)
    result = client.get_text("https://example.com/")
    assert result == "ok"
    fake_session.get.assert_called_once()
    # Module-level requests.get must NOT be called when a session is provided.
    call_count = {"n": 0}

    def fail_if_called(*args, **kwargs):
        call_count["n"] += 1
        raise AssertionError("module-level requests.get should not be called")

    monkeypatch.setattr(requests, "get", fail_if_called)
    # Re-run; should still use session
    fake_session.get.reset_mock()
    fake_session.get.return_value = fake_response
    client.get_text("https://example.com/again")
    assert call_count["n"] == 0


# ============================================================================
# discovery — pure-function HTML extraction
# ============================================================================


def test_normalize_extensions_strips_dots_and_lowercases():
    assert _normalize_extensions({"CSV", ".xlsx"}) == {"csv", "xlsx"}
    assert _normalize_extensions(None) is None


def test_filename_of_simple_url():
    assert _filename_of("https://example.com/path/to/file.csv") == "file.csv"


def test_filename_of_url_with_query():
    # urlparse drops the query string; basename is just the path's last segment
    assert _filename_of("https://example.com/file.csv?x=1") == "file.csv"


def test_extract_links_basic():
    html = """
        <html><body>
            <a href="data.csv">Data</a>
            <a href="report.xlsx">Report</a>
            <a href="about.html">About</a>
        </body></html>
    """
    links = _extract_links(
        html=html,
        base_url="https://example.com/",
        extensions={"csv", "xlsx"},
        href_pattern=None,
        text_pattern=None,
    )
    assert len(links) == 2
    assert {link.filename for link in links} == {"data.csv", "report.xlsx"}


def test_extract_links_resolves_relative():
    html = '<a href="data.csv">d</a>'
    links = _extract_links(
        html=html,
        base_url="https://example.com/dir/",
        extensions=None,
        href_pattern=None,
        text_pattern=None,
    )
    assert links[0].url == "https://example.com/dir/data.csv"


def test_extract_links_dedupes_same_url():
    html = '<a href="x.csv">A</a><a href="x.csv">B</a>'
    links = _extract_links(
        html=html, base_url="https://e.com/", extensions=None, href_pattern=None, text_pattern=None
    )
    assert len(links) == 1


def test_extract_links_href_filter():
    import re

    html = '<a href="rdar0825.csv">A</a><a href="other.csv">B</a>'
    pattern = re.compile(r"^rdar\d{4}\.csv$")
    links = _extract_links(
        html=html,
        base_url="https://e.com/",
        extensions=None,
        href_pattern=pattern,
        text_pattern=None,
    )
    assert len(links) == 1
    assert links[0].filename == "rdar0825.csv"


def test_extract_links_text_filter():
    import re

    html = '<a href="a.csv">2025 data</a><a href="b.csv">other</a>'
    pattern = re.compile(r"^\d{4}")
    links = _extract_links(
        html=html,
        base_url="https://e.com/",
        extensions=None,
        href_pattern=None,
        text_pattern=pattern,
    )
    assert len(links) == 1
    assert links[0].text == "2025 data"


def test_extract_links_skips_empty_href():
    html = '<a href="">empty</a><a href="real.csv">r</a>'
    links = _extract_links(
        html=html, base_url="https://e.com/", extensions=None, href_pattern=None, text_pattern=None
    )
    assert len(links) == 1


def test_extract_links_returns_dataclass():
    html = '<a href="x.csv">X</a>'
    links = _extract_links(
        html=html, base_url="https://e.com/", extensions=None, href_pattern=None, text_pattern=None
    )
    assert isinstance(links[0], DiscoveredLink)


# ============================================================================
# HTMLDiscoverer — wiring
# ============================================================================


def test_discoverer_uses_client_get_text(monkeypatch):
    """Discoverer must delegate to client.get_text — not bypass it."""
    fake_client = MagicMock()
    fake_client.get_text.return_value = '<a href="data.csv">d</a>'

    discoverer = HTMLDiscoverer(fake_client)
    links = discoverer.discover("https://example.com/", extensions={"csv"})

    fake_client.get_text.assert_called_once_with("https://example.com/")
    assert len(links) == 1
    assert links[0].filename == "data.csv"


def test_discoverer_returns_empty_on_fetch_failure():
    fake_client = MagicMock()
    fake_client.get_text.return_value = None  # fetch failed

    discoverer = HTMLDiscoverer(fake_client)
    links = discoverer.discover("https://example.com/")
    assert links == []


# ============================================================================
# DownloadResult / DownloadStatus
# ============================================================================


def test_download_status_enum_values():
    assert DownloadStatus.OK.value == "ok"
    assert DownloadStatus.HTML_DISGUISE.value == "html_disguise"
    assert DownloadStatus.FAILED.value == "failed"


def test_download_result_is_frozen():
    from pathlib import Path

    r = DownloadResult(
        url="https://x.com/", dest_path=Path("/tmp/x"), status=DownloadStatus.OK, size=1, attempts=1
    )
    # frozen=True dataclasses raise FrozenInstanceError (a subclass of AttributeError)
    # on attribute mutation; AttributeError is the closest assertable supertype.
    with pytest.raises(AttributeError):
        r.size = 9  # type: ignore
