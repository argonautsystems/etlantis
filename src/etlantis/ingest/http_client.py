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

"""etlantis.ingest.http_client — Fremen-Protocol-honoring HTTP client.

Plain HTTP client for downloading public-records files. Lifted from the
RiskyEats cleanroom `download_dbpr.py` and generalized: any base URL, any
filename list, any destination directory. The DBPR-specific URL constant
moves into the consuming app's manifest.

Honors the Fremen Protocol (see README §"Fremen Protocol" and
`etlantis.config.schema.FremenProtocol`):

  * Polite User-Agent identifying project + contact URL
  * Configurable rate limit between requests (default 2.5s + jitter)
  * Exponential backoff on retry (default 5s base, doubling per attempt + jitter)
  * Configurable retry count (default 3)
  * HTML-disguise detection — many gov sites return their CMS homepage at
    HTTP 200 instead of a real 404 when a file is missing. Without this
    check, downstream parsers would crash on unexpected HTML inside what's
    supposed to be CSV
  * Atomic writes via .part + rename so an interrupted download doesn't
    leave a half-written file passing as complete

Usage:
    from etlantis.ingest.http_client import HTTPClient
    from etlantis.config import FremenProtocol

    client = HTTPClient(FremenProtocol(user_agent="MyApp/1.0 (+https://my.app)"))
    result = client.download_one(
        url="https://example.gov/data/file.csv",
        dest_path=Path("data/staging/file.csv"),
    )
    if result.status == DownloadStatus.OK:
        print(f"got {result.size} bytes")
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import requests

from etlantis.config.schema import FremenProtocol

logger = logging.getLogger(__name__)

# Stream chunk size for downloading large CSVs without loading into memory.
_STREAM_CHUNK = 16384


class DownloadStatus(str, Enum):
    """Outcome of a single download attempt.

    Values are strings (str-derived) so they serialize cleanly in JSON
    audit logs without bespoke encoders.
    """

    OK = "ok"
    """The download completed and the file was written atomically."""

    HTML_DISGUISE = "html_disguise"
    """The server returned 200 but the body is HTML, not the requested file.
    Common with WordPress-fronted government sites that show their homepage
    instead of a 404 for missing files. The .part file is removed."""

    FAILED = "failed"
    """All retry attempts exhausted with network / HTTP errors. The .part
    file is removed."""


@dataclass(frozen=True)
class DownloadResult:
    """Outcome record from a single download attempt.

    Suitable for inclusion in a clio.track.Fingerprint metadata field, an
    audit log line, or a pipeline summary report.
    """

    url: str
    """The fully-qualified URL that was attempted."""

    dest_path: Path
    """Where the file landed (or would have landed)."""

    status: DownloadStatus
    """Final status; see DownloadStatus enum."""

    size: int
    """Bytes written. 0 for HTML_DISGUISE and FAILED."""

    attempts: int
    """How many attempts were made (1 to retry_attempts+1 inclusive)."""

    content_type: str | None = None
    """The Content-Type header from the final attempt's response, if any."""


def _is_transient_error(exc: requests.exceptions.RequestException) -> bool:
    """Decide whether an HTTP exception is worth retrying.

    Retry semantics per Fremen Protocol principle 3 (exponential backoff
    on transient failure): retry network-class issues and server-class
    HTTP responses, but NOT permanent client-class responses. A 404 or
    410 will not become a 200 by waiting longer; retrying just wastes
    the source's bandwidth and our rate-limit budget.

    Classification:
      * Connection / timeout / SSL / chunked-encoding errors -> transient
        (response object absent; network issue assumed recoverable)
      * HTTPError with 5xx status -> transient (server hiccup)
      * HTTPError with 429 -> transient (explicit rate-limit signal)
      * HTTPError with 408 -> transient (request timeout)
      * HTTPError with any other 4xx -> permanent (caller's URL/auth/
        request is wrong; won't recover by retrying)
      * Any other unclassified RequestException -> transient (be charitable;
        the caller should fix recurring patterns out-of-band)
    """
    if not isinstance(exc, requests.exceptions.HTTPError):
        return True
    response = exc.response
    if response is None:
        return True
    status = response.status_code
    if status >= 500:
        return True
    if status in (408, 429):
        return True
    if 400 <= status < 500:
        return False
    return True


def _safe_unlink(path: Path) -> None:
    """Best-effort unlink. Logs and swallows OSError so failure-path cleanup
    can never mask the underlying download exception or skip rate-limit
    bookkeeping in the caller.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not remove %s: %s", path, exc)


def _is_html_disguise(content_type: str | None, head_bytes: bytes) -> bool:
    """Detect HTML masquerading as a non-HTML resource.

    Two-channel check: the Content-Type header AND the first bytes of the
    body. Some servers send a generic Content-Type (octet-stream / text/plain)
    while still returning HTML. Two-channel catches both.
    """
    if content_type and "text/html" in content_type.lower():
        return True
    head_lower = head_bytes.lstrip()[:32].lower()
    if head_lower.startswith(b"<!doctype html") or head_lower.startswith(b"<html"):
        return True
    return False


class HTTPClient:
    """Fremen-Protocol-honoring HTTP client for public-records ingestion.

    The client is stateless across calls EXCEPT for the rate-limit timer:
    consecutive `download_one` calls on the same instance respect
    `fremen.rate_limit_delay_seconds` between requests. Each call resets
    its own retry counter independently.

    Rate-limit jitter: a random ±25% jitter is added to the configured
    delay to avoid traffic-shape patterns that downstream WAFs sometimes
    flag as abuse heuristics. Configurable via `jitter` constructor arg.

    Args:
        fremen: FremenProtocol instance from a loaded manifest's
            global_settings. Carries User-Agent, rate-limit delay, timeout,
            retry attempts, retry backoff.
        jitter: Multiplier range for rate-limit and backoff jitter. Default
            0.25 means actual delay is 75-125% of nominal. Set to 0.0 to
            disable jitter (useful in tests).
        session: Optional requests.Session for connection-pool reuse.
            Useful when an adapter does many downloads to the same host.
            Caller is responsible for closing the session.
    """

    def __init__(
        self,
        fremen: FremenProtocol | None = None,
        *,
        jitter: float = 0.25,
        session: requests.Session | None = None,
    ):
        self.fremen = fremen if fremen is not None else FremenProtocol()
        self.jitter = max(0.0, jitter)
        self._session = session
        self._last_request_at: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_one(self, url: str, dest_path: Path) -> DownloadResult:
        """Download a single URL to a destination path.

        Atomic write: streams to `<dest_path>.part`, renames to `dest_path`
        on success, removes the `.part` on any failure. Honors rate-limit
        delay before the request fires; honors retry-with-backoff on
        transient failures.

        Args:
            url: Fully-qualified URL.
            dest_path: Destination Path. Parent directories are created
                automatically.

        Returns:
            DownloadResult with status + size + attempt count.
        """
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        last_attempt = 0
        for attempt in range(1, self.fremen.retry_attempts + 2):
            last_attempt = attempt
            # Rate-limit window applies before EVERY request, including
            # retries. If retry_backoff_seconds < rate_limit_delay_seconds,
            # awaiting only before the first attempt would let the retry
            # burst the source faster than configured. The first iteration
            # is a no-op when _last_request_at is None.
            self._await_rate_limit()
            try:
                logger.info(
                    "downloading %s (attempt %d/%d)",
                    url,
                    attempt,
                    self.fremen.retry_attempts + 1,
                )
                content_type, bytes_written, html_disguise = self._stream_to(url, tmp_path)
                self._mark_request()

                if html_disguise:
                    # Use _safe_unlink so a cleanup OSError doesn't escape
                    # into the outer `except OSError` branch and silently
                    # downgrade an otherwise-correct HTML_DISGUISE outcome
                    # to FAILED.
                    _safe_unlink(tmp_path)
                    logger.warning(
                        "%s: HTML disguise detected (content-type=%r); not saved",
                        url,
                        content_type,
                    )
                    return DownloadResult(
                        url=url,
                        dest_path=dest_path,
                        status=DownloadStatus.HTML_DISGUISE,
                        size=0,
                        attempts=attempt,
                        content_type=content_type,
                    )

                tmp_path.replace(dest_path)
                logger.info("downloaded %s (%d bytes)", url, bytes_written)
                return DownloadResult(
                    url=url,
                    dest_path=dest_path,
                    status=DownloadStatus.OK,
                    size=bytes_written,
                    attempts=attempt,
                    content_type=content_type,
                )

            except requests.exceptions.RequestException as exc:
                logger.warning("attempt %d for %s failed: %s", attempt, url, exc)
                # Mark BEFORE cleanup. If the unlink ever raises (rare —
                # PermissionError / IsADirectoryError on a tmp file we own),
                # we still want _last_request_at populated so the next
                # caller honors the Fremen rate-limit window.
                self._mark_request()
                _safe_unlink(tmp_path)
                if not _is_transient_error(exc):
                    logger.info("not retrying permanent failure for %s", url)
                    break
                if attempt > self.fremen.retry_attempts:
                    break
                self._sleep_backoff(attempt)
            except OSError as exc:
                # Local file-system error while writing `.part` (disk full,
                # read-only mount, etc). Disk errors are not the source's
                # fault and won't typically recover under retry, so mark and
                # bail out to FAILED rather than burning attempts. The HTTP
                # response did succeed, so `_mark_request()` is correct.
                logger.warning("attempt %d for %s disk error: %s", attempt, url, exc)
                self._mark_request()
                _safe_unlink(tmp_path)
                break

        return DownloadResult(
            url=url,
            dest_path=dest_path,
            status=DownloadStatus.FAILED,
            size=0,
            attempts=last_attempt,
        )

    def download_many(
        self,
        urls_and_dests: Iterable[tuple[str, Path]],
    ) -> list[DownloadResult]:
        """Download multiple files in sequence.

        Inter-request rate limiting is enforced per the Fremen Protocol
        delay setting. The delay applies BETWEEN requests, not before the
        first one.

        Args:
            urls_and_dests: Iterable of (url, dest_path) tuples.

        Returns:
            List of DownloadResult records in the same order as input.
        """
        return [self.download_one(url, dest) for url, dest in urls_and_dests]

    def get_text(self, url: str) -> str | None:
        """Fetch a URL and return decoded text, or None on failure.

        For pages we want to parse in-memory rather than write to disk
        (e.g. HTML index pages for `etlantis.ingest.discovery`). Honors the
        same Fremen Protocol discipline as download_one(): rate limit
        before the request, retry with exponential backoff on transient
        failures, mark the request timestamp for rate-limit enforcement
        regardless of success/failure outcome.

        Uses the configured session (if any) so cookie + proxy + adapter
        configuration is honored consistently with download_one().

        Args:
            url: Fully-qualified URL.

        Returns:
            Decoded response text on success. None on any failure (after
            all retries exhausted) — caller treats as "skip" per Fremen
            Protocol's graceful-degradation principle.
        """
        sender = self._session if self._session is not None else requests
        headers = {"User-Agent": self.fremen.user_agent}

        last_exc: requests.exceptions.RequestException | None = None
        for attempt in range(1, self.fremen.retry_attempts + 2):
            # Rate-limit window applies before every attempt; see download_one().
            self._await_rate_limit()
            try:
                response = sender.get(  # type: ignore[union-attr]
                    url,
                    headers=headers,
                    timeout=self.fremen.request_timeout_seconds,
                )
                response.raise_for_status()
                self._mark_request()
                return response.text
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                logger.warning("get_text attempt %d for %s failed: %s", attempt, url, exc)
                # See download_one(): mark BEFORE break/continue so every
                # exit path leaves the rate-limit timestamp populated.
                self._mark_request()
                if not _is_transient_error(exc):
                    logger.info("not retrying permanent failure for %s", url)
                    break
                if attempt > self.fremen.retry_attempts:
                    break
                self._sleep_backoff(attempt)

        if last_exc is not None:
            logger.warning(
                "get_text gave up on %s after %d attempts", url, self.fremen.retry_attempts + 1
            )
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _stream_to(self, url: str, tmp_path: Path) -> tuple[str | None, int, bool]:
        """Stream a URL response to a tmp path. Returns (content_type, bytes, html_disguise)."""
        headers = {"User-Agent": self.fremen.user_agent}
        sender = self._session if self._session is not None else requests
        with sender.get(  # type: ignore[union-attr]
            url,
            headers=headers,
            timeout=self.fremen.request_timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type")

            # Header-channel check fires before streaming. If the server
            # already declares text/html, we don't need a body sample to
            # know it's a disguise — and a zero-byte HTML response would
            # otherwise slip through the body-channel check below.
            if _is_html_disguise(content_type, b""):
                return content_type, 0, True

            disguise_check_pending = True
            bytes_written = 0
            with tmp_path.open("wb") as fh:
                for chunk in response.iter_content(_STREAM_CHUNK):
                    # Skip empty keep-alive frames; only run body-channel
                    # disguise detection against the first non-empty chunk.
                    # Otherwise an initial empty chunk would flip the flag
                    # and let an HTML body slip through silently.
                    if not chunk:
                        continue
                    if disguise_check_pending:
                        disguise_check_pending = False
                        if _is_html_disguise(content_type, chunk):
                            return content_type, 0, True
                    fh.write(chunk)
                    bytes_written += len(chunk)
            return content_type, bytes_written, False

    def _await_rate_limit(self) -> None:
        """Block until the configured rate-limit delay has elapsed since the last request."""
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        delay = self._jittered(self.fremen.rate_limit_delay_seconds)
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def _mark_request(self) -> None:
        """Record the moment of a completed (success or failure) request for rate-limit calc."""
        self._last_request_at = time.monotonic()

    def _sleep_backoff(self, attempt: int) -> None:
        """Sleep for an exponentially-backed-off duration with jitter."""
        base = self.fremen.retry_backoff_seconds * (2 ** (attempt - 1))
        time.sleep(self._jittered(base))

    def _jittered(self, value: float) -> float:
        """Apply +/- self.jitter * value random multiplier. Clamped to >= 0."""
        if self.jitter <= 0 or value <= 0:
            return max(0.0, value)
        offset = value * self.jitter * (2.0 * random.random() - 1.0)
        return max(0.0, value + offset)


__all__ = ["HTTPClient", "DownloadResult", "DownloadStatus"]
