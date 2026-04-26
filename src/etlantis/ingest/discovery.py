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

"""etlantis.ingest.discovery — discover downloadable files on a public records portal.

Generic HTML link extraction. Given a URL of an index page on a public-
records portal (DBPR, BLM, OMMU, etc.), fetch the page and return the
links that look like data downloads.

Generalized from the RiskyEats cleanroom `E0_discovery.py`. The cleanroom
version was DBPR-specific (hardcoded hotset, RDAR-filename date parsing).
This version takes:

  * The index URL
  * An optional file-extension allowlist (e.g. {"csv", "xlsx", "json"})
  * An optional regex filter on the link's href or display text

The DBPR-specific patterns (hotset, RDAR date parsing) belong in the
consuming app's manifest, not in etlantis.

Usage:
    from etlantis.ingest.discovery import HTMLDiscoverer
    from etlantis.ingest.http_client import HTTPClient

    client = HTTPClient()
    discoverer = HTMLDiscoverer(client)
    links = discoverer.discover(
        url="https://example.gov/data/",
        extensions={"csv", "xlsx"},
    )
    for link in links:
        print(link.url, link.text)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from re import Pattern
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from etlantis.ingest.http_client import HTTPClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredLink:
    """A single link found on an index page."""

    url: str
    """Fully-qualified URL (relative paths in the source HTML are resolved
    against the page URL via urljoin)."""

    text: str
    """The link's display text, stripped of leading/trailing whitespace."""

    filename: str
    """The basename of the URL path. For 'https://x.com/a/b.csv' this is 'b.csv'.
    Useful for downstream extension filtering and naming."""


class HTMLDiscoverer:
    """Discover downloadable files via HTML link extraction.

    Args:
        client: HTTPClient instance honoring the Fremen Protocol. The
            discoverer reuses the client's rate-limit + User-Agent +
            timeout settings.
    """

    def __init__(self, client: HTTPClient):
        self.client = client

    def discover(
        self,
        url: str,
        *,
        extensions: set[str] | None = None,
        href_pattern: Pattern[str] | str | None = None,
        text_pattern: Pattern[str] | str | None = None,
    ) -> list[DiscoveredLink]:
        """Fetch a URL, parse <a href> tags, return matching links.

        Args:
            url: Index page URL.
            extensions: If provided, keep only links whose URL path ends
                with one of these (case-insensitive, dot-stripped). Common
                values: {"csv"}, {"csv", "xlsx"}, {"json", "geojson"}.
            href_pattern: If provided (compiled regex or string), keep only
                links whose href matches via `re.search`.
            text_pattern: If provided (compiled regex or string), keep only
                links whose display text matches via `re.search`.

        Filters are AND-combined when multiple are provided.

        Returns:
            List of DiscoveredLink, in document order. Duplicates (same URL)
            are de-duped, keeping the first occurrence.
        """
        html = self._fetch(url)
        if html is None:
            return []
        return _extract_links(
            html=html,
            base_url=url,
            extensions=_normalize_extensions(extensions),
            href_pattern=_compile_pattern(href_pattern),
            text_pattern=_compile_pattern(text_pattern),
        )

    def _fetch(self, url: str) -> str | None:
        """Fetch a URL and return decoded HTML, or None on failure.

        Delegates to `HTTPClient.get_text()` so the request honors the
        client's configured session (cookies / proxies / adapters / test
        doubles), rate-limit timer, retry/backoff, and User-Agent
        consistently with `HTTPClient.download_one()`.
        """
        return self.client.get_text(url)


def _normalize_extensions(exts: set[str] | None) -> set[str] | None:
    """Lowercase + strip leading dots from extension allowlist."""
    if exts is None:
        return None
    return {e.lower().lstrip(".") for e in exts}


def _compile_pattern(pat: Pattern[str] | str | None) -> Pattern[str] | None:
    """Compile a string regex; pass through compiled patterns; None passes through."""
    if pat is None:
        return None
    if isinstance(pat, str):
        return re.compile(pat)
    return pat


def _extract_links(
    *,
    html: str,
    base_url: str,
    extensions: set[str] | None,
    href_pattern: Pattern[str] | None,
    text_pattern: Pattern[str] | None,
) -> list[DiscoveredLink]:
    """Pure-function HTML parse + filter. Separable for testability."""
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    out: list[DiscoveredLink] = []
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"]).strip()
        if not href:
            continue
        absolute_url = urljoin(base_url, href)

        if href_pattern is not None and not href_pattern.search(href):
            continue

        text = tag.get_text(strip=True)
        if text_pattern is not None and not text_pattern.search(text):
            continue

        filename = _filename_of(absolute_url)
        if extensions is not None:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext not in extensions:
                continue

        if absolute_url in seen:
            continue
        seen.add(absolute_url)

        out.append(DiscoveredLink(url=absolute_url, text=text, filename=filename))

    logger.info("discovered %d links from %s", len(out), base_url)
    return out


def _filename_of(url: str) -> str:
    """Extract the basename from a URL path. 'https://x.com/a/b.csv' -> 'b.csv'."""
    from urllib.parse import urlparse

    path = urlparse(url).path
    if "/" in path:
        return path.rsplit("/", 1)[-1]
    return path


__all__ = ["HTMLDiscoverer", "DiscoveredLink"]
