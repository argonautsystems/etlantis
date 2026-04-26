# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""etlantis.ingest.scrape — JS render + paid proxy tiers.

Lands Phase 1+ as needs surface. Plain HTTP via etlantis.ingest.http_client
is the always-available default; this subpackage is for scrape targets that
need more.

Planned modules:

    js_render         Playwright wrapper for SPA pages. Optional dep:
                      pip install etlantis[scrape].

    bright_data/      Paid Bright Data tiers: residential proxies (proxies),
                      scraping browser API (scraping_browser), Web Unlocker
                      (web_unlocker). Each is opt-in via manifest config.
"""
