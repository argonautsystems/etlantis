# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""etlantis.ingest.scrape.bright_data — paid Bright Data tiers.

Lands when the first concrete consumer needs them. Apps declare which tier
to use via manifest source config; this package routes the request to the
right Bright Data API.
"""
