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

"""etlantis.config.schema — pydantic models for manifest validation.

These schemas constrain the shape of an etlantis manifest. Apps using
etlantis declare their pipelines in JSON/YAML; the loader validates the
loaded dict via these models before handing it off to the runner.

Three primary documents:

    - **Pipeline manifest** (top-level): metadata, global_settings (with
      Fremen Protocol parameters), directories, sources, stages.
    - **Source manifest** (per-source): endpoint config, rate limits,
      output paths, fallback strategy. Sources can also be inlined in
      the pipeline manifest under `sources:`.
    - **Stage manifest** (per-stage): runner module, dependencies, config
      passed to the runner. Stages can also be inlined.

The schemas are intentionally permissive — `extra = "allow"` so app-
specific fields don't trigger validation errors. Strict validation is
applied only to fields etlantis itself reads.

Lifted-and-generalized from the rvmaps prototype's implicit JSON shape
(rvmaps had no pydantic; this codifies the pattern).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================================
# Fremen Protocol — respectful-extraction settings
# ============================================================================


class FremenProtocol(BaseModel):
    """Per-pipeline Fremen Protocol settings.

    Defaults are the cleanroom + rvmaps converged values: 2.5s rate-limit
    delay, 30s request timeout, 3 retries with 5s base backoff. Apps can
    override per-source via `Source.rate_limit`.
    """

    model_config = ConfigDict(extra="allow")

    user_agent: str = Field(
        default="etlantis/0.1 (+https://gitlab.com/perlowja/etlantis)",
        description="User-Agent string for HTTP requests. Apps should override "
        "with a project-identifying value plus a contact URL.",
    )
    rate_limit_delay_seconds: float = Field(
        default=2.5,
        ge=0.0,
        description="Default delay between consecutive requests to the same source.",
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="HTTP request timeout in seconds.",
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts on transient failure.",
    )
    retry_backoff_seconds: float = Field(
        default=5.0,
        ge=0.0,
        description="Base backoff for exponential retry. Actual backoff is "
        "this value times 2^attempt with jitter.",
    )


class GlobalSettings(BaseModel):
    """Top-level pipeline settings."""

    model_config = ConfigDict(extra="allow")

    fremen_protocol: FremenProtocol = Field(default_factory=FremenProtocol)
    log_level: str = Field(default="INFO")


# ============================================================================
# Source manifest
# ============================================================================


class SourceEndpoint(BaseModel):
    """HTTP endpoint config for an api_fetch / http_csv source."""

    model_config = ConfigDict(extra="allow")

    base_url: str
    paths: list[str] = Field(default_factory=list)
    method: str = "GET"
    headers: dict[str, str] = Field(default_factory=dict)


class SourceRateLimit(BaseModel):
    """Per-source rate-limit override of the global Fremen settings."""

    model_config = ConfigDict(extra="allow")

    requests_per_minute: int | None = None
    delay_between_requests_seconds: float | None = None


class SourceOutput(BaseModel):
    """Where a source's raw extracted data lands."""

    model_config = ConfigDict(extra="allow")

    format: str = Field(
        default="json", description="Format tag: 'json' | 'csv' | 'parquet' | other"
    )
    path: str = Field(description="Destination path; supports ${VAR} substitution.")
    compression: str | None = Field(default=None, description="'gzip' | 'snappy' | 'zstd' | None")


class SourceFallback(BaseModel):
    """Fallback config for when a source is unreachable.

    Per Fremen Protocol principle 4: graceful fallback to representative
    cached data rather than failing the pipeline. The operator decides
    when stale data is unacceptable.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    source: str = Field(default="representative_data")
    path: str | None = None


class Source(BaseModel):
    """A single declared data source.

    Sources are referenced by `source_id` from stage configs. The runner
    looks up sources by id and passes them to the stage that consumes them.
    """

    model_config = ConfigDict(extra="allow")

    source_id: str
    source_name: str | None = None
    type: str = Field(description="'api_fetch' | 'http_csv' | 'http_xlsx' | 'sftp' | other")
    description: str | None = None
    category: str | None = None
    enabled: bool = True
    priority: int = 999
    endpoint: SourceEndpoint | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    rate_limit: SourceRateLimit | None = None
    output: SourceOutput | None = None
    fallback: SourceFallback | None = None


# ============================================================================
# Stage manifest
# ============================================================================


class Stage(BaseModel):
    """A single pipeline stage declaration.

    Stages reference a `runner` (Python module path) and declare their
    `depends_on` graph. The pipeline runner orchestrates execution in
    topological order.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    runner: str = Field(
        description="Fully-qualified module path of the stage runner, e.g. "
        "'etlantis.ingest.http_client' or 'etlantis.transform.consolidate'."
    )
    depends_on: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


# ============================================================================
# Top-level manifest
# ============================================================================


class ManifestMetadata(BaseModel):
    """Manifest metadata header. Carries app + version info for audit logs."""

    model_config = ConfigDict(extra="allow")

    version: str
    description: str | None = None
    app: str | None = Field(
        default=None,
        description="App name consuming etlantis (e.g. 'riskyeats', 'rvmaps').",
    )


class PipelineManifest(BaseModel):
    """Top-level pipeline manifest.

    Apps validate their loaded manifest against this model before passing
    it to `etlantis.pipeline.stage_runner`. Unknown fields are preserved
    (extra='allow') to support app-specific extensions.
    """

    model_config = ConfigDict(extra="allow")

    metadata: ManifestMetadata
    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)
    directories: dict[str, str] = Field(default_factory=dict)
    sources: list[Source] = Field(default_factory=list)
    stages: list[Stage] = Field(default_factory=list)

    @field_validator("stages")
    @classmethod
    def _stage_names_unique(cls, stages: list[Stage]) -> list[Stage]:
        seen: set[str] = set()
        for stage in stages:
            if stage.name in seen:
                raise ValueError(f"duplicate stage name: {stage.name!r}")
            seen.add(stage.name)
        return stages

    @field_validator("sources")
    @classmethod
    def _source_ids_unique(cls, sources: list[Source]) -> list[Source]:
        seen: set[str] = set()
        for source in sources:
            if source.source_id in seen:
                raise ValueError(f"duplicate source_id: {source.source_id!r}")
            seen.add(source.source_id)
        return sources


__all__ = [
    "FremenProtocol",
    "GlobalSettings",
    "SourceEndpoint",
    "SourceRateLimit",
    "SourceOutput",
    "SourceFallback",
    "Source",
    "Stage",
    "ManifestMetadata",
    "PipelineManifest",
]
