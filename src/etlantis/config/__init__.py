# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""etlantis.config — manifest-driven configuration.

Subsystems (Phase 1):

    manifest_loader   ManifestLoader / ManifestLoadError. Load JSON or YAML
                      manifests from a configured directory, substitute
                      ${VAR} / ${VAR:default} env vars, validate via
                      pydantic, expose typed accessors. Generalized from
                      the rvmaps prototype manifest_loader.py with YAML +
                      pydantic + caching additions.

    schema            PipelineManifest, Source, Stage, FremenProtocol, and
                      supporting pydantic models. Apps validate their
                      manifests against PipelineManifest at load time.

    env_substitute    substitute() — recursively resolves ${VAR} and
                      ${VAR:default} tokens in dict/list/str inputs against
                      os.environ. Pure function; no side effects beyond
                      reading the environment.
"""

from etlantis.config.env_substitute import substitute
from etlantis.config.manifest_loader import ManifestLoader, ManifestLoadError
from etlantis.config.schema import (
    FremenProtocol,
    GlobalSettings,
    ManifestMetadata,
    PipelineManifest,
    Source,
    SourceEndpoint,
    SourceFallback,
    SourceOutput,
    SourceRateLimit,
    Stage,
)

__all__ = [
    "ManifestLoader",
    "ManifestLoadError",
    "PipelineManifest",
    "ManifestMetadata",
    "GlobalSettings",
    "FremenProtocol",
    "Source",
    "SourceEndpoint",
    "SourceFallback",
    "SourceOutput",
    "SourceRateLimit",
    "Stage",
    "substitute",
]
