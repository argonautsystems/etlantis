# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""etlantis.config — manifest-driven configuration.

Lands Phase 1. Lifts the working manifest_loader pattern from the rvmaps
prototype (rvmaps/etlantis/manifest_loader.py on argonas) and generalizes it.

Planned modules:

    manifest_loader   Load JSON/YAML manifests from a configured directory,
                      substitute ${VAR} and ${VAR:default} env vars, cache
                      loaded manifests, validate via pydantic schemas,
                      provide typed accessors (sources, stages, directories).

    schema            Pydantic models for the etlantis manifest format.
                      Apps validate their manifests on load via these
                      models. See docs/PHASE_3_ETLANTIS_DESIGN.md §7 for
                      the schema shape.

    env_substitute    Recursive ${VAR} / ${VAR:default} substitution over
                      JSON/YAML loaded structures. Generalized from rvmaps.
"""
