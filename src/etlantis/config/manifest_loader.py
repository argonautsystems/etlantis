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

"""etlantis.config.manifest_loader — load and validate JSON/YAML pipeline manifests.

Generalized from the rvmaps prototype `manifest_loader.py`, with three
additions:

  1. **YAML support** alongside JSON. Files are auto-detected by extension
     (`.yaml` / `.yml` -> yaml.safe_load; `.json` -> json.load). YAML is
     supported via PyYAML which is already an etlantis core dep.
  2. **Pydantic validation** via `etlantis.config.schema.PipelineManifest`.
     Apps that don't want validation can use the lower-level `load_raw()`
     directly; apps that do want validation use `load()` to get a
     PipelineManifest instance.
  3. **Caching** keyed by absolute path so a manifest loaded twice from
     the same file path doesn't re-parse.

Lifecycle:

    loader = ManifestLoader(manifest_dir="manifests/")
    manifest = loader.load("pipeline.yaml")  # PipelineManifest
    sources = manifest.sources              # list[Source]
    stages = manifest.stages                # list[Stage]

Or for the lower-level escape hatch:

    raw_dict = loader.load_raw("pipeline.json")  # dict, env vars substituted
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from etlantis.config.env_substitute import substitute
from etlantis.config.schema import PipelineManifest

logger = logging.getLogger(__name__)


class ManifestLoadError(Exception):
    """Raised when a manifest file is unreadable, malformed, or fails validation."""


class ManifestLoader:
    """Load JSON/YAML manifests from a configured directory.

    Args:
        manifest_dir: Directory containing manifest files. Relative paths
            are resolved against the current working directory at
            construction time. The loader does NOT auto-create the
            directory; if it doesn't exist, all `load*` calls raise
            `ManifestLoadError`.
    """

    def __init__(self, manifest_dir: Path | str):
        self.manifest_dir = Path(manifest_dir).resolve()
        self._cache: dict[Path, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, filename: str) -> PipelineManifest:
        """Load a manifest, substitute env vars, validate via pydantic.

        Args:
            filename: Manifest filename relative to manifest_dir (e.g.
                'pipeline.yaml' or 'extraction-config.json').

        Returns:
            A validated PipelineManifest instance.

        Raises:
            ManifestLoadError: if the file is missing, unreadable,
                malformed JSON/YAML, or fails pydantic validation.
        """
        raw = self.load_raw(filename)
        try:
            return PipelineManifest.model_validate(raw)
        except Exception as exc:
            raise ManifestLoadError(
                f"manifest {filename!r} failed schema validation: {exc}"
            ) from exc

    def load_raw(self, filename: str) -> dict[str, Any]:
        """Load a manifest file as a plain dict (no schema validation).

        Env-var substitution IS applied (so callers always get the
        resolved values). Cached by absolute path.

        Args:
            filename: Manifest filename relative to manifest_dir.

        Returns:
            Dict from the parsed JSON/YAML, with env vars substituted.

        Raises:
            ManifestLoadError: if the file is missing, unreadable, or
                malformed.
        """
        path = self.manifest_dir / filename
        abs_path = path.resolve()

        # Boundary enforcement: the resolved path must remain within
        # manifest_dir. Without this check, an attacker-controlled (or
        # accidentally-traversal'd) filename like '../../etc/passwd' or an
        # absolute path could read files outside the configured manifest
        # directory, which the API explicitly documents as relative-only.
        try:
            abs_path.relative_to(self.manifest_dir)
        except ValueError as exc:
            raise ManifestLoadError(
                f"manifest filename {filename!r} resolves outside manifest_dir "
                f"({self.manifest_dir}); path traversal and absolute paths are not allowed"
            ) from exc

        if abs_path in self._cache:
            return self._cache[abs_path]

        if not abs_path.exists():
            raise ManifestLoadError(f"manifest not found: {abs_path}")

        try:
            text = abs_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ManifestLoadError(f"could not read {abs_path}: {exc}") from exc

        suffix = abs_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            data = self._parse_yaml(text, abs_path)
        elif suffix == ".json":
            data = self._parse_json(text, abs_path)
        else:
            raise ManifestLoadError(
                f"unsupported manifest extension {suffix!r} for {abs_path}; "
                f"expected .json, .yaml, or .yml"
            )

        if not isinstance(data, dict):
            raise ManifestLoadError(
                f"manifest {abs_path} must be a mapping at the top level; got {type(data).__name__}"
            )

        resolved = substitute(data)
        self._cache[abs_path] = resolved
        logger.debug("loaded manifest %s (%d bytes)", abs_path, len(text))
        return resolved

    def list_manifests(self) -> list[str]:
        """List all JSON/YAML files in the manifest directory.

        Returns:
            Sorted list of filenames (not full paths). Empty list if the
            directory doesn't exist.
        """
        if not self.manifest_dir.exists():
            return []
        candidates: list[str] = []
        for path in self.manifest_dir.iterdir():
            if path.is_file() and path.suffix.lower() in (".json", ".yaml", ".yml"):
                candidates.append(path.name)
        return sorted(candidates)

    def clear_cache(self) -> None:
        """Drop all cached manifest data. Next `load()` re-reads from disk."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Lower-level lookups (compatible with rvmaps prototype API)
    # ------------------------------------------------------------------

    def get_source(self, manifest: PipelineManifest, source_id: str) -> Any | None:
        """Look up a source by id from a loaded manifest.

        Returns the Source dataclass if found, else None and logs a
        warning. Does not raise; callers handle missing sources how
        they prefer.
        """
        for src in manifest.sources:
            if src.source_id == source_id:
                return src
        logger.warning("source not found: %s", source_id)
        return None

    def get_enabled_sources(self, manifest: PipelineManifest) -> list[Any]:
        """Return enabled sources from a manifest, sorted by priority ascending.

        Lower priority numbers come first (rvmaps convention). Disabled
        sources are filtered out entirely.
        """
        enabled = [s for s in manifest.sources if s.enabled]
        return sorted(enabled, key=lambda s: s.priority)

    def get_stage(self, manifest: PipelineManifest, name: str) -> Any | None:
        """Look up a stage by name from a loaded manifest."""
        for stage in manifest.stages:
            if stage.name == name:
                return stage
        logger.warning("stage not found: %s", name)
        return None

    def get_directory(self, manifest: PipelineManifest, key: str) -> Path | None:
        """Resolve a directories[key] entry to an absolute Path.

        Relative paths in the manifest are resolved against the manifest's
        parent directory (i.e. one level up from manifest_dir). Absolute
        paths pass through unchanged. Returns None if the key isn't
        configured.
        """
        path_str = manifest.directories.get(key)
        if path_str is None:
            logger.debug("directory %r not configured in manifest", key)
            return None
        path = Path(path_str)
        if path.is_absolute():
            return path
        # rvmaps convention: directories are relative to the project root,
        # which is one level up from etlantis/manifests/. We don't enforce
        # that layout, so resolve relative to manifest_dir.parent.
        return (self.manifest_dir.parent / path).resolve()

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str, path: Path) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestLoadError(f"invalid JSON in {path}: {exc}") from exc

    @staticmethod
    def _parse_yaml(text: str, path: Path) -> Any:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover — yaml is in core deps
            raise ManifestLoadError(
                "PyYAML is required to load YAML manifests but is not installed"
            ) from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ManifestLoadError(f"invalid YAML in {path}: {exc}") from exc


__all__ = ["ManifestLoader", "ManifestLoadError"]
