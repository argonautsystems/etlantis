# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""Tests for etlantis.config — manifest_loader, schema, env_substitute."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from etlantis.config import (
    ManifestLoader,
    ManifestLoadError,
    PipelineManifest,
    Source,
    Stage,
    substitute,
)

# ============================================================================
# env_substitute
# ============================================================================


def test_substitute_simple_env_var(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "hello")
    assert substitute("${TEST_VAR}") == "hello"


def test_substitute_with_default(monkeypatch):
    monkeypatch.delenv("UNSET_VAR", raising=False)
    assert substitute("${UNSET_VAR:fallback}") == "fallback"


def test_substitute_unset_no_default_returns_empty(monkeypatch):
    monkeypatch.delenv("UNSET_VAR", raising=False)
    assert substitute("${UNSET_VAR}") == ""


def test_substitute_recursive_dict(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    raw = {"auth": {"key": "${API_KEY}"}, "timeout": 30}
    out = substitute(raw)
    assert out["auth"]["key"] == "secret123"
    assert out["timeout"] == 30  # non-string passes through


def test_substitute_in_list(monkeypatch):
    monkeypatch.setenv("HOST", "argonas")
    raw = ["${HOST}.example.com", "static.example.com"]
    out = substitute(raw)
    assert out == ["argonas.example.com", "static.example.com"]


def test_substitute_multiple_vars_in_string(monkeypatch):
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("DOMAIN", "example.com")
    out = substitute("${USER}@${DOMAIN}")
    assert out == "alice@example.com"


def test_substitute_does_not_mutate_input():
    raw = {"static": "value"}
    out = substitute(raw)
    assert out is not raw  # new dict


# ============================================================================
# Schema validation
# ============================================================================


def test_pipeline_manifest_minimum():
    """A manifest with just metadata should validate."""
    raw = {"metadata": {"version": "1.0.0"}}
    manifest = PipelineManifest.model_validate(raw)
    assert manifest.metadata.version == "1.0.0"
    assert manifest.global_settings.fremen_protocol.rate_limit_delay_seconds == 2.5
    assert manifest.sources == []
    assert manifest.stages == []


def test_pipeline_manifest_with_sources_and_stages():
    raw = {
        "metadata": {"version": "1.0.0", "app": "test"},
        "sources": [
            {"source_id": "src_a", "type": "http_csv"},
            {"source_id": "src_b", "type": "api_fetch", "priority": 1},
        ],
        "stages": [
            {"name": "ingest", "runner": "etlantis.ingest.http_client"},
            {
                "name": "transform",
                "runner": "etlantis.transform.consolidate",
                "depends_on": ["ingest"],
            },
        ],
    }
    manifest = PipelineManifest.model_validate(raw)
    assert len(manifest.sources) == 2
    assert manifest.sources[0].source_id == "src_a"
    assert len(manifest.stages) == 2
    assert manifest.stages[1].depends_on == ["ingest"]


def test_pipeline_manifest_rejects_duplicate_stage_names():
    raw = {
        "metadata": {"version": "1.0.0"},
        "stages": [
            {"name": "ingest", "runner": "x"},
            {"name": "ingest", "runner": "y"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate stage name"):
        PipelineManifest.model_validate(raw)


def test_pipeline_manifest_rejects_duplicate_source_ids():
    raw = {
        "metadata": {"version": "1.0.0"},
        "sources": [
            {"source_id": "dup", "type": "http_csv"},
            {"source_id": "dup", "type": "api_fetch"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate source_id"):
        PipelineManifest.model_validate(raw)


def test_fremen_protocol_defaults():
    raw = {"metadata": {"version": "1.0.0"}}
    manifest = PipelineManifest.model_validate(raw)
    fp = manifest.global_settings.fremen_protocol
    assert fp.rate_limit_delay_seconds >= 0
    assert fp.retry_attempts >= 0
    assert fp.retry_backoff_seconds >= 0


def test_fremen_protocol_overridable():
    raw = {
        "metadata": {"version": "1.0.0"},
        "global_settings": {
            "fremen_protocol": {
                "rate_limit_delay_seconds": 5.0,
                "user_agent": "MyApp/1.0",
            }
        },
    }
    manifest = PipelineManifest.model_validate(raw)
    assert manifest.global_settings.fremen_protocol.rate_limit_delay_seconds == 5.0
    assert manifest.global_settings.fremen_protocol.user_agent == "MyApp/1.0"


def test_fremen_protocol_rejects_negative_rate_limit():
    raw = {
        "metadata": {"version": "1.0.0"},
        "global_settings": {"fremen_protocol": {"rate_limit_delay_seconds": -1}},
    }
    with pytest.raises(ValueError):
        PipelineManifest.model_validate(raw)


def test_extra_fields_pass_through():
    """App-specific extensions don't trigger validation errors."""
    raw = {
        "metadata": {"version": "1.0.0", "custom_app_field": "ok"},
        "app_specific": {"anything": "goes"},
    }
    manifest = PipelineManifest.model_validate(raw)
    assert manifest.metadata.version == "1.0.0"


# ============================================================================
# ManifestLoader (filesystem-backed)
# ============================================================================


def _write_json(dir: Path, name: str, data: dict) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / name
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _write_yaml(dir: Path, name: str, content: str) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / name
    path.write_text(content, encoding="utf-8")
    return path


def test_loader_load_json(tmp_path):
    _write_json(
        tmp_path / "manifests",
        "pipeline.json",
        {
            "metadata": {"version": "1.0.0"},
            "sources": [{"source_id": "a", "type": "http_csv"}],
        },
    )
    loader = ManifestLoader(tmp_path / "manifests")
    manifest = loader.load("pipeline.json")
    assert isinstance(manifest, PipelineManifest)
    assert manifest.sources[0].source_id == "a"


def test_loader_load_yaml(tmp_path):
    _write_yaml(
        tmp_path / "manifests",
        "pipeline.yaml",
        """metadata:
  version: "1.0.0"
sources:
  - source_id: a
    type: http_csv
""",
    )
    loader = ManifestLoader(tmp_path / "manifests")
    manifest = loader.load("pipeline.yaml")
    assert manifest.sources[0].source_id == "a"


def test_loader_substitutes_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "abcdef")
    _write_json(
        tmp_path / "manifests",
        "pipeline.json",
        {
            "metadata": {"version": "1.0.0"},
            "sources": [
                {
                    "source_id": "a",
                    "type": "api_fetch",
                    "endpoint": {
                        "base_url": "https://example.com",
                        "headers": {"Authorization": "Bearer ${API_KEY}"},
                    },
                }
            ],
        },
    )
    loader = ManifestLoader(tmp_path / "manifests")
    manifest = loader.load("pipeline.json")
    assert manifest.sources[0].endpoint.headers["Authorization"] == "Bearer abcdef"


def test_loader_load_raw_returns_dict(tmp_path):
    _write_json(tmp_path / "manifests", "pipeline.json", {"metadata": {"version": "1.0.0"}})
    loader = ManifestLoader(tmp_path / "manifests")
    raw = loader.load_raw("pipeline.json")
    assert isinstance(raw, dict)
    assert raw["metadata"]["version"] == "1.0.0"


def test_loader_missing_file_raises(tmp_path):
    (tmp_path / "manifests").mkdir()
    loader = ManifestLoader(tmp_path / "manifests")
    with pytest.raises(ManifestLoadError, match="not found"):
        loader.load("nonexistent.json")


def test_loader_invalid_json_raises(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "broken.json").write_text("{not valid json}", encoding="utf-8")
    loader = ManifestLoader(manifest_dir)
    with pytest.raises(ManifestLoadError, match="invalid JSON"):
        loader.load("broken.json")


def test_loader_rejects_absolute_path(tmp_path):
    """An absolute filename arg should not let the caller bypass manifest_dir."""
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    other_file = tmp_path / "outside" / "secret.json"
    other_file.parent.mkdir()
    other_file.write_text('{"metadata": {"version": "x"}}')
    loader = ManifestLoader(manifest_dir)
    with pytest.raises(ManifestLoadError, match="outside manifest_dir"):
        loader.load_raw(str(other_file))


def test_loader_rejects_path_traversal(tmp_path):
    """A relative filename with `../` segments must not escape manifest_dir."""
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    other_file = tmp_path / "secret.json"
    other_file.write_text('{"metadata": {"version": "x"}}')
    loader = ManifestLoader(manifest_dir)
    with pytest.raises(ManifestLoadError, match="outside manifest_dir"):
        loader.load_raw("../secret.json")


def test_loader_unknown_extension_raises(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "wrong.txt").write_text("unused", encoding="utf-8")
    loader = ManifestLoader(manifest_dir)
    with pytest.raises(ManifestLoadError, match="unsupported manifest extension"):
        loader.load("wrong.txt")


def test_loader_top_level_must_be_mapping(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "list.json").write_text(json.dumps(["not", "a", "dict"]))
    loader = ManifestLoader(manifest_dir)
    with pytest.raises(ManifestLoadError, match="must be a mapping"):
        loader.load("list.json")


def test_loader_caches_by_path(tmp_path):
    _write_json(tmp_path / "manifests", "pipeline.json", {"metadata": {"version": "1.0.0"}})
    loader = ManifestLoader(tmp_path / "manifests")
    raw1 = loader.load_raw("pipeline.json")
    # Mutate the file on disk; cached load should still return the original
    _write_json(tmp_path / "manifests", "pipeline.json", {"metadata": {"version": "9.9.9"}})
    raw2 = loader.load_raw("pipeline.json")
    assert raw1["metadata"]["version"] == raw2["metadata"]["version"]
    # After clear_cache, the new content surfaces
    loader.clear_cache()
    raw3 = loader.load_raw("pipeline.json")
    assert raw3["metadata"]["version"] == "9.9.9"


def test_loader_list_manifests(tmp_path):
    md = tmp_path / "manifests"
    md.mkdir()
    (md / "a.json").write_text("{}")
    (md / "b.yaml").write_text("")
    (md / "c.txt").write_text("ignored")
    loader = ManifestLoader(md)
    files = loader.list_manifests()
    assert files == ["a.json", "b.yaml"]


def test_loader_list_manifests_missing_dir(tmp_path):
    loader = ManifestLoader(tmp_path / "missing")
    assert loader.list_manifests() == []


def test_loader_get_source(tmp_path):
    _write_json(
        tmp_path / "manifests",
        "pipeline.json",
        {
            "metadata": {"version": "1.0.0"},
            "sources": [
                {"source_id": "src_a", "type": "http_csv", "priority": 5},
                {"source_id": "src_b", "type": "api_fetch", "enabled": False},
                {"source_id": "src_c", "type": "http_csv", "priority": 1},
            ],
        },
    )
    loader = ManifestLoader(tmp_path / "manifests")
    manifest = loader.load("pipeline.json")
    assert loader.get_source(manifest, "src_a").type == "http_csv"
    assert loader.get_source(manifest, "nonexistent") is None
    enabled = loader.get_enabled_sources(manifest)
    assert [s.source_id for s in enabled] == ["src_c", "src_a"]  # priority asc, src_b filtered


def test_loader_get_stage(tmp_path):
    _write_json(
        tmp_path / "manifests",
        "pipeline.json",
        {
            "metadata": {"version": "1.0.0"},
            "stages": [
                {"name": "ingest", "runner": "etlantis.ingest.http_client"},
            ],
        },
    )
    loader = ManifestLoader(tmp_path / "manifests")
    manifest = loader.load("pipeline.json")
    assert isinstance(loader.get_stage(manifest, "ingest"), Stage)
    assert loader.get_stage(manifest, "missing") is None


def test_loader_get_directory_relative_resolves_against_parent(tmp_path):
    _write_json(
        tmp_path / "project" / "manifests",
        "pipeline.json",
        {
            "metadata": {"version": "1.0.0"},
            "directories": {
                "data_root": "data/",
                "absolute_path": "/tmp/etlantis-test",
            },
        },
    )
    loader = ManifestLoader(tmp_path / "project" / "manifests")
    manifest = loader.load("pipeline.json")
    data_root = loader.get_directory(manifest, "data_root")
    assert data_root == (tmp_path / "project" / "data").resolve()
    abs_path = loader.get_directory(manifest, "absolute_path")
    assert abs_path == Path("/tmp/etlantis-test")
    assert loader.get_directory(manifest, "missing") is None


def test_loader_source_endpoint_typed(tmp_path):
    _write_json(
        tmp_path / "manifests",
        "pipeline.json",
        {
            "metadata": {"version": "1.0.0"},
            "sources": [
                {
                    "source_id": "blm",
                    "type": "api_fetch",
                    "endpoint": {
                        "base_url": "https://ridb.recreation.gov/api/v1",
                        "paths": ["/facilities", "/campsites"],
                        "method": "GET",
                        "headers": {"Accept": "application/json"},
                    },
                    "fallback": {"enabled": True, "path": "data/cache/blm_seed.json"},
                }
            ],
        },
    )
    loader = ManifestLoader(tmp_path / "manifests")
    manifest = loader.load("pipeline.json")
    src = manifest.sources[0]
    assert isinstance(src, Source)
    assert src.endpoint.base_url.startswith("https://")
    assert src.endpoint.paths == ["/facilities", "/campsites"]
    assert src.fallback.enabled is True
