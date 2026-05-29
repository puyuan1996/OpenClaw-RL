from __future__ import annotations

import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path

import pytest


DRIVER_PATH = Path(__file__).resolve().parents[1] / "a3s_code_agent_traffic_driver.py"


def _load_driver_module(monkeypatch, tmp_path: Path, *, config_mode: str, include_tags: str = ""):
    fake_a3s_code = types.ModuleType("a3s_code")
    fake_a3s_code.Agent = object
    fake_a3s_code.SessionOptions = object
    monkeypatch.setitem(sys.modules, "a3s_code", fake_a3s_code)

    monkeypatch.setenv("A3S_CODE_CONFIG_ROOT", str(tmp_path / "configs"))
    monkeypatch.setenv("A3S_CODE_WORKSPACE_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv(
        "A3S_CODE_WORKSPACE_TEMPLATE_CACHE_ROOT", str(tmp_path / "template_cache")
    )
    monkeypatch.setenv("A3S_CODE_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("A3S_CODE_AGENT_CONFIG_MODE", config_mode)
    monkeypatch.setenv("A3S_CODE_SHARED_CONFIG_NAME", "shared-agent.hcl")
    monkeypatch.setenv("A3S_CODE_SESSION_ID_HEADER_NAME", "X-Session-Id")
    monkeypatch.setenv("RL_BASE_URL", "http://127.0.0.1:30000")
    monkeypatch.setenv("A3S_MODEL_NAME", "test-model")
    monkeypatch.setenv("A3S_API_KEY", "test-key")
    monkeypatch.setenv("A3S_CODE_INCLUDE_SEED_TAGS", include_tags)

    module_name = f"a3s_code_agent_traffic_driver_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_build_agent_config_shared_mode(monkeypatch, tmp_path: Path):
    driver = _load_driver_module(monkeypatch, tmp_path, config_mode="shared")

    config_path = driver._build_agent_config("sess-123")
    content = config_path.read_text(encoding="utf-8")

    assert config_path.name == "shared-agent.hcl"
    assert 'base_url = "http://127.0.0.1:30000"' in content
    assert 'sessionIdHeader = "X-Session-Id"' in content
    assert "/session/sess-123" not in content


def test_build_agent_config_per_session_mode(monkeypatch, tmp_path: Path):
    driver = _load_driver_module(monkeypatch, tmp_path, config_mode="per_session")

    config_path = driver._build_agent_config("sess-123")
    content = config_path.read_text(encoding="utf-8")

    assert config_path.name == "sess-123.hcl"
    assert 'base_url = "http://127.0.0.1:30000/session/sess-123"' in content
    assert "sessionIdHeader" not in content


def test_load_seed_tasks_defaults_to_all_tags(monkeypatch, tmp_path: Path):
    driver = _load_driver_module(monkeypatch, tmp_path, config_mode="shared")

    seeds = driver._load_seed_tasks()
    raw = json.loads((driver.DATA_DIR / "code_task_seeds.json").read_text(encoding="utf-8"))

    assert len(seeds) == len(raw)
    assert {seed.seed_id for seed in seeds} == {item["id"] for item in raw}


def test_load_seed_tasks_filters_by_included_tags(monkeypatch, tmp_path: Path):
    driver = _load_driver_module(
        monkeypatch,
        tmp_path,
        config_mode="shared",
        include_tags="docs,automation",
    )

    seeds = driver._load_seed_tasks()

    assert seeds
    for seed in seeds:
        assert {"docs", "automation"}.intersection(seed.tags)


def test_load_seed_tasks_raises_when_include_tags_match_nothing(monkeypatch, tmp_path: Path):
    driver = _load_driver_module(
        monkeypatch,
        tmp_path,
        config_mode="shared",
        include_tags="nonexistent-tag",
    )

    with pytest.raises(RuntimeError, match="matched no seeds"):
        driver._load_seed_tasks()
