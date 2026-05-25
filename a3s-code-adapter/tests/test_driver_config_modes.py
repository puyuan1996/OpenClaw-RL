from __future__ import annotations

import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path

import pytest


DRIVER_PATH = Path(__file__).resolve().parents[1] / "a3s_code_agent_traffic_driver.py"


def _load_driver_module(
    monkeypatch,
    tmp_path: Path,
    *,
    config_mode: str,
    include_tags: str = "",
    extra_env: dict[str, str] | None = None,
):
    fake_a3s_code = types.ModuleType("a3s_code")
    fake_a3s_code.Agent = object
    fake_a3s_code.PermissionPolicy = object
    fake_a3s_code.SessionOptions = object
    monkeypatch.setitem(sys.modules, "a3s_code", fake_a3s_code)
    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = object
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    monkeypatch.setenv("A3S_CODE_CONFIG_ROOT", str(tmp_path / "configs"))
    monkeypatch.setenv("A3S_CODE_WORKSPACE_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv(
        "A3S_CODE_WORKSPACE_TEMPLATE_CACHE_ROOT", str(tmp_path / "template_cache")
    )
    monkeypatch.setenv("A3S_CODE_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("A3S_CODE_AGENT_CONFIG_MODE", config_mode)
    monkeypatch.setenv("A3S_CODE_SHARED_CONFIG_NAME", "shared-agent.acl")
    monkeypatch.setenv("A3S_CODE_SESSION_ID_HEADER_NAME", "X-Session-Id")
    monkeypatch.setenv("RL_BASE_URL", "http://127.0.0.1:30000")
    monkeypatch.setenv("A3S_MODEL_NAME", "test-model")
    monkeypatch.setenv("A3S_API_KEY", "test-key")
    monkeypatch.setenv("A3S_CODE_INCLUDE_SEED_TAGS", include_tags)
    for key, value in (extra_env or {}).items():
        monkeypatch.setenv(key, value)

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

    assert config_path.name == "shared-agent.acl"
    assert 'providers "openai"' in content
    assert 'models "test-model"' in content
    assert 'base_url = "http://127.0.0.1:30000"' in content
    assert 'sessionIdHeader = "X-Session-Id"' in content
    assert "limit = {" in content
    assert "context = " in content
    assert "output = " in content
    assert "context_tokens =" not in content
    assert "max_tokens =" not in content
    assert "/session/sess-123" not in content


def test_build_agent_config_per_session_mode(monkeypatch, tmp_path: Path):
    driver = _load_driver_module(monkeypatch, tmp_path, config_mode="per_session")

    config_path = driver._build_agent_config("sess-123")
    content = config_path.read_text(encoding="utf-8")

    assert config_path.name == "sess-123.acl"
    assert 'providers "openai"' in content
    assert 'models "test-model"' in content
    assert 'base_url = "http://127.0.0.1:30000/session/sess-123"' in content
    assert "limit = {" in content
    assert "context = " in content
    assert "output = " in content
    assert "context_tokens =" not in content
    assert "max_tokens =" not in content
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


def test_load_seed_tasks_filters_by_included_seed_ids(monkeypatch, tmp_path: Path):
    raw = json.loads(
        ((DRIVER_PATH.parent / "seed_data" / "code_task_seeds.json")).read_text(
            encoding="utf-8"
        )
    )
    target_id = raw[0]["id"]
    monkeypatch.setenv("A3S_CODE_INCLUDE_SEED_IDS", target_id)
    driver = _load_driver_module(monkeypatch, tmp_path, config_mode="shared")

    seeds = driver._load_seed_tasks()

    assert [seed.seed_id for seed in seeds] == [target_id]


def test_build_docker_session_command_for_single_session(monkeypatch, tmp_path: Path):
    driver = _load_driver_module(
        monkeypatch,
        tmp_path,
        config_mode="shared",
        extra_env={
            "A3S_CODE_AGENT_ENV_BACKEND": "docker",
            "A3S_CODE_AGENT_DOCKER_IMAGE": "registry.local/a3s-agent:probe",
            "A3S_CODE_AGENT_DOCKER_NETWORK": "bridge",
            "A3S_CODE_AGENT_DOCKER_PULL_POLICY": "never",
            "A3S_CODE_AGENT_DOCKER_PYTHON_BIN": "/opt/conda/bin/python",
            "A3S_CODE_AGENT_DOCKER_WORKDIR": "/repo",
            "A3S_CODE_AGENT_DOCKER_MOUNTS": "/host:/container:ro",
            "PYTHONPATH": "/extra/site",
        },
    )

    command = driver._build_docker_session_command(worker_id=2, session_index=7)

    assert command[:3] == ["docker", "run", "--rm"]
    assert "--pull=never" in command
    assert command[command.index("--network") + 1] == "bridge"
    assert "/host:/container:ro" in command
    assert command[command.index("-w") + 1] == "/repo"
    assert "A3S_CODE_AGENT_ENV_BACKEND=local" in command
    assert "A3S_CODE_TRAFFIC_CONCURRENCY=1" in command
    assert "A3S_CODE_TRAFFIC_SESSION_LIMIT=1" in command
    assert "A3S_CODE_TRAFFIC_SESSION_START_INDEX=7" in command
    assert "A3S_CODE_WORKER_LOCAL_DOCKER=0" in command
    pythonpath_env = next(item for item in command if item.startswith("PYTHONPATH="))
    assert str(driver.SCRIPT_DIR) in pythonpath_env
    assert "/extra/site" in pythonpath_env
    assert command[-4:] == [
        "registry.local/a3s-agent:probe",
        "/opt/conda/bin/python",
        "-u",
        str(driver.SCRIPT_DIR / "a3s_code_agent_traffic_driver.py"),
    ]


def test_session_limit_counts_from_session_start_index(monkeypatch, tmp_path: Path):
    driver = _load_driver_module(
        monkeypatch,
        tmp_path,
        config_mode="shared",
        extra_env={
            "A3S_CODE_TRAFFIC_SESSION_START_INDEX": "7",
            "A3S_CODE_TRAFFIC_SESSION_LIMIT": "2",
        },
    )

    assert driver._next_session_index() == 7
    assert driver._next_session_index() == 8
    assert driver._next_session_index() is None


def test_load_seed_tasks_raises_when_include_tags_match_nothing(monkeypatch, tmp_path: Path):
    driver = _load_driver_module(
        monkeypatch,
        tmp_path,
        config_mode="shared",
        include_tags="nonexistent-tag",
    )

    with pytest.raises(RuntimeError, match="matched no seeds"):
        driver._load_seed_tasks()
