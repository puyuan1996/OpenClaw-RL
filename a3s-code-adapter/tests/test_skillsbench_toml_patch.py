from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[import-not-found]

from a3s_code_benchmarks.official.official_benchmark_eval import (
    _network_env_overrides,
    _patch_network_relaxed_files,
    _patch_task_toml_section_values,
    _subprocess_env,
)


def test_patch_section_values_merges_inline_env_table(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text(
        "\n".join(
            [
                'version = "1.0"',
                "",
                "[verifier]",
                'env = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }',
                "",
                "[environment]",
                "allow_internet = true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _patch_task_toml_section_values(
        task_toml,
        "verifier.env",
        {
            "HTTP_PROXY": "http://httpproxy-headless.kubebrain.svc.pjlab.local:3128",
            "PLAYWRIGHT_DOWNLOAD_HOST": "https://playwright-akamai.azureedge.net",
        },
    )

    payload = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    assert payload["verifier"]["env"]["OPENAI_API_KEY"] == "${OPENAI_API_KEY}"
    assert payload["verifier"]["env"]["HTTP_PROXY"] == "http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"
    assert payload["verifier"]["env"]["PLAYWRIGHT_DOWNLOAD_HOST"] == "https://playwright-akamai.azureedge.net"
    assert "[verifier.env]" not in task_toml.read_text(encoding="utf-8")


def test_patch_section_values_appends_explicit_env_section(tmp_path: Path) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text(
        "\n".join(
            [
                'version = "1.0"',
                "",
                "[verifier]",
                'command = "pytest"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _patch_task_toml_section_values(task_toml, "verifier.env", {"HTTP_PROXY": "http://proxy:3128"})

    payload = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    assert payload["verifier"]["env"]["HTTP_PROXY"] == "http://proxy:3128"
    assert "[verifier.env]" in task_toml.read_text(encoding="utf-8")


def test_network_env_overrides_include_openai_base_url() -> None:
    overrides = _network_env_overrides(
        {
            "A3S_CODE_BENCHMARK_PROXY": "http://proxy:3128",
            "OPENAI_BASE_URL": "http://boyue.example/v1",
        }
    )

    assert overrides["OPENAI_BASE_URL"] == "http://boyue.example/v1"
    assert overrides["OPENAI_API_BASE"] == "http://boyue.example/v1"


def test_network_relaxed_files_guard_redundant_coursier_setup(tmp_path: Path) -> None:
    test_sh = tmp_path / "test.sh"
    test_sh.write_text(
        "curl -fL https://github.com/coursier/coursier/releases/download/v2.1.25-M23/"
        "cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup --yes\n",
        encoding="utf-8",
    )

    _patch_network_relaxed_files(tmp_path, {})

    patched = test_sh.read_text(encoding="utf-8")
    assert "Scala toolchain already available; skipping coursier setup" in patched
    assert "timeout \"${A3S_CODE_COURSIER_SETUP_TIMEOUT_SEC:-300}\" ./cs setup --yes" in patched


def test_subprocess_env_respects_model_no_proxy_flag(monkeypatch) -> None:
    monkeypatch.setenv("A3S_CODE_MODEL_NO_PROXY", "0")
    env = _subprocess_env({"A3S_CODE_MODEL_BASE_URL": "http://35.220.164.252:3888/v1"})

    assert "35.220.164.252" not in env["NO_PROXY"].split(",")


def test_subprocess_env_exports_a3s_code_no_proxy(monkeypatch) -> None:
    monkeypatch.delenv("A3S_CODE_MODEL_NO_PROXY", raising=False)
    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    env = _subprocess_env({"A3S_CODE_MODEL_BASE_URL": "http://10.102.232.67:18080/v1"})

    assert "10.102.232.67" in env["NO_PROXY"].split(",")
    assert env["A3S_CODE_NO_PROXY"] == env["NO_PROXY"]
    assert env["no_proxy"] == env["NO_PROXY"]


def test_agent_env_does_not_embed_model_api_key(monkeypatch) -> None:
    pytest.importorskip("harbor")
    from a3s_code_benchmarks.official.skillsbench_harbor_a3s_agent import A3SCodeHarbor

    monkeypatch.setenv("A3S_CODE_MODEL_API_KEY", "secret-token")

    env = A3SCodeHarbor._default_env()

    assert "A3S_CODE_MODEL_API_KEY" not in env
    assert env["A3S_CODE_MODEL_API_KEY_FILE"] == "/installed-agent/.model_api_key"
    assert "secret-token" not in "\n".join(f"{key}={value}" for key, value in env.items())
