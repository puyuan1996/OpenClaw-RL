#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import errno
import json
import logging
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Iterable
from urllib.parse import urlparse

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

if __package__ in {None, ""}:  # pragma: no cover - script entrypoint path bootstrap
    PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from a3s_code_benchmarks.benchmark_runtime_utils import (
    REPO_ROOT,
    detect_model_base_url,
    ensure_a3s_code_wheel,
    ensure_skillsbench_a3s_code_wheel,
    render_openai_agent_config,
    shell_join,
)
from a3s_code_benchmarks.official.worker_local_docker import start_worker_local_docker


LOGGER = logging.getLogger("official_benchmark_eval")
CODE_RL_DIR = REPO_ROOT
OFFICIAL_DIR = Path(__file__).resolve().parent
DEFAULT_SKILLSBENCH_ROOT = Path.home() / "workspace" / "skillsbench"
DEFAULT_CLAWMARK_ROOT = Path.home() / "workspace" / "ClawMark"
DEFAULT_LAUNCH_LOG_MAX_CHARS = 2_000_000
DEFAULT_UV_BIN = Path(os.getenv("A3S_CODE_UV_BIN", shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")))
DEFAULT_BENCHMARK_PROXY = "http://closeai-proxy.pjlab.org.cn:23128"
DISABLE_PROXY_VALUES = {"", "0", "false", "no", "none", "off", "direct"}
SKILLSBENCH_MODES = {"without-skills", "with-skills"}
SKILLSBENCH_RERUN_MARKERS = {
    "ENDPOINT_CIRCUIT_BREAKER_NEEDS_RERUN.md",
    "PAUSED_FOR_PROXY_DIAGNOSTICS_NEEDS_RERUN.md",
    "CLEANUP_EXIT137_NEEDS_RERUN.md",
    "AGENT_EXIT137_NEEDS_RERUN.md",
    "AGENT_TIMEOUT_NEEDS_RERUN.md",
    "VERIFIER_PYTHON_RUNTIME_NEEDS_RERUN.md",
    "VERIFIER_APT_NETWORK_NEEDS_RERUN.md",
    "VERIFIER_PYTEST_PLUGIN_NEEDS_RERUN.md",
    "VERIFIER_COURSIER_SETUP_NEEDS_RERUN.md",
    "VERIFIER_PLAYWRIGHT_DOWNLOAD_NEEDS_RERUN.md",
    "VERIFIER_INTERNAL_SERVICE_PROXY_NEEDS_RERUN.md",
    "TASK_OPENAI_AUDIO_ENV_NEEDS_RERUN.md",
    "CONTEXT_WINDOW_NEEDS_RERUN.md",
    "DOCKER_IMAGE_PULL_NEEDS_RERUN.md",
    "MODEL_QUOTA_EXHAUSTED_NEEDS_RERUN.md",
    "MANUAL_STOPPED_NEEDS_RERUN.md",
}


@dataclass
class TrialRecord:
    suite: str
    task_id: str
    task_path: str
    repeat_index: int
    score: float
    success: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    execution_time_sec: float | None = None
    error: str = ""
    metadata: dict[str, Any] | None = None
    reused: bool = False


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values]
    if not values:
        return None
    return sum(values) / len(values)


def _round_robin_take(grouped: dict[str, list[Path]], limit: int) -> list[Path]:
    if limit <= 0:
        return [item for _, items in sorted(grouped.items()) for item in items]

    buckets: dict[str, deque[Path]] = {key: deque(items) for key, items in sorted(grouped.items())}
    selected: list[Path] = []
    order = deque(sorted(buckets))
    while order and len(selected) < limit:
        key = order.popleft()
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.popleft())
        if bucket:
            order.append(key)
    return selected


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    worker_local_docker = env.get("A3S_CODE_WORKER_LOCAL_DOCKER_ACTIVE") == "1"
    if worker_local_docker:
        proxy = ""
        for env_name in ("A3S_CODE_BENCHMARK_PROXY", "A3S_CODE_HTTP_PROXY", "BENCHMARK_HTTP_PROXY"):
            if env_name in env:
                proxy = env.get(env_name, "").strip()
                break
        for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            if not proxy or proxy.lower() in DISABLE_PROXY_VALUES:
                env.pop(key, None)
            else:
                env[key] = proxy
    else:
        for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env.pop(key, None)
    if worker_local_docker:
        # Worker-local dockerd has a daemon proxy configured. BuildKit/Bake has
        # been observed to fetch registry OAuth tokens directly from PJLab
        # workers, so keep Harbor on the daemon builder path.
        env["DOCKER_BUILDKIT"] = "0"
        env["COMPOSE_DOCKER_CLI_BUILD"] = "0"
        env["COMPOSE_BAKE"] = "false"
    else:
        # Harbor's compose-based build path is more reliable on the shared
        # remote Docker daemon with the legacy builder than with Compose/Bake.
        env["DOCKER_BUILDKIT"] = "0"
        env["COMPOSE_DOCKER_CLI_BUILD"] = "0"
        env["COMPOSE_BAKE"] = "false"
    env.setdefault("GOOGLE_AUTH_PATH", str((Path.home() / ".config" / "gcloud").resolve()))
    if "A3S_CODE_UV_BIN" not in env and DEFAULT_UV_BIN.exists():
        env["A3S_CODE_UV_BIN"] = str(DEFAULT_UV_BIN)
    if DEFAULT_UV_BIN.parent.exists():
        env["PATH"] = f"{DEFAULT_UV_BIN.parent}:{env.get('PATH', '')}"
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{CODE_RL_DIR}:{py_path}" if py_path else str(CODE_RL_DIR)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    no_proxy_entries = {
        entry.strip()
        for entry in (env.get("NO_PROXY", "") + "," + env.get("no_proxy", "")).split(",")
        if entry.strip() and entry.strip() != "*"
    }
    no_proxy_entries.update(
        {
            "127.0.0.1",
            "localhost",
            "0.0.0.0",
            "api",
            "app",
            "backend",
            "chrome",
            "db",
            "frontend",
            "mongo",
            "mysql",
            "postgres",
            "redis",
            "server",
            "selenium",
            "web",
            ".i.h.pjlab.org.cn",
            "mirrors.i.h.pjlab.org.cn",
            "pypi.i.h.pjlab.org.cn",
        }
    )
    model_base_url = env.get("A3S_CODE_MODEL_BASE_URL", "").strip()
    if _env_flag("A3S_CODE_MODEL_NO_PROXY", True) and model_base_url:
        parsed = urlparse(model_base_url)
        if parsed.hostname:
            no_proxy_entries.add(parsed.hostname)
    merged_no_proxy = ",".join(sorted(no_proxy_entries))
    env["A3S_CODE_NO_PROXY"] = merged_no_proxy
    env["NO_PROXY"] = merged_no_proxy
    env["no_proxy"] = merged_no_proxy
    return env


def _launch_log_max_chars() -> int:
    raw = os.getenv("BENCHMARK_LAUNCH_LOG_MAX_CHARS", "").strip()
    if not raw:
        return DEFAULT_LAUNCH_LOG_MAX_CHARS
    try:
        return max(0, int(raw))
    except ValueError:
        LOGGER.warning("Ignoring invalid BENCHMARK_LAUNCH_LOG_MAX_CHARS=%r", raw)
        return DEFAULT_LAUNCH_LOG_MAX_CHARS


def _bounded_log_text(text: str) -> str:
    max_chars = _launch_log_max_chars()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"[truncated {omitted} chars from the beginning]\n{text[-max_chars:]}"


def _coerce_subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _safe_write_text(path: Path, text: str, *, bounded: bool = False) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_bounded_log_text(text) if bounded else text, encoding="utf-8")
    except OSError as exc:
        if exc.errno == errno.EDQUOT:
            LOGGER.warning("Skipping supplementary log write due to disk quota: %s", path)
            return
        LOGGER.warning("Skipping supplementary log write after OSError(%s): %s", exc.errno, path)


def _write_command_logs(log_dir: Path, stdout: str, stderr: str) -> None:
    _safe_write_text(log_dir / "stdout.txt", stdout, bounded=True)
    _safe_write_text(log_dir / "stderr.txt", stderr, bounded=True)


def _run_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_sec: int | None,
    log_dir: Path,
) -> tuple[int, str, str, str]:
    LOGGER.info("Running command: %s", shell_join(argv))
    started = time.time()
    try:
        result = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        return_code = result.returncode
        error = ""
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_subprocess_text(exc.stdout)
        stderr = _coerce_subprocess_text(exc.stderr)
        return_code = -1
        error = f"TimeoutExpired after {timeout_sec}s"
    duration = time.time() - started
    _write_command_logs(log_dir, stdout, stderr)
    _safe_write_text(log_dir / "command.txt", shell_join(argv))
    _safe_write_text(log_dir / "return_code.txt", str(return_code))
    _safe_write_text(log_dir / "duration_sec.txt", f"{duration:.3f}")
    if error:
        _safe_write_text(log_dir / "error.txt", error)
    return return_code, stdout, stderr, error


def _read_text_tail(path: Path, max_chars: int) -> str | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _extract_harbor_diagnostics(trial_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    ctrf_path = trial_dir / "verifier" / "ctrf.json"
    if ctrf_path.exists():
        try:
            ctrf = json.loads(ctrf_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            ctrf = None
        if isinstance(ctrf, dict):
            results = ctrf.get("results") or {}
            summary = results.get("summary") or {}
            if summary:
                metadata["verifier_summary"] = {
                    key: summary.get(key)
                    for key in ("tests", "passed", "failed", "skipped", "pending", "other")
                    if key in summary
                }
            failures = []
            for test in results.get("tests") or []:
                if not isinstance(test, dict):
                    continue
                if str(test.get("status", "")).lower() not in {"failed", "errored", "error"}:
                    continue
                failures.append(
                    {
                        "name": test.get("name"),
                        "message": test.get("message"),
                        "trace_tail": str(test.get("trace") or "")[-2000:],
                    }
                )
                if len(failures) >= int(os.getenv("A3S_CODE_VERIFIER_FAILURE_LIMIT", "8")):
                    break
            if failures:
                metadata["verifier_failures"] = failures

    tail_chars = int(os.getenv("A3S_CODE_VERIFIER_STDOUT_TAIL_CHARS", "8000"))
    stdout_tail = _read_text_tail(trial_dir / "verifier" / "test-stdout.txt", tail_chars)
    if stdout_tail:
        metadata["verifier_stdout_tail"] = stdout_tail
    stderr_tail = _read_text_tail(trial_dir / "verifier" / "test-stderr.txt", tail_chars)
    if stderr_tail:
        metadata["verifier_stderr_tail"] = stderr_tail
    return metadata


def _extract_harbor_score(trial_dir: Path) -> tuple[float, dict[str, Any]]:
    verifier_reward_json = trial_dir / "verifier" / "reward.json"
    verifier_reward_txt = trial_dir / "verifier" / "reward.txt"
    result_json = trial_dir / "result.json"
    metadata: dict[str, Any] = _extract_harbor_diagnostics(trial_dir)
    result_payload: dict[str, Any] | None = None

    if result_json.exists():
        result_payload = json.loads(result_json.read_text(encoding="utf-8"))
        metadata["result"] = result_payload

    if verifier_reward_json.exists():
        rewards = json.loads(verifier_reward_json.read_text(encoding="utf-8"))
        metadata["verifier_rewards"] = rewards
        for key in ("reward", "score", "success"):
            if key in rewards:
                return float(rewards[key]), metadata
        first_value = next(iter(rewards.values()), 0.0)
        return float(first_value), metadata

    if verifier_reward_txt.exists():
        raw_reward = verifier_reward_txt.read_text(encoding="utf-8").strip()
        if raw_reward:
            try:
                value = float(raw_reward)
            except ValueError:
                metadata["verifier_reward_parse_error"] = raw_reward[:2000]
            else:
                metadata["verifier_reward_text"] = value
                return value, metadata
        else:
            metadata["verifier_reward_parse_error"] = "empty reward.txt"

    if result_payload is not None:
        rewards = ((result_payload.get("verifier_result") or {}).get("rewards") or {})
        for key in ("reward", "score", "success"):
            if key in rewards:
                return float(rewards[key]), metadata
        first_value = next(iter(rewards.values()), 0.0)
        return float(first_value), metadata

    return 0.0, metadata


def _extract_harbor_tokens(trial_dir: Path) -> tuple[int | None, int | None, int | None]:
    result_json = trial_dir / "result.json"
    if not result_json.exists():
        return None, None, None
    result_payload = json.loads(result_json.read_text(encoding="utf-8"))
    agent_result = result_payload.get("agent_result") or {}
    input_tokens = agent_result.get("n_input_tokens")
    output_tokens = agent_result.get("n_output_tokens")
    total_tokens = None
    if input_tokens is not None or output_tokens is not None:
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
    return input_tokens, output_tokens, total_tokens


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _extract_trial_success_and_duration(trial_dir: Path) -> tuple[bool, float | None]:
    result_json = trial_dir / "result.json"
    if not result_json.exists():
        return False, None
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    success = payload.get("exception_info") is None
    started_at = _parse_iso_datetime(payload.get("started_at"))
    finished_at = _parse_iso_datetime(payload.get("finished_at"))
    duration = None
    if started_at and finished_at:
        duration = max(0.0, (finished_at - started_at).total_seconds())
    return success, duration


def _duration_from_payload_block(payload: dict[str, Any], block_name: str) -> float | None:
    block = payload.get(block_name) or {}
    if not isinstance(block, dict):
        return None
    started_at = _parse_iso_datetime(block.get("started_at"))
    finished_at = _parse_iso_datetime(block.get("finished_at"))
    if not started_at or not finished_at:
        return None
    return max(0.0, (finished_at - started_at).total_seconds())


def _extract_trial_timing(trial_dir: Path) -> dict[str, float | None]:
    result_json = trial_dir / "result.json"
    if not result_json.exists():
        return {
            "total_duration_sec": None,
            "environment_setup_duration_sec": None,
            "agent_setup_duration_sec": None,
            "agent_execution_duration_sec": None,
            "verifier_duration_sec": None,
        }
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    _, total_duration_sec = _extract_trial_success_and_duration(trial_dir)
    return {
        "total_duration_sec": total_duration_sec,
        "environment_setup_duration_sec": _duration_from_payload_block(payload, "environment_setup"),
        "agent_setup_duration_sec": _duration_from_payload_block(payload, "agent_setup"),
        "agent_execution_duration_sec": _duration_from_payload_block(payload, "agent_execution"),
        "verifier_duration_sec": _duration_from_payload_block(payload, "verifier"),
    }


def _official_agent_timeout_sec(task_dir: Path) -> int | None:
    return _skillsbench_task_agent_timeout_sec(task_dir)


def _effective_agent_timeout_sec(
    *,
    official_agent_timeout_sec: int | None,
    agent_timeout_floor_sec: int | None,
    agent_timeout_multiplier: float,
) -> int | None:
    candidates: list[float] = []
    if official_agent_timeout_sec is not None:
        candidates.append(float(official_agent_timeout_sec) * agent_timeout_multiplier)
    if agent_timeout_floor_sec:
        candidates.append(float(agent_timeout_floor_sec))
    if not candidates:
        return None
    return int(math.ceil(max(candidates)))


def _timeout_budget_metadata(
    *,
    official_agent_timeout_sec: int | None,
    effective_agent_timeout_sec: int | None,
    agent_timeout_floor_sec: int | None,
    agent_timeout_multiplier: float,
    agent_execution_duration_sec: float | None,
) -> dict[str, Any]:
    extension_sec = None
    extension_ratio = None
    runtime_over_official_sec = None
    runtime_over_official_ratio = None
    if official_agent_timeout_sec is not None:
        if effective_agent_timeout_sec is not None:
            extension_sec = max(0.0, float(effective_agent_timeout_sec) - float(official_agent_timeout_sec))
            extension_ratio = round(float(effective_agent_timeout_sec) / float(official_agent_timeout_sec), 4)
        if agent_execution_duration_sec is not None:
            runtime_over_official_sec = max(0.0, float(agent_execution_duration_sec) - float(official_agent_timeout_sec))
            runtime_over_official_ratio = round(float(agent_execution_duration_sec) / float(official_agent_timeout_sec), 4)
    return {
        "official_agent_timeout_sec": official_agent_timeout_sec,
        "effective_agent_timeout_sec": effective_agent_timeout_sec,
        "agent_timeout_floor_sec": agent_timeout_floor_sec,
        "agent_timeout_multiplier": agent_timeout_multiplier,
        "agent_timeout_extension_sec": extension_sec,
        "agent_timeout_extension_ratio": extension_ratio,
        "agent_runtime_over_official_sec": runtime_over_official_sec,
        "agent_runtime_over_official_ratio": runtime_over_official_ratio,
    }


def _load_existing_skillsbench_record(
    *,
    trial_dir: Path,
    task_id: str,
    task_dir: Path,
    repeat_index: int,
    skillsbench_mode: str,
    agent_timeout_floor_sec: int | None,
    agent_timeout_multiplier: float,
    force_allow_internet: bool = False,
) -> TrialRecord:
    score, metadata = _extract_harbor_score(trial_dir)
    input_tokens, output_tokens, total_tokens = _extract_harbor_tokens(trial_dir)
    completed_without_exception, execution_time_sec = _extract_trial_success_and_duration(trial_dir)
    timing = _extract_trial_timing(trial_dir)
    official_timeout = _official_agent_timeout_sec(task_dir)
    effective_timeout = _effective_agent_timeout_sec(
        official_agent_timeout_sec=official_timeout,
        agent_timeout_floor_sec=agent_timeout_floor_sec,
        agent_timeout_multiplier=agent_timeout_multiplier,
    )
    metadata = dict(metadata)
    metadata["reused_from_disk"] = True
    metadata["skillsbench_mode"] = skillsbench_mode
    metadata["force_allow_internet"] = force_allow_internet
    metadata["agent_completed_without_exception"] = completed_without_exception
    metadata["positive_score"] = score > 0.0
    metadata["timing"] = timing
    metadata["timeout_budget"] = _timeout_budget_metadata(
        official_agent_timeout_sec=official_timeout,
        effective_agent_timeout_sec=effective_timeout,
        agent_timeout_floor_sec=agent_timeout_floor_sec,
        agent_timeout_multiplier=agent_timeout_multiplier,
        agent_execution_duration_sec=timing.get("agent_execution_duration_sec"),
    )
    return TrialRecord(
        suite="skillsbench",
        task_id=task_id,
        task_path=str(task_dir),
        repeat_index=repeat_index,
        score=score,
        success=score > 0.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        execution_time_sec=execution_time_sec,
        metadata=metadata,
        reused=True,
    )


def _skillsbench_task_agent_timeout_sec(task_dir: Path) -> int | None:
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return None
    try:
        cfg = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        value = (cfg.get("agent") or {}).get("timeout_sec")
    except Exception as exc:
        LOGGER.warning("Failed to read task agent timeout from %s: %s", task_toml, exc)
        return None
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        LOGGER.warning("Invalid task agent timeout in %s: %r", task_toml, value)
    return None


def _skillsbench_task_docker_images(task_dir: Path) -> list[str]:
    images: list[str] = []
    task_toml = task_dir / "task.toml"
    if task_toml.exists():
        try:
            cfg = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            cfg = {}
        image = str((cfg.get("environment") or {}).get("docker_image") or "").strip()
        if image:
            images.append(image)

    compose_path = task_dir / "environment" / "docker-compose.yaml"
    if compose_path.exists() and yaml is not None:
        try:
            compose = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - diagnostic path
            LOGGER.warning("Could not parse compose file for image pre-pull: %s (%s)", compose_path, exc)
            compose = {}
        services = compose.get("services") if isinstance(compose, dict) else {}
        if isinstance(services, dict):
            for service in services.values():
                if not isinstance(service, dict):
                    continue
                image = str(service.get("image") or "").strip()
                if image and "$" not in image:
                    images.append(image)

    deduped: list[str] = []
    seen: set[str] = set()
    for image in images:
        if image not in seen:
            seen.add(image)
            deduped.append(image)
    return deduped


def _docker_failure_text(*parts: str | None) -> str:
    return "\n".join(part for part in parts if part).strip()


def _docker_pull_failure_retryable(text: str) -> bool:
    lowered = text.lower()
    non_retryable_markers = (
        "manifest unknown",
        "not found",
        "pull access denied",
        "requested access to the resource is denied",
        "unauthorized:",
        "authentication required",
    )
    if any(marker in lowered for marker in non_retryable_markers):
        return False
    retryable_markers = (
        "unexpected eof",
        "connection reset",
        "connection refused",
        "connection timed out",
        "tls handshake timeout",
        "temporary failure",
        "i/o timeout",
        "service unavailable",
        "too many requests",
        "cannot connect to the docker daemon",
        "is the docker daemon running",
        "dial unix",
        "broken pipe",
    )
    return any(marker in lowered for marker in retryable_markers)


def _docker_daemon_unavailable(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "dial unix",
            "connection refused",
        )
    )


def _sync_docker_env_from_process(env: dict[str, str]) -> None:
    for key in (
        "DOCKER_HOST",
        "DOCKER_CONFIG",
        "A3S_CODE_WORKER_LOCAL_DOCKER_ACTIVE",
        "DOCKER_BUILDKIT",
        "COMPOSE_DOCKER_CLI_BUILD",
        "COMPOSE_BAKE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
        else:
            env.pop(key, None)


def _pull_skillsbench_prebuilt_images(
    *,
    selected_tasks: list[Path],
    env: dict[str, str],
    suite_dir: Path,
    concurrency: int,
) -> dict[str, Any]:
    if not _env_flag("A3S_CODE_SKILLSBENCH_PULL_PREBUILT_IMAGES", True):
        return {"enabled": False, "images": [], "records": [], "counts": {}}

    images: list[str] = []
    seen: set[str] = set()
    for task_dir in selected_tasks:
        for image in _skillsbench_task_docker_images(task_dir):
            if image not in seen:
                seen.add(image)
                images.append(image)

    if not images:
        return {"enabled": True, "images": [], "records": [], "counts": {}}

    max_workers = max(
        1,
        int(os.getenv("A3S_CODE_SKILLSBENCH_IMAGE_PULL_CONCURRENCY", str(min(4, max(1, concurrency))))),
    )
    max_workers = min(max_workers, len(images))
    pull_dir = suite_dir / "prebuilt_image_pulls"
    attempts = max(1, int(os.getenv("A3S_CODE_SKILLSBENCH_IMAGE_PULL_ATTEMPTS", "3")))
    retry_sleep_sec = max(0.0, float(os.getenv("A3S_CODE_SKILLSBENCH_IMAGE_PULL_RETRY_SLEEP_SEC", "10")))
    recover_worker_docker = _env_flag("A3S_CODE_SKILLSBENCH_DOCKER_RECOVER_ON_PULL_FAILURE", True)
    recovery_lock = Lock()
    LOGGER.info("Pulling %d prebuilt SkillsBench image(s) with concurrency=%d", len(images), max_workers)

    def pull_one(image: str) -> dict[str, Any]:
        log_name = _sanitize_name(image.replace("/", "_").replace(":", "__"))
        last_record: dict[str, Any] | None = None
        for attempt in range(1, attempts + 1):
            attempt_log_dir = pull_dir / log_name if attempts == 1 else pull_dir / log_name / f"attempt_{attempt:02d}"
            return_code, stdout, stderr, error = _run_subprocess(
                ["docker", "pull", "--quiet", image],
                cwd=suite_dir,
                env=env,
                timeout_sec=int(os.getenv("A3S_CODE_SKILLSBENCH_IMAGE_PULL_TIMEOUT_SEC", "1800")),
                log_dir=attempt_log_dir,
            )
            status = "pulled" if return_code == 0 else "failed"
            recovery_note = ""
            if return_code != 0:
                inspect_code, inspect_stdout, inspect_stderr, inspect_error = _run_subprocess(
                    ["docker", "image", "inspect", image],
                    cwd=suite_dir,
                    env=env,
                    timeout_sec=60,
                    log_dir=attempt_log_dir / "inspect",
                )
                if inspect_code == 0:
                    status = "local_after_pull_failed"
                    stdout = (stdout + "\n" + inspect_stdout)[-4000:]
                    stderr = (
                        stderr
                        + "\n"
                        + "docker pull failed, but the image is already present in worker-local Docker."
                        + "\n"
                        + inspect_stderr
                    )[-4000:]
                    error = error or inspect_error
                else:
                    failure_text = _docker_failure_text(error, stderr, stdout, inspect_error, inspect_stderr, inspect_stdout)
                    if (
                        attempt < attempts
                        and recover_worker_docker
                        and env.get("A3S_CODE_WORKER_LOCAL_DOCKER_ACTIVE") == "1"
                        and _docker_daemon_unavailable(failure_text)
                    ):
                        with recovery_lock:
                            try:
                                start_worker_local_docker(log_dir=pull_dir / "worker_local_docker_recovery")
                                _sync_docker_env_from_process(env)
                                recovery_note = "worker-local dockerd recovered before retry"
                            except Exception as exc:  # pragma: no cover - depends on worker privilege/runtime
                                recovery_note = f"worker-local dockerd recovery failed: {exc}"
                                LOGGER.warning("%s", recovery_note)
                    retryable = _docker_pull_failure_retryable(failure_text)
                    if attempt < attempts and retryable:
                        LOGGER.warning(
                            "docker pull failed for %s on attempt %d/%d; retrying. %s",
                            image,
                            attempt,
                            attempts,
                            recovery_note,
                        )
                        if retry_sleep_sec > 0:
                            time.sleep(retry_sleep_sec)
                        last_record = {
                            "image": image,
                            "status": "retrying",
                            "returncode": return_code,
                            "attempt": attempt,
                            "attempts": attempts,
                            "stdout_tail": stdout[-2000:],
                            "stderr_tail": stderr[-2000:],
                            "error": error,
                            "recovery_note": recovery_note,
                            "log_dir": str(attempt_log_dir),
                        }
                        continue
            record = {
                "image": image,
                "status": status,
                "returncode": return_code,
                "attempt": attempt,
                "attempts": attempts,
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
                "error": error,
                "recovery_note": recovery_note,
                "log_dir": str(attempt_log_dir),
            }
            if status in {"pulled", "local_after_pull_failed"} or attempt >= attempts:
                return record
            last_record = record
        assert last_record is not None
        last_record["status"] = "failed"
        return last_record

    if max_workers == 1:
        records = [pull_one(image) for image in images]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            records = list(executor.map(pull_one, images))

    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    manifest = {
        "enabled": True,
        "image_count": len(images),
        "concurrency": max_workers,
        "attempts": attempts,
        "retry_sleep_sec": retry_sleep_sec,
        "recover_worker_docker": recover_worker_docker,
        "images": images,
        "records": records,
        "counts": counts,
    }
    _safe_write_text(pull_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    successful_statuses = {"pulled", "local_after_pull_failed"}
    failed = [record for record in records if record["status"] not in successful_statuses]
    if failed:
        examples = "; ".join(f"{item['image']} rc={item['returncode']} {item['error'] or item['stderr_tail'][:200]}" for item in failed[:3])
        raise RuntimeError(f"Failed to pull {len(failed)} prebuilt SkillsBench image(s): {examples}")
    return manifest


def _pull_skillsbench_prebuilt_images_for_trial(*, task_dir: Path, env: dict[str, str], suite_dir: Path, trial_name: str) -> None:
    if not _env_flag("A3S_CODE_SKILLSBENCH_PULL_PREBUILT_IMAGES_PER_TRIAL", True):
        return
    images = _skillsbench_task_docker_images(task_dir)
    if not images:
        return

    pull_dir = suite_dir / "prebuilt_image_pulls_per_trial" / trial_name
    timeout_sec = int(os.getenv("A3S_CODE_SKILLSBENCH_IMAGE_PULL_TIMEOUT_SEC", "1800"))
    for image in images:
        log_name = _sanitize_name(image.replace("/", "_").replace(":", "__"))
        image_dir = pull_dir / log_name
        inspect_code, inspect_stdout, inspect_stderr, inspect_error = _run_subprocess(
            ["docker", "image", "inspect", image],
            cwd=suite_dir,
            env=env,
            timeout_sec=60,
            log_dir=image_dir / "inspect_before",
        )
        if inspect_code == 0:
            continue

        return_code, stdout, stderr, error = _run_subprocess(
            ["docker", "pull", image],
            cwd=suite_dir,
            env=env,
            timeout_sec=timeout_sec,
            log_dir=image_dir / "pull",
        )
        if return_code == 0:
            continue

        after_code, after_stdout, after_stderr, after_error = _run_subprocess(
            ["docker", "image", "inspect", image],
            cwd=suite_dir,
            env=env,
            timeout_sec=60,
            log_dir=image_dir / "inspect_after",
        )
        if after_code == 0:
            continue
        failure_text = _docker_failure_text(
            error,
            stderr,
            stdout,
            inspect_error,
            inspect_stderr,
            inspect_stdout,
            after_error,
            after_stderr,
            after_stdout,
        )
        raise RuntimeError(
            f"Prebuilt SkillsBench image pull failed for {image} before trial {trial_name}; "
            f"return_code={return_code}; {failure_text[-4000:]}"
        )


def _skillsbench_task_skills_dir(task_dir: Path) -> Path:
    return task_dir / "environment" / "skills"


def _patch_task_toml_allow_internet(task_toml: Path) -> None:
    lines = task_toml.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    in_environment = False
    saw_environment = False
    patched = False
    inserted = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_environment and not patched and not inserted:
                output.append("allow_internet = true")
                inserted = True
            in_environment = stripped == "[environment]"
            saw_environment = saw_environment or in_environment
            output.append(line)
            continue

        if in_environment and stripped.startswith("allow_internet"):
            output.append("allow_internet = true")
            patched = True
            continue
        output.append(line)

    if saw_environment and in_environment and not patched and not inserted:
        output.append("allow_internet = true")
    elif not saw_environment:
        output.extend(["", "[environment]", "allow_internet = true"])

    task_toml.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


def _toml_string(value: str) -> str:
    return json.dumps(str(value))


def _toml_inline_table(values: dict[str, str]) -> str:
    return "{ " + ", ".join(f"{key} = {_toml_string(value)}" for key, value in values.items()) + " }"


def _merge_toml_inline_env_line(line: str, values: dict[str, str]) -> str | None:
    prefix, separator, raw_value = line.partition("=")
    if not separator or prefix.strip() != "env":
        return None
    try:
        parsed = tomllib.loads(f"env = {raw_value.strip()}")
    except tomllib.TOMLDecodeError:
        return None
    existing = parsed.get("env")
    if not isinstance(existing, dict):
        return None
    merged = {str(key): str(value) for key, value in existing.items()}
    merged.update(values)
    leading = line[: len(line) - len(line.lstrip())]
    return f"{leading}env = {_toml_inline_table(merged)}"


def _patch_task_toml_section_values(task_toml: Path, section: str, values: dict[str, str]) -> None:
    if not values:
        return
    parent_section, _, child_key = section.partition(".")
    lines = task_toml.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    current_section: str | None = None
    saw_section = False
    saw_inline_parent_env = False
    seen: set[str] = set()

    def append_missing() -> None:
        for key, value in values.items():
            if key not in seen:
                output.append(f"{key} = {_toml_string(value)}")
                seen.add(key)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_section == section:
                append_missing()
            current_section = stripped.strip("[]")
            if current_section == section:
                saw_section = True
                seen = set()
            output.append(line)
            continue

        if child_key == "env" and current_section == parent_section and "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key == child_key:
                merged_line = _merge_toml_inline_env_line(line, values)
                if merged_line is not None:
                    output.append(merged_line)
                    saw_inline_parent_env = True
                    continue

        if current_section == section and "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                output.append(f"{key} = {_toml_string(values[key])}")
                seen.add(key)
                continue
        output.append(line)

    if current_section == section:
        append_missing()
    elif not saw_section and not saw_inline_parent_env:
        if output and output[-1].strip():
            output.append("")
        output.append(f"[{section}]")
        for key, value in values.items():
            output.append(f"{key} = {_toml_string(value)}")

    task_toml.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


def _docker_compose_service_names(task_dir: Path) -> list[str]:
    compose_paths = [
        task_dir / "environment" / "docker-compose.yaml",
        task_dir / "environment" / "docker-compose.yml",
        task_dir / "docker-compose.yaml",
        task_dir / "docker-compose.yml",
    ]
    names: list[str] = []
    for compose_path in compose_paths:
        if not compose_path.exists():
            continue
        try:
            import yaml  # type: ignore[import-untyped]

            payload = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001 - malformed task YAML should not abort evaluation setup.
            LOGGER.warning("Failed to read docker-compose services from %s: %s", compose_path, exc)
            continue
        services = payload.get("services") if isinstance(payload, dict) else None
        if not isinstance(services, dict):
            continue
        for raw_name in services:
            name = str(raw_name).strip()
            if name and name not in names:
                names.append(name)
    return names


def _merge_no_proxy_entries(no_proxy: str, extra_entries: Iterable[str]) -> str:
    entries: list[str] = []
    for raw_entry in no_proxy.split(","):
        entry = raw_entry.strip()
        if entry and entry != "*" and entry not in entries:
            entries.append(entry)
    for raw_entry in extra_entries:
        entry = raw_entry.strip()
        if entry and entry != "*" and entry not in entries:
            entries.append(entry)
    return ",".join(entries)


def _network_env_overrides(env: dict[str, str], *, task_dir: Path | None = None) -> dict[str, str]:
    proxy = ""
    for env_name in ("A3S_CODE_BENCHMARK_PROXY", "A3S_CODE_HTTP_PROXY", "BENCHMARK_HTTP_PROXY"):
        if env_name in env:
            proxy = env.get(env_name, "").strip()
            break
    if not proxy or proxy.lower() in DISABLE_PROXY_VALUES:
        return {}

    no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or (
        "localhost,127.0.0.1,0.0.0.0,::1,*.local,.pjlab.org.cn,"
        ".i.h.pjlab.org.cn,mirrors.i.h.pjlab.org.cn,pypi.i.h.pjlab.org.cn"
    )
    compose_services = _docker_compose_service_names(task_dir) if task_dir is not None else []
    no_proxy = _merge_no_proxy_entries(
        no_proxy,
        [
            "api",
            "app",
            "backend",
            "chrome",
            "db",
            "frontend",
            "mongo",
            "mysql",
            "postgres",
            "redis",
            "server",
            "selenium",
            "web",
            *compose_services,
        ],
    )
    pip_extra = env.get(
        "A3S_CODE_PIP_EXTRA_INDEX_URL",
        "http://pypi.i.h.pjlab.org.cn/brain/dev/+simple",
    )
    pip_trusted = env.get(
        "A3S_CODE_PIP_TRUSTED_HOST",
        "mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn",
    )
    playwright_download_host = env.get(
        "A3S_CODE_PLAYWRIGHT_DOWNLOAD_HOST",
        "https://playwright-akamai.azureedge.net",
    )
    overrides = {
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "http_proxy": proxy,
        "https_proxy": proxy,
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
        "PIP_EXTRA_INDEX_URL": pip_extra,
        "PIP_TRUSTED_HOST": pip_trusted,
        "PIP_DEFAULT_TIMEOUT": env.get("A3S_CODE_PIP_DEFAULT_TIMEOUT", "120"),
        "PLAYWRIGHT_DOWNLOAD_HOST": playwright_download_host,
    }
    openai_base_url = env.get("OPENAI_BASE_URL") or env.get("A3S_CODE_OPENAI_BASE_URL")
    if openai_base_url:
        overrides["OPENAI_BASE_URL"] = openai_base_url
        overrides["OPENAI_API_BASE"] = openai_base_url
    return overrides


def _patch_network_relaxed_files(target: Path, env: dict[str, str]) -> None:
    """Keep verifier/solution package installs on the lab mirror in relaxed runs."""

    pip_index = env.get("A3S_CODE_PIP_INDEX_URL", "http://mirrors.i.h.pjlab.org.cn/pypi/simple/")
    replacements = {
        "https://pypi.org/simple": pip_index,
        "http://pypi.org/simple": pip_index,
        (
            "curl -fL https://github.com/coursier/coursier/releases/download/v2.1.25-M23/"
            "cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup --yes"
        ): (
            "curl -fL https://github.com/coursier/coursier/releases/download/v2.1.25-M23/"
            "cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs\n"
            "if command -v sbt >/dev/null 2>&1 && command -v scalac >/dev/null 2>&1 && "
            "command -v scala >/dev/null 2>&1; then\n"
            "  echo \"Scala toolchain already available; skipping coursier setup\"\n"
            "else\n"
            "  timeout \"${A3S_CODE_COURSIER_SETUP_TIMEOUT_SEC:-300}\" ./cs setup --yes\n"
            "fi"
        ),
        "pytest -p pytest_jsonreport.plugin --ctrf": "pytest --ctrf",
    }
    allowed_suffixes = {
        ".py",
        ".sh",
        ".bash",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
        ".md",
    }
    for path in target.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix and path.suffix.lower() not in allowed_suffixes:
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _materialize_shadow_task_links(target: Path) -> None:
    for relative in ("tests", "solution", "environment", "instruction.md"):
        path = target / relative
        if not path.is_symlink():
            continue
        source = path.resolve()
        path.unlink()
        if source.is_dir():
            shutil.copytree(source, path, symlinks=True)
        else:
            shutil.copy2(source, path)


def _network_relaxed_task_dir(*, task_dir: Path, suite_dir: Path, trial_name: str, env: dict[str, str]) -> Path:
    target = suite_dir / "network_relaxed_tasks" / trial_name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(task_dir, target, symlinks=True)
    _materialize_shadow_task_links(target)
    task_toml = target / "task.toml"
    _patch_task_toml_allow_internet(task_toml)
    network_env = _network_env_overrides(env, task_dir=target)
    _patch_task_toml_section_values(task_toml, "verifier.env", network_env)
    _patch_task_toml_section_values(task_toml, "solution.env", network_env)
    _patch_network_relaxed_files(target, env)
    return target


def _write_skillsbench_rerun_marker(trial_dir: Path, marker: str, reason: str) -> None:
    if marker not in SKILLSBENCH_RERUN_MARKERS:
        raise ValueError(f"Unknown SkillsBench rerun marker: {marker}")
    path = trial_dir / marker
    if not path.exists():
        path.write_text(reason.rstrip() + "\n", encoding="utf-8")


def _infer_skillsbench_rerun_marker(trial_dir: Path) -> tuple[str, str] | None:
    result_json = trial_dir / "result.json"
    payload: dict[str, Any] = {}
    if result_json.exists():
        try:
            payload = json.loads(result_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    exception_info = payload.get("exception_info") or {}
    exception_type = str(exception_info.get("exception_type") or "")
    exception_message = str(exception_info.get("exception_message") or "")
    exception_txt = _read_text_tail(trial_dir / "exception.txt", 8000) or ""
    verifier_stdout = _read_text_tail(trial_dir / "verifier" / "test-stdout.txt", 8000) or ""
    verifier_stderr = _read_text_tail(trial_dir / "verifier" / "test-stderr.txt", 8000) or ""
    combined = "\n".join([exception_type, exception_message, exception_txt, verifier_stdout, verifier_stderr])
    if "RewardFileNotFoundError" in combined and "externally-managed-environment" in combined:
        return (
            "VERIFIER_PYTHON_RUNTIME_NEEDS_RERUN.md",
            "Verifier pip failed with PEP 668 after the A3S installer shadowed the task image Python. "
            "This is a harness/runtime issue; archive and rerun with the non-shadowing installer.",
        )
    if (
        "Plugin already registered under a different name" in combined
        and "pytest_jsonreport" in combined
    ):
        return (
            "VERIFIER_PYTEST_PLUGIN_NEEDS_RERUN.md",
            "The verifier failed before writing report.json because pytest-json-report was loaded twice. "
            "Archive and rerun with the shadow-task pytest command patched to avoid duplicate plugin registration.",
        )
    if "RewardFileNotFoundError" in combined and (
        "Unable to connect to" in combined
        or "Could not connect to" in combined
        or "Failed to fetch" in combined
        or "does not have a Release file" in combined
        or "404  Not Found" in combined
        or "404 Not Found" in combined
    ):
        return (
            "VERIFIER_APT_NETWORK_NEEDS_RERUN.md",
            "The verifier failed before writing a reward because package-manager network access failed. "
            "Archive and rerun with verifier apt/http proxy configuration enabled.",
        )
    if "ContextWindowExceededError" in combined or "maximum context length" in combined:
        return (
            "CONTEXT_WINDOW_NEEDS_RERUN.md",
            "The model rejected the request because prompt tokens plus max output tokens exceeded its context window. "
            "Archive and rerun with a smaller A3S_CODE_OUTPUT_TOKENS / model max_tokens setting.",
        )
    if "Environment variable 'OPENAI_API_KEY' not found" in combined:
        return (
            "TASK_OPENAI_AUDIO_ENV_NEEDS_RERUN.md",
            "A SkillsBench task requested OPENAI_API_KEY for its verifier/solution environment, but the runner "
            "host environment did not provide it. Archive and rerun with an OpenAI-compatible audio endpoint "
            "available through OPENAI_API_KEY and OPENAI_BASE_URL.",
        )
    if "LLM circuit breaker triggered" in combined or "Timeout on reading data from socket" in combined:
        return (
            "ENDPOINT_CIRCUIT_BREAKER_NEEDS_RERUN.md",
            "The model endpoint timed out or returned repeated transient failures. "
            "Archive and rerun this trial instead of counting it as a task failure.",
        )
    if "exit 137" in combined or " Killed " in combined:
        return (
            "AGENT_EXIT137_NEEDS_RERUN.md",
            "The agent process was killed with exit 137. Treat this as a resource/runtime interruption and rerun.",
        )
    if "Command timed out after" in combined:
        return (
            "AGENT_TIMEOUT_NEEDS_RERUN.md",
            "The agent reached the enlarged timeout before verifier execution. "
            "Archive and rerun with a larger timeout multiplier, recording the runtime over official limit.",
        )
    if "Prebuilt SkillsBench image pull failed" in combined or (
        "Docker compose command failed" in combined and (" Pulling " in combined or "Pulling fs layer" in combined)
    ):
        return (
            "DOCKER_IMAGE_PULL_NEEDS_RERUN.md",
            "The SkillsBench task image failed or stalled during Docker image pull before agent execution. "
            "Archive and rerun after the image is present or with per-trial image pre-pull enabled.",
        )
    return None


def _skillsbench_rerun_markers(trial_dir: Path) -> list[str]:
    markers = sorted(marker for marker in SKILLSBENCH_RERUN_MARKERS if (trial_dir / marker).exists())
    inferred = _infer_skillsbench_rerun_marker(trial_dir)
    if inferred is not None:
        marker, reason = inferred
        if marker not in markers:
            _write_skillsbench_rerun_marker(trial_dir, marker, reason)
            markers.append(marker)
            markers.sort()
    return markers


def _archive_skillsbench_trial_dir(trial_dir: Path, *, reason: str) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = trial_dir.with_name(f"{trial_dir.name}__archived_{timestamp}_{reason}")
    target = base
    suffix = 1
    while target.exists():
        suffix += 1
        target = trial_dir.with_name(f"{base.name}_{suffix}")
    shutil.move(str(trial_dir), str(target))
    return target


def _run_skillsbench_trial(
    *,
    skillsbench_root: Path,
    env: dict[str, str],
    suite_dir: Path,
    trials_dir: Path,
    task_dir: Path,
    step: int,
    repeat_index: int,
    model_name: str,
    timeout_sec: int | None,
    agent_timeout_sec: int | None,
    agent_timeout_multiplier: float,
    keep_images: bool,
    resume_existing: bool,
    skillsbench_mode: str,
    force_allow_internet: bool,
) -> TrialRecord:
    task_id = task_dir.name
    base_trial_name = f"{task_id}__step{step:07d}__r{repeat_index}"
    trial_prefix = _sanitize_name(env.get("A3S_CODE_SKILLSBENCH_TRIAL_PREFIX", "").strip())
    trial_name = _sanitize_name(f"{trial_prefix}__{base_trial_name}" if trial_prefix else base_trial_name)
    trial_dir = trials_dir / trial_name
    result_json = trial_dir / "result.json"
    rerun_markers = _skillsbench_rerun_markers(trial_dir)

    if resume_existing and result_json.exists() and not rerun_markers:
        try:
            record = _load_existing_skillsbench_record(
                trial_dir=trial_dir,
                task_id=task_id,
                task_dir=task_dir,
                repeat_index=repeat_index,
                skillsbench_mode=skillsbench_mode,
                agent_timeout_floor_sec=agent_timeout_sec,
                agent_timeout_multiplier=agent_timeout_multiplier,
                force_allow_internet=force_allow_internet,
            )
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring corrupt SkillsBench trial result %s: %s", trial_name, exc)
        else:
            LOGGER.info("Reusing existing SkillsBench trial result: %s", trial_name)
            return record
    elif resume_existing and result_json.exists() and rerun_markers:
        LOGGER.info(
            "Not reusing SkillsBench trial %s because rerun marker(s) exist: %s",
            trial_name,
            ", ".join(rerun_markers),
        )

    if trial_dir.exists():
        if rerun_markers:
            archived = _archive_skillsbench_trial_dir(trial_dir, reason="rerun_marker")
            LOGGER.info("Archived SkillsBench trial %s before rerun: %s", trial_name, archived)
        elif resume_existing and not result_json.exists():
            archived = _archive_skillsbench_trial_dir(trial_dir, reason="incomplete")
            LOGGER.info("Archived incomplete SkillsBench trial %s before rerun: %s", trial_name, archived)
        else:
            shutil.rmtree(trial_dir)

    harbor_task_dir = (
        _network_relaxed_task_dir(task_dir=task_dir, suite_dir=suite_dir, trial_name=trial_name, env=env)
        if force_allow_internet
        else task_dir
    )
    if not _env_flag("A3S_CODE_SKILLSBENCH_PULL_PREBUILT_IMAGES", True):
        _pull_skillsbench_prebuilt_images_for_trial(
            task_dir=harbor_task_dir,
            env=env,
            suite_dir=suite_dir,
            trial_name=trial_name,
        )
    uv_bin = env.get("A3S_CODE_UV_BIN") or shutil.which("uv") or "uv"
    cmd = [
        uv_bin,
        "run",
        "--project",
        str(skillsbench_root),
        "harbor",
        "trials",
        "start",
        "--path",
        str(harbor_task_dir),
        "--trial-name",
        trial_name,
        "--trials-dir",
        str(trials_dir),
        "--agent-import-path",
        "a3s_code_benchmarks.official.skillsbench_harbor_a3s_agent:A3SCodeHarbor",
        "--model",
        model_name,
    ]
    task_agent_timeout_sec = _official_agent_timeout_sec(task_dir)
    effective_agent_timeout_sec = _effective_agent_timeout_sec(
        official_agent_timeout_sec=task_agent_timeout_sec,
        agent_timeout_floor_sec=agent_timeout_sec,
        agent_timeout_multiplier=agent_timeout_multiplier,
    )
    if effective_agent_timeout_sec:
        cmd.extend(["--agent-timeout", str(effective_agent_timeout_sec)])
    harbor_timeout_multiplier = float(
        env.get("A3S_CODE_SKILLSBENCH_HARBOR_TIMEOUT_MULTIPLIER", str(agent_timeout_multiplier))
    )
    if harbor_timeout_multiplier > 0 and harbor_timeout_multiplier != 1.0:
        cmd.extend(["--timeout-multiplier", str(harbor_timeout_multiplier)])
    if keep_images:
        cmd.append("--no-delete")
    skills_dir_for_metadata: str | None = None
    if skillsbench_mode == "with-skills":
        skills_dir = _skillsbench_task_skills_dir(harbor_task_dir)
        if not skills_dir.is_dir():
            raise FileNotFoundError(f"SkillsBench task has no skills directory for with-skills mode: {skills_dir}")
        skills_dir_for_metadata = str(skills_dir)
        cmd.extend(["--agent-kwarg", f"skills_dir={skills_dir}"])
    elif skillsbench_mode == "without-skills":
        cmd.extend(["--agent-kwarg", "skills_dir=none"])
    else:
        raise ValueError(f"Unsupported SkillsBench mode: {skillsbench_mode}")

    log_dir = suite_dir / "launch_logs" / trial_name
    return_code, _, stderr, error = _run_subprocess(
        cmd,
        cwd=skillsbench_root,
        env=env,
        timeout_sec=timeout_sec,
        log_dir=log_dir,
    )
    score, metadata = _extract_harbor_score(trial_dir)
    input_tokens, output_tokens, total_tokens = _extract_harbor_tokens(trial_dir)
    completed_without_exception, execution_time_sec = _extract_trial_success_and_duration(trial_dir)
    timing = _extract_trial_timing(trial_dir)
    metadata = dict(metadata)
    metadata["skillsbench_mode"] = skillsbench_mode
    metadata["force_allow_internet"] = force_allow_internet
    if harbor_task_dir != task_dir:
        metadata["harbor_task_path"] = str(harbor_task_dir)
    if skills_dir_for_metadata:
        metadata["skills_dir"] = skills_dir_for_metadata
    metadata["agent_completed_without_exception"] = completed_without_exception
    metadata["positive_score"] = score > 0.0
    metadata["timing"] = timing
    metadata["timeout_budget"] = _timeout_budget_metadata(
        official_agent_timeout_sec=task_agent_timeout_sec,
        effective_agent_timeout_sec=effective_agent_timeout_sec,
        agent_timeout_floor_sec=agent_timeout_sec,
        agent_timeout_multiplier=agent_timeout_multiplier,
        agent_execution_duration_sec=timing.get("agent_execution_duration_sec"),
    )
    if return_code != 0 and not error:
        error = stderr.strip() or f"subprocess exited with code {return_code}"
    return TrialRecord(
        suite="skillsbench",
        task_id=task_id,
        task_path=str(task_dir),
        repeat_index=repeat_index,
        score=score if return_code == 0 or trial_dir.exists() else 0.0,
        success=score > 0.0 if trial_dir.exists() else False,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        execution_time_sec=execution_time_sec,
        error=error,
        metadata=metadata,
        reused=False,
    )


def _skillsbench_tasks(skillsbench_root: Path, max_tasks: int, tasks_dir: Path | None = None) -> tuple[list[Path], int]:
    tasks_root = (tasks_dir or (skillsbench_root / "tasks")).expanduser().resolve()
    grouped: dict[str, list[Path]] = defaultdict(list)
    all_tasks: list[Path] = []
    for task_dir in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        task_toml = task_dir / "task.toml"
        instruction = task_dir / "instruction.md"
        if not task_toml.exists() or not instruction.exists():
            continue
        cfg = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        category = str((cfg.get("metadata") or {}).get("category") or "unknown")
        resolved_task_dir = task_dir.resolve()
        grouped[category].append(resolved_task_dir)
        all_tasks.append(resolved_task_dir)
    return _round_robin_take(grouped, max_tasks), len(all_tasks)


def run_skillsbench_suite(
    *,
    skillsbench_root: Path,
    skillsbench_tasks_dir: Path | None,
    output_dir: Path,
    step: int,
    wheel_path: Path,
    model_provider: str,
    model_name: str,
    model_base_url: str,
    model_api_key: str,
    session_id_header: str,
    max_tasks: int,
    repeats: int,
    timeout_sec: int | None,
    agent_timeout_sec: int | None,
    agent_timeout_multiplier: float,
    concurrency: int,
    keep_images: bool,
    resume_existing: bool,
    skillsbench_mode: str,
    force_allow_internet: bool,
) -> dict[str, Any]:
    if skillsbench_mode not in SKILLSBENCH_MODES:
        raise ValueError(f"Unsupported SkillsBench mode: {skillsbench_mode}")
    selected_tasks, total_tasks = _skillsbench_tasks(skillsbench_root, max_tasks, skillsbench_tasks_dir)
    reference_total_tasks = total_tasks
    if skillsbench_tasks_dir is not None:
        _, reference_total_tasks = _skillsbench_tasks(skillsbench_root, 0, None)
    suite_dir = output_dir / "skillsbench"
    suite_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = suite_dir / "trials"
    records: list[TrialRecord] = []

    env = _subprocess_env(
        {
            "A3S_CODE_WHEEL_PATH": str(wheel_path),
            "A3S_CODE_MODEL_PROVIDER": model_provider,
            "A3S_CODE_MODEL_NAME": model_name,
            "A3S_CODE_MODEL_BASE_URL": model_base_url,
            "A3S_CODE_MODEL_API_KEY": model_api_key,
            "A3S_CODE_SESSION_ID_HEADER": session_id_header,
        }
    )
    pull_manifest = _pull_skillsbench_prebuilt_images(
        selected_tasks=selected_tasks,
        env=env,
        suite_dir=suite_dir,
        concurrency=concurrency,
    )

    trial_specs = [
        {
            "skillsbench_root": skillsbench_root,
            "env": env,
            "suite_dir": suite_dir,
            "trials_dir": trials_dir,
            "task_dir": task_dir,
            "step": step,
            "repeat_index": repeat_index,
            "model_name": model_name,
            "timeout_sec": timeout_sec,
            "agent_timeout_sec": agent_timeout_sec,
            "agent_timeout_multiplier": agent_timeout_multiplier,
            "keep_images": keep_images,
            "resume_existing": resume_existing,
            "skillsbench_mode": skillsbench_mode,
            "force_allow_internet": force_allow_internet,
        }
        for task_dir in selected_tasks
        for repeat_index in range(1, repeats + 1)
    ]

    max_workers = max(1, int(concurrency))
    if max_workers == 1:
        records = [_run_skillsbench_trial(**spec) for spec in trial_specs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_skillsbench_trial, **spec) for spec in trial_specs]
            for future in as_completed(futures):
                records.append(future.result())

    records.sort(key=lambda record: (record.task_id, record.repeat_index))

    return _summarize_suite(
        suite="skillsbench",
        records=records,
        selected_tasks=selected_tasks,
        total_tasks=reference_total_tasks,
        repeats=repeats,
        leaderboard_comparable=len(selected_tasks) == reference_total_tasks and repeats >= 1 and not force_allow_internet,
        extra={
            "skillsbench_mode": skillsbench_mode,
            "skillsbench_task_source_total_tasks": total_tasks,
            "agent_timeout_multiplier": agent_timeout_multiplier,
            "agent_timeout_floor_sec": agent_timeout_sec,
            "force_allow_internet": force_allow_internet,
            "prebuilt_image_pull": pull_manifest,
        },
    )


def _load_clawmark_modules(clawmark_root: Path):
    clawmark_src = clawmark_root / "src"
    if str(clawmark_src) not in sys.path:
        sys.path.insert(0, str(clawmark_src))

    from clawmark.main import StageResult  # type: ignore
    from clawmark.orchestrator import Orchestrator  # type: ignore
    from clawmark.sandbox.docker import DockerSandbox  # type: ignore
    from clawmark.sandbox.dry_run import DryRunSandbox  # type: ignore
    from clawmark.state.composite import CompositeStateManager  # type: ignore
    from clawmark.task_loader import load_task_py  # type: ignore

    return {
        "StageResult": StageResult,
        "Orchestrator": Orchestrator,
        "DockerSandbox": DockerSandbox,
        "DryRunSandbox": DryRunSandbox,
        "CompositeStateManager": CompositeStateManager,
        "load_task_py": load_task_py,
    }


def _clawmark_tasks(clawmark_root: Path, max_tasks: int) -> tuple[list[Path], int]:
    tasks_root = (clawmark_root / "tasks").expanduser().resolve()
    grouped: dict[str, list[Path]] = defaultdict(list)
    all_tasks: list[Path] = []
    for task_py in sorted(tasks_root.glob("*/*/task.py")):
        task_dir = task_py.parent
        category = task_dir.parent.name
        resolved_task_dir = task_dir.resolve()
        grouped[category].append(resolved_task_dir)
        all_tasks.append(resolved_task_dir)
    return _round_robin_take(grouped, max_tasks), len(all_tasks)


class _ClawMarkA3SRuntime:
    def __init__(
        self,
        *,
        clawmark_root: Path,
        wheel_path: Path,
        model_name: str,
        model_base_url: str,
        model_api_key: str,
        session_id_header: str,
    ):
        modules = _load_clawmark_modules(clawmark_root)
        self._Orchestrator = modules["Orchestrator"]
        self._DockerSandbox = modules["DockerSandbox"]
        self._DryRunSandbox = modules["DryRunSandbox"]
        self._CompositeStateManager = modules["CompositeStateManager"]
        self._load_task_py = modules["load_task_py"]
        self._StageResult = modules["StageResult"]
        self._clawmark_root = clawmark_root
        self._wheel_path = wheel_path
        self._model_name = model_name
        self._model_base_url = model_base_url
        self._model_api_key = model_api_key
        self._session_id_header = session_id_header
        self._runner_path = OFFICIAL_DIR / "clawmark_a3s_code_runner.py"

    def _make_orchestrator_class(self):
        wheel_path = self._wheel_path
        runner_path = self._runner_path
        model_name = self._model_name
        model_base_url = self._model_base_url
        model_api_key = self._model_api_key
        session_id_header = self._session_id_header

        class A3SCodeOrchestrator(self._Orchestrator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._trace_remote_path = f"/root/.a3s/traces/{self.session_id}.jsonl"

            @property
            def trace_remote_path(self) -> str:
                return self._trace_remote_path

            async def _setup_openclaw_config(
                self,
                *,
                model: str,
                api_key: str,
                api_base: str,
                api_format: str = "anthropic",
            ) -> None:
                await self.sandbox.exec("mkdir -p /root/.a3s /root/.a3s/sessions /root/.a3s/traces")

                config_text = render_openai_agent_config(
                    base_url=model_base_url,
                    model_name=model_name,
                    api_key=model_api_key,
                    context_tokens=int(os.getenv("A3S_CODE_CONTEXT_TOKENS", "131072")),
                    output_tokens=int(os.getenv("A3S_CODE_OUTPUT_TOKENS", "8192")),
                    session_id_header=session_id_header,
                )
                with tempfile.TemporaryDirectory(prefix="clawmark-a3s-config-") as tmp:
                    local_config = Path(tmp) / "config.acl"
                    local_install = Path(tmp) / "install_a3s_code.sh"
                    local_config.write_text(config_text, encoding="utf-8")
                    local_install.write_text(
                        """#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! python3 - <<'PY' >/dev/null 2>&1
import venv
PY
then
  apt-get update
  apt-get install -y --no-install-recommends python3 python3-pip python3-venv
  rm -rf /var/lib/apt/lists/*
fi
python3 -m venv /opt/a3s-code-venv
. /opt/a3s-code-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install /root/.a3s/a3s_code.whl
""",
                        encoding="utf-8",
                    )
                    await self.sandbox.upload_file(local_config, "/root/.a3s/config.acl")
                    await self.sandbox.upload_file(local_install, "/root/.a3s/install_a3s_code.sh")

                await self.sandbox.upload_file(wheel_path, "/root/.a3s/a3s_code.whl")
                await self.sandbox.upload_file(runner_path, "/root/.a3s/clawmark_a3s_code_runner.py")
                await self.sandbox.exec("bash /root/.a3s/install_a3s_code.sh", timeout_sec=1800)
                LOGGER.info("Configured a3s-code runtime inside ClawMark sandbox")

            async def _send_to_agent(self, *, message: str, timeout_sec: int):
                env = {
                    "A3S_CODE_CONFIG": "/root/.a3s/config.acl",
                    "A3S_CODE_WORKSPACE": "/workspace",
                    "A3S_CODE_INSTRUCTION": message,
                    "A3S_CODE_SESSION_ID": self.session_id,
                    "A3S_CODE_SESSION_STORE_DIR": "/root/.a3s/sessions",
                    "A3S_CODE_TRACE_PATH": self._trace_remote_path,
                    "A3S_CODE_BUILTIN_SKILLS": os.getenv("A3S_CODE_BUILTIN_SKILLS", "true"),
                    "A3S_CODE_PLANNING": os.getenv("A3S_CODE_PLANNING", "true"),
                    "A3S_CODE_PERMISSIVE": os.getenv("A3S_CODE_PERMISSIVE", "true"),
                    "A3S_CODE_THINKING_BUDGET": os.getenv("A3S_CODE_THINKING_BUDGET", "32000"),
                    "A3S_CODE_MAX_TOOL_ROUNDS": os.getenv("A3S_CODE_MAX_TOOL_ROUNDS", "64"),
                    "A3S_CODE_TOOL_TIMEOUT_MS": os.getenv("A3S_CODE_TOOL_TIMEOUT_MS", "300000"),
                    "A3S_CODE_SKILL_DIRS_JSON": json.dumps(["/root/.openclaw/skills"]),
                }
                cmd = ". /opt/a3s-code-venv/bin/activate && python /root/.a3s/clawmark_a3s_code_runner.py"
                LOGGER.info("Sending to a3s-code session=%s (%d chars)", self.session_id, len(message))
                return await self.sandbox.exec(cmd, timeout_sec=timeout_sec + 60, env=env)

        return A3SCodeOrchestrator

    async def run_task(
        self,
        *,
        task_dir: Path,
        compose_file: Path,
        results_dir: Path,
        dry_run: bool,
    ) -> dict[str, Any]:
        task = self._load_task_py(task_dir)

        if dry_run:
            sandbox = self._DryRunSandbox(workspace_dir=results_dir / task.id / "dryrun_workspace")
        else:
            session_id = f"clawmark-{task.id}-{uuid.uuid4().hex[:8]}"
            sandbox = self._DockerSandbox(session_id=session_id, compose_file=compose_file)

        state_manager = self._CompositeStateManager(
            environments=task.environments,
            env_config=task.env_config,
        )
        orchestrator_cls = self._make_orchestrator_class()
        orchestrator = orchestrator_cls(
            sandbox=sandbox,
            state_manager=state_manager,
            openclaw_config_path=None,
        )

        local_workspace = results_dir / task.id / "workspace"
        stage_results: list[Any] = []
        started_at = time.time()

        try:
            await sandbox.start()
            await state_manager.setup(sandbox=sandbox)
            ctx = state_manager.create_context(task_dir=task.task_dir, sandbox=sandbox)

            stage_results = await orchestrator.run(
                task=task,
                ctx=ctx,
                model=self._model_name,
                api_key=self._model_api_key,
                api_base=self._model_base_url,
                api_format="openrouter",
            )

            await sandbox.download_dir("/workspace", local_workspace)
            trace_local = results_dir / task.id / "messages.jsonl"
            try:
                await sandbox.download_file(orchestrator.trace_remote_path, trace_local)
            except Exception as exc:
                LOGGER.warning("Could not download a3s-code trace for %s: %s", task.id, exc)
        except Exception as exc:
            LOGGER.error("ClawMark task %s failed: %s", task.id, exc, exc_info=True)
            stage_results.append(self._StageResult(stage_id="FRAMEWORK_ERROR", success=False, error=str(exc)))
        finally:
            try:
                await state_manager.cleanup()
            except Exception as exc:
                LOGGER.warning("Cleanup error for %s: %s", task.id, exc)
            await sandbox.stop(delete=True)

        elapsed = time.time() - started_at
        all_items = [item for stage in stage_results for item in stage.verification]
        total_weight = sum(item.weight for item in all_items)
        passed_weight = sum(item.weight for item in all_items if item.passed)
        score = passed_weight / total_weight if total_weight > 0 else 0.0

        result_json = results_dir / task.id / "result.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "task_id": task.id,
            "score": score,
            "execution_time": elapsed,
            "stages": [
                {
                    "id": stage.stage_id,
                    "success": stage.success,
                    "error": stage.error,
                    "verification_score": stage.verification_score,
                    "verification": [
                        {
                            "id": item.item_id,
                            "passed": item.passed,
                            "weight": item.weight,
                            "detail": item.detail,
                            "method": item.method.value,
                        }
                        for item in stage.verification
                    ],
                }
                for stage in stage_results
            ],
            "rubric": [
                {
                    "id": item.item_id,
                    "passed": item.passed,
                    "weight": item.weight,
                    "detail": item.detail,
                }
                for item in all_items
            ],
        }
        result_json.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        trace_path = results_dir / task.id / "messages.jsonl"
        input_tokens = 0
        output_tokens = 0
        turns = 0
        if trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("kind") != "assistant_result":
                    continue
                turns += 1
                input_tokens += int(record.get("prompt_tokens") or 0)
                output_tokens += int(record.get("completion_tokens") or 0)

        return {
            "task_id": task.id,
            "task_path": str(task_dir),
            "score": score,
            "success": all(stage.success for stage in stage_results if stage.stage_id != "final"),
            "input_tokens": input_tokens or None,
            "output_tokens": output_tokens or None,
            "total_tokens": (input_tokens + output_tokens) or None,
            "turns": turns,
            "execution_time_sec": elapsed,
            "result_payload": result_payload,
        }


def run_clawmark_suite(
    *,
    clawmark_root: Path,
    output_dir: Path,
    step: int,
    wheel_path: Path,
    model_name: str,
    model_base_url: str,
    model_api_key: str,
    session_id_header: str,
    max_tasks: int,
    repeats: int,
    compose_file: Path,
    dry_run: bool,
) -> dict[str, Any]:
    selected_tasks, total_tasks = _clawmark_tasks(clawmark_root, max_tasks)
    suite_dir = output_dir / "clawmark"
    suite_dir.mkdir(parents=True, exist_ok=True)
    records: list[TrialRecord] = []
    helper_script = OFFICIAL_DIR / "clawmark_official_eval_runner.py"
    env = _subprocess_env()
    for task_dir in selected_tasks:
        task_id = f"{task_dir.parent.name}_{task_dir.name}"
        for repeat_index in range(1, repeats + 1):
            repeat_dir = suite_dir / f"{_sanitize_name(task_id)}__step{step:07d}__r{repeat_index}"
            summary_path = repeat_dir / "summary.json"
            cmd = [
                "uv",
                "run",
                "--project",
                str(clawmark_root),
                "python",
                str(helper_script),
                "--clawmark-root",
                str(clawmark_root),
                "--task-dir",
                str(task_dir),
                "--results-dir",
                str(repeat_dir),
                "--compose-file",
                str(compose_file),
                "--wheel-path",
                str(wheel_path),
                "--model-name",
                model_name,
                "--model-base-url",
                model_base_url,
                "--model-api-key",
                model_api_key,
                "--session-id-header",
                session_id_header,
                "--summary-path",
                str(summary_path),
            ]
            if dry_run:
                cmd.append("--dry-run")
            log_dir = suite_dir / "launch_logs" / repeat_dir.name
            return_code, _, stderr, error = _run_subprocess(
                cmd,
                cwd=clawmark_root,
                env=env,
                timeout_sec=None,
                log_dir=log_dir,
            )
            if summary_path.exists():
                result = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                result = {
                    "task_id": task_id,
                    "task_path": str(task_dir),
                    "score": 0.0,
                    "success": False,
                    "execution_time_sec": None,
                    "turns": 0,
                }
            if return_code != 0 and not error:
                error = stderr.strip() or f"subprocess exited with code {return_code}"
            records.append(
                TrialRecord(
                    suite="clawmark",
                    task_id=str(result["task_id"]),
                    task_path=str(result["task_path"]),
                    repeat_index=repeat_index,
                    score=float(result["score"]),
                    success=bool(result["success"]) and return_code == 0,
                    input_tokens=result.get("input_tokens"),
                    output_tokens=result.get("output_tokens"),
                    total_tokens=result.get("total_tokens"),
                    execution_time_sec=result.get("execution_time_sec"),
                    error=error,
                    metadata={
                        "turns": result.get("turns"),
                    },
                )
            )

    summary = _summarize_suite(
        suite="clawmark",
        records=records,
        selected_tasks=selected_tasks,
        total_tasks=total_tasks,
        repeats=repeats,
        leaderboard_comparable=len(selected_tasks) == total_tasks and repeats == 3,
    )
    turns = [int((record.metadata or {}).get("turns") or 0) for record in records]
    summary["metrics"]["clawmark_official_turns_per_trial_mean"] = _mean(turns) or 0.0
    return summary


def _summarize_suite(
    *,
    suite: str,
    records: list[TrialRecord],
    selected_tasks: list[Path],
    total_tasks: int,
    repeats: int,
    leaderboard_comparable: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    total_tokens: list[int] = []
    execution_times: list[float] = []
    agent_execution_times: list[float] = []
    agent_timeout_extensions: list[float] = []
    agent_runtime_over_official: list[float] = []
    successes = 0
    positive_score_runs = 0
    exception_free_runs = 0
    reused_runs = 0

    for record in records:
        grouped_scores[record.task_id].append(float(record.score))
        if float(record.score) > 0.0:
            positive_score_runs += 1
        if record.input_tokens is not None:
            input_tokens.append(int(record.input_tokens))
        if record.output_tokens is not None:
            output_tokens.append(int(record.output_tokens))
        if record.total_tokens is not None:
            total_tokens.append(int(record.total_tokens))
        if record.execution_time_sec is not None:
            execution_times.append(float(record.execution_time_sec))
        timeout_budget = (record.metadata or {}).get("timeout_budget") or {}
        timing = (record.metadata or {}).get("timing") or {}
        if timing.get("agent_execution_duration_sec") is not None:
            agent_execution_times.append(float(timing["agent_execution_duration_sec"]))
        if timeout_budget.get("agent_timeout_extension_sec") is not None:
            agent_timeout_extensions.append(float(timeout_budget["agent_timeout_extension_sec"]))
        if timeout_budget.get("agent_runtime_over_official_sec") is not None:
            agent_runtime_over_official.append(float(timeout_budget["agent_runtime_over_official_sec"]))
        if record.success:
            successes += 1
        exception_free = (record.metadata or {}).get("agent_completed_without_exception")
        if exception_free is True or (exception_free is None and record.success):
            exception_free_runs += 1
        if record.reused:
            reused_runs += 1

    task_level_scores = {task_id: _mean(scores) or 0.0 for task_id, scores in grouped_scores.items()}
    suite_score = _mean(task_level_scores.values()) or 0.0
    metrics = {
        f"{suite}_official_score": suite_score,
        f"{suite}_official_completed_runs": float(len(records)),
        f"{suite}_official_successful_runs": float(successes),
        f"{suite}_official_positive_score_runs": float(positive_score_runs),
        f"{suite}_official_exception_free_runs": float(exception_free_runs),
        f"{suite}_official_reused_runs": float(reused_runs),
        f"{suite}_official_selected_tasks": float(len(selected_tasks)),
        f"{suite}_official_total_tasks": float(total_tasks),
        f"{suite}_official_repeats": float(repeats),
        f"{suite}_official_input_tokens_mean": _mean(input_tokens) or 0.0,
        f"{suite}_official_output_tokens_mean": _mean(output_tokens) or 0.0,
        f"{suite}_official_total_tokens_mean": _mean(total_tokens) or 0.0,
        f"{suite}_official_execution_time_sec_mean": _mean(execution_times) or 0.0,
        f"{suite}_official_agent_execution_time_sec_mean": _mean(agent_execution_times) or 0.0,
        f"{suite}_official_agent_timeout_extension_sec_mean": _mean(agent_timeout_extensions) or 0.0,
        f"{suite}_official_agent_runtime_over_official_sec_mean": _mean(agent_runtime_over_official) or 0.0,
        f"{suite}_official_agent_runtime_over_official_runs": float(
            sum(1 for value in agent_runtime_over_official if value > 0.0)
        ),
        f"{suite}_official_leaderboard_comparable": 1.0 if leaderboard_comparable else 0.0,
    }
    payload = {
        "suite": suite,
        "metrics": metrics,
        "records": [asdict(record) for record in records],
        "task_level_scores": task_level_scores,
        "selected_task_paths": [str(path) for path in selected_tasks],
        "selected_task_count": len(selected_tasks),
        "total_task_count": total_tasks,
        "repeats": repeats,
        "leaderboard_comparable": leaderboard_comparable,
    }
    if extra:
        payload.update(extra)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official SkillsBench / ClawMark eval through a3s-code.")
    parser.add_argument("--step", type=int, default=int(os.getenv("A3S_CODE_OFFICIAL_BENCHMARK_STEP", "0")))
    parser.add_argument(
        "--suites",
        type=str,
        default=os.getenv("A3S_CODE_OFFICIAL_BENCHMARK_SUITES", "skillsbench,clawmark"),
        help="Comma-separated subset of: skillsbench, clawmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("A3S_CODE_OFFICIAL_BENCHMARK_OUTPUT_DIR", REPO_ROOT / "runs" / "official_benchmark_eval")),
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--skillsbench-root", type=Path, default=Path(os.getenv("A3S_CODE_SKILLSBENCH_ROOT", DEFAULT_SKILLSBENCH_ROOT)))
    parser.add_argument(
        "--skillsbench-tasks-dir",
        type=Path,
        default=Path(os.environ["A3S_CODE_SKILLSBENCH_TASKS_DIR"]) if os.getenv("A3S_CODE_SKILLSBENCH_TASKS_DIR") else None,
    )
    parser.add_argument("--clawmark-root", type=Path, default=Path(os.getenv("A3S_CODE_CLAWMARK_ROOT", DEFAULT_CLAWMARK_ROOT)))
    parser.add_argument("--skillsbench-max-tasks", type=int, default=int(os.getenv("A3S_CODE_OFFICIAL_SKILLSBENCH_MAX_TASKS", "0")))
    parser.add_argument("--clawmark-max-tasks", type=int, default=int(os.getenv("A3S_CODE_OFFICIAL_CLAWMARK_MAX_TASKS", "0")))
    parser.add_argument("--skillsbench-repeats", type=int, default=int(os.getenv("A3S_CODE_OFFICIAL_SKILLSBENCH_REPEATS", "1")))
    parser.add_argument("--clawmark-repeats", type=int, default=int(os.getenv("A3S_CODE_OFFICIAL_CLAWMARK_REPEATS", "1")))
    parser.add_argument("--skillsbench-timeout-sec", type=int, default=int(os.getenv("A3S_CODE_OFFICIAL_SKILLSBENCH_TIMEOUT_SEC", "0")))
    parser.add_argument(
        "--skillsbench-agent-timeout-sec",
        type=int,
        default=int(os.getenv("A3S_CODE_OFFICIAL_SKILLSBENCH_AGENT_TIMEOUT_SEC", "0")),
        help="Absolute floor for agent execution timeout. 0 preserves each task's official timeout before applying the multiplier.",
    )
    parser.add_argument(
        "--skillsbench-agent-timeout-multiplier",
        type=float,
        default=float(os.getenv("A3S_CODE_OFFICIAL_SKILLSBENCH_AGENT_TIMEOUT_MULTIPLIER", os.getenv("A3S_CODE_SKILLSBENCH_AGENT_TIMEOUT_MULTIPLIER", "1.0"))),
        help="Multiply each task's official [agent].timeout_sec. Values above 1.0 are recorded in per-trial timeout_budget metadata.",
    )
    parser.add_argument("--skillsbench-concurrency", type=int, default=int(os.getenv("A3S_CODE_OFFICIAL_SKILLSBENCH_CONCURRENCY", "1")))
    parser.add_argument(
        "--skillsbench-mode",
        choices=sorted(SKILLSBENCH_MODES),
        default=os.getenv("A3S_CODE_SKILLSBENCH_MODE", "without-skills"),
        help="SkillsBench evaluation condition. with-skills injects task environment/skills; without-skills disables task skill dirs.",
    )
    parser.add_argument(
        "--skillsbench-force-allow-internet",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("A3S_CODE_SKILLSBENCH_FORCE_ALLOW_INTERNET", False),
        help=(
            "Run trials from a per-trial shadow task directory whose task.toml sets "
            "environment.allow_internet=true. This is an infrastructure-relaxed mode "
            "and is not official-leaderboard comparable."
        ),
    )
    parser.add_argument(
        "--skillsbench-keep-images",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("A3S_CODE_OFFICIAL_SKILLSBENCH_KEEP_IMAGES", False),
    )
    parser.add_argument(
        "--worker-local-docker",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("A3S_CODE_WORKER_LOCAL_DOCKER", False),
        help="Start dockerd inside the current privileged worker and override any injected remote DOCKER_HOST.",
    )
    parser.add_argument(
        "--skillsbench-resume-existing",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("A3S_CODE_OFFICIAL_SKILLSBENCH_RESUME_EXISTING", True),
    )
    parser.add_argument("--model-provider", type=str, default=os.getenv("A3S_CODE_MODEL_PROVIDER", "openai"))
    parser.add_argument("--model-name", type=str, default=os.getenv("A3S_CODE_MODEL_NAME", os.getenv("SERVED_MODEL_NAME", "qwen3-4b-2507")))
    parser.add_argument("--model-base-url", type=str, default=os.getenv("A3S_CODE_BENCHMARK_BASE_URL", detect_model_base_url()))
    parser.add_argument("--model-api-key", type=str, default=os.getenv("A3S_CODE_MODEL_API_KEY", os.getenv("SGLANG_API_KEY", "apiKey")))
    parser.add_argument("--session-id-header", type=str, default=os.getenv("A3S_CODE_SESSION_ID_HEADER", "X-Session-Id"))
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=Path(os.getenv("A3S_CODE_CLAWMARK_COMPOSE_FILE", DEFAULT_CLAWMARK_ROOT / "docker" / "docker-compose.yaml")),
    )
    parser.add_argument("--dry-run", action="store_true", default=_env_flag("A3S_CODE_OFFICIAL_CLAWMARK_DRY_RUN", False))
    parser.add_argument("--log-level", type=str, default=os.getenv("A3S_CODE_OFFICIAL_BENCHMARK_LOG_LEVEL", "INFO"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.skillsbench_agent_timeout_multiplier <= 0:
        raise ValueError("--skillsbench-agent-timeout-multiplier must be positive")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    suites = {item.strip().lower() for item in args.suites.split(",") if item.strip()}
    output_dir = (args.output_dir / f"step_{args.step:07d}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_path or (output_dir / "summary.json")
    worker_docker = None
    if args.worker_local_docker:
        worker_docker = start_worker_local_docker(log_dir=output_dir / "worker_local_docker")
    wheel_path = (
        ensure_skillsbench_a3s_code_wheel()
        if "skillsbench" in suites and suites <= {"skillsbench"}
        else ensure_a3s_code_wheel()
    )

    summary: dict[str, Any] = {
        "step": args.step,
        "model_provider": args.model_provider,
        "model_name": args.model_name,
        "model_base_url": args.model_base_url,
        "wheel_path": str(wheel_path),
        "skillsbench_mode": args.skillsbench_mode if "skillsbench" in suites else None,
        "skillsbench_agent_timeout_multiplier": (
            args.skillsbench_agent_timeout_multiplier if "skillsbench" in suites else None
        ),
        "skillsbench_agent_timeout_floor_sec": (
            (args.skillsbench_agent_timeout_sec or None) if "skillsbench" in suites else None
        ),
        "skillsbench_force_allow_internet": (
            args.skillsbench_force_allow_internet if "skillsbench" in suites else None
        ),
        "started_at": time.time(),
        "suites": {},
        "metrics": {},
    }

    try:
        if "skillsbench" in suites:
            skillsbench_summary = run_skillsbench_suite(
                skillsbench_root=args.skillsbench_root,
                skillsbench_tasks_dir=args.skillsbench_tasks_dir,
                output_dir=output_dir,
                step=args.step,
                wheel_path=wheel_path,
                model_provider=args.model_provider,
                model_name=args.model_name,
                model_base_url=args.model_base_url,
                model_api_key=args.model_api_key,
                session_id_header=args.session_id_header,
                max_tasks=args.skillsbench_max_tasks,
                repeats=args.skillsbench_repeats,
                timeout_sec=args.skillsbench_timeout_sec or None,
                agent_timeout_sec=args.skillsbench_agent_timeout_sec or None,
                agent_timeout_multiplier=args.skillsbench_agent_timeout_multiplier,
                concurrency=args.skillsbench_concurrency,
                keep_images=args.skillsbench_keep_images,
                resume_existing=args.skillsbench_resume_existing,
                skillsbench_mode=args.skillsbench_mode,
                force_allow_internet=args.skillsbench_force_allow_internet,
            )
            summary["suites"]["skillsbench"] = skillsbench_summary
            summary["metrics"].update(skillsbench_summary["metrics"])

        if "clawmark" in suites:
            clawmark_summary = run_clawmark_suite(
                clawmark_root=args.clawmark_root,
                output_dir=output_dir,
                step=args.step,
                wheel_path=wheel_path,
                model_name=args.model_name,
                model_base_url=args.model_base_url,
                model_api_key=args.model_api_key,
                session_id_header=args.session_id_header,
                max_tasks=args.clawmark_max_tasks,
                repeats=args.clawmark_repeats,
                compose_file=args.compose_file,
                dry_run=args.dry_run,
            )
            summary["suites"]["clawmark"] = clawmark_summary
            summary["metrics"].update(clawmark_summary["metrics"])
    finally:
        if worker_docker is not None:
            worker_docker.stop()

    summary["finished_at"] = time.time()
    summary["duration_sec"] = summary["finished_at"] - summary["started_at"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
