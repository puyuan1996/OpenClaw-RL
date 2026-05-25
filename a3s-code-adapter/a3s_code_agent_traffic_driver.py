#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.metadata as importlib_metadata
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[import-not-found]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _json_env_dict(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must be a valid JSON object") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object, got {type(value).__name__}")
    return value


def _clear_proxy_env_for_local_rl() -> None:
    # a3s_code's reqwest client consumes explicit proxy env vars but does not
    # honor no_proxy for localhost. Clear them so local RL calls stay local.
    for key in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        os.environ.pop(key, None)


def _bootstrap_a3s_code() -> None:
    try:
        import a3s_code  # noqa: F401

        return
    except ImportError:
        pass

    repo_root = Path(
        os.getenv(
            "A3S_CODE_REPO_ROOT",
            str(Path(__file__).resolve().parents[2] / "a3s-lab" / "Code"),
        )
    )
    sdk_python = repo_root / "sdk" / "python"
    import sys

    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    extra_sites = [
        Path(item).expanduser()
        for item in os.getenv("A3S_CODE_EXTRA_SITE_PACKAGES", "").split(":")
        if item.strip()
    ]
    candidates = [
        Path(sys.prefix) / "lib" / version_dir / "site-packages",
        Path(os.getenv("CONDA_PREFIX", "")) / "lib" / version_dir / "site-packages",
        sdk_python / ".venv" / "lib" / "python3.13" / "site-packages",
        sdk_python / ".venv" / "lib" / "python3.12" / "site-packages",
    ] + extra_sites
    for site in candidates:
        if (site / "a3s_code").exists():
            sys.path.insert(0, str(site))
            return

    raise RuntimeError(
        "a3s_code is not importable. Install a packaged SDK first, for example:\n"
        "  python -m pip install --upgrade a3s-code\n"
        "If the required a3s-code PR has not been released yet, build a wheel from "
        "the a3s-code repository and install that wheel into this environment."
    )


def _latest_a3s_code_version_from_repo() -> str | None:
    repo_root = Path(
        os.getenv(
            "A3S_CODE_REPO_ROOT",
            str(Path(__file__).resolve().parents[2] / "a3s-lab" / "Code"),
        )
    )
    pyproject = repo_root / "sdk" / "python" / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Could not read a3s-code version from {pyproject}") from exc
    version = str(project.get("version", "")).strip()
    return version or None


def _enforce_required_a3s_code_version() -> None:
    required = os.getenv("A3S_CODE_REQUIRED_VERSION", "").strip()
    if not required:
        return
    if required.lower() in {"latest", "current"}:
        required = _latest_a3s_code_version_from_repo() or ""
    try:
        actual = importlib_metadata.version("a3s-code")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "A3S_CODE_REQUIRED_VERSION is set but installed distribution a3s-code is not visible"
        ) from exc
    if not required:
        return
    if actual != required:
        raise RuntimeError(
            f"a3s-code version mismatch: required {required}, imported distribution {actual}. "
            "Install the latest AI45Lab/Code wheel or PyPI package into this environment before running agent-RL."
        )


_clear_proxy_env_for_local_rl()
_bootstrap_a3s_code()
_enforce_required_a3s_code_version()
from a3s_code import Agent, PermissionPolicy, SessionOptions  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "seed_data"
SEED_DATA_FILE = Path(
    os.getenv("A3S_CODE_SEED_DATA_FILE", str(DATA_DIR / "code_task_seeds.json"))
).expanduser()
TEMPLATE_ROOT = Path(
    os.getenv("A3S_CODE_TASK_TEMPLATE_ROOT", str(SCRIPT_DIR / "task_templates"))
).expanduser()
WORKSPACE_ROOT = Path(os.getenv("A3S_CODE_WORKSPACE_ROOT", str(SCRIPT_DIR / "generated_workspaces")))
CONFIG_ROOT = Path(os.getenv("A3S_CODE_CONFIG_ROOT", str(SCRIPT_DIR / "generated_configs")))
WORKSPACE_TEMPLATE_CACHE_ROOT = Path(
    os.getenv("A3S_CODE_WORKSPACE_TEMPLATE_CACHE_ROOT", str(SCRIPT_DIR / "workspace_template_cache"))
)
RESULTS_DIR = Path(os.getenv("A3S_CODE_RESULTS_DIR", str(SCRIPT_DIR / "results")))
RECORD_FILE = Path(
    os.getenv("A3S_CODE_TRAFFIC_RECORD_FILE", str(RESULTS_DIR / "a3s_code_agent_traffic.jsonl"))
)
SIMULATED_USER_BACKENDS_FILE = Path(
    os.getenv(
        "A3S_CODE_SIMULATED_USER_BACKENDS_FILE",
        str(SCRIPT_DIR / "simulated_user_backends.json"),
    )
)

RL_BASE_URL = os.getenv("RL_BASE_URL", "http://127.0.0.1:30000").rstrip("/")
RL_HEALTH_URL = os.getenv("A3S_CODE_RL_HEALTH_URL", f"{RL_BASE_URL}/healthz").strip()
RL_FALLBACK_HEALTH_URL = os.getenv("A3S_CODE_RL_FALLBACK_HEALTH_URL", f"{RL_BASE_URL}/health").strip()
A3S_MODEL_NAME = os.getenv("A3S_MODEL_NAME", os.getenv("SERVED_MODEL_NAME", "qwen3.5-4b"))
SIMULATED_USER_MODEL_URL = os.getenv(
    "SIMULATED_USER_MODEL_URL",
    "",
)
SIMULATED_USER_MODEL_NAME = os.getenv(
    "SIMULATED_USER_MODEL_NAME",
    "kimi-k2.5",
)
SIMULATED_USER_API_KEY = os.getenv(
    "SIMULATED_USER_API_KEY",
    "",
)
SIMULATED_USER_OPENAI_EXTRA_BODY = _json_env_dict("SIMULATED_USER_OPENAI_EXTRA_BODY")
A3S_API_KEY = os.getenv("A3S_API_KEY", os.getenv("SGLANG_API_KEY", "apiKey"))
TASK_VERIFIER_FEEDBACK_TURN_ID = int(os.getenv("A3S_CODE_TASK_VERIFIER_FEEDBACK_TURN_ID", "1"))

CONCURRENCY = int(os.getenv("A3S_CODE_TRAFFIC_CONCURRENCY", "1"))
SESSION_LIMIT = int(os.getenv("A3S_CODE_TRAFFIC_SESSION_LIMIT", "0"))
SESSION_START_INDEX = max(1, int(os.getenv("A3S_CODE_TRAFFIC_SESSION_START_INDEX", "1")))
SESSION_GROUP_SIZE = max(1, int(os.getenv("A3S_CODE_SESSION_GROUP_SIZE", "1")))
MAX_MAIN_TURNS = max(1, int(os.getenv("A3S_CODE_MAX_MAIN_TURNS", "3")))
SESSION_DELAY_SEC = float(os.getenv("A3S_CODE_SESSION_DELAY_SEC", "0.5"))
SIMULATED_USER_TIMEOUT_SEC = float(
    os.getenv("A3S_CODE_SIMULATED_USER_TIMEOUT_SEC", "45")
)
SIMULATED_USER_BACKEND_COOLDOWN_SEC = float(
    os.getenv("A3S_CODE_SIMULATED_USER_BACKEND_COOLDOWN_SEC", "60")
)
SIMULATED_USER_MAX_ATTEMPTS = int(
    os.getenv("A3S_CODE_SIMULATED_USER_MAX_ATTEMPTS", "0")
)
REQUEST_TIMEOUT_SEC = float(os.getenv("A3S_CODE_REQUEST_TIMEOUT_SEC", "600"))
RL_HEALTH_CHECK_INTERVAL_SEC = max(1.0, float(os.getenv("A3S_CODE_RL_HEALTH_CHECK_INTERVAL_SEC", "15")))
RL_UNAVAILABLE_EXIT_SEC = max(0.0, float(os.getenv("A3S_CODE_RL_UNAVAILABLE_EXIT_SEC", "300")))
KEEP_WORKSPACES = _env_flag("A3S_CODE_KEEP_WORKSPACES", False)
KEEP_WORKSPACES_ON_ERROR = _env_flag("A3S_CODE_KEEP_WORKSPACES_ON_ERROR", KEEP_WORKSPACES)
KEEP_CONFIGS = _env_flag("A3S_CODE_KEEP_CONFIGS", False)
AGENT_CONFIG_MODE = os.getenv("A3S_CODE_AGENT_CONFIG_MODE", "shared").strip().lower()
if AGENT_CONFIG_MODE not in {"shared", "per_session"}:
    raise RuntimeError(
        "A3S_CODE_AGENT_CONFIG_MODE must be 'shared' or 'per_session', "
        f"got {AGENT_CONFIG_MODE!r}"
    )
SESSION_ID_HEADER_NAME = os.getenv("A3S_CODE_SESSION_ID_HEADER_NAME", "X-Session-Id").strip()
if not SESSION_ID_HEADER_NAME:
    SESSION_ID_HEADER_NAME = "X-Session-Id"
SHARED_CONFIG_NAME = os.getenv("A3S_CODE_SHARED_CONFIG_NAME", "a3s-code-shared.acl").strip()
if not SHARED_CONFIG_NAME:
    SHARED_CONFIG_NAME = "a3s-code-shared.acl"
WORKSPACE_COPY_MODE = os.getenv("A3S_CODE_WORKSPACE_COPY_MODE", "reflink_auto").strip().lower()
MAX_TOOL_ROUNDS = int(os.getenv("A3S_CODE_MAX_TOOL_ROUNDS", "8"))
TURN_TIMEOUT_SEC = float(os.getenv("A3S_CODE_TURN_TIMEOUT_SEC", "240"))
TOOL_TIMEOUT_MS = int(os.getenv("A3S_CODE_TOOL_TIMEOUT_MS", "240000"))
MAX_PARSE_RETRIES = int(os.getenv("A3S_CODE_MAX_PARSE_RETRIES", "4"))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("A3S_CODE_CIRCUIT_BREAKER_THRESHOLD", "5"))
BUILTIN_SKILLS = _env_flag("A3S_CODE_BUILTIN_SKILLS", True)
PLANNING_MODE = os.getenv("A3S_CODE_PLANNING_MODE", "").strip().lower()
if PLANNING_MODE and PLANNING_MODE not in {"auto", "enabled", "disabled"}:
    raise RuntimeError(
        "A3S_CODE_PLANNING_MODE must be one of auto/enabled/disabled, "
        f"got {PLANNING_MODE!r}"
    )
PLANNING = _env_flag("A3S_CODE_PLANNING", True)
AUTO_COMPACT = _env_flag("A3S_CODE_AUTO_COMPACT", True)
AUTO_COMPACT_THRESHOLD = float(os.getenv("A3S_CODE_AUTO_COMPACT_THRESHOLD", "0.85"))
THINKING_BUDGET = int(os.getenv("A3S_CODE_THINKING_BUDGET", "24000"))
CONTINUATION_ENABLED = _env_flag("A3S_CODE_CONTINUATION_ENABLED", True)
MAX_CONTINUATION_TURNS = int(os.getenv("A3S_CODE_MAX_CONTINUATION_TURNS", "5"))
ENABLE_TASK_VERIFIER_REWARD = _env_flag("A3S_CODE_ENABLE_TASK_VERIFIER_REWARD", True)
VERIFIER_FALLBACK_TO_TEST_COMMAND = _env_flag(
    "A3S_CODE_VERIFIER_FALLBACK_TO_TEST_COMMAND",
    True,
)
VERIFIER_TIMEOUT_SEC = float(os.getenv("A3S_CODE_TASK_VERIFIER_TIMEOUT_SEC", "180"))
MODEL_CONTEXT_TOKENS = int(
    os.getenv(
        "A3S_CODE_MODEL_CONTEXT_TOKENS",
        str(
            max(
                1024,
                int(os.getenv("CODE_RL_MATCHED_CONTEXT_TOKENS", "16384"))
                - int(os.getenv("A3S_CODE_CONTEXT_HEADROOM_TOKENS", "2048")),
            )
        ),
    )
)
MODEL_OUTPUT_TOKENS = int(os.getenv("A3S_CODE_MODEL_OUTPUT_TOKENS", "4096"))
GIT_USER_NAME = os.getenv("A3S_CODE_GIT_USER_NAME", "A3S Code RL")
GIT_USER_EMAIL = os.getenv("A3S_CODE_GIT_USER_EMAIL", "a3s-code-adapter@example.com")
INCLUDED_SEED_TAGS = {
    tag.strip()
    for tag in os.getenv("A3S_CODE_INCLUDE_SEED_TAGS", "").split(",")
    if tag.strip()
}
INCLUDED_SEED_IDS = {
    seed_id.strip()
    for seed_id in os.getenv("A3S_CODE_INCLUDE_SEED_IDS", "").split(",")
    if seed_id.strip()
}
AGENT_ENV_BACKEND = os.getenv("A3S_CODE_AGENT_ENV_BACKEND", "local").strip().lower()
if AGENT_ENV_BACKEND not in {"local", "docker"}:
    raise RuntimeError(
        "A3S_CODE_AGENT_ENV_BACKEND must be 'local' or 'docker', "
        f"got {AGENT_ENV_BACKEND!r}"
    )
AGENT_DOCKER_IMAGE = os.getenv(
    "A3S_CODE_AGENT_DOCKER_IMAGE",
    os.getenv("A3S_CODE_DOCKER_IMAGE", ""),
).strip()
AGENT_DOCKER_NETWORK = os.getenv("A3S_CODE_AGENT_DOCKER_NETWORK", "host").strip()
AGENT_DOCKER_PULL_POLICY = os.getenv("A3S_CODE_AGENT_DOCKER_PULL_POLICY", "missing").strip()
AGENT_DOCKER_PYTHON_BIN = os.getenv("A3S_CODE_AGENT_DOCKER_PYTHON_BIN", sys.executable).strip()
AGENT_DOCKER_WORKDIR = os.getenv(
    "A3S_CODE_AGENT_DOCKER_WORKDIR",
    str(SCRIPT_DIR.parent),
).strip()
WORKER_LOCAL_DOCKER = _env_flag("A3S_CODE_WORKER_LOCAL_DOCKER", AGENT_ENV_BACKEND == "docker")
DOCKER_CONTAINER_TIMEOUT_SEC = float(os.getenv("A3S_CODE_AGENT_DOCKER_TIMEOUT_SEC", "0"))

RECORD_LOCK = threading.Lock()
COUNTER_LOCK = threading.Lock()
TEMPLATE_CACHE_LOCK = threading.Lock()
CONFIG_BUILD_LOCK = threading.Lock()
SESSION_COUNTER = SESSION_START_INDEX - 1
SHUTDOWN_EVENT = threading.Event()


@dataclass
class SimulatedUserBackend:
    url: str
    model: str
    api_key: str
    label: str
    priority: int = 100
    cooldown_until: float = 0.0
    failures: int = 0


@dataclass(frozen=True)
class SeedTask:
    seed_id: str
    template: str
    seed: str
    acceptance: list[str]
    tags: list[str]
    complexity: str
    sampling_weight: int
    scenario: str
    target_skills: list[str]
    constraints: list[str]
    benchmark_refs: list[str]
    followup_axes: list[str]


@dataclass(frozen=True)
class TemplateMeta:
    template: str
    repo_summary: str
    key_files: list[str]
    test_command: str
    realism_notes: list[str]
    verifier_command: str
    verifier_score_key: str


class SimulatedUserBackendPool:
    def __init__(
        self,
        backends: list[SimulatedUserBackend],
        *,
        cooldown_sec: float,
        max_attempts: int,
    ) -> None:
        self._backends = backends
        self._cooldown_sec = max(0.0, cooldown_sec)
        self._max_attempts = max(0, max_attempts)
        self._lock = threading.Lock()
        self._next_index_by_priority: dict[int, int] = {}

    @property
    def backends(self) -> list[SimulatedUserBackend]:
        return self._backends

    def candidate_order(self) -> list[SimulatedUserBackend]:
        with self._lock:
            if not self._backends:
                return []
            now = time.monotonic()
            ready = [backend for backend in self._backends if backend.cooldown_until <= now]
            if not ready:
                ready = list(self._backends)
            grouped: dict[int, list[SimulatedUserBackend]] = {}
            for backend in ready:
                grouped.setdefault(backend.priority, []).append(backend)
            ordered: list[SimulatedUserBackend] = []
            for priority in sorted(grouped):
                group = grouped[priority]
                start = self._next_index_by_priority.get(priority, 0) % len(group)
                self._next_index_by_priority[priority] = (start + 1) % len(group)
                ordered.extend(group[start:] + group[:start])
            if self._max_attempts > 0:
                ordered = ordered[: self._max_attempts]
            return ordered

    def mark_success(self, backend: SimulatedUserBackend) -> None:
        with self._lock:
            backend.failures = 0
            backend.cooldown_until = 0.0

    def mark_failure(self, backend: SimulatedUserBackend) -> None:
        with self._lock:
            backend.failures += 1
            if self._cooldown_sec > 0:
                backend.cooldown_until = time.monotonic() + self._cooldown_sec


def _split_csv_env(raw: str, *, keep_empty: bool = False) -> list[str]:
    if raw == "":
        return []
    items = [item.strip() for item in raw.split(",")]
    if keep_empty:
        return items
    return [item for item in items if item]


def _split_path_list(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(":") if item.strip()]


def _docker_mount_specs() -> list[str]:
    specs: list[str] = []
    for source, target, mode in (
        ("/mnt/shared-storage-user", "/mnt/shared-storage-user", "rw"),
        ("/mnt/shared-storage-gpfs2", "/mnt/shared-storage-gpfs2", "ro"),
    ):
        if Path(source).exists():
            specs.append(f"{source}:{target}:{mode}")

    for spec in _split_csv_env(os.getenv("A3S_CODE_AGENT_DOCKER_MOUNTS", "")):
        if spec:
            specs.append(spec)
    return list(dict.fromkeys(specs))


def _docker_env_keys() -> list[str]:
    explicit = {
        "A3S_API_KEY",
        "A3S_MODEL_NAME",
        "CONDA_PREFIX",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "NO_PROXY",
        "PATH",
        "PYTHONPATH",
        "SERVED_MODEL_NAME",
        "SGLANG_API_KEY",
        "no_proxy",
    }
    prefixes = (
        "A3S_CODE_",
        "CODE_RL_",
        "OPENAI_",
        "RL_",
        "SIMULATED_USER_",
        "VLLM_",
    )
    skip = {
        "A3S_CODE_AGENT_DOCKER_MOUNTS",
        "A3S_CODE_AGENT_ENV_BACKEND",
        "A3S_CODE_WORKER_LOCAL_DOCKER",
        "DOCKER_CONFIG",
        "DOCKER_HOST",
    }
    keys = [
        key
        for key in os.environ
        if key not in skip and (key in explicit or key.startswith(prefixes))
    ]
    return sorted(keys)


def _build_docker_session_command(worker_id: int, session_index: int) -> list[str]:
    if not AGENT_DOCKER_IMAGE:
        raise RuntimeError("A3S_CODE_AGENT_DOCKER_IMAGE is required for docker agent backend")
    if not AGENT_DOCKER_PYTHON_BIN:
        raise RuntimeError("A3S_CODE_AGENT_DOCKER_PYTHON_BIN is required for docker agent backend")

    container_name = (
        f"a3s-code-adapter-w{worker_id}-s{session_index}-{uuid.uuid4().hex[:8]}"
    )[:120]
    command = ["docker", "run", "--rm", "--name", container_name]
    if AGENT_DOCKER_PULL_POLICY:
        command.append(f"--pull={AGENT_DOCKER_PULL_POLICY}")
    if AGENT_DOCKER_NETWORK:
        command.extend(["--network", AGENT_DOCKER_NETWORK])

    for spec in _docker_mount_specs():
        command.extend(["-v", spec])

    if AGENT_DOCKER_WORKDIR:
        command.extend(["-w", AGENT_DOCKER_WORKDIR])

    for key in _docker_env_keys():
        command.extend(["-e", key])

    child_pythonpath = ":".join(
        [str(SCRIPT_DIR)] + _split_path_list(os.getenv("PYTHONPATH", ""))
    )
    overrides = {
        "A3S_CODE_AGENT_ENV_BACKEND": "local",
        "A3S_CODE_TRAFFIC_CONCURRENCY": "1",
        "A3S_CODE_TRAFFIC_SESSION_LIMIT": "1",
        "A3S_CODE_TRAFFIC_SESSION_START_INDEX": str(session_index),
        "A3S_CODE_WORKER_LOCAL_DOCKER": "0",
        "PYTHONPATH": child_pythonpath,
        "PYTHONUNBUFFERED": "1",
    }
    for key, value in overrides.items():
        command.extend(["-e", f"{key}={value}"])

    command.extend(
        [
            AGENT_DOCKER_IMAGE,
            AGENT_DOCKER_PYTHON_BIN,
            "-u",
            str(SCRIPT_DIR / "a3s_code_agent_traffic_driver.py"),
        ]
    )
    return command


def _run_one_session_in_docker(worker_id: int, session_index: int) -> None:
    command = _build_docker_session_command(worker_id, session_index)
    print(
        f"[a3s-code-driver] worker={worker_id} session_index={session_index} "
        f"agent_env_backend=docker image={AGENT_DOCKER_IMAGE} network={AGENT_DOCKER_NETWORK}",
        flush=True,
    )
    timeout = DOCKER_CONTAINER_TIMEOUT_SEC if DOCKER_CONTAINER_TIMEOUT_SEC > 0 else None
    subprocess.run(command, check=True, timeout=timeout)


def _make_backend_label(index: int, model: str, url: str) -> str:
    return f"{index}:{model}@{url.replace('http://', '').replace('https://', '')}"


def _load_simulated_user_backends_from_config(
    path: Path,
) -> tuple[bool, list[SimulatedUserBackend]]:
    if not path.exists():
        return False, []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[a3s-code-driver] simulated_user config unreadable path={path} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return True, []

    raw_backends = payload.get("backends", []) if isinstance(payload, dict) else []
    if not isinstance(raw_backends, list):
        print(
            f"[a3s-code-driver] simulated_user config malformed path={path} backends_type={type(raw_backends).__name__}",
            flush=True,
        )
        return True, []

    backends: list[SimulatedUserBackend] = []
    for index, item in enumerate(raw_backends, start=1):
        if not isinstance(item, dict):
            continue
        enabled = bool(item.get("enabled", True))
        healthy = bool(item.get("healthy", item.get("probe", {}).get("chat_ok", True)))
        if not enabled or not healthy:
            continue
        url = str(item.get("url", "")).strip()
        model = str(item.get("model", "")).strip()
        if not url or not model:
            continue
        api_key = str(item.get("api_key", "") or "")
        label = str(item.get("label", "")).strip() or _make_backend_label(index, model, url)
        priority = int(item.get("priority", 100) or 100)
        backends.append(
            SimulatedUserBackend(
                url=url,
                model=model,
                api_key=api_key,
                label=label,
                priority=priority,
            )
        )
    return True, backends


def _expand_backend_values(values: list[str], size: int, default: str) -> list[str]:
    if size <= 0:
        return []
    if not values:
        values = [default]
    if len(values) == 1 and size > 1:
        return values * size
    if len(values) != size:
        raise RuntimeError(
            f"simulated-user backend config length mismatch: expected 1 or {size}, got {len(values)}"
        )
    return values


def _build_simulated_user_backends() -> list[SimulatedUserBackend]:
    urls = _split_csv_env(os.getenv("SIMULATED_USER_MODEL_URLS", ""))
    if not urls and SIMULATED_USER_MODEL_URL:
        urls = [SIMULATED_USER_MODEL_URL]
    if not urls:
        return []

    names = _split_csv_env(os.getenv("SIMULATED_USER_MODEL_NAMES", ""))
    keys = _split_csv_env(os.getenv("SIMULATED_USER_API_KEYS", ""), keep_empty=True)
    names = _expand_backend_values(names, len(urls), SIMULATED_USER_MODEL_NAME)
    keys = _expand_backend_values(keys, len(urls), SIMULATED_USER_API_KEY)

    backends: list[SimulatedUserBackend] = []
    for index, (url, model, api_key) in enumerate(zip(urls, names, keys), start=1):
        backends.append(
            SimulatedUserBackend(
                url=url,
                model=model,
                api_key=api_key,
                label=_make_backend_label(index, model, url),
                priority=100,
            )
        )
    return backends


def _resolve_simulated_user_backends() -> tuple[list[SimulatedUserBackend], str]:
    if _env_flag("A3S_CODE_DISABLE_SIMULATED_USER", False):
        return [], "disabled-env"

    env_backends = _build_simulated_user_backends()
    if env_backends:
        return env_backends, "env"

    loaded_from_config, config_backends = _load_simulated_user_backends_from_config(
        SIMULATED_USER_BACKENDS_FILE
    )
    if loaded_from_config:
        if config_backends:
            return config_backends, f"config:{SIMULATED_USER_BACKENDS_FILE}"
        return [], f"config-empty:{SIMULATED_USER_BACKENDS_FILE}"
    return [], "fallback-only"


SIMULATED_USER_BACKENDS, SIMULATED_USER_BACKENDS_SOURCE = _resolve_simulated_user_backends()
SIMULATED_USER_POOL = SimulatedUserBackendPool(
    SIMULATED_USER_BACKENDS,
    cooldown_sec=SIMULATED_USER_BACKEND_COOLDOWN_SEC,
    max_attempts=SIMULATED_USER_MAX_ATTEMPTS,
)


def _ensure_dirs() -> None:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    WORKSPACE_TEMPLATE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / (
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _render_agent_config(base_url: str, *, session_id_header: str | None = None) -> str:
    session_header_line = ""
    if session_id_header:
        session_header_line = f'  sessionIdHeader = "{session_id_header}"\n'
    return (
        f'default_model = "openai/{A3S_MODEL_NAME}"\n\n'
        'providers "openai" {\n'
        f'  api_key = "{A3S_API_KEY}"\n'
        f'  base_url = "{base_url}"\n'
        f"{session_header_line}\n"
        f'  models "{A3S_MODEL_NAME}" {{\n'
        f'    name = "{A3S_MODEL_NAME}"\n'
        "    tool_call = true\n\n"
        "    limit = {\n"
        f"      context = {MODEL_CONTEXT_TOKENS}\n"
        f"      output = {MODEL_OUTPUT_TOKENS}\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def _load_seed_tasks() -> list[SeedTask]:
    raw = json.loads(SEED_DATA_FILE.read_text(encoding="utf-8"))
    seeds = [
        SeedTask(
            seed_id=str(item["id"]),
            template=str(item["template"]),
            seed=str(item["seed"]),
            acceptance=[str(x) for x in item.get("acceptance", [])],
            tags=[str(x) for x in item.get("tags", [])],
            complexity=str(item.get("complexity", "standard")),
            sampling_weight=max(1, int(item.get("sampling_weight", 1))),
            scenario=str(item.get("scenario", "")),
            target_skills=[str(x) for x in item.get("target_skills", [])],
            constraints=[str(x) for x in item.get("constraints", [])],
            benchmark_refs=[str(x) for x in item.get("benchmark_refs", [])],
            followup_axes=[str(x) for x in item.get("followup_axes", [])],
        )
        for item in raw
    ]
    filtered = seeds
    if INCLUDED_SEED_IDS:
        filtered = [seed for seed in filtered if seed.seed_id in INCLUDED_SEED_IDS]
        if not filtered:
            raise RuntimeError(
                "A3S_CODE_INCLUDE_SEED_IDS matched no seeds. "
                f"requested={sorted(INCLUDED_SEED_IDS)}"
            )

    if INCLUDED_SEED_TAGS:
        filtered = [seed for seed in filtered if INCLUDED_SEED_TAGS.intersection(set(seed.tags))]
        if not filtered:
            raise RuntimeError(
                "A3S_CODE_INCLUDE_SEED_TAGS matched no seeds. "
                f"requested={sorted(INCLUDED_SEED_TAGS)}"
            )

    return filtered


def _load_template_meta(template_name: str) -> TemplateMeta:
    meta_path = TEMPLATE_ROOT / template_name / "template_meta.json"
    item = json.loads(meta_path.read_text(encoding="utf-8"))
    return TemplateMeta(
        template=str(item["template"]),
        repo_summary=str(item["repo_summary"]),
        key_files=[str(x) for x in item.get("key_files", [])],
        test_command=str(item.get("test_command", "pytest -q")),
        realism_notes=[str(x) for x in item.get("realism_notes", [])],
        verifier_command=str(item.get("verifier_command", "")),
        verifier_score_key=str(item.get("verifier_score_key", "")),
    )


def _append_record(payload: dict[str, Any]) -> None:
    with RECORD_LOCK:
        with RECORD_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _probe_rl_health(client: httpx.Client) -> tuple[bool, str]:
    last_error = "no health endpoints configured"
    for url in (RL_HEALTH_URL, RL_FALLBACK_HEALTH_URL):
        if not url:
            continue
        try:
            response = client.get(url, timeout=5.0)
            if response.status_code == 200:
                return True, url
            last_error = f"status={response.status_code} url={url}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return False, last_error


def _wait_for_rl_service(worker_id: int) -> bool:
    if SHUTDOWN_EVENT.is_set():
        return False

    start = time.monotonic()
    warned = False
    with httpx.Client(trust_env=False) as client:
        while not SHUTDOWN_EVENT.is_set():
            healthy, detail = _probe_rl_health(client)
            if healthy:
                if warned:
                    print(
                        f"[a3s-code-driver] worker={worker_id} rl_proxy_restored detail={detail}",
                        flush=True,
                    )
                return True

            elapsed = time.monotonic() - start
            if RL_UNAVAILABLE_EXIT_SEC > 0 and elapsed >= RL_UNAVAILABLE_EXIT_SEC:
                print(
                    f"[a3s-code-driver] worker={worker_id} rl_proxy_unavailable_for={elapsed:.0f}s "
                    f"detail={detail} action=exit",
                    flush=True,
                )
                SHUTDOWN_EVENT.set()
                return False

            if not warned:
                print(
                    f"[a3s-code-driver] worker={worker_id} waiting_for_rl_proxy "
                    f"health_url={RL_HEALTH_URL} detail={detail}",
                    flush=True,
                )
                warned = True
            time.sleep(RL_HEALTH_CHECK_INTERVAL_SEC)

    return False


def _next_session_index() -> int | None:
    global SESSION_COUNTER
    with COUNTER_LOCK:
        if SESSION_LIMIT and SESSION_COUNTER >= SESSION_START_INDEX + SESSION_LIMIT - 1:
            return None
        SESSION_COUNTER += 1
        return SESSION_COUNTER


def _extract_text(resp_json: dict[str, Any]) -> str:
    return str(resp_json.get("choices", [{}])[0].get("message", {}).get("content", "") or "")


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _chat_completion(
    client: httpx.Client,
    url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.6,
    max_tokens: int = 1024,
    api_key: str = "",
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    body.update(SIMULATED_USER_OPENAI_EXTRA_BODY)
    resp = client.post(url, headers=headers, json=body, timeout=SIMULATED_USER_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()


def _simulated_user_chat_completion(
    client: httpx.Client,
    *,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    if not SIMULATED_USER_POOL.backends:
        raise RuntimeError("no simulated-user backends configured")

    last_exc: Exception | None = None
    for backend in SIMULATED_USER_POOL.candidate_order():
        try:
            resp = _chat_completion(
                client,
                backend.url,
                model=backend.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=backend.api_key,
            )
            SIMULATED_USER_POOL.mark_success(backend)
            return resp
        except Exception as exc:
            SIMULATED_USER_POOL.mark_failure(backend)
            last_exc = exc
            if len(SIMULATED_USER_POOL.backends) > 1:
                print(
                    f"[a3s-code-driver] simulated_user backend_fail "
                    f"backend={backend.label} error={type(exc).__name__}: {exc}",
                    flush=True,
                )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("simulated-user request failed without a concrete backend error")


def _fallback_seed_prompt(seed: SeedTask, template_meta: TemplateMeta) -> str:
    acceptance = "; ".join(seed.acceptance[:3])
    constraints = "; ".join(seed.constraints[:3])
    skill_focus = ", ".join(seed.target_skills[:4])
    return (
        f"I'm working in a small repo for {template_meta.repo_summary.lower()}. "
        f"Please handle this request: {seed.seed}. "
        f"{seed.scenario + ' ' if seed.scenario else ''}"
        f"Keep the change scoped, update tests or docs if they are affected, "
        f"and tell me how you verified it. "
        f"{f'Complexity target: {seed.complexity}. ' if seed.complexity else ''}"
        f"{f'Skill focus: {skill_focus}. ' if skill_focus else ''}"
        f"{f'Extra constraints: {constraints}. ' if constraints else ''}"
        f"Acceptance notes: {acceptance}."
    )


def _rewrite_seed_task(client: httpx.Client, seed: SeedTask, template_meta: TemplateMeta) -> str:
    if not SIMULATED_USER_POOL.backends:
        return _fallback_seed_prompt(seed, template_meta)

    system = (
        "You rewrite terse issue notes into realistic, benchmark-inspired user requests for an autonomous coding agent. "
        "Keep the request natural, concrete, and scoped to the repo summary. "
        "The request should sound like it comes from a busy engineer with real constraints, not from a benchmark author. "
        "Include 2-4 concrete deliverables or constraints when helpful, and mention testing or verification when appropriate. "
        "Return strict JSON only with keys user_request and success_checks."
    )
    user = (
        f"Repo summary:\n{template_meta.repo_summary}\n\n"
        f"Key files:\n- " + "\n- ".join(template_meta.key_files) + "\n\n"
        f"Suggested test command: {template_meta.test_command}\n\n"
        f"Realism notes:\n- " + "\n- ".join(template_meta.realism_notes) + "\n\n"
        f"Scenario:\n{seed.scenario or 'General maintenance request for this repo.'}\n\n"
        f"Complexity target: {seed.complexity}\n"
        f"Seed task: {seed.seed}\n"
        f"Acceptance hints: {json.dumps(seed.acceptance, ensure_ascii=False)}\n"
        f"Tags: {json.dumps(seed.tags, ensure_ascii=False)}\n"
        f"Target skills: {json.dumps(seed.target_skills, ensure_ascii=False)}\n"
        f"Constraints: {json.dumps(seed.constraints, ensure_ascii=False)}\n"
        f"Benchmark references: {json.dumps(seed.benchmark_refs, ensure_ascii=False)}"
    )
    try:
        resp = _simulated_user_chat_completion(
            client,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.7,
            max_tokens=900,
        )
        parsed = _extract_json_obj(_extract_text(resp)) or {}
        prompt = str(parsed.get("user_request", "")).strip()
        if prompt:
            return prompt
    except Exception as exc:
        print(
            f"[a3s-code-driver] simulated_user rewrite fallback "
            f"seed={seed.seed_id} error={type(exc).__name__}: {exc}",
            flush=True,
        )

    return _fallback_seed_prompt(seed, template_meta)


def _generate_followup(
    client: httpx.Client,
    *,
    seed: SeedTask,
    template_meta: TemplateMeta,
    original_request: str,
    latest_response: str,
    next_turn_number: int,
    is_final_turn: bool,
) -> tuple[str, bool]:
    system = (
        "You simulate a realistic engineer following up on a coding task. "
        "Look at the original request and the latest assistant message, then write the next user message only. "
        "Usually ask for one concrete verification step, edge case, risk check, or small polish item. "
        "Prefer follow-ups that feel like a real project review: regression concerns, docs drift, edge-case coverage, or benchmark-style realism gaps. "
        "If this is the final turn, the reply should close the loop naturally. "
        "Return strict JSON with keys reply and done."
    )
    user = (
        f"Repo summary:\n{template_meta.repo_summary}\n\n"
        f"Original request:\n{original_request}\n\n"
        f"Latest assistant response:\n{latest_response}\n\n"
        f"Next user turn number: {next_turn_number}\n"
        f"Final turn: {json.dumps(is_final_turn)}\n"
        f"Suggested test command: {template_meta.test_command}\n"
        f"Acceptance hints: {json.dumps(seed.acceptance, ensure_ascii=False)}\n"
        f"Target skills: {json.dumps(seed.target_skills, ensure_ascii=False)}\n"
        f"Constraints: {json.dumps(seed.constraints, ensure_ascii=False)}\n"
        f"Benchmark references: {json.dumps(seed.benchmark_refs, ensure_ascii=False)}\n"
        f"Preferred followup axes: {json.dumps(seed.followup_axes, ensure_ascii=False)}"
    )

    try:
        resp = _simulated_user_chat_completion(
            client,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.5,
            max_tokens=500,
        )
        parsed = _extract_json_obj(_extract_text(resp)) or {}
        reply = str(parsed.get("reply", "")).strip()
        done = bool(parsed.get("done", False))
        if reply:
            if is_final_turn:
                done = True
            return reply, done
    except Exception as exc:
        print(
            f"[a3s-code-driver] simulated_user followup fallback "
            f"turn={next_turn_number} seed={seed.seed_id} error={type(exc).__name__}: {exc}",
            flush=True,
        )

    if is_final_turn:
        return (
            f"Looks good. Please do one final verification pass, mention the exact command you ran "
            f"({template_meta.test_command}), and summarize what changed.",
            True,
        )
    if seed.followup_axes:
        followup_axis = seed.followup_axes[(next_turn_number - 2) % len(seed.followup_axes)]
        return (
            f"Please do one more pass focused on this review angle: {followup_axis}. "
            f"Run the relevant verification, call out any remaining risk, and update docs or examples if they changed.",
            False,
        )
    return (
        "Please run the relevant verification, cover one edge case if it is missing, "
        "and update any user-facing docs or help text that changed.",
        False,
    )


def _copy_dir(src: Path, dst: Path, *, mode: str, strip_template_meta: bool = False) -> None:
    if dst.exists():
        shutil.rmtree(dst)

    normalized = mode.strip().lower()
    reflink_mode = None
    if normalized == "reflink_auto":
        reflink_mode = "auto"
    elif normalized == "reflink_always":
        reflink_mode = "always"

    copied = False
    if reflink_mode is not None:
        dst.mkdir(parents=True, exist_ok=True)
        cmd = ["cp", "-a", f"--reflink={reflink_mode}", f"{src}/.", str(dst)]
        result = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        copied = result.returncode == 0
        if not copied:
            shutil.rmtree(dst, ignore_errors=True)

    if not copied:
        shutil.copytree(src, dst)

    if strip_template_meta:
        meta = dst / "template_meta.json"
        if meta.exists():
            meta.unlink()


def _prepare_workspace_template_cache(template_name: str) -> Path:
    src = TEMPLATE_ROOT / template_name
    if not src.exists():
        raise FileNotFoundError(f"template {template_name!r} not found at {src}")
    cache_dir = WORKSPACE_TEMPLATE_CACHE_ROOT / template_name
    with TEMPLATE_CACHE_LOCK:
        if cache_dir.exists():
            return cache_dir

        tmp_dir = WORKSPACE_TEMPLATE_CACHE_ROOT / f".{template_name}.tmp-{uuid.uuid4().hex[:8]}"
        try:
            _copy_dir(src, tmp_dir, mode=WORKSPACE_COPY_MODE, strip_template_meta=True)
            _init_git_repo(tmp_dir)
            tmp_dir.rename(cache_dir)
        except FileExistsError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return cache_dir


def _copy_workspace_template(template_name: str, session_id: str) -> Path:
    src = _prepare_workspace_template_cache(template_name)
    dst = WORKSPACE_ROOT / session_id
    _copy_dir(src, dst, mode=WORKSPACE_COPY_MODE)
    return dst


def _init_git_repo(workspace: Path) -> None:
    commands = [
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            f"user.name={GIT_USER_NAME}",
            "-c",
            f"user.email={GIT_USER_EMAIL}",
            "commit",
            "-qm",
            "template baseline",
        ],
    ]
    for cmd in commands:
        subprocess.run(
            cmd,
            cwd=workspace,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _build_agent_config(session_id: str) -> Path:
    if AGENT_CONFIG_MODE == "shared":
        config_path = CONFIG_ROOT / SHARED_CONFIG_NAME
        config_text = _render_agent_config(RL_BASE_URL, session_id_header=SESSION_ID_HEADER_NAME)
        with CONFIG_BUILD_LOCK:
            if not config_path.exists() or config_path.read_text(encoding="utf-8") != config_text:
                _write_text_atomic(config_path, config_text)
        return config_path

    config_path = CONFIG_ROOT / f"{session_id}.acl"
    _write_text_atomic(
        config_path,
        _render_agent_config(f"{RL_BASE_URL}/session/{session_id}"),
    )
    return config_path


def _mark_session_done(client: httpx.Client, session_id: str) -> None:
    headers = {"Authorization": f"Bearer {A3S_API_KEY}"} if A3S_API_KEY else {}
    resp = client.post(
        f"{RL_BASE_URL}/session_done",
        headers=headers,
        json={"session_id": session_id},
        timeout=REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()


def _post_task_verifier_feedback(
    client: httpx.Client,
    session_id: str,
    *,
    score: float,
    details: dict[str, Any],
) -> None:
    headers = {"Authorization": f"Bearer {A3S_API_KEY}"} if A3S_API_KEY else {}
    resp = client.post(
        f"{RL_BASE_URL}/feedback",
        headers=headers,
        json={
            "session_id": session_id,
            "turn_id": TASK_VERIFIER_FEEDBACK_TURN_ID,
            "event_type": "task_verifier_reward",
            "details": {
                "score": score,
                **details,
            },
        },
        timeout=REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()


def _find_json_object(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = stripped.find(start_char)
        end = stripped.rfind(end_char)
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except Exception:
                pass
    return None


def _lookup_nested_key(payload: Any, dotted_key: str) -> Any:
    current = payload
    for part in dotted_key.split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _numeric_score_values(payload: Any) -> list[float]:
    if isinstance(payload, bool):
        return [1.0 if payload else 0.0]
    if isinstance(payload, (int, float)):
        return [float(payload)]
    if isinstance(payload, list):
        values: list[float] = []
        for item in payload:
            values.extend(_numeric_score_values(item))
        return values
    if isinstance(payload, dict):
        values = []
        ignored_keys = {
            "elapsed",
            "elapsed_sec",
            "execution_time",
            "execution_time_sec",
            "input_tokens",
            "output_tokens",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "turns",
        }
        for key, value in payload.items():
            key_l = str(key).lower()
            if key_l in ignored_keys or key_l.endswith("_tokens") or key_l.endswith("_sec"):
                continue
            values.extend(_numeric_score_values(value))
        return values
    return []


def _score_from_verifier_output(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    score_key: str,
) -> tuple[float, dict[str, Any]]:
    parsed = _find_json_object(stdout)
    details: dict[str, Any] = {
        "returncode": returncode,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }
    if parsed is not None:
        details["parsed"] = parsed
        if score_key:
            raw = _lookup_nested_key(parsed, score_key)
            try:
                return float(raw), details
            except (TypeError, ValueError):
                pass
        if isinstance(parsed, dict):
            for key in ("score", "reward", "mean_score", "avg_score", "accuracy"):
                if key in parsed:
                    try:
                        return float(parsed[key]), details
                    except (TypeError, ValueError):
                        pass
            if isinstance(parsed.get("scores"), dict):
                values = _numeric_score_values(parsed["scores"])
                if values:
                    return sum(values) / len(values), details
        values = _numeric_score_values(parsed)
        if values:
            return sum(values) / len(values), details

    return (1.0 if returncode == 0 else -1.0), details


def _render_verifier_command(
    command: str,
    *,
    workspace: Path,
    seed: SeedTask,
    template_meta: TemplateMeta,
) -> str:
    return command.format(
        workspace=str(workspace),
        seed_id=seed.seed_id,
        template=template_meta.template,
    )


def _run_task_verifier(
    *,
    workspace: Path,
    seed: SeedTask,
    template_meta: TemplateMeta,
) -> dict[str, Any] | None:
    if not ENABLE_TASK_VERIFIER_REWARD:
        return None

    command = template_meta.verifier_command.strip()
    command_source = "template_meta.verifier_command"
    if not command and VERIFIER_FALLBACK_TO_TEST_COMMAND:
        command = template_meta.test_command.strip()
        command_source = "template_meta.test_command"
    if not command:
        return None

    rendered = _render_verifier_command(
        command,
        workspace=workspace,
        seed=seed,
        template_meta=template_meta,
    )
    start = time.time()
    try:
        proc = subprocess.run(
            rendered,
            cwd=workspace,
            shell=True,
            text=True,
            capture_output=True,
            timeout=VERIFIER_TIMEOUT_SEC,
        )
        elapsed = time.time() - start
        score, details = _score_from_verifier_output(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            score_key=template_meta.verifier_score_key,
        )
        details.update(
            {
                "command": rendered,
                "command_source": command_source,
                "elapsed_sec": elapsed,
                "template": template_meta.template,
                "seed_id": seed.seed_id,
            }
        )
        return {"score": score, "details": details}
    except subprocess.TimeoutExpired as exc:
        return {
            "score": -1.0,
            "details": {
                "command": rendered,
                "command_source": command_source,
                "elapsed_sec": time.time() - start,
                "returncode": None,
                "timeout_sec": VERIFIER_TIMEOUT_SEC,
                "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                "template": template_meta.template,
                "seed_id": seed.seed_id,
            },
        }


def _send_with_timeout(session: Any, prompt: str) -> Any:
    done = threading.Event()
    result_box: dict[str, Any] = {}

    def _target() -> None:
        try:
            result_box["result"] = session.send(prompt)
        except Exception as exc:
            result_box["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()

    if done.wait(timeout=TURN_TIMEOUT_SEC):
        if "error" in result_box:
            raise result_box["error"]
        return result_box["result"]

    try:
        session.cancel()
    except Exception:
        pass
    done.wait(timeout=5)
    raise TimeoutError(f"session.send timed out after {TURN_TIMEOUT_SEC:.0f}s")


def _select_seed_for_session(session_index: int, seeds: list[SeedTask]) -> tuple[SeedTask, int | None, int]:
    if SESSION_GROUP_SIZE <= 1:
        seed = random.choices(seeds, weights=[seed.sampling_weight for seed in seeds], k=1)[0]
        return seed, None, 0

    group_index = (session_index - 1) // SESSION_GROUP_SIZE
    replica_index = (session_index - 1) % SESSION_GROUP_SIZE
    seed = seeds[group_index % len(seeds)]
    return seed, group_index, replica_index


def _run_one_session(worker_id: int, session_index: int, seeds: list[SeedTask]) -> None:
    seed, sample_group_index, sample_replica_index = _select_seed_for_session(session_index, seeds)
    template_meta = _load_template_meta(seed.template)
    group_fragment = (
        f"grp{sample_group_index:06d}-rep{sample_replica_index:02d}-"
        if sample_group_index is not None
        else ""
    )
    session_id = (
        f"a3s-code-{int(time.time())}-{worker_id}-{session_index}-"
        f"{group_fragment}{uuid.uuid4().hex[:8]}"
    )
    workspace = _copy_workspace_template(seed.template, session_id)
    config_path = _build_agent_config(session_id)
    config_is_ephemeral = AGENT_CONFIG_MODE == "per_session"

    record: dict[str, Any] = {
        "session_id": session_id,
        "worker_id": worker_id,
        "session_index": session_index,
        "seed_id": seed.seed_id,
        "seed": seed.seed,
        "template": seed.template,
        "sample_group_index": sample_group_index,
        "sample_replica_index": sample_replica_index,
        "session_group_size": SESSION_GROUP_SIZE,
        "agent_config_mode": AGENT_CONFIG_MODE,
        "agent_config_path": str(config_path),
        "workspace": str(workspace),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "main_turns": [],
        "status": "started",
    }

    simulated_user_client = httpx.Client(timeout=None, trust_env=False)
    print(
        f"[a3s-code-driver] worker={worker_id} session_index={session_index} "
        f"session_id={session_id} seed={seed.seed_id} template={seed.template} "
        f"group={sample_group_index} replica={sample_replica_index}",
        flush=True,
    )
    original_request = _rewrite_seed_task(simulated_user_client, seed, template_meta)
    record["original_request"] = original_request

    try:
        agent = Agent.create(str(config_path))
        opts = SessionOptions()
        opts.session_id = session_id
        opts.builtin_skills = BUILTIN_SKILLS
        opts.planning = PLANNING
        if PLANNING_MODE:
            opts.planning_mode = PLANNING_MODE
        opts.auto_compact = AUTO_COMPACT
        opts.auto_compact_threshold = AUTO_COMPACT_THRESHOLD
        opts.tool_timeout_ms = TOOL_TIMEOUT_MS
        opts.max_parse_retries = MAX_PARSE_RETRIES
        opts.max_tool_rounds = MAX_TOOL_ROUNDS
        opts.circuit_breaker_threshold = CIRCUIT_BREAKER_THRESHOLD
        opts.permission_policy = PermissionPolicy(default_decision="allow")
        opts.thinking_budget = THINKING_BUDGET if THINKING_BUDGET > 0 else None
        opts.continuation_enabled = CONTINUATION_ENABLED
        opts.max_continuation_turns = MAX_CONTINUATION_TURNS
        session = agent.session(str(workspace), opts)

        latest_response = ""
        for main_turn_number in range(1, MAX_MAIN_TURNS + 1):
            if main_turn_number == 1:
                user_prompt = original_request
                done_after_response = False
            else:
                user_prompt, done_after_response = _generate_followup(
                    simulated_user_client,
                    seed=seed,
                    template_meta=template_meta,
                    original_request=original_request,
                    latest_response=latest_response,
                    next_turn_number=main_turn_number,
                    is_final_turn=main_turn_number == MAX_MAIN_TURNS,
                )

            print(
                f"[a3s-code-driver] worker={worker_id} session_id={session_id} "
                f"turn={main_turn_number} prompt_chars={len(user_prompt)}",
                flush=True,
            )
            result = _send_with_timeout(session, user_prompt)
            latest_response = result.text
            print(
                f"[a3s-code-driver] worker={worker_id} session_id={session_id} "
                f"turn={main_turn_number} tool_calls={result.tool_calls_count} "
                f"response_chars={len(result.text)}",
                flush=True,
            )
            record["main_turns"].append(
                {
                    "turn": main_turn_number,
                    "user": user_prompt,
                    "assistant": result.text,
                    "tool_calls_count": result.tool_calls_count,
                    "done_after_response": done_after_response,
                }
            )
            if done_after_response:
                break

        verifier_result = _run_task_verifier(
            workspace=workspace,
            seed=seed,
            template_meta=template_meta,
        )
        if verifier_result is not None:
            record["task_verifier_reward"] = verifier_result
            _post_task_verifier_feedback(
                simulated_user_client,
                session_id,
                score=float(verifier_result["score"]),
                details=verifier_result["details"],
            )
            print(
                f"[a3s-code-driver] worker={worker_id} session_id={session_id} "
                f"task_verifier_score={float(verifier_result['score']):.4f}",
                flush=True,
            )

        _mark_session_done(simulated_user_client, session_id)
        record["status"] = "completed"
        print(
            f"[a3s-code-driver] worker={worker_id} session_id={session_id} status=completed "
            f"main_turns={len(record['main_turns'])}",
            flush=True,
        )
    except Exception as exc:
        failure_details = {
            "error": f"{type(exc).__name__}: {exc}",
            "status": "error",
            "template": template_meta.template,
            "seed_id": seed.seed_id,
            "session_index": session_index,
            "sample_group_index": sample_group_index,
            "sample_replica_index": sample_replica_index,
        }
        record["task_verifier_reward"] = {
            "score": -1.0,
            "details": failure_details,
        }
        try:
            _post_task_verifier_feedback(
                simulated_user_client,
                session_id,
                score=-1.0,
                details=failure_details,
            )
        except Exception:
            pass
        try:
            _mark_session_done(simulated_user_client, session_id)
        except Exception:
            pass
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(
            f"[a3s-code-driver] worker={worker_id} session_id={session_id} status=error "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
    finally:
        simulated_user_client.close()
        _append_record(record)
        should_keep_workspace = KEEP_WORKSPACES or (
            record["status"] != "completed" and KEEP_WORKSPACES_ON_ERROR
        )
        if not should_keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
        if config_is_ephemeral and not KEEP_CONFIGS:
            try:
                config_path.unlink(missing_ok=True)
            except TypeError:
                if config_path.exists():
                    config_path.unlink()


def _worker_main(worker_id: int, seeds: list[SeedTask]) -> None:
    while not SHUTDOWN_EVENT.is_set():
        if not _wait_for_rl_service(worker_id):
            return
        session_index = _next_session_index()
        if session_index is None:
            return
        try:
            if AGENT_ENV_BACKEND == "docker":
                _run_one_session_in_docker(worker_id, session_index)
            else:
                _run_one_session(worker_id, session_index, seeds)
            print(
                f"[a3s-code-driver] worker={worker_id} completed session_index={session_index}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[a3s-code-driver] worker={worker_id} crashed session_index={session_index} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
        time.sleep(SESSION_DELAY_SEC)


def main() -> None:
    _ensure_dirs()
    worker_docker = None
    if AGENT_ENV_BACKEND == "docker" and WORKER_LOCAL_DOCKER:
        from a3s_code_benchmarks.official.worker_local_docker import start_worker_local_docker

        worker_docker = start_worker_local_docker(log_dir=RESULTS_DIR / "worker_local_docker")
        print(
            f"[a3s-code-driver] worker_local_docker_active docker_host={worker_docker.docker_host}",
            flush=True,
        )
    seeds = _load_seed_tasks()
    backend_labels = [backend.label for backend in SIMULATED_USER_POOL.backends]
    print(
        "[a3s-code-driver] "
        f"rl_base={RL_BASE_URL} simulated_user_backends={backend_labels or ['fallback-only']} "
        f"simulated_user_backend_source={SIMULATED_USER_BACKENDS_SOURCE} "
        f"model={A3S_MODEL_NAME} "
        f"agent_config_mode={AGENT_CONFIG_MODE} "
        f"session_id_header={SESSION_ID_HEADER_NAME if AGENT_CONFIG_MODE == 'shared' else 'path'} "
        f"concurrency={CONCURRENCY} session_start_index={SESSION_START_INDEX} "
        f"session_limit={SESSION_LIMIT or 'inf'} "
        f"session_group_size={SESSION_GROUP_SIZE} "
        f"max_main_turns={MAX_MAIN_TURNS} max_tool_rounds={MAX_TOOL_ROUNDS} "
        f"agent_env_backend={AGENT_ENV_BACKEND} "
        f"agent_docker_image={AGENT_DOCKER_IMAGE if AGENT_ENV_BACKEND == 'docker' else 'n/a'} "
        f"tool_timeout_ms={TOOL_TIMEOUT_MS} turn_timeout_sec={TURN_TIMEOUT_SEC:.0f} "
        f"thinking_budget={THINKING_BUDGET} auto_compact={AUTO_COMPACT} "
        f"context={MODEL_CONTEXT_TOKENS} output={MODEL_OUTPUT_TOKENS} "
        f"rl_health_url={RL_HEALTH_URL} rl_unavailable_exit_sec={RL_UNAVAILABLE_EXIT_SEC:.0f} "
        f"sim_user_timeout_sec={SIMULATED_USER_TIMEOUT_SEC:.0f} "
        f"sim_user_cooldown_sec={SIMULATED_USER_BACKEND_COOLDOWN_SEC:.0f} "
        f"include_seed_tags={sorted(INCLUDED_SEED_TAGS) if INCLUDED_SEED_TAGS else 'all'} "
        f"record_file={RECORD_FILE}",
        flush=True,
    )

    threads = [
        threading.Thread(target=_worker_main, args=(worker_id, seeds), daemon=False)
        for worker_id in range(CONCURRENCY)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
