from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from a3s_code_benchmarks.benchmark_runtime_utils import (
    detect_model_base_url,
    ensure_a3s_code_wheel,
)


RESULT_BEGIN = "A3S_CODE_RESULT_BEGIN"
RESULT_END = "A3S_CODE_RESULT_END"
REMOTE_SKILLS_DIR = "/workspace/.skillsbench-skills"
REMOTE_AGENT_DIR = "/installed-agent"
REMOTE_RUNNER_PATH = f"{REMOTE_AGENT_DIR}/skillsbench_a3s_code_runner.py"
REMOTE_MODEL_API_KEY_PATH = f"{REMOTE_AGENT_DIR}/.model_api_key"
REMOTE_UV_PATH = f"{REMOTE_AGENT_DIR}/uv"
REMOTE_UV_REAL_PATH = "/root/.local/bin/uv-real"
REMOTE_UV_PYTHON_ARCHIVE_PATH = f"{REMOTE_AGENT_DIR}/uv-python-runtime.tar.gz"
REMOTE_AGENT_UV_PYTHON_ARCHIVE_PATH = f"{REMOTE_AGENT_DIR}/uv-python-agent-runtime.tar.gz"
REMOTE_UV_PYTHON_ROOT = "/opt/a3s-uv-python"
REMOTE_VENV_DIR = "/opt/a3s-code-venv"
REMOTE_VENV_PYTHON = f"{REMOTE_VENV_DIR}/bin/python"
REMOTE_AGENT_LOG_DIR = "/logs/agent"
REMOTE_A3S_LOG_DIR = f"{REMOTE_AGENT_LOG_DIR}/a3s"
DEFAULT_BENCHMARK_PROXY = "http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"
DISABLE_PROXY_VALUES = {"", "0", "false", "no", "none", "off", "direct"}
DEFAULT_NO_PROXY_ENTRIES = [
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
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
    "*.local",
    ".pjlab.org.cn",
    ".i.h.pjlab.org.cn",
    "mirrors.i.h.pjlab.org.cn",
    "pypi.i.h.pjlab.org.cn",
]
DEFAULT_MAVEN_NON_PROXY_HOSTS = (
    "localhost|127.*|0.0.0.0|10.*|100.*|172.16.*|192.168.*|"
    "*.local|*.pjlab.org.cn|*.i.h.pjlab.org.cn"
)
DISABLE_SKILLS_DIR_VALUES = {"", "0", "false", "no", "none", "null", "off", "disable", "disabled"}
DEFAULT_SKILLSBENCH_GUIDANCE = """\
任务模式: coding execution. You may edit files, run commands, install missing
dependencies when needed, and generate the requested benchmark artifacts.

You are running inside an official SkillsBench container. Treat the task as a
strict benchmark, not an open-ended coding request.

Before writing the final answer:
1. Read the task instruction and any machine-readable task files such as
   problem.json, config files, schemas, examples, reference directories, or
   README files that are present in the workspace.
2. Extract an exact output-path checklist before generating files. Produce every
   required artifact at the exact path requested by the task. Respect absolute
   paths such as /root/output or /app/output exactly. Do not replace requested
   names with near synonyms, singular/plural variants, or your own taxonomy
   such as meshes, parts, part_mesh, output_files, manifest.json, or links.json.
   If the task says links/*.obj, the files in links must be OBJ files.
   If an output tree shows nested placeholders such as
   part_meshes/<part_name>/<mesh_name>.obj, then <part_name> is a real directory
   level; do not flatten those files into part_meshes/.
   A links/ directory in an OBJ export task usually expects one OBJ file per
   part/link, not documentation files.
   Do not aggregate indexed parts such as cabin_0 ... cabin_15 into one plural
   directory like cabins/. Do not create extra top-level output files that are
   not on the checklist, because exact file-set tests may reject them.
3. After generation, run shell/Python checks against the exact checklist paths
   from step 2. A check against a similar directory does not count.
4. For each required output, verify existence, parseability, schema, shape,
   and key names using small local checks. If visible reference or ground-truth
   files are included in the task workspace, use them as validation oracles and
   fix mismatches before finishing.
5. When a reference, expected, answer, oracle, or ground_truth directory is
   visible, compare every generated artifact against the corresponding expected
   artifact. For problem.json tasks with .npy outputs, run a script that loads
   each output and reference file with numpy and asserts np.allclose with strict
   tolerances. Shape-only or "reasonable value" checks are not sufficient.
6. Do not stop after explaining a solution. Actually run the code/scripts that
   generate the artifacts, then inspect the generated files.
7. If a generated artifact is wrong, iterate until the local checks pass or the
   remaining blocker is explicit.
8. Before installing language or package dependencies, inspect the task image
   for preinstalled dependencies such as package.json, node_modules, virtualenvs,
   Maven caches, or system packages. Prefer those pinned, preinstalled packages
   unless a required dependency is actually missing. For Node.js tasks, check
   /workspace/node_modules and /root/node_modules before running npm install.
9. If external network access is needed from shell commands and standard proxy
   variables are unset, use the URL in A3S_CODE_BENCHMARK_PROXY for HTTP_PROXY
   and HTTPS_PROXY on that command.
"""
WHEEL_VERSION_RE = re.compile(r"^a3s_code-(?P<version>[^-]+)-")
WHEEL_PYTHON_TAG_RE = re.compile(r"^a3s_code-[^-]+-(?P<tag>cp\d+)-")
UV_PYTHON_RUNTIME_RE = re.compile(r"^cpython-(?P<version>3\.\d+)\.")
UV_PYTHON_REQUEST_RE = re.compile(
    r"(?:--python(?:=|\s+)|uv\s+python\s+install\s+|python(?:\s+|=|-))"
    r"(?P<version>3\.\d+)(?:\.\d+)?",
    re.IGNORECASE,
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _benchmark_proxy_url() -> str | None:
    for name in ("A3S_CODE_BENCHMARK_PROXY", "A3S_CODE_HTTP_PROXY", "BENCHMARK_HTTP_PROXY"):
        if name in os.environ:
            value = os.environ.get(name, "").strip()
            return None if value.lower() in DISABLE_PROXY_VALUES else value
    value = DEFAULT_BENCHMARK_PROXY.strip()
    return None if value.lower() in DISABLE_PROXY_VALUES else value


def _proxy_host_port(proxy_url: str) -> tuple[str, int]:
    parsed = urlparse(proxy_url)
    if not parsed.hostname:
        raise ValueError(f"Invalid benchmark proxy URL: {proxy_url!r}")
    if parsed.port:
        return parsed.hostname, parsed.port
    return parsed.hostname, 443 if parsed.scheme == "https" else 80


def _wheel_version(path: Path) -> str:
    match = WHEEL_VERSION_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot infer a3s-code version from wheel name: {path.name}")
    return match.group("version")


def _wheel_python_version(path: Path) -> str | None:
    match = WHEEL_PYTHON_TAG_RE.match(path.name)
    if not match:
        return None
    tag = match.group("tag")
    if not tag.startswith("cp") or len(tag) < 4:
        return None
    major = tag[2]
    minor = tag[3:]
    if major.isdigit() and minor.isdigit():
        return f"{major}.{int(minor)}"
    return None


def _runtime_version(path: Path) -> str | None:
    match = UV_PYTHON_RUNTIME_RE.match(path.name)
    return match.group("version") if match else None


def _available_uv_python_runtimes() -> dict[str, Path]:
    root = Path(
        os.getenv(
            "A3S_CODE_UV_PYTHON_RUNTIME_ROOT",
            str(Path.home() / ".local" / "share" / "uv" / "python"),
        )
    ).expanduser()
    if not root.exists():
        return {}
    runtimes: dict[str, Path] = {}
    for path in sorted(root.glob("cpython-3.*-linux-x86_64-gnu")):
        version = _runtime_version(path)
        if version and (path / "bin" / f"python{version}").exists():
            runtimes[version] = path
    return runtimes


def _ensure_uv_python_runtime_archive(runtime_dir: Path) -> Path:
    archive_root = Path(
        os.getenv(
            "A3S_CODE_UV_PYTHON_RUNTIME_ARCHIVE_DIR",
            str(Path(__file__).resolve().parents[2] / ".artifacts" / "uv_python_runtime"),
        )
    ).expanduser()
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / f"{runtime_dir.name}.tar.gz"
    if archive_path.exists() and archive_path.stat().st_mtime >= runtime_dir.stat().st_mtime:
        return archive_path
    tmp_path = archive_path.with_suffix(".tar.gz.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    subprocess.run(
        ["tar", "-C", str(runtime_dir.parent), "-czf", str(tmp_path), runtime_dir.name],
        check=True,
    )
    tmp_path.replace(archive_path)
    return archive_path


class A3SCodeHarbor(BaseInstalledAgent):
    """Harbor installed agent wrapper that runs tasks with a3s_code."""

    SUPPORTS_ATIF = False

    def __init__(self, *args, **kwargs):
        version = kwargs.pop("version", None) or os.getenv("A3S_CODE_VERSION")
        wheel_path = kwargs.pop("wheel_path", None) or os.getenv("A3S_CODE_WHEEL_PATH", "")
        super().__init__(*args, version=version, **kwargs)
        self._wheel_path = Path(wheel_path).expanduser().resolve() if wheel_path else ensure_a3s_code_wheel()
        wheel_version = _wheel_version(self._wheel_path)
        if _env_flag("A3S_CODE_UPLOAD_ALL_COMPATIBLE_WHEELS", False):
            self._wheel_paths = [
                path
                for path in sorted(self._wheel_path.parent.glob("a3s_code-*.whl"))
                if _wheel_version(path) == wheel_version
            ] or [self._wheel_path]
        else:
            self._wheel_paths = [self._wheel_path]

    @staticmethod
    def name() -> str:
        return "a3s-code"

    def version(self) -> str | None:
        return super().version() or os.getenv("A3S_CODE_VERSION")

    def _configured_skills_dir(self) -> str | None:
        value = getattr(self, "skills_dir", None)
        if value is None:
            return None
        text = str(value).strip()
        if text.lower() in DISABLE_SKILLS_DIR_VALUES:
            return ""
        return text

    @property
    def _install_agent_template_path(self) -> Path:
        # Unused: setup() is fully custom.
        return Path(__file__)

    async def install(self, environment: BaseEnvironment) -> None:
        # setup() is intentionally overridden below to upload local wheels and
        # runner code into the task container. This method only satisfies the
        # current BaseInstalledAgent abstract interface.
        return None

    async def _requested_uv_python_versions(self, environment: BaseEnvironment) -> list[str]:
        if not _env_flag("A3S_CODE_UPLOAD_UV_PYTHON_RUNTIME", True):
            return []
        versions = self._requested_uv_python_versions_from_task_config()
        if versions:
            return versions
        default_version = os.getenv("A3S_CODE_DEFAULT_UV_PYTHON_VERSION", "3.11").strip()
        if re.fullmatch(r"3\.\d+", default_version):
            return [default_version]
        command = r"""find -L /tests /home/travis /workspace -maxdepth 5 -type f -size -2000k -exec grep -Eho '(--python[= ]+|uv[[:space:]]+python[[:space:]]+install[[:space:]]+|python([[:space:]]|=|-))3\.[0-9]+(\.[0-9]+)?' {} + 2>/dev/null | sed -E 's/.*(3\.[0-9]+)(\.[0-9]+)?.*/\1/' | sort -u || true"""
        result = await environment.exec(command=command, timeout_sec=30)
        for line in (result.stdout or "").splitlines():
            cleaned = line.strip()
            if re.fullmatch(r"3\.\d+", cleaned) and cleaned not in versions:
                versions.append(cleaned)
        return versions

    def _requested_uv_python_versions_from_task_config(self) -> list[str]:
        config_path = self.logs_dir.parent / "config.json"
        if not config_path.exists():
            return []
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        task_path = Path(str(payload.get("task", {}).get("path", ""))).expanduser()
        if not task_path.exists():
            return []

        candidates: list[Path] = []
        for relative in ("tests", "solution"):
            root = task_path / relative
            if root.exists():
                for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
                    dirnames[:] = [
                        name
                        for name in dirnames
                        if name not in {".git", ".venv", "__pycache__", "node_modules"}
                    ]
                    for filename in filenames:
                        candidates.append(Path(dirpath) / filename)
        for relative in ("task.toml", "instruction.md", "environment/Dockerfile", "environment/README.md"):
            path = task_path / relative
            if path.exists():
                candidates.append(path)

        versions: list[str] = []
        seen_realpaths: set[Path] = set()
        for path in candidates:
            try:
                realpath = path.resolve()
                if realpath in seen_realpaths or path.stat().st_size > 2_000_000:
                    continue
                seen_realpaths.add(realpath)
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in UV_PYTHON_REQUEST_RE.finditer(text):
                version = match.group("version")
                if version not in versions:
                    versions.append(version)
        return versions

    async def _select_uv_python_runtime(self, environment: BaseEnvironment) -> tuple[str, str, Path] | None:
        requested_versions = await self._requested_uv_python_versions(environment)
        if not requested_versions:
            return None
        runtimes = _available_uv_python_runtimes()
        for requested in requested_versions:
            runtime = runtimes.get(requested)
            if runtime is not None:
                return requested, requested, runtime
        for requested in requested_versions:
            if requested == "3.10":
                for fallback in ("3.11", "3.12"):
                    runtime = runtimes.get(fallback)
                    if runtime is not None:
                        return requested, fallback, runtime
        for requested in requested_versions:
            for fallback in ("3.12", "3.11", "3.9", "3.8"):
                runtime = runtimes.get(fallback)
                if runtime is not None:
                    return requested, fallback, runtime
        return None

    @staticmethod
    def _model_api_key() -> str:
        return os.getenv("A3S_CODE_MODEL_API_KEY", os.getenv("SGLANG_API_KEY", "apiKey"))

    @staticmethod
    def _default_env() -> dict[str, str]:
        model_name = os.getenv("A3S_CODE_MODEL_NAME", os.getenv("SERVED_MODEL_NAME", "qwen3-4b-2507"))
        model_provider = os.getenv("A3S_CODE_MODEL_PROVIDER", "openai").strip() or "openai"
        base_url = detect_model_base_url()
        parsed_base_url = urlparse(base_url)
        proxy_url = _benchmark_proxy_url()
        runtime_proxy_env = _env_flag("A3S_CODE_AGENT_RUNTIME_PROXY", False)
        no_proxy_entries: list[str] = []
        for entry in DEFAULT_NO_PROXY_ENTRIES:
            if entry not in no_proxy_entries:
                no_proxy_entries.append(entry)
        for env_name in ("A3S_CODE_NO_PROXY", "NO_PROXY", "no_proxy"):
            raw = os.getenv(env_name, "").strip()
            if not raw:
                continue
            for entry in raw.split(","):
                cleaned = entry.strip()
                if cleaned == "*":
                    continue
                if cleaned and cleaned not in no_proxy_entries:
                    no_proxy_entries.append(cleaned)
        if (
            _env_flag("A3S_CODE_MODEL_NO_PROXY", True)
            and parsed_base_url.hostname
            and parsed_base_url.hostname not in no_proxy_entries
        ):
            no_proxy_entries.append(parsed_base_url.hostname)
        no_proxy = ",".join(no_proxy_entries)
        maven_non_proxy_hosts = os.getenv("A3S_CODE_MAVEN_NON_PROXY_HOSTS", DEFAULT_MAVEN_NON_PROXY_HOSTS)
        env = {
            "A3S_CODE_BUILTIN_SKILLS": os.getenv("A3S_CODE_BUILTIN_SKILLS", "true"),
            "A3S_CODE_PLANNING": os.getenv("A3S_CODE_PLANNING", "false"),
            "A3S_CODE_PERMISSIVE": os.getenv("A3S_CODE_PERMISSIVE", "true"),
            "A3S_CODE_MODEL_PROVIDER": model_provider,
            "A3S_CODE_MODEL_NAME": model_name,
            "A3S_CODE_MODEL_API_KEY_FILE": REMOTE_MODEL_API_KEY_PATH,
            "A3S_CODE_MODEL_BASE_URL": base_url,
            "A3S_CODE_SESSION_ID_HEADER": os.getenv("A3S_CODE_SESSION_ID_HEADER", "X-Session-Id"),
            "A3S_CODE_CONTEXT_TOKENS": os.getenv("A3S_CODE_CONTEXT_TOKENS", "131072"),
            "A3S_CODE_OUTPUT_TOKENS": os.getenv("A3S_CODE_OUTPUT_TOKENS", "8192"),
            "A3S_CODE_THINKING_BUDGET": os.getenv("A3S_CODE_THINKING_BUDGET", "32000"),
            "A3S_CODE_OPENAI_ENABLE_THINKING": os.getenv("A3S_CODE_OPENAI_ENABLE_THINKING", ""),
            "A3S_CODE_OPENAI_EXTRA_BODY_JSON": os.getenv("A3S_CODE_OPENAI_EXTRA_BODY_JSON", ""),
            "A3S_CODE_DISABLE_STREAMING": os.getenv("A3S_CODE_DISABLE_STREAMING", "0"),
            "A3S_CODE_LLM_STREAM_IDLE_TIMEOUT_SEC": os.getenv("A3S_CODE_LLM_STREAM_IDLE_TIMEOUT_SEC", "900"),
            "A3S_CODE_MAX_TOOL_ROUNDS": os.getenv("A3S_CODE_MAX_TOOL_ROUNDS", "64"),
            "A3S_CODE_TOOL_TIMEOUT_MS": os.getenv("A3S_CODE_TOOL_TIMEOUT_MS", "300000"),
            "A3S_CODE_MAX_PARSE_RETRIES": os.getenv("A3S_CODE_MAX_PARSE_RETRIES", "4"),
            "A3S_CODE_CIRCUIT_BREAKER_THRESHOLD": os.getenv("A3S_CODE_CIRCUIT_BREAKER_THRESHOLD", "5"),
            "A3S_CODE_MAX_EXECUTION_TIME_MS": os.getenv("A3S_CODE_MAX_EXECUTION_TIME_MS", "0"),
            "A3S_CODE_PROXY_BRIDGE_TIMEOUT_SEC": os.getenv("A3S_CODE_PROXY_BRIDGE_TIMEOUT_SEC", "0"),
            "A3S_CODE_PROXY_BRIDGE_CONNECT_TIMEOUT_SEC": os.getenv(
                "A3S_CODE_PROXY_BRIDGE_CONNECT_TIMEOUT_SEC",
                "120",
            ),
            "A3S_CODE_WORKSPACE": os.getenv("A3S_CODE_WORKSPACE", "/workspace"),
            "A3S_CODE_MAVEN_NETWORK_TIMEOUT_MS": os.getenv("A3S_CODE_MAVEN_NETWORK_TIMEOUT_MS", "15000"),
            "A3S_CODE_MAVEN_COMMAND_TIMEOUT_SEC": os.getenv("A3S_CODE_MAVEN_COMMAND_TIMEOUT_SEC", "120"),
            "A3S_CODE_MAVEN_MIRROR_URL": os.getenv("A3S_CODE_MAVEN_MIRROR_URL", ""),
            "A3S_CODE_MAVEN_MIRROR_OF": os.getenv("A3S_CODE_MAVEN_MIRROR_OF", "central"),
            "NODE_PATH": os.getenv(
                "A3S_CODE_NODE_PATH",
                "/workspace/node_modules:/root/node_modules:/app/node_modules",
            ),
            "UV_INDEX_STRATEGY": os.getenv("UV_INDEX_STRATEGY", "unsafe-best-match"),
            "A3S_CODE_TRACE_PATH": os.getenv("A3S_CODE_TRACE_PATH", f"{REMOTE_A3S_LOG_DIR}/messages.jsonl"),
            "A3S_CODE_SESSION_STORE_DIR": os.getenv("A3S_CODE_SESSION_STORE_DIR", f"{REMOTE_A3S_LOG_DIR}/sessions"),
            "A3S_CODE_RUN_METADATA_PATH": os.getenv("A3S_CODE_RUN_METADATA_PATH", f"{REMOTE_A3S_LOG_DIR}/run_metadata.json"),
            "A3S_CODE_WORKSPACE_MANIFEST_PATH": os.getenv(
                "A3S_CODE_WORKSPACE_MANIFEST_PATH",
                f"{REMOTE_A3S_LOG_DIR}/workspace_manifest.json",
            ),
            "A3S_CODE_WORKSPACE_MANIFEST_MAX_FILES": os.getenv(
                "A3S_CODE_WORKSPACE_MANIFEST_MAX_FILES",
                "5000",
            ),
            "A3S_CODE_AUTO_SAVE_SESSION": os.getenv("A3S_CODE_AUTO_SAVE_SESSION", "true"),
            "A3S_CODE_AGENT_RUNTIME_PROXY": "1" if runtime_proxy_env else "0",
            "A3S_CODE_AGENT_PROXY_MODE": os.getenv("A3S_CODE_AGENT_PROXY_MODE", "bridge"),
            "A3S_CODE_NO_PROXY": no_proxy,
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }
        if proxy_url:
            proxy_host, proxy_port = _proxy_host_port(proxy_url)
            pip_extra_index_url = os.getenv(
                "A3S_CODE_PIP_EXTRA_INDEX_URL",
                "http://pypi.i.h.pjlab.org.cn/brain/dev/+simple",
            )
            pip_trusted_host = os.getenv(
                "A3S_CODE_PIP_TRUSTED_HOST",
                "mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn",
            )
            maven_opts = (
                f"-Dhttp.proxyHost={proxy_host} -Dhttp.proxyPort={proxy_port} "
                f"-Dhttps.proxyHost={proxy_host} -Dhttps.proxyPort={proxy_port} "
                f"-Dhttp.nonProxyHosts={maven_non_proxy_hosts} "
                f"-Dhttps.nonProxyHosts={maven_non_proxy_hosts}"
            )
            java_proxy_opts = os.getenv("A3S_CODE_JAVA_PROXY_OPTS", maven_opts)
            env.update(
                {
                    "A3S_CODE_BENCHMARK_PROXY": proxy_url,
                    "MAVEN_OPTS": os.getenv("A3S_CODE_MAVEN_OPTS", maven_opts),
                    "JAVA_OPTS": os.getenv("A3S_CODE_JAVA_OPTS", java_proxy_opts),
                    "SBT_OPTS": os.getenv("A3S_CODE_SBT_OPTS", java_proxy_opts),
                    "COURSIER_OPTS": os.getenv("A3S_CODE_COURSIER_OPTS", java_proxy_opts),
                    "PIP_INDEX_URL": os.getenv(
                        "A3S_CODE_PIP_INDEX_URL",
                        "http://mirrors.i.h.pjlab.org.cn/pypi/simple/",
                    ),
                    "PIP_EXTRA_INDEX_URL": pip_extra_index_url,
                    "PIP_TRUSTED_HOST": pip_trusted_host,
                    "PIP_DEFAULT_TIMEOUT": os.getenv("A3S_CODE_PIP_DEFAULT_TIMEOUT", "120"),
                    "UV_INDEX_URL": os.getenv(
                        "A3S_CODE_UV_INDEX_URL",
                        "http://mirrors.i.h.pjlab.org.cn/pypi/simple/",
                    ),
                    "UV_EXTRA_INDEX_URL": os.getenv(
                        "A3S_CODE_UV_EXTRA_INDEX_URL",
                        pip_extra_index_url,
                    ),
                    "UV_INSECURE_HOST": os.getenv("A3S_CODE_UV_INSECURE_HOST", pip_trusted_host),
                }
            )
            if runtime_proxy_env:
                env.update(
                    {
                        "HTTP_PROXY": proxy_url,
                        "http_proxy": proxy_url,
                        "HTTPS_PROXY": proxy_url,
                        "https_proxy": proxy_url,
                        "ALL_PROXY": proxy_url,
                        "all_proxy": proxy_url,
                    }
                )
            else:
                env.update(
                    {
                        "BASH_ENV": "/etc/profile.d/a3s-code-proxy.sh",
                        "SHELL": "/bin/bash",
                        "A3S_CODE_SHELL_PROXY_ENV": "/etc/profile.d/a3s-code-proxy.sh",
                    }
                )
        else:
            env["A3S_CODE_BENCHMARK_PROXY"] = ""
        return env

    @staticmethod
    def _default_skill_dirs() -> list[str]:
        candidates = [
            REMOTE_SKILLS_DIR,
            "/root/.codex/skills",
            "/root/.claude/skills",
            "/root/.agents/skills",
            "/root/.factory/skills",
            "/root/.goose/skills",
            "/root/.gemini/skills",
            "/app/skills",
            "environment/skills",
        ]
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    @staticmethod
    def _jsonable(value):
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): A3SCodeHarbor._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [A3SCodeHarbor._jsonable(v) for v in value]
        if hasattr(value, "model_dump"):
            return A3SCodeHarbor._jsonable(value.model_dump())
        if hasattr(value, "dict"):
            return A3SCodeHarbor._jsonable(value.dict())
        if hasattr(value, "__dict__"):
            return A3SCodeHarbor._jsonable(vars(value))
        return str(value)

    @staticmethod
    def _augment_instruction(instruction: str) -> str:
        if not _env_flag("A3S_CODE_SKILLSBENCH_GUIDANCE", True):
            return instruction

        extra_guidance = os.getenv("A3S_CODE_EXTRA_SKILLSBENCH_GUIDANCE", "").strip()
        sections = [DEFAULT_SKILLSBENCH_GUIDANCE.strip()]
        if extra_guidance:
            sections.append(extra_guidance)
        sections.append("Original task instruction:\n" + instruction.strip())
        return "\n\n".join(sections).strip() + "\n"

    async def _upload_skills_dir(self, environment: BaseEnvironment, source_dir: Path) -> str:
        await environment.exec(command=f"mkdir -p {REMOTE_SKILLS_DIR}")
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_dir).as_posix()
            target_path = f"{REMOTE_SKILLS_DIR}/{relative}"
            parent = str(Path(target_path).parent)
            await environment.exec(command=f"mkdir -p {parent}")
            await environment.upload_file(source_path=path, target_path=target_path)
        return REMOTE_SKILLS_DIR

    async def setup(self, environment: BaseEnvironment) -> None:
        remote_wheels_dir = f"{REMOTE_AGENT_DIR}/wheels"
        wheel_version = _wheel_version(self._wheel_path)
        await environment.exec(command=f"mkdir -p {REMOTE_AGENT_DIR}")
        await environment.exec(command=f"mkdir -p {remote_wheels_dir}")
        tmp_key_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp_key:
                tmp_key.write(self._model_api_key())
                tmp_key.write("\n")
                tmp_key_path = Path(tmp_key.name)
            await environment.upload_file(tmp_key_path, REMOTE_MODEL_API_KEY_PATH)
            await environment.exec(command=f"chmod 600 {shlex.quote(REMOTE_MODEL_API_KEY_PATH)}")
        finally:
            if tmp_key_path is not None:
                tmp_key_path.unlink(missing_ok=True)
        for wheel_path in self._wheel_paths:
            await environment.upload_file(wheel_path, f"{remote_wheels_dir}/{wheel_path.name}")
        await environment.upload_file(Path(__file__).with_name("skillsbench_a3s_code_runner.py"), REMOTE_RUNNER_PATH)
        uv_path = Path(os.getenv("A3S_CODE_UV_BIN", shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv"))).expanduser()
        if uv_path.exists():
            await environment.upload_file(uv_path, REMOTE_UV_PATH)
        runtime_requested = ""
        runtime_actual = ""
        runtime_dirname = ""
        agent_runtime_requested = ""
        agent_runtime_actual = ""
        agent_runtime_dirname = ""
        runtime = await self._select_uv_python_runtime(environment)
        if runtime is not None:
            runtime_requested, runtime_actual, runtime_dir = runtime
            runtime_dirname = runtime_dir.name
            runtime_archive = _ensure_uv_python_runtime_archive(runtime_dir)
            await environment.upload_file(runtime_archive, REMOTE_UV_PYTHON_ARCHIVE_PATH)
        agent_runtime_version = (
            os.getenv("A3S_CODE_AGENT_UV_PYTHON_VERSION", "").strip()
            or _wheel_python_version(self._wheel_path)
            or os.getenv("A3S_CODE_DEFAULT_UV_PYTHON_VERSION", "3.11").strip()
        )
        agent_runtime_dir = _available_uv_python_runtimes().get(agent_runtime_version)
        if agent_runtime_dir is not None:
            agent_runtime_requested = agent_runtime_version
            agent_runtime_actual = agent_runtime_version
            agent_runtime_dirname = agent_runtime_dir.name
            agent_runtime_archive = _ensure_uv_python_runtime_archive(agent_runtime_dir)
            await environment.upload_file(agent_runtime_archive, REMOTE_AGENT_UV_PYTHON_ARCHIVE_PATH)

        install_proxy_url = _benchmark_proxy_url() or ""
        install_proxy_host = "httpproxy-headless.kubebrain.svc.pjlab.local"
        install_proxy_port = 3128
        if install_proxy_url:
            install_proxy_host, install_proxy_port = _proxy_host_port(install_proxy_url)
        install_script = f"""#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy
benchmark_proxy={shlex.quote(install_proxy_url)}
case "${{benchmark_proxy,,}}" in
  ""|0|false|no|none|off|direct) benchmark_proxy="" ;;
esac
proxy_host="${{A3S_CODE_BENCHMARK_PROXY_HOST:-{install_proxy_host}}}"
proxy_port="${{A3S_CODE_BENCHMARK_PROXY_PORT:-{install_proxy_port}}}"
maven_non_proxy_hosts="${{A3S_CODE_MAVEN_NON_PROXY_HOSTS:-{DEFAULT_MAVEN_NON_PROXY_HOSTS}}}"
maven_network_timeout_ms="${{A3S_CODE_MAVEN_NETWORK_TIMEOUT_MS:-15000}}"
maven_mirror_url="${{A3S_CODE_MAVEN_MIRROR_URL:-}}"
maven_mirror_of="${{A3S_CODE_MAVEN_MIRROR_OF:-central}}"
mirror_block=""
if [ -n "$maven_mirror_url" ]; then
  mirror_block="$(cat <<MIRROR
  <mirrors>
    <mirror>
      <id>a3s-code-maven-mirror</id>
      <name>a3s-code configured Maven mirror</name>
      <url>$maven_mirror_url</url>
      <mirrorOf>$maven_mirror_of</mirrorOf>
    </mirror>
  </mirrors>
MIRROR
)"
fi
configure_a3s_apt_sources() {{
  if [ ! -r /etc/os-release ]; then
    return 0
  fi
  . /etc/os-release
  mkdir -p /etc/apt/apt.conf.d /etc/apt/sources.list.d
  if [ "${{ID:-}}" = "ubuntu" ] && [ -n "${{VERSION_CODENAME:-}}" ]; then
    mirror="${{A3S_CODE_UBUNTU_APT_MIRROR:-http://mirrors.i.h.pjlab.org.cn/ubuntu}}"
    signed_by_line=""
    if [ -r /usr/share/keyrings/ubuntu-archive-keyring.gpg ]; then
      signed_by_line="Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg"
    fi
    rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources
    cat > /etc/apt/sources.list.d/a3s-ubuntu.sources <<EOF
Types: deb
URIs: $mirror
Suites: $VERSION_CODENAME $VERSION_CODENAME-security $VERSION_CODENAME-updates $VERSION_CODENAME-backports
Components: main restricted universe multiverse
$signed_by_line
EOF
  elif [ "${{ID:-}}" = "debian" ] && [ -n "${{VERSION_CODENAME:-}}" ]; then
    debian_mirror="${{A3S_CODE_DEBIAN_APT_MIRROR:-http://mirrors.tuna.tsinghua.edu.cn/debian}}"
    debian_security_mirror="${{A3S_CODE_DEBIAN_SECURITY_APT_MIRROR:-http://mirrors.tuna.tsinghua.edu.cn/debian-security}}"
    for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
      [ -e "$file" ] || continue
      sed -i \
        -e "s#http://deb.debian.org/debian-security#$debian_security_mirror#g" \
        -e "s#https://deb.debian.org/debian-security#$debian_security_mirror#g" \
        -e "s#http://security.debian.org/debian-security#$debian_security_mirror#g" \
        -e "s#https://security.debian.org/debian-security#$debian_security_mirror#g" \
        -e "s#http://deb.debian.org/debian#$debian_mirror#g" \
        -e "s#https://deb.debian.org/debian#$debian_mirror#g" \
        "$file"
    done
  fi
  if [ -n "$benchmark_proxy" ]; then
    printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' "$benchmark_proxy" "$benchmark_proxy" > /etc/apt/apt.conf.d/99a3s-benchmark-proxy
    rm -f /etc/apt/apt.conf.d/99a3s-direct-proxy
  else
    printf '%s\n' 'Acquire::http::Proxy "false";' 'Acquire::https::Proxy "false";' > /etc/apt/apt.conf.d/99a3s-direct-proxy
    rm -f /etc/apt/apt.conf.d/99a3s-benchmark-proxy
  fi
}}
configure_a3s_apt_sources
cat > /etc/pip.conf <<'EOF'
[global]
index-url = http://mirrors.i.h.pjlab.org.cn/pypi/simple/
extra-index-url = http://pypi.i.h.pjlab.org.cn/brain/dev/+simple
trusted-host =
    mirrors.i.h.pjlab.org.cn
    pypi.i.h.pjlab.org.cn
disable-pip-version-check = true
EOF
mkdir -p /root/.local/bin /usr/local/bin
link_a3s_python_if_missing() {{
  local target="$1"
  local name="$2"
  [ -n "$target" ] || return 0
  [ -n "$name" ] || return 0
  if command -v "$name" >/dev/null 2>&1; then
    return 0
  fi
  ln -sf "$target" "/usr/local/bin/$name"
}}
python_runtime_requested={shlex.quote(runtime_requested)}
python_runtime_actual={shlex.quote(runtime_actual)}
python_runtime_dirname={shlex.quote(runtime_dirname)}
agent_runtime_requested={shlex.quote(agent_runtime_requested)}
agent_runtime_actual={shlex.quote(agent_runtime_actual)}
agent_runtime_dirname={shlex.quote(agent_runtime_dirname)}
python_runtime_bin=""
agent_runtime_bin=""
if [ -n "$python_runtime_dirname" ] && [ -s {REMOTE_UV_PYTHON_ARCHIVE_PATH} ]; then
  mkdir -p {REMOTE_UV_PYTHON_ROOT}
  tar -xzf {REMOTE_UV_PYTHON_ARCHIVE_PATH} -C {REMOTE_UV_PYTHON_ROOT}
  python_runtime_home="{REMOTE_UV_PYTHON_ROOT}/$python_runtime_dirname"
  if [ -x "$python_runtime_home/bin/python$python_runtime_actual" ]; then
    python_runtime_bin="$python_runtime_home/bin/python$python_runtime_actual"
  elif [ -x "$python_runtime_home/bin/python3" ]; then
    python_runtime_bin="$python_runtime_home/bin/python3"
  elif [ -x "$python_runtime_home/bin/python" ]; then
    python_runtime_bin="$python_runtime_home/bin/python"
  fi
  if [ -n "$python_runtime_bin" ]; then
    link_a3s_python_if_missing "$python_runtime_bin" "python$python_runtime_requested"
    link_a3s_python_if_missing "$python_runtime_bin" "python$python_runtime_actual"
    link_a3s_python_if_missing "$python_runtime_bin" python3
  fi
fi
if [ -n "$agent_runtime_dirname" ] && [ -s {REMOTE_AGENT_UV_PYTHON_ARCHIVE_PATH} ]; then
  mkdir -p {REMOTE_UV_PYTHON_ROOT}
  tar -xzf {REMOTE_AGENT_UV_PYTHON_ARCHIVE_PATH} -C {REMOTE_UV_PYTHON_ROOT}
  agent_runtime_home="{REMOTE_UV_PYTHON_ROOT}/$agent_runtime_dirname"
  if [ -x "$agent_runtime_home/bin/python$agent_runtime_actual" ]; then
    agent_runtime_bin="$agent_runtime_home/bin/python$agent_runtime_actual"
  elif [ -x "$agent_runtime_home/bin/python3" ]; then
    agent_runtime_bin="$agent_runtime_home/bin/python3"
  elif [ -x "$agent_runtime_home/bin/python" ]; then
    agent_runtime_bin="$agent_runtime_home/bin/python"
  fi
fi
if [ -s {REMOTE_UV_PATH} ]; then
  cp {REMOTE_UV_PATH} {REMOTE_UV_REAL_PATH}
  chmod +x {REMOTE_UV_REAL_PATH}
  {{
    printf 'A3S_UV_RUNTIME_BIN=%q\n' "$python_runtime_bin"
    printf 'A3S_UV_RUNTIME_REQUESTED=%q\n' "$python_runtime_requested"
  }} > /root/.local/bin/uv-runtime.env
  cat > /root/.local/bin/uv <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export PIP_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export UV_DEFAULT_INDEX="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export UV_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export UV_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export UV_INSECURE_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export UV_INDEX_STRATEGY="${{UV_INDEX_STRATEGY:-unsafe-best-match}}"
export UV_PYTHON_DOWNLOADS=never
if [ -f /root/.local/bin/uv-runtime.env ]; then
  . /root/.local/bin/uv-runtime.env
fi
runtime_bin="${{A3S_UV_RUNTIME_BIN:-}}"
runtime_requested="${{A3S_UV_RUNTIME_REQUESTED:-}}"
matches_runtime() {{
  local requested="$1"
  [ -n "$runtime_bin" ] || return 1
  [ -n "$runtime_requested" ] || return 1
  case "$requested" in
    "$runtime_requested"|"$runtime_requested".*) return 0 ;;
    *) return 1 ;;
  esac
}}
if [ "${{1:-}}" = "python" ] && [ "${{2:-}}" = "install" ] && [ -n "$runtime_bin" ]; then
  shift 2
  for requested in "$@"; do
    if matches_runtime "$requested"; then
      exit 0
    fi
  done
  set -- python install "$@"
fi
rewritten=()
rewrite_next=0
for arg in "$@"; do
  if [ "$rewrite_next" = "1" ]; then
    if matches_runtime "$arg"; then
      rewritten+=("$runtime_bin")
    else
      rewritten+=("$arg")
    fi
    rewrite_next=0
    continue
  fi
  case "$arg" in
    --python|-p)
      rewritten+=("$arg")
      rewrite_next=1
      ;;
    --python=*)
      requested="${{arg#--python=}}"
      if matches_runtime "$requested"; then
        rewritten+=("--python=$runtime_bin")
      else
        rewritten+=("$arg")
      fi
      ;;
    *)
      rewritten+=("$arg")
      ;;
  esac
done
exec {REMOTE_UV_REAL_PATH} "${{rewritten[@]}}"
EOF
  chmod +x /root/.local/bin/uv
  cat > /root/.local/bin/uvx <<'EOF'
#!/usr/bin/env sh
exec /root/.local/bin/uv tool run "$@"
EOF
  chmod +x /root/.local/bin/uvx
fi
cat > /root/.local/bin/env <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
export PIP_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export UV_DEFAULT_INDEX="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export UV_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export UV_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export UV_INSECURE_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export UV_INDEX_STRATEGY="${{UV_INDEX_STRATEGY:-unsafe-best-match}}"
export UV_PYTHON_DOWNLOADS=never
EOF
if [ -x /usr/bin/curl ]; then
  cat > /usr/local/bin/curl <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
  case "$arg" in
    https://astral.sh/uv/*/install.sh|http://astral.sh/uv/*/install.sh)
      cat <<'INSTALL'
#!/usr/bin/env sh
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/env" <<'ENV'
export PATH="$HOME/.local/bin:$PATH"
export PIP_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export UV_DEFAULT_INDEX="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export UV_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export UV_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export UV_INSECURE_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export UV_INDEX_STRATEGY="${{UV_INDEX_STRATEGY:-unsafe-best-match}}"
export UV_PYTHON_DOWNLOADS=never
ENV
exit 0
INSTALL
      exit 0
      ;;
  esac
done
exec /usr/bin/curl "$@"
EOF
  chmod +x /usr/local/bin/curl
fi
agent_python_bin="$agent_runtime_bin"
if [ -z "$agent_python_bin" ]; then
  agent_python_bin="$python_runtime_bin"
fi
if [ -z "$agent_python_bin" ]; then
  if ! command -v python3 >/dev/null 2>&1 || ! python3 -m venv /tmp/a3s-code-venv-check >/dev/null 2>&1
  then
    apt-get update
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv
    rm -rf /var/lib/apt/lists/*
  fi
  agent_python_bin="$(command -v python3)"
elif [ -z "$python_runtime_bin" ]; then
  link_a3s_python_if_missing "$agent_python_bin" python3
fi
if [ -z "$agent_python_bin" ] || ! "$agent_python_bin" -m venv /tmp/a3s-code-venv-check >/dev/null 2>&1
then
  echo "No usable Python runtime for a3s-code agent venv" >&2
  exit 1
fi
mkdir -p /root/.m2
cat > /usr/local/bin/mvn <<'EOF'
#!/usr/bin/env bash
timeout_ms="${{A3S_CODE_MAVEN_NETWORK_TIMEOUT_MS:-15000}}"
command_timeout_sec="${{A3S_CODE_MAVEN_COMMAND_TIMEOUT_SEC:-120}}"
export MAVEN_OPTS="${{MAVEN_OPTS:-}} -Dsun.net.client.defaultConnectTimeout=$timeout_ms -Dsun.net.client.defaultReadTimeout=$timeout_ms -Dmaven.wagon.rto=$timeout_ms -Dmaven.wagon.http.retryHandler.count=1"
if command -v timeout >/dev/null 2>&1; then
  exec timeout --preserve-status "$command_timeout_sec" /usr/bin/mvn "$@"
fi
exec /usr/bin/mvn "$@"
EOF
chmod +x /usr/local/bin/mvn
cat > /root/.mavenrc <<'EOF'
timeout_ms="${{A3S_CODE_MAVEN_NETWORK_TIMEOUT_MS:-15000}}"
export MAVEN_OPTS="${{MAVEN_OPTS:-}} -Dsun.net.client.defaultConnectTimeout=$timeout_ms -Dsun.net.client.defaultReadTimeout=$timeout_ms -Dmaven.wagon.rto=$timeout_ms -Dmaven.wagon.http.retryHandler.count=1"
EOF
mkdir -p /workspace
if [ -d /root/node_modules ]; then
  if [ ! -e /workspace/node_modules ]; then
    ln -s /root/node_modules /workspace/node_modules
  fi
  if [ -d /app ] && [ ! -e /app/node_modules ]; then
    ln -s /root/node_modules /app/node_modules
  fi
fi
if [ -n "$benchmark_proxy" ]; then
cat > /root/.m2/settings.xml <<EOF
<settings>
${{mirror_block}}
  <proxies>
    <proxy>
      <id>a3s-code-http</id>
      <active>true</active>
      <protocol>http</protocol>
      <host>$proxy_host</host>
      <port>$proxy_port</port>
      <nonProxyHosts>$maven_non_proxy_hosts</nonProxyHosts>
    </proxy>
    <proxy>
      <id>a3s-code-https</id>
      <active>true</active>
      <protocol>https</protocol>
      <host>$proxy_host</host>
      <port>$proxy_port</port>
      <nonProxyHosts>$maven_non_proxy_hosts</nonProxyHosts>
    </proxy>
  </proxies>
</settings>
EOF
cat > /etc/profile.d/a3s-code-proxy.sh <<EOF
export HTTP_PROXY="$benchmark_proxy"
export http_proxy="$benchmark_proxy"
export HTTPS_PROXY="$benchmark_proxy"
export https_proxy="$benchmark_proxy"
export ALL_PROXY="$benchmark_proxy"
export all_proxy="$benchmark_proxy"
export NO_PROXY="${{A3S_CODE_NO_PROXY:-{','.join(DEFAULT_NO_PROXY_ENTRIES)}}}"
export no_proxy="${{A3S_CODE_NO_PROXY:-{','.join(DEFAULT_NO_PROXY_ENTRIES)}}}"
export MAVEN_OPTS="-Dhttp.proxyHost=$proxy_host -Dhttp.proxyPort=$proxy_port -Dhttps.proxyHost=$proxy_host -Dhttps.proxyPort=$proxy_port -Dhttp.nonProxyHosts=$maven_non_proxy_hosts -Dhttps.nonProxyHosts=$maven_non_proxy_hosts"
export JAVA_OPTS="-Dhttp.proxyHost=$proxy_host -Dhttp.proxyPort=$proxy_port -Dhttps.proxyHost=$proxy_host -Dhttps.proxyPort=$proxy_port -Dhttp.nonProxyHosts=$maven_non_proxy_hosts -Dhttps.nonProxyHosts=$maven_non_proxy_hosts"
export SBT_OPTS="-Dhttp.proxyHost=$proxy_host -Dhttp.proxyPort=$proxy_port -Dhttps.proxyHost=$proxy_host -Dhttps.proxyPort=$proxy_port -Dhttp.nonProxyHosts=$maven_non_proxy_hosts -Dhttps.nonProxyHosts=$maven_non_proxy_hosts"
export COURSIER_OPTS="-Dhttp.proxyHost=$proxy_host -Dhttp.proxyPort=$proxy_port -Dhttps.proxyHost=$proxy_host -Dhttps.proxyPort=$proxy_port -Dhttp.nonProxyHosts=$maven_non_proxy_hosts -Dhttps.nonProxyHosts=$maven_non_proxy_hosts"
export PLAYWRIGHT_DOWNLOAD_HOST="${{A3S_CODE_PLAYWRIGHT_DOWNLOAD_HOST:-https://playwright-akamai.azureedge.net}}"
export NODE_PATH="${{NODE_PATH:-/workspace/node_modules:/root/node_modules:/app/node_modules}}"
EOF
grep -q "a3s-code-proxy.sh" /root/.bashrc 2>/dev/null || echo '. /etc/profile.d/a3s-code-proxy.sh' >> /root/.bashrc
else
cat > /root/.m2/settings.xml <<EOF
<settings>
${{mirror_block}}
</settings>
EOF
rm -f /etc/profile.d/a3s-code-proxy.sh
if [ -f /root/.bashrc ]; then
  sed -i '/a3s-code-proxy\\.sh/d' /root/.bashrc
fi
fi
rm -rf /tmp/a3s-code-venv-check {REMOTE_VENV_DIR}
"$agent_python_bin" -m venv {REMOTE_VENV_DIR}
{REMOTE_VENV_PYTHON} -m pip install --no-index --find-links {remote_wheels_dir} a3s-code=={wheel_version}
chmod +x /installed-agent/skillsbench_a3s_code_runner.py
"""
        local_script = self.logs_dir / "install-a3s-code.sh"
        local_script.write_text(install_script, encoding="utf-8")
        await environment.upload_file(local_script, f"{REMOTE_AGENT_DIR}/install-a3s-code.sh")
        install_result = await environment.exec(command=f"bash {REMOTE_AGENT_DIR}/install-a3s-code.sh", timeout_sec=1800)
        install_log_path = self.logs_dir / "install-a3s-code.log"
        install_log_path.write_text(
            "\n".join(
                [
                    f"return_code={install_result.return_code}",
                    "stdout:",
                    install_result.stdout or "",
                    "stderr:",
                    install_result.stderr or "",
                ]
            ),
            encoding="utf-8",
        )
        if install_result.return_code != 0:
            raise RuntimeError(
                "Failed to install a3s-code in SkillsBench environment. "
                f"See {install_log_path} for details."
            )

        skills_dir = self._configured_skills_dir()
        if skills_dir:
            source_dir = Path(skills_dir)
            if source_dir.exists():
                await self._upload_skills_dir(environment, source_dir)

    def _create_run_agent_command(self, instruction: str) -> tuple[str, dict[str, str], str | None, int | None]:
        env = self._default_env()
        augmented_instruction = self._augment_instruction(instruction)
        encoded_instruction = base64.b64encode(augmented_instruction.encode("utf-8")).decode("ascii")
        env["A3S_CODE_INSTRUCTION_B64"] = encoded_instruction
        env["A3S_CODE_INSTRUCTION_SHA256"] = hashlib.sha256(
            augmented_instruction.encode("utf-8")
        ).hexdigest()
        env["A3S_CODE_SESSION_ID"] = os.getenv("A3S_CODE_SESSION_ID", f"skillsbench-{self.logs_dir.parent.name}")

        skills_dir = self._configured_skills_dir()
        if skills_dir:
            source_dir = Path(skills_dir)
            env["A3S_CODE_SKILL_DIRS_JSON"] = json.dumps(
                [REMOTE_SKILLS_DIR if source_dir.exists() else str(skills_dir)]
            )
        elif skills_dir == "":
            env["A3S_CODE_SKILL_DIRS_JSON"] = "[]"
        else:
            env["A3S_CODE_SKILL_DIRS_JSON"] = json.dumps(self._default_skill_dirs())

        mcp_servers = getattr(self, "mcp_servers", None)
        if mcp_servers:
            env["A3S_CODE_MCP_SERVERS_JSON"] = json.dumps(self._jsonable(mcp_servers))

        env["A3S_CODE_TRACE_PATH"] = f"{REMOTE_A3S_LOG_DIR}/messages.jsonl"
        env["A3S_CODE_LLM_TRACE_PATH"] = f"{REMOTE_A3S_LOG_DIR}/llm_trace.jsonl"
        env["A3S_CODE_LLM_TRACE_FULL"] = os.getenv("A3S_CODE_LLM_TRACE_FULL", "1")
        env["A3S_CODE_SESSION_STORE_DIR"] = f"{REMOTE_A3S_LOG_DIR}/sessions"
        env["A3S_CODE_RUN_METADATA_PATH"] = f"{REMOTE_A3S_LOG_DIR}/run_metadata.json"
        env["A3S_CODE_WORKSPACE_MANIFEST_PATH"] = f"{REMOTE_A3S_LOG_DIR}/workspace_manifest.json"
        timeout_sec = self._agent_command_timeout_sec()
        if timeout_sec is not None:
            env["A3S_CODE_AGENT_COMMAND_TIMEOUT_SEC"] = str(timeout_sec)
        command = f"{REMOTE_VENV_PYTHON} /installed-agent/skillsbench_a3s_code_runner.py"
        return command, env, env.get("A3S_CODE_WORKSPACE") or None, timeout_sec

    def _agent_command_timeout_sec(self) -> int | None:
        configured = os.getenv("A3S_CODE_AGENT_COMMAND_TIMEOUT_SEC", "").strip()
        if configured:
            try:
                value = int(float(configured))
            except ValueError:
                value = 0
            return value if value > 0 else None

        config_path = self.logs_dir.parent / "config.json"
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                value = (payload.get("agent") or {}).get("override_timeout_sec")
                if value:
                    return max(1, int(float(value)) + 60)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        return 1260

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        command, env, cwd, timeout_sec = self._create_run_agent_command(instruction)
        command_dir = f"{REMOTE_AGENT_LOG_DIR}/command-0"
        stdout_path = f"{command_dir}/stdout.txt"
        stderr_path = f"{command_dir}/stderr.txt"
        script = "\n".join(
            [
                "set -euo pipefail",
                f"mkdir -p {shlex.quote(command_dir)} {shlex.quote(REMOTE_A3S_LOG_DIR)}",
                f"if [ -r {shlex.quote(REMOTE_MODEL_API_KEY_PATH)} ]; then export A3S_CODE_MODEL_API_KEY=\"$(cat {shlex.quote(REMOTE_MODEL_API_KEY_PATH)})\"; fi",
                f"{command} > >(tee {shlex.quote(stdout_path)}) 2> >(tee {shlex.quote(stderr_path)} >&2)",
            ]
        )
        await self.exec_as_agent(
            environment,
            command=script,
            env=env,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )
        self.populate_context_post_run(context)

    def populate_context_post_run(self, context: AgentContext) -> None:
        stdout_path = self.logs_dir / "command-0" / "stdout.txt"
        a3s_log_dir = self.logs_dir / "a3s"
        trace_path = a3s_log_dir / "messages.jsonl"
        llm_trace_path = a3s_log_dir / "llm_trace.jsonl"
        run_metadata_path = a3s_log_dir / "run_metadata.json"
        workspace_manifest_path = a3s_log_dir / "workspace_manifest.json"
        session_store_dir = a3s_log_dir / "sessions"
        metadata = {
            "agent": self.name(),
            "version": self.version(),
            "wheel_path": str(self._wheel_path),
            "wheel_paths": [str(path) for path in self._wheel_paths],
            "trace_local_path": str(trace_path),
            "trace_exists": trace_path.exists(),
            "trace_bytes": trace_path.stat().st_size if trace_path.exists() else 0,
            "trace_lines": sum(1 for _ in trace_path.open(encoding="utf-8")) if trace_path.exists() else 0,
            "llm_trace_local_path": str(llm_trace_path),
            "llm_trace_exists": llm_trace_path.exists(),
            "llm_trace_bytes": llm_trace_path.stat().st_size if llm_trace_path.exists() else 0,
            "llm_trace_lines": sum(1 for _ in llm_trace_path.open(encoding="utf-8")) if llm_trace_path.exists() else 0,
            "run_metadata_local_path": str(run_metadata_path),
            "run_metadata_exists": run_metadata_path.exists(),
            "workspace_manifest_local_path": str(workspace_manifest_path),
            "workspace_manifest_exists": workspace_manifest_path.exists(),
            "workspace_manifest_bytes": workspace_manifest_path.stat().st_size
            if workspace_manifest_path.exists()
            else 0,
            "session_store_local_dir": str(session_store_dir),
            "session_store_exists": session_store_dir.exists(),
            "session_store_files": sum(1 for path in session_store_dir.rglob("*") if path.is_file())
            if session_store_dir.exists()
            else 0,
        }
        if not stdout_path.exists():
            context.metadata = metadata
            return

        stdout = stdout_path.read_text(encoding="utf-8", errors="ignore")
        start = stdout.find(RESULT_BEGIN)
        end = stdout.find(RESULT_END)
        if start == -1 or end == -1 or end <= start:
            metadata["raw_stdout"] = stdout
            context.metadata = metadata
            return

        payload_text = stdout[start + len(RESULT_BEGIN):end].strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            metadata["raw_stdout"] = stdout
            context.metadata = metadata
            return

        context.n_input_tokens = payload.get("prompt_tokens")
        context.n_output_tokens = payload.get("completion_tokens")
        context.rollout_details = payload.get("rollout_details")
        metadata.update(payload)
        context.metadata = metadata
