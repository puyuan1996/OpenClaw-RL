#!/usr/bin/env python3
from __future__ import annotations

import ast
import base64
import hashlib
import ipaddress
import json
import os
import selectors
import shlex
import socket
import socketserver
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    from a3s_code import Agent, ConfirmationPolicy, FileSessionStore, PermissionPolicy, SessionOptions
except ImportError:
    from a3s_code._native import Agent, ConfirmationPolicy, PermissionPolicy, SessionOptions

    try:
        from a3s_code._native import FileSessionStore
    except ImportError:  # pragma: no cover - depends on installed a3s-code build
        FileSessionStore = None  # type: ignore[assignment]


RESULT_BEGIN = "A3S_CODE_RESULT_BEGIN"
RESULT_END = "A3S_CODE_RESULT_END"
DEFAULT_BENCHMARK_PROXY = "http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"
DISABLE_PROXY_VALUES = {"", "0", "false", "no", "none", "off", "direct"}
DEFAULT_NO_PROXY = (
    "localhost,127.0.0.1,0.0.0.0,::1,*.local,.pjlab.org.cn,"
    ".i.h.pjlab.org.cn,mirrors.i.h.pjlab.org.cn,pypi.i.h.pjlab.org.cn"
)
DEFAULT_AGENT_LOG_DIR = Path("/logs/agent")
DEFAULT_A3S_LOG_SUBDIR = "a3s"
DEFAULT_SESSION_AGENT = "general"
MANIFEST_HASH_MAX_BYTES = 1_000_000
_PROXY_BRIDGE = None


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(float(value))
    except ValueError:
        print(f"[a3s-code-runner] ignoring invalid {name}={value!r}", file=sys.stderr)
        return default


def _env_json(name: str, default):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, str):
            stripped = parsed.strip()
            if stripped.startswith(("[", "{")) and stripped.endswith(("]", "}")):
                return json.loads(stripped)
        return parsed
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            if isinstance(default, list):
                for separator in (os.pathsep, ",", "\n"):
                    parts = [part.strip() for part in raw.split(separator) if part.strip()]
                    if len(parts) > 1:
                        return parts
            return default


def _default_skill_dirs() -> list[str]:
    candidates = [
        "/workspace/.skillsbench-skills",
        "/root/.codex/skills",
        "/root/.claude/skills",
        "/root/.agents/skills",
        "/root/.factory/skills",
        "/root/.goose/skills",
        "/root/.gemini/skills",
        "/app/skills",
        "environment/skills",
    ]
    found: list[str] = []
    for candidate in candidates:
        if Path(candidate).exists():
            found.append(candidate)
    return found


def _ensure_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                return _ensure_str_list(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return [value] if stripped else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _render_config(
    *,
    provider: str,
    base_url: str,
    model_name: str,
    api_key: str,
    session_id_header: str | None,
) -> str:
    session_header_line = ""
    if session_id_header:
        session_header_line = f'  sessionIdHeader = "{session_id_header}"\n'
    context_tokens = int(os.getenv("A3S_CODE_CONTEXT_TOKENS", "131072"))
    output_tokens = int(os.getenv("A3S_CODE_OUTPUT_TOKENS", "8192"))
    return (
        f'default_model = "{provider}/{model_name}"\n\n'
        f'providers "{provider}" {{\n'
        f'  apiKey = "{api_key}"\n'
        f'  baseUrl = "{base_url}"\n'
        f"{session_header_line}"
        f'  models "{model_name}" {{\n'
        f'    name = "{model_name}"\n'
        "    toolCall = true\n\n"
        "    limit = {\n"
        f"      context = {context_tokens}\n"
        f"      output = {output_tokens}\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def _resolve_config_path() -> Path:
    configured = os.getenv("A3S_CODE_CONFIG", "").strip()
    if configured and Path(configured).exists():
        return Path(configured)

    base_url = os.getenv("A3S_CODE_MODEL_BASE_URL", "").strip()
    model_name = os.getenv("A3S_CODE_MODEL_NAME", "").strip()
    provider = os.getenv("A3S_CODE_MODEL_PROVIDER", "openai").strip() or "openai"
    api_key = os.getenv("A3S_CODE_MODEL_API_KEY", "apiKey")
    if not base_url or not model_name:
        raise RuntimeError(
            "A3S_CODE_CONFIG is missing and A3S_CODE_MODEL_BASE_URL / "
            "A3S_CODE_MODEL_NAME were not provided"
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="a3s-code-config-"))
    config_path = temp_dir / "config.acl"
    config_path.write_text(
        _render_config(
            provider=provider,
            base_url=base_url.rstrip("/"),
            model_name=model_name,
            api_key=api_key,
            session_id_header=os.getenv("A3S_CODE_SESSION_ID_HEADER", "X-Session-Id").strip() or None,
        ),
        encoding="utf-8",
    )
    return config_path


def _normalize_mcp_server(server: dict) -> dict:
    transport = server.get("transport", "stdio")
    if isinstance(transport, dict):
        kind = transport.get("type", "stdio")
        merged = dict(server)
        merged["transport"] = kind
        for key in ("command", "args", "url", "headers"):
            if key in transport and key not in merged:
                merged[key] = transport[key]
        return merged
    return server


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _default_log_base() -> Path:
    if DEFAULT_AGENT_LOG_DIR.exists():
        return DEFAULT_AGENT_LOG_DIR / DEFAULT_A3S_LOG_SUBDIR
    return Path("/root/.a3s")


def _append_jsonl(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": _utc_now(), **payload}, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - logging must not fail the agent
        print(f"[a3s-code-runner] failed to append trace {path}: {exc}", file=sys.stderr)


def _write_json(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:  # pragma: no cover - logging must not fail the agent
        print(f"[a3s-code-runner] failed to write json {path}: {exc}", file=sys.stderr)


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw) if raw else default


def _split_host_port(authority: str, default_port: int) -> tuple[str, int]:
    value = authority.strip()
    if "@" in value:
        value = value.rsplit("@", 1)[1]
    if value.startswith("[") and "]" in value:
        host, _, rest = value[1:].partition("]")
        if rest.startswith(":") and rest[1:].isdigit():
            return host, int(rest[1:])
        return host, default_port
    if ":" in value:
        host, port = value.rsplit(":", 1)
        if port.isdigit():
            return host, int(port)
    return value, default_port


def _proxy_bypass_host(host: str, no_proxy: str) -> bool:
    normalized = host.strip("[]").strip().rstrip(".").lower()
    if not normalized:
        return False
    model_host = (urlparse(os.getenv("A3S_CODE_MODEL_BASE_URL", "")).hostname or "").lower()
    if model_host and normalized == model_host:
        return True
    if normalized in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        ip = None
    if ip and (ip.is_loopback or ip.is_private or str(ip).startswith("100.96.")):
        return True
    if normalized.endswith((".local", ".pjlab.local", ".pjlab.org.cn")):
        return True
    for raw_entry in no_proxy.split(","):
        entry = raw_entry.strip().lower()
        if not entry or entry == "*":
            continue
        if entry.startswith("*.") and normalized.endswith(entry[1:]):
            return True
        if entry.startswith(".") and normalized.endswith(entry):
            return True
        if normalized == entry:
            return True
        if ip and "/" in entry:
            try:
                if ip in ipaddress.ip_network(entry, strict=False):
                    return True
            except ValueError:
                pass
    return False


def _proxy_bridge_timeout_sec() -> int:
    configured = _env_int("A3S_CODE_PROXY_BRIDGE_TIMEOUT_SEC", 0)
    if configured > 0:
        return max(30, configured)
    agent_timeout = _env_int("A3S_CODE_AGENT_COMMAND_TIMEOUT_SEC", 0)
    return max(600, agent_timeout + 600)


def _relay_sockets(left: socket.socket, right: socket.socket, timeout_sec: int) -> None:
    left.settimeout(None)
    right.settimeout(None)
    selector_cls = getattr(selectors, "PollSelector", selectors.DefaultSelector)
    selector = selector_cls()
    try:
        selector.register(left, selectors.EVENT_READ, right)
        selector.register(right, selectors.EVENT_READ, left)
        while True:
            events = selector.select(timeout_sec)
            if not events:
                return
            for key, _ in events:
                sock = key.fileobj
                other = key.data
                try:
                    data = sock.recv(65536)
                except OSError:
                    return
                if not data:
                    return
                try:
                    other.sendall(data)
                except OSError:
                    return
    finally:
        selector.close()


class _ProxyBridgeHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server  # type: ignore[assignment]
        upstream_host, upstream_port = server.upstream_proxy  # type: ignore[attr-defined]
        no_proxy = server.no_proxy  # type: ignore[attr-defined]
        timeout_sec = server.timeout_sec  # type: ignore[attr-defined]
        connect_timeout_sec = min(timeout_sec, max(30, _env_int("A3S_CODE_PROXY_BRIDGE_CONNECT_TIMEOUT_SEC", 120)))
        client = self.request
        client.settimeout(timeout_sec)
        try:
            first_chunk = b""
            while b"\r\n\r\n" not in first_chunk and len(first_chunk) < 131072:
                piece = client.recv(65536)
                if not piece:
                    return
                first_chunk += piece
            header_end = first_chunk.find(b"\r\n\r\n")
            if header_end < 0:
                return
            header = first_chunk[: header_end + 4]
            rest = first_chunk[header_end + 4 :]
            lines = header.split(b"\r\n")
            request_line = lines[0].decode("latin-1", errors="replace")
            parts = request_line.split()
            if len(parts) != 3:
                return
            method, target, version = parts
            method_upper = method.upper()

            if method_upper == "CONNECT":
                host, port = _split_host_port(target, 443)
                bypass = _proxy_bypass_host(host, no_proxy)
                if bypass:
                    with socket.create_connection((host, port), timeout=connect_timeout_sec) as upstream:
                        client.sendall(f"{version} 200 Connection Established\r\n\r\n".encode("latin-1"))
                        if rest:
                            upstream.sendall(rest)
                        _relay_sockets(client, upstream, timeout_sec=timeout_sec)
                else:
                    with socket.create_connection((upstream_host, upstream_port), timeout=connect_timeout_sec) as upstream:
                        upstream.sendall(header + rest)
                        _relay_sockets(client, upstream, timeout_sec=timeout_sec)
                return

            parsed = urlparse(target)
            if not (parsed.scheme and parsed.hostname):
                decoded_target = unquote(target)
                if decoded_target != target:
                    decoded = urlparse(decoded_target)
                    if decoded.scheme and decoded.hostname:
                        parsed = decoded
            if parsed.scheme and parsed.hostname:
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                path = parsed.path or "/"
                if parsed.params:
                    path += ";" + parsed.params
                if parsed.query:
                    path += "?" + parsed.query
            else:
                host_header = ""
                for line in lines[1:]:
                    if line.lower().startswith(b"host:"):
                        host_header = line.split(b":", 1)[1].decode("latin-1", errors="replace").strip()
                        break
                host, port = _split_host_port(host_header, 80)
                path = target or "/"

            bypass = _proxy_bypass_host(host, no_proxy)
            if bypass:
                rewritten_first = f"{method} {path} {version}\r\n".encode("latin-1")
                header_lines = list(lines[1:])
                while header_lines and header_lines[-1] == b"":
                    header_lines.pop()
                rewritten_header = rewritten_first + b"\r\n".join(header_lines) + b"\r\n\r\n"
                with socket.create_connection((host, port), timeout=connect_timeout_sec) as upstream:
                    upstream.sendall(rewritten_header + rest)
                    _relay_sockets(client, upstream, timeout_sec=timeout_sec)
            else:
                with socket.create_connection((upstream_host, upstream_port), timeout=connect_timeout_sec) as upstream:
                    upstream.sendall(header + rest)
                    _relay_sockets(client, upstream, timeout_sec=timeout_sec)
        except Exception as exc:  # pragma: no cover - diagnostic path inside worker container
            print(f"[a3s-code-runner] proxy bridge error: {exc}", file=sys.stderr)


class _ProxyBridgeServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _start_proxy_bridge(proxy_url: str, no_proxy: str) -> str:
    global _PROXY_BRIDGE
    parsed = urlparse(proxy_url)
    if parsed.scheme not in {"http", ""} or not parsed.hostname:
        raise RuntimeError(f"Unsupported proxy bridge upstream: {proxy_url!r}")
    upstream_port = parsed.port or 80
    server = _ProxyBridgeServer(("127.0.0.1", 0), _ProxyBridgeHandler)
    server.upstream_proxy = (parsed.hostname, upstream_port)  # type: ignore[attr-defined]
    server.no_proxy = no_proxy  # type: ignore[attr-defined]
    server.timeout_sec = _proxy_bridge_timeout_sec()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, name="a3s-proxy-bridge", daemon=True)
    thread.start()
    _PROXY_BRIDGE = server
    host, port = server.server_address
    return f"http://{host}:{port}"


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _maybe_unquote_env_value(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return stripped
    try:
        parts = shlex.split(stripped)
    except ValueError:
        return stripped
    if len(parts) == 1:
        return parts[0]
    return stripped


def _instruction_from_env() -> tuple[str, str]:
    encoded = os.getenv("A3S_CODE_INSTRUCTION_B64", "").strip()
    if encoded:
        encoded = _maybe_unquote_env_value(encoded)
        return base64.b64decode(encoded).decode("utf-8").strip(), "A3S_CODE_INSTRUCTION_B64"
    instruction = os.getenv("A3S_CODE_INSTRUCTION", "").strip()
    if _env_flag("A3S_CODE_UNQUOTE_INSTRUCTION", True):
        instruction = _maybe_unquote_env_value(instruction)
    return instruction.strip(), "A3S_CODE_INSTRUCTION"


def _session_agent_name() -> str:
    for env_name in ("A3S_CODE_SESSION_AGENT", "A3S_CODE_AGENT_NAME"):
        value = os.getenv(env_name, "").strip()
        if value:
            return "" if value.lower() in DISABLE_PROXY_VALUES else value
    return DEFAULT_SESSION_AGENT


def _logged_env_snapshot() -> dict[str, str | None]:
    sensitive_fragments = ("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN", "SECRET", "PASSWORD")
    redacted_names = {"A3S_CODE_INSTRUCTION_B64"}
    snapshot: dict[str, str | None] = {}
    for key in sorted(os.environ):
        if not (
            key.startswith("A3S_CODE_")
            or key
            in {
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "NO_PROXY",
                "BASH_ENV",
                "SHELL",
                "http_proxy",
                "https_proxy",
                "no_proxy",
            }
        ):
            continue
        value = os.getenv(key)
        if key in redacted_names or any(fragment in key for fragment in sensitive_fragments):
            snapshot[key] = "<redacted>" if value else value
        else:
            snapshot[key] = value
    return snapshot


def _file_digest(path: Path, max_bytes: int = MANIFEST_HASH_MAX_BYTES) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _workspace_manifest(workspace: Path, *, max_files: int) -> dict:
    skip_names = {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "node_modules",
    }
    files: list[dict] = []
    dirs_seen = 0
    truncated = False
    if not workspace.exists():
        return {"workspace": str(workspace), "exists": False, "files": [], "truncated": False}

    for root, dirnames, filenames in os.walk(workspace):
        dirs_seen += 1
        dirnames[:] = [name for name in dirnames if name not in skip_names]
        root_path = Path(root)
        for filename in sorted(filenames):
            path = root_path / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            try:
                relative = path.relative_to(workspace).as_posix()
            except ValueError:
                relative = str(path)
            files.append(
                {
                    "path": relative,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "sha256": _file_digest(path),
                }
            )
            if len(files) >= max_files:
                truncated = True
                break
        if truncated:
            break

    return {
        "workspace": str(workspace),
        "exists": True,
        "dirs_seen": dirs_seen,
        "files_count": len(files),
        "truncated": truncated,
        "hash_max_bytes": MANIFEST_HASH_MAX_BYTES,
        "files": files,
    }


def _configure_proxy_env() -> None:
    proxy = DEFAULT_BENCHMARK_PROXY.strip()
    for env_name in ("A3S_CODE_BENCHMARK_PROXY", "A3S_CODE_HTTP_PROXY", "BENCHMARK_HTTP_PROXY"):
        if env_name in os.environ:
            proxy = os.environ.get(env_name, "").strip()
            break
    if proxy.lower() in DISABLE_PROXY_VALUES:
        proxy = ""
    runtime_proxy_env = _env_flag("A3S_CODE_AGENT_RUNTIME_PROXY", False)
    no_proxy_raw = (os.getenv("A3S_CODE_NO_PROXY") or os.getenv("NO_PROXY") or DEFAULT_NO_PROXY).strip()
    no_proxy_entries = [
        entry.strip()
        for entry in no_proxy_raw.split(",")
        if entry.strip() and entry.strip() != "*"
    ]
    model_host = (urlparse(os.getenv("A3S_CODE_MODEL_BASE_URL", "")).hostname or "").strip()
    if _env_flag("A3S_CODE_MODEL_NO_PROXY", True) and model_host:
        no_proxy_entries.append(model_host)
    no_proxy = ",".join(
        dict.fromkeys(
            no_proxy_entries
        )
    )
    proxy_mode = os.getenv("A3S_CODE_AGENT_PROXY_MODE", "bridge").strip().lower()
    if proxy and runtime_proxy_env:
        for env_name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ[env_name] = proxy
    elif proxy and proxy_mode == "bridge":
        bridge_proxy = _start_proxy_bridge(proxy, no_proxy)
        os.environ["A3S_CODE_PROXY_BRIDGE_UPSTREAM"] = proxy
        os.environ["A3S_CODE_PROXY_BRIDGE_URL"] = bridge_proxy
        for env_name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ[env_name] = bridge_proxy
    else:
        for env_name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(env_name, None)
    if no_proxy:
        os.environ["A3S_CODE_NO_PROXY"] = no_proxy
        os.environ["NO_PROXY"] = no_proxy
        os.environ["no_proxy"] = no_proxy


def main() -> int:
    _configure_proxy_env()

    instruction, instruction_source = _instruction_from_env()
    if not instruction:
        raise RuntimeError("A3S_CODE_INSTRUCTION or A3S_CODE_INSTRUCTION_B64 is required")

    workspace = os.getenv("A3S_CODE_WORKSPACE", os.getcwd()).strip() or os.getcwd()
    workspace_path = Path(workspace)
    session_id = os.getenv("A3S_CODE_SESSION_ID", "skillsbench-a3s-code").strip() or "skillsbench-a3s-code"
    log_base = _default_log_base()
    trace_path = _path_from_env("A3S_CODE_TRACE_PATH", log_base / "messages.jsonl")
    llm_trace_path = _path_from_env("A3S_CODE_LLM_TRACE_PATH", log_base / "llm_trace.jsonl")
    run_metadata_path = _path_from_env("A3S_CODE_RUN_METADATA_PATH", log_base / "run_metadata.json")
    workspace_manifest_path = _path_from_env("A3S_CODE_WORKSPACE_MANIFEST_PATH", log_base / "workspace_manifest.json")
    session_store_dir = _path_from_env("A3S_CODE_SESSION_STORE_DIR", log_base / "sessions")
    manifest_max_files = int(os.getenv("A3S_CODE_WORKSPACE_MANIFEST_MAX_FILES", "5000"))
    config_path = _resolve_config_path()
    skill_dirs = _ensure_str_list(_env_json("A3S_CODE_SKILL_DIRS_JSON", _default_skill_dirs()))
    mcp_servers = _env_json("A3S_CODE_MCP_SERVERS_JSON", [])
    session_agent_name = _session_agent_name()

    _append_jsonl(
        trace_path,
        {
            "kind": "runner_start",
            "session_id": session_id,
            "workspace": workspace,
            "config_path": str(config_path),
            "skill_dirs": skill_dirs,
            "session_agent_name": session_agent_name or None,
            "mcp_servers": [server.get("name") for server in mcp_servers if isinstance(server, dict)],
            "pid": os.getpid(),
            "argv": sys.argv,
            "instruction_source": instruction_source,
            "instruction_chars": len(instruction),
            "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        },
    )
    _append_jsonl(
        trace_path,
        {
            "kind": "user_message",
            "session_id": session_id,
            "workspace": workspace,
            "instruction": instruction,
        },
    )

    agent = Agent.create(str(config_path))
    opts = SessionOptions()
    provider = os.getenv("A3S_CODE_MODEL_PROVIDER", "openai").strip() or "openai"
    model_name = os.getenv("A3S_CODE_MODEL_NAME", "").strip()
    if provider and model_name:
        opts.model = f"{provider}/{model_name}"
    opts.builtin_skills = _env_flag("A3S_CODE_BUILTIN_SKILLS", True)
    opts.planning = _env_flag("A3S_CODE_PLANNING", False)
    opts.thinking_budget = int(os.getenv("A3S_CODE_THINKING_BUDGET", "32000"))
    opts.max_tool_rounds = int(os.getenv("A3S_CODE_MAX_TOOL_ROUNDS", "64"))
    opts.tool_timeout_ms = int(os.getenv("A3S_CODE_TOOL_TIMEOUT_MS", "300000"))
    opts.max_parse_retries = int(os.getenv("A3S_CODE_MAX_PARSE_RETRIES", "4"))
    opts.circuit_breaker_threshold = _env_int("A3S_CODE_CIRCUIT_BREAKER_THRESHOLD", 5)
    max_execution_time_ms = _env_int("A3S_CODE_MAX_EXECUTION_TIME_MS", 0)
    if max_execution_time_ms > 0:
        opts.max_execution_time_ms = max_execution_time_ms
    if _env_flag("A3S_CODE_PERMISSIVE", True):
        opts.permission_policy = PermissionPolicy(default_decision="allow")
        opts.confirmation_policy = ConfirmationPolicy(
            enabled=True,
            timeout_action="auto_approve",
            yolo_lanes=["control", "query", "execute", "generate"],
        )
    if skill_dirs:
        opts.skill_dirs = skill_dirs
    auto_save_session = bool(FileSessionStore) and _env_flag("A3S_CODE_AUTO_SAVE_SESSION", True)
    if auto_save_session:
        session_store_dir.mkdir(parents=True, exist_ok=True)
        opts.session_store = FileSessionStore(str(session_store_dir))
        opts.session_id = session_id
        opts.auto_save = True

    _write_json(
        run_metadata_path,
        {
            "timestamp": _utc_now(),
            "session_id": session_id,
            "workspace": workspace,
            "config_path": str(config_path),
            "trace_path": str(trace_path),
            "llm_trace_path": str(llm_trace_path),
            "session_store_dir": str(session_store_dir),
            "auto_save_session": auto_save_session,
            "skill_dirs": skill_dirs,
            "mcp_servers": [server.get("name") for server in mcp_servers if isinstance(server, dict)],
            "instruction_source": instruction_source,
            "instruction_chars": len(instruction),
            "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            "session_agent_name": session_agent_name or None,
            "options": {
                "builtin_skills": opts.builtin_skills,
                "planning": opts.planning,
                "thinking_budget": opts.thinking_budget,
                "max_tool_rounds": opts.max_tool_rounds,
                "tool_timeout_ms": opts.tool_timeout_ms,
                "max_parse_retries": opts.max_parse_retries,
                "circuit_breaker_threshold": opts.circuit_breaker_threshold,
                "max_execution_time_ms": getattr(opts, "max_execution_time_ms", None),
            },
            "env": _logged_env_snapshot(),
        },
    )

    if session_agent_name:
        try:
            session = agent.session_for_agent(workspace, session_agent_name, [], opts)
        except AttributeError:
            print(
                "[a3s-code-runner] Agent.session_for_agent unavailable; falling back to auto style",
                file=sys.stderr,
            )
            session = agent.session(workspace, opts)
    else:
        session = agent.session(workspace, opts)

    for raw_server in mcp_servers:
        server = _normalize_mcp_server(raw_server)
        session.add_mcp_server(
            server["name"],
            transport=server.get("transport", "stdio"),
            command=server.get("command"),
            args=server.get("args"),
            url=server.get("url"),
            headers=server.get("headers"),
            env=server.get("env"),
            timeout_ms=server.get("timeout_ms"),
        )

    started_at = time.time()
    try:
        result = session.send(instruction)
    except Exception as exc:
        _append_jsonl(
            trace_path,
            {
                "kind": "runner_error",
                "session_id": session_id,
                "elapsed_sec": time.time() - started_at,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    elapsed_sec = time.time() - started_at
    manifest = _workspace_manifest(workspace_path, max_files=manifest_max_files)
    _write_json(workspace_manifest_path, manifest)
    rollout_details = _jsonable(getattr(result, "rollout_details", None))
    payload = {
        "session_id": session_id,
        "text": result.text,
        "tool_calls_count": result.tool_calls_count,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "rollout_details": rollout_details,
        "result_class": type(result).__name__,
        "elapsed_sec": elapsed_sec,
        "workspace": workspace,
        "config_path": str(config_path),
        "trace_path": str(trace_path),
        "llm_trace_path": str(llm_trace_path),
        "session_store_dir": str(session_store_dir),
        "run_metadata_path": str(run_metadata_path),
        "workspace_manifest_path": str(workspace_manifest_path),
        "workspace_manifest_files": manifest.get("files_count"),
        "workspace_manifest_truncated": manifest.get("truncated"),
        "auto_save_session": auto_save_session,
        "skill_dirs": skill_dirs,
        "session_agent_name": session_agent_name or None,
        "mcp_servers": [server.get("name") for server in mcp_servers if isinstance(server, dict)],
    }
    _append_jsonl(
        trace_path,
        {
            "kind": "assistant_result",
            **payload,
        },
    )
    sys.stdout.write(f"{RESULT_BEGIN}\n")
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.write(f"\n{RESULT_END}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
