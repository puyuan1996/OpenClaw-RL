#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from a3s_code import Agent, FileSessionStore, PermissionPolicy, SessionOptions
except ImportError:
    from a3s_code._native import Agent, FileSessionStore, PermissionPolicy, SessionOptions


RESULT_BEGIN = "A3S_CODE_RESULT_BEGIN"
RESULT_END = "A3S_CODE_RESULT_END"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        "/root/.openclaw/skills",
        "/root/.codex/skills",
        "/root/.claude/skills",
        "/root/.agents/skills",
        "/root/.factory/skills",
        "/root/.goose/skills",
        "/root/.gemini/skills",
    ]
    return [candidate for candidate in candidates if Path(candidate).exists()]


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


def _acl_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _render_config(base_url: str, model_name: str, api_key: str, session_id_header: str | None) -> str:
    session_header_line = ""
    if session_id_header:
        session_header_line = f"  sessionIdHeader = {_acl_string(session_id_header)}\n"
    return (
        f"default_model = {_acl_string(f'openai/{model_name}')}\n\n"
        'providers "openai" {\n'
        f"  apiKey = {_acl_string(api_key)}\n"
        f"  baseUrl = {_acl_string(base_url)}\n"
        f"{session_header_line}"
        f"  models {_acl_string(model_name)} {{\n"
        f"    name = {_acl_string(model_name)}\n"
        "    toolCall = true\n"
        "  }\n"
        "}\n"
        "\n"
        'storage_backend = "memory"\n'
    )


def _resolve_config_path() -> Path:
    configured = os.getenv("A3S_CODE_CONFIG", "").strip()
    if configured and Path(configured).exists():
        return Path(configured)

    base_url = os.getenv("A3S_CODE_MODEL_BASE_URL", "").strip()
    model_name = os.getenv("A3S_CODE_MODEL_NAME", "").strip()
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


def _append_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def main() -> int:
    for env_name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(env_name, None)

    instruction = os.getenv("A3S_CODE_INSTRUCTION", "").strip()
    if not instruction:
        raise RuntimeError("A3S_CODE_INSTRUCTION is required")

    workspace = os.getenv("A3S_CODE_WORKSPACE", "/workspace").strip() or "/workspace"
    session_id = os.getenv("A3S_CODE_SESSION_ID", "clawmark-a3s-code").strip() or "clawmark-a3s-code"
    config_path = _resolve_config_path()
    session_store_dir = Path(os.getenv("A3S_CODE_SESSION_STORE_DIR", "/root/.a3s/sessions"))
    session_store_dir.mkdir(parents=True, exist_ok=True)
    trace_path = Path(os.getenv("A3S_CODE_TRACE_PATH", "/root/.a3s/messages.jsonl"))
    skill_dirs = _ensure_str_list(_env_json("A3S_CODE_SKILL_DIRS_JSON", _default_skill_dirs()))
    mcp_servers = _env_json("A3S_CODE_MCP_SERVERS_JSON", [])

    agent = Agent.create(str(config_path))
    opts = SessionOptions()
    opts.builtin_skills = _env_flag("A3S_CODE_BUILTIN_SKILLS", True)
    opts.planning = _env_flag("A3S_CODE_PLANNING", True)
    opts.thinking_budget = int(os.getenv("A3S_CODE_THINKING_BUDGET", "32000"))
    opts.max_tool_rounds = int(os.getenv("A3S_CODE_MAX_TOOL_ROUNDS", "64"))
    opts.tool_timeout_ms = int(os.getenv("A3S_CODE_TOOL_TIMEOUT_MS", "300000"))
    opts.skill_dirs = skill_dirs
    opts.session_store = FileSessionStore(str(session_store_dir))
    opts.session_id = session_id
    opts.auto_save = True
    if _env_flag("A3S_CODE_PERMISSIVE", True):
        opts.permission_policy = PermissionPolicy(default_decision="allow")

    session = None
    try:
        if any(session_store_dir.iterdir()):
            session = agent.resume_session(session_id, opts)
    except Exception:
        session = None

    if session is None:
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

    _append_trace(
        trace_path,
        {
            "kind": "user_message",
            "session_id": session_id,
            "workspace": workspace,
            "instruction": instruction,
        },
    )

    started_at = time.time()
    result = session.send(instruction)
    elapsed_sec = time.time() - started_at

    payload = {
        "session_id": session_id,
        "workspace": workspace,
        "config_path": str(config_path),
        "session_store_dir": str(session_store_dir),
        "skill_dirs": skill_dirs,
        "mcp_servers": [server.get("name") for server in mcp_servers if isinstance(server, dict)],
        "text": result.text,
        "tool_calls_count": result.tool_calls_count,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "elapsed_sec": elapsed_sec,
    }

    _append_trace(
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
