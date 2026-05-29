from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from a3s_code import Agent, PermissionPolicy, SessionOptions, SessionQueueConfig

from custom_types import Interaction, TurnResult

logger = logging.getLogger(__name__)


def _clear_proxy_env_for_local_sdk_bridge() -> None:
    for key in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        os.environ.pop(key, None)


@dataclass
class A3SCodeResponse:
    msg: str
    terminated: bool = False
    info: dict[str, Any] = field(default_factory=lambda: {"termination_reasons": []})


class A3SOpenAIModelBridge:
    def __init__(self, *, sglang_client: Any, model_name: str) -> None:
        self._sglang_client = sglang_client
        self._model_name = model_name
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._interactions: list[Interaction] = []

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("A3S OpenAI bridge is not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        if self._server is not None:
            return

        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("a3s-code bridge: " + fmt, *args)

            def do_GET(self) -> None:
                if self.path == "/v1/models":
                    self._write_json({"object": "list", "data": [{"id": parent._model_name}]})
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    if payload.get("stream"):
                        self._write_sse(parent._stream_chunks(payload))
                        return
                    response = parent._complete(payload)
                    self._write_json(response)
                except Exception as exc:
                    logger.exception("a3s-code bridge request failed")
                    self._write_json({"error": str(exc)}, status=500)

            def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_sse(self, chunks: list[dict[str, Any]]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for chunk in chunks:
                    body = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode(
                        "utf-8"
                    )
                    self.wfile.write(body)
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        host = "127.0.0.1"
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, 0))
        _, port = sock.getsockname()
        sock.close()
        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"a3s-openai-bridge-{port}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def interactions(self) -> list[Interaction]:
        with self._lock:
            return list(self._interactions)

    def _complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = self._normalize_messages(payload.get("messages") or [])
        tools = payload.get("tools") or None
        with self._lock:
            turn_idx = len(self._interactions)

        chat_completion, interaction = asyncio.run(
            self._sglang_client.generate_turn(
                messages=messages,
                tools=tools,
                turn_idx=turn_idx,
            )
        )
        with self._lock:
            self._interactions.append(interaction)

        response = chat_completion.model_dump(mode="json", exclude_none=True)
        response["model"] = str(payload.get("model") or self._model_name)
        return response

    def _stream_chunks(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        full_payload = dict(payload)
        full_payload["stream"] = False
        response = self._complete(full_payload)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "stop"
        base = {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion.chunk",
            "created": response.get("created", int(time.time())),
            "model": response.get("model", self._model_name),
        }

        chunks = [
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        ]
        delta: dict[str, Any] = {}
        content = message.get("content")
        if content:
            delta["content"] = str(content)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            delta["tool_calls"] = [
                self._stream_tool_call_delta(index, tool_call)
                for index, tool_call in enumerate(tool_calls)
            ]
            finish_reason = "tool_calls"
        if delta:
            chunks.append(
                {
                    **base,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": None}
                    ],
                }
            )
        chunks.append(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": finish_reason}
                ],
            }
        )
        return chunks

    @staticmethod
    def _stream_tool_call_delta(index: int, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = dict(tool_call.get("function") or {})
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return {
            "index": index,
            "id": str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
            "type": str(tool_call.get("type") or "function"),
            "function": {
                "name": str(function.get("name") or ""),
                "arguments": arguments,
            },
        }

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            item = dict(message)
            content = item.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if text is None:
                            text = block.get("content")
                        if text is not None:
                            parts.append(str(text))
                    else:
                        parts.append(str(block))
                item["content"] = "\n".join(parts)
            elif content is None:
                item["content"] = ""
            normalized.append(item)
        return normalized


class A3SCodeAgent:
    def __init__(
        self,
        *,
        model_type: str,
        sglang_client: Any,
        non_think_mode: bool,
        max_total_tokens: int,
        max_parse_errors: int | None = None,
        env_client: Any | None = None,
        lease_id: str | None = None,
        tool_timeout_ms: int | None = None,
    ) -> None:
        _ = non_think_mode
        self.model_type = model_type or "slime-sglang"
        self._sglang_client = sglang_client
        self._max_total_tokens = int(max_total_tokens)
        self._env_client = env_client
        self._lease_id = lease_id
        self._tool_timeout_ms = int(
            tool_timeout_ms or os.getenv("A3S_CODE_TOOL_TIMEOUT_MS", "7200000")
        )
        self._max_tool_rounds = max(1, int(os.getenv("A3S_CODE_MAX_TOOL_ROUNDS", "10")))
        self.max_parse_errors = max(1, int(max_parse_errors or 3))
        self.parse_error_count = 0
        self._prompt = ""
        self._session_id = ""
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._bridge: A3SOpenAIModelBridge | None = None
        self._agent: Agent | None = None
        self._session: Any | None = None
        self._last_response: A3SCodeResponse | None = None
        self._tool_call_records: list[dict[str, Any]] = []

    def set_max_parse_errors(self, max_parse_errors: int) -> None:
        self.max_parse_errors = max(1, int(max_parse_errors))

    def set_max_iterations(self, max_iterations: int) -> None:
        _ = max_iterations

    def start_turn_loop(self, input_message: Any) -> None:
        self.parse_error_count = 0
        self._tool_call_records = []
        self._last_response = None
        self._prompt = str(input_message)
        self._session_id = f"terminal-rl-a3s-{uuid.uuid4().hex[:16]}"
        self._close_session()

    async def get_turn_context(self) -> tuple[list[dict[str, Any]] | None, A3SCodeResponse | None]:
        if self._last_response is not None:
            return None, self._last_response
        return [{"role": "user", "content": self._prompt}], None

    async def consume_completion(
        self, chat_completion: Any
    ) -> tuple[Any | None, list[Any], bool, A3SCodeResponse | None]:
        _ = chat_completion
        raise RuntimeError("A3SCodeAgent uses the a3s-code SDK run path")

    def record_tool_result(self, tool_call_request: Any, raw_result: Any) -> None:
        _ = (tool_call_request, raw_result)
        raise RuntimeError("A3SCodeAgent executes tools through the SDK external queue")

    async def run_model_turn(self, context_messages: list[dict[str, Any]]) -> TurnResult:
        _ = context_messages
        result = await asyncio.to_thread(self._send_with_external_tools)
        interactions = self._bridge.interactions() if self._bridge is not None else []
        if not interactions:
            raise RuntimeError("a3s-code SDK returned no model interactions")
        self._last_response = self._response_from_result(result)
        turn_result = TurnResult(
            interaction=interactions[-1],
            model_response=self._last_response,
            tool_call_requests=[],
            parse_error_recorded=False,
            terminated_response=None,
            interactions=interactions,
        )
        return turn_result

    def finalize_response(self, model_response: Any) -> A3SCodeResponse:
        if isinstance(model_response, A3SCodeResponse):
            return model_response
        return self._last_response or A3SCodeResponse(
            msg="",
            terminated=True,
            info={
                "termination_reasons": ["missing_a3s_code_response"],
                "harness": "a3s-code",
            },
        )

    def close(self) -> None:
        self._close_session()

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        _clear_proxy_env_for_local_sdk_bridge()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="terminal-rl-a3s-")
        tmpdir = Path(self._tmpdir.name)
        workspace = tmpdir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        self._bridge = A3SOpenAIModelBridge(
            sglang_client=self._sglang_client,
            model_name=self.model_type,
        )
        self._bridge.start()
        config_path = tmpdir / "agent.acl"
        config_path.write_text(self._render_agent_config(self._bridge.base_url), encoding="utf-8")

        opts = SessionOptions()
        opts.session_id = self._session_id
        opts.builtin_skills = False
        opts.max_parse_retries = self.max_parse_errors
        opts.max_tool_rounds = getattr(self, "_max_tool_rounds", None)
        opts.tool_timeout_ms = self._tool_timeout_ms
        opts.circuit_breaker_threshold = int(os.getenv("A3S_CODE_CIRCUIT_BREAKER", "3"))
        opts.planning_mode = os.getenv("A3S_CODE_PLANNING_MODE", "disabled")
        thinking_budget = os.getenv("A3S_CODE_THINKING_BUDGET", "").strip()
        if thinking_budget:
            opts.thinking_budget = int(thinking_budget)
        opts.permission_policy = PermissionPolicy(default_decision="allow")

        queue = SessionQueueConfig()
        queue.set_lane_handler("query", "external", self._tool_timeout_ms)
        queue.set_lane_handler("execute", "external", self._tool_timeout_ms)
        opts.queue_config = queue

        self._agent = Agent.create(str(config_path))
        self._session = self._agent.session(str(workspace), opts)

    def _send_with_external_tools(self) -> Any:
        self._ensure_session()
        assert self._session is not None

        result_box: dict[str, Any] = {}

        def send_target() -> None:
            try:
                result_box["result"] = self._session.send(self._prompt)
            except Exception as exc:
                result_box["error"] = exc

        sender = threading.Thread(target=send_target, daemon=True)
        sender.start()

        handled: set[str] = set()
        while sender.is_alive():
            self._drain_external_tasks(handled)
            sender.join(timeout=0.1)

        self._drain_external_tasks(handled)
        if "error" in result_box:
            raise result_box["error"]
        return result_box["result"]

    def _drain_external_tasks(self, handled: set[str]) -> None:
        assert self._session is not None
        for task in list(self._session.pending_external_tasks()):
            task_id = str(task.get("task_id") or "")
            if not task_id or task_id in handled:
                continue
            handled.add(task_id)
            tool_name = str(task.get("command_type") or "")
            args = task.get("payload") if isinstance(task.get("payload"), dict) else {}
            try:
                output = asyncio.run(self._exec_terminal_tool(tool_name, args))
                self._session.complete_external_task(
                    task_id,
                    True,
                    {"output": output, "exit_code": 0},
                )
                self._tool_call_records.append(
                    {
                        "tool_call_id": task_id,
                        "tool_name": tool_name,
                        "args": args,
                        "result": output[:4096],
                    }
                )
            except Exception as exc:
                self._session.complete_external_task(task_id, False, None, str(exc))
                self._tool_call_records.append(
                    {
                        "tool_call_id": task_id,
                        "tool_name": tool_name,
                        "args": args,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    async def _exec_terminal_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        if self._env_client is None or self._lease_id is None:
            raise RuntimeError("terminal env client is required for a3s-code tool execution")

        mapped_name, mapped_args = self._map_tool_call(tool_name, args)
        return await self._env_client.exec_tool(self._lease_id, mapped_name, mapped_args)

    @staticmethod
    def _map_tool_call(tool_name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if tool_name in {
            "shell_exec",
            "shell_view",
            "shell_write_to_process",
            "shell_write_content_to_file",
        }:
            return tool_name, args

        if tool_name in {"bash", "execute"}:
            command = str(args.get("command") or args.get("cmd") or "")
            return "shell_exec", {"command": command}

        if tool_name == "read":
            path = str(args.get("path") or args.get("file_path") or "")
            offset = int(args.get("offset") or args.get("line_offset") or 1)
            limit = int(args.get("limit") or args.get("line_limit") or 200)
            command = (
                f"sed -n '{max(1, offset)},{max(1, offset) + max(1, limit) - 1}p' "
                f"{shlex.quote(path)}"
            )
            return "shell_exec", {"command": command}

        if tool_name == "ls":
            path = str(args.get("path") or ".")
            return "shell_exec", {"command": f"ls -la {shlex.quote(path)}"}

        if tool_name == "grep":
            pattern = str(args.get("pattern") or args.get("query") or "")
            path = str(args.get("path") or ".")
            command = (
                f"grep -RIn -- {shlex.quote(pattern)} {shlex.quote(path)} | head -200"
            )
            return "shell_exec", {"command": command}

        if tool_name == "glob":
            pattern = str(args.get("pattern") or args.get("glob") or "*")
            command = (
                "python3 - <<'PY'\n"
                "import glob\n"
                f"for p in glob.glob({json.dumps(pattern)}, recursive=True)[:200]: print(p)\n"
                "PY"
            )
            return "shell_exec", {"command": command}

        if tool_name == "write":
            path = str(args.get("path") or args.get("file_path") or "")
            content = str(args.get("content") or "")
            return "shell_write_content_to_file", {"file_path": path, "content": content}

        if tool_name == "edit":
            path = str(args.get("path") or args.get("file_path") or "")
            old = str(args.get("old_string") or args.get("old") or "")
            new = str(args.get("new_string") or args.get("new") or "")
            command = (
                "python3 - <<'PY'\n"
                "from pathlib import Path\n"
                f"path = Path({json.dumps(path)})\n"
                f"old = {json.dumps(old)}\n"
                f"new = {json.dumps(new)}\n"
                "text = path.read_text()\n"
                "if old not in text:\n"
                "    raise SystemExit('old_string not found')\n"
                "path.write_text(text.replace(old, new, 1))\n"
                "PY"
            )
            return "shell_exec", {"command": command}

        return "shell_exec", {"command": f"echo unsupported a3s-code tool: {shlex.quote(tool_name)}"}

    def _response_from_result(self, result: Any) -> A3SCodeResponse:
        text = str(getattr(result, "text", "") or "")
        return A3SCodeResponse(
            msg=text,
            info={
                "termination_reasons": [],
                "harness": "a3s-code",
                "tool_calls_count": int(getattr(result, "tool_calls_count", 0) or 0),
                "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(result, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(result, "total_tokens", 0) or 0),
                "parse_error_count": self.parse_error_count,
                "tool_calls": list(self._tool_call_records),
            },
        )

    def _render_agent_config(self, base_url: str) -> str:
        output_tokens = int(os.getenv("A3S_CODE_OUTPUT_TOKENS", "8192"))
        return (
            f'default_model = "openai/{self.model_type}"\n\n'
            'providers "openai" {\n'
            '  api_key = "terminal-rl"\n'
            f'  base_url = "{base_url}"\n\n'
            f'  models "{self.model_type}" {{\n'
            f'    name = "{self.model_type}"\n'
            "    tool_call = true\n\n"
            "    limit = {\n"
            f"      context = {self._max_total_tokens}\n"
            f"      output = {output_tokens}\n"
            "    }\n"
            "  }\n"
            "}\n"
        )

    def _close_session(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
        self._bridge = None
        self._agent = None
        self._session = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
        self._tmpdir = None
