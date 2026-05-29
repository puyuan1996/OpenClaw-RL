from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


TERMINAL_RL = Path(__file__).resolve().parents[1]


@dataclass
class StubInteraction:
    turn_idx: int = 0
    input_ids: list[int] | None = None
    output_token_ids: list[int] | None = None
    output_token_logprobs: list[float] | None = None
    output_text: str = ""
    finish_reason: str = "stop"
    messages: list[dict[str, Any]] | None = None
    latency_ms: float = 0.0


@dataclass
class StubTurnResult:
    interaction: Any
    model_response: Any
    tool_call_requests: list[Any]
    parse_error_recorded: bool
    terminated_response: Any = None
    interactions: list[Any] | None = None


class FakeSessionOptions:
    pass


class FakeSessionQueueConfig:
    def __init__(self) -> None:
        self.handlers: list[tuple[str, str, int]] = []

    def set_lane_handler(self, lane: str, mode: str, timeout_ms: int) -> None:
        self.handlers.append((lane, mode, timeout_ms))


class FakePermissionPolicy:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeSession:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    def send(self, prompt: Any) -> Any:
        self.sent.append(prompt)
        return SimpleNamespace(
            text="final answer",
            tool_calls_count=0,
            prompt_tokens=3,
            completion_tokens=4,
            total_tokens=7,
        )

    def pending_external_tasks(self) -> list[dict[str, Any]]:
        return []


class FakeAgent:
    created_config: str | None = None
    last_session: FakeSession | None = None
    last_opts: Any | None = None

    @staticmethod
    def create(config_path: str) -> "FakeAgent":
        FakeAgent.created_config = config_path
        return FakeAgent()

    def session(self, workspace: str, opts: Any) -> FakeSession:
        _ = workspace
        FakeAgent.last_opts = opts
        FakeAgent.last_session = FakeSession()
        return FakeAgent.last_session


class FakeBridge:
    def __init__(self, *, sglang_client: Any, model_name: str) -> None:
        _ = (sglang_client, model_name)
        self.base_url = "http://127.0.0.1:9"
        self.closed = False

    def start(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def interactions(self) -> list[StubInteraction]:
        return [
            StubInteraction(
                turn_idx=0,
                input_ids=[1, 2],
                output_token_ids=[3, 4],
                output_token_logprobs=[-0.1, -0.2],
                output_text="final answer",
                messages=[{"role": "user", "content": "fix"}],
            )
        ]


def _load_module(monkeypatch):
    custom_types = ModuleType("custom_types")
    custom_types.Interaction = StubInteraction
    custom_types.TurnResult = StubTurnResult
    monkeypatch.setitem(sys.modules, "custom_types", custom_types)

    fake_a3s = ModuleType("a3s_code")
    fake_a3s.Agent = FakeAgent
    fake_a3s.PermissionPolicy = FakePermissionPolicy
    fake_a3s.SessionOptions = FakeSessionOptions
    fake_a3s.SessionQueueConfig = FakeSessionQueueConfig
    monkeypatch.setitem(sys.modules, "a3s_code", fake_a3s)

    spec = importlib.util.spec_from_file_location(
        "a3s_code_agent_sdk_test", TERMINAL_RL / "agent" / "a3s_code_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.A3SOpenAIModelBridge = FakeBridge
    return module


def test_a3s_code_agent_uses_pip_sdk_agent_session_send(monkeypatch) -> None:
    module = _load_module(monkeypatch)
    agent = module.A3SCodeAgent(
        model_type="slime-sglang",
        sglang_client=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: [])),
        non_think_mode=True,
        max_total_tokens=32768,
        env_client=object(),
        lease_id="lease-1",
    )
    agent.start_turn_loop("fix")

    result = asyncio.run(agent.run_model_turn([{"role": "user", "content": "fix"}]))

    assert FakeAgent.created_config is not None
    assert FakeAgent.last_session is not None
    assert FakeAgent.last_session.sent == ["fix"]
    assert result.model_response.info["harness"] == "a3s-code"
    assert result.interactions[0].output_token_logprobs == [-0.1, -0.2]


def test_a3s_code_tool_rounds_are_independent_from_terminal_iterations(monkeypatch) -> None:
    monkeypatch.setenv("A3S_CODE_MAX_TOOL_ROUNDS", "12")
    module = _load_module(monkeypatch)
    agent = module.A3SCodeAgent(
        model_type="slime-sglang",
        sglang_client=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: [])),
        non_think_mode=True,
        max_total_tokens=32768,
    )
    agent.set_max_iterations(2)
    agent.start_turn_loop("fix")

    asyncio.run(agent.run_model_turn([{"role": "user", "content": "fix"}]))

    assert FakeAgent.last_opts is not None
    assert FakeAgent.last_opts.max_tool_rounds == 12


def test_a3s_code_maps_sdk_tools_to_terminal_env(monkeypatch) -> None:
    module = _load_module(monkeypatch)

    assert module.A3SCodeAgent._map_tool_call("bash", {"command": "pwd"}) == (
        "shell_exec",
        {"command": "pwd"},
    )
    assert module.A3SCodeAgent._map_tool_call("write", {"path": "x.txt", "content": "hi"}) == (
        "shell_write_content_to_file",
        {"file_path": "x.txt", "content": "hi"},
    )
