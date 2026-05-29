from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


TERMINAL_RL = Path(__file__).resolve().parents[1]


@dataclass
class StubTurnContext:
    context_messages: list[dict[str, Any]] | None
    terminated_response: Any = None


@dataclass
class StubTurnResult:
    interaction: Any
    model_response: Any
    tool_call_requests: list[Any]
    parse_error_recorded: bool
    terminated_response: Any = None


def _load_agent_runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    custom_types = ModuleType("custom_types")
    custom_types.TurnContext = StubTurnContext
    custom_types.TurnResult = StubTurnResult
    inference_client = ModuleType("inference_client")
    inference_client.SGLangTurnClient = object
    monkeypatch.setitem(sys.modules, "custom_types", custom_types)
    monkeypatch.setitem(sys.modules, "inference_client", inference_client)

    spec = importlib.util.spec_from_file_location(
        "agent_runner_harness_option_test", TERMINAL_RL / "agent_runner.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_harness_option_accepts_camel_and_a3s_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_runner = _load_agent_runner(monkeypatch)

    assert agent_runner.normalize_harness_option(None) == "camel-agent"
    assert agent_runner.normalize_harness_option("camel_agent") == "camel-agent"
    assert agent_runner.normalize_harness_option("camel-agent") == "camel-agent"
    assert agent_runner.normalize_harness_option("a3s") == "a3s-code"
    assert agent_runner.normalize_harness_option("a3s_code") == "a3s-code"
    assert agent_runner.normalize_harness_option("a3s-code-harness") == "a3s-code"


def test_normalize_harness_option_rejects_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_runner = _load_agent_runner(monkeypatch)

    with pytest.raises(ValueError):
        agent_runner.normalize_harness_option("adapter")
