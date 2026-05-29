from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from custom_types import TurnContext, TurnResult
from inference_client import SGLangTurnClient


class RolloutAgent(Protocol):
    @property
    def parse_error_count(self) -> int: ...

    def set_max_parse_errors(self, max_parse_errors: int) -> None: ...

    def start_turn_loop(self, input_message: Any) -> None: ...

    async def get_turn_context(
        self,
    ) -> tuple[Optional[List[dict[str, Any]]], Optional[Any]]: ...

    async def consume_completion(
        self, chat_completion: Any
    ) -> tuple[Optional[Any], List[Any], bool, Optional[Any]]: ...

    def record_tool_result(self, tool_call_request: Any, raw_result: Any) -> None: ...

    def finalize_response(self, model_response: Any) -> Any: ...


class AgentRunner:
    def __init__(
        self,
        *,
        rollout_agent: RolloutAgent,
        sglang_client: SGLangTurnClient,
        tool_schemas: List[Dict[str, Any]],
    ) -> None:
        self._rollout_agent = rollout_agent
        self._sglang_client = sglang_client
        self._tool_schemas = tool_schemas
        self._model_turn_count = 0
        self._max_iterations = 10
        self._max_parse_errors = 3

    @property
    def model_turn_count(self) -> int:
        return self._model_turn_count

    @property
    def parse_error_count(self) -> int:
        return self._rollout_agent.parse_error_count

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def max_parse_errors(self) -> int:
        return self._max_parse_errors

    def reset(self, input_message: Any) -> None:
        self._model_turn_count = 0
        self._rollout_agent.start_turn_loop(input_message)

    def set_max_parse_errors(self, max_parse_errors: int) -> None:
        self._max_parse_errors = max(1, int(max_parse_errors))
        self._rollout_agent.set_max_parse_errors(self._max_parse_errors)

    def set_max_iterations(self, max_iterations: int) -> None:
        self._max_iterations = max(1, int(max_iterations))
        if hasattr(self._rollout_agent, "set_max_iterations"):
            self._rollout_agent.set_max_iterations(self._max_iterations)  # type: ignore[attr-defined]

    def reached_iteration_limit(self) -> bool:
        return self._model_turn_count >= self._max_iterations

    def reached_parse_error_limit(self) -> bool:
        return self.parse_error_count >= self._max_parse_errors

    async def get_turn_context(self) -> TurnContext:
        messages, terminated = await self._rollout_agent.get_turn_context()
        return TurnContext(context_messages=messages, terminated_response=terminated)

    async def run_model_turn(
        self, context_messages: List[dict[str, Any]]
    ) -> TurnResult:
        if hasattr(self._rollout_agent, "run_model_turn"):
            turn_result = await self._rollout_agent.run_model_turn(context_messages)  # type: ignore[attr-defined]
            interactions = getattr(turn_result, "interactions", None) or [
                turn_result.interaction
            ]
            self._model_turn_count += max(1, len(interactions))
            return turn_result

        chat_completion, interaction = await self._sglang_client.generate_turn(
            messages=context_messages,
            tools=self._tool_schemas,
            turn_idx=self._model_turn_count,
        )
        self._model_turn_count += 1

        model_response, tool_call_requests, parse_error_recorded, terminated = (
            await self._rollout_agent.consume_completion(chat_completion)
        )
        return TurnResult(
            interaction=interaction,
            model_response=model_response,
            tool_call_requests=tool_call_requests,
            parse_error_recorded=parse_error_recorded,
            terminated_response=terminated,
            interactions=[interaction],
        )

    def record_tool_result(self, tool_call_request: Any, raw_result: Any) -> None:
        self._rollout_agent.record_tool_result(tool_call_request, raw_result)

    def finalize_response(self, model_response: Any) -> Any:
        return self._rollout_agent.finalize_response(model_response)

    def close(self) -> None:
        close = getattr(self._rollout_agent, "close", None)
        if close is not None:
            close()


def normalize_harness_option(value: str | None) -> str:
    normalized = (value or "camel-agent").strip().lower().replace("_", "-")
    aliases = {
        "camel": "camel-agent",
        "camel-agent": "camel-agent",
        "a3s": "a3s-code",
        "a3s-code": "a3s-code",
        "a3s-code-agent": "a3s-code",
        "a3s-code-harness": "a3s-code",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported harness_option: {value!r}. Expected camel-agent or a3s-code."
        )
    return aliases[normalized]


def create_agent_runner(
    *,
    agent_type: str,
    sglang_client: SGLangTurnClient,
    model_type: str,
    tool_schemas: List[Dict[str, Any]],
    non_think_mode: bool,
    max_total_tokens: int,
    env_client: Any | None = None,
    lease_id: str | None = None,
) -> AgentRunner:
    harness_option = normalize_harness_option(agent_type)
    if harness_option == "camel-agent":
        from agent.camel_agent import CamelAgent

        rollout_agent = CamelAgent(
            model_type=model_type,
            sglang_client=sglang_client,
            non_think_mode=non_think_mode,
            max_total_tokens=max_total_tokens,
        )
    elif harness_option == "a3s-code":
        from agent.a3s_code_agent import A3SCodeAgent

        rollout_agent = A3SCodeAgent(
            model_type=model_type,
            sglang_client=sglang_client,
            non_think_mode=non_think_mode,
            max_total_tokens=max_total_tokens,
            env_client=env_client,
            lease_id=lease_id,
        )
    else:
        raise AssertionError(f"unreachable harness_option={harness_option!r}")

    return AgentRunner(
        rollout_agent=rollout_agent,
        sglang_client=sglang_client,
        tool_schemas=tool_schemas,
    )
