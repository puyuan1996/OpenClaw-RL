from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Protocol
from inference_client import create_sglang_client

logger = logging.getLogger(__name__)


class RolloutAgent(Protocol):
    @property
    def parse_error_count(self) -> int: ...

    def start_turn_loop(self, input_message: Any) -> None: ...

    async def get_turn_context(
        self,
    ) -> tuple[Optional[List[dict[str, Any]]], Optional[Any]]: ...

    async def consume_completion(
        self, chat_completion: Any
    ) -> tuple[Optional[Any], List[Any], bool]: ...

    def record_tool_result(self, tool_call_request: Any, raw_result: Any) -> None: ...

    def finalize_response(self, model_response: Any) -> Any: ...


class PRMAgent(Protocol):
    async def judge_turn(self, turn_idx: int) -> tuple[str, int]: ...

    def record_model_turn(
        self,
        turn_idx: int,
        *,
        assistant_text: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        parse_error_recorded: bool = False,
        finish_reason: Optional[str] = None,
    ) -> None: ...

    def record_tool_result(
        self, turn_idx: int, tool_call_request: Any, raw_result: Any
    ) -> None: ...


def resolve_client_template_kwargs(args: Any) -> Dict[str, Any]:
    client_template_kwargs = {
        "chat_template_type": getattr(args, "chat_template_type", "hf"),
        "chat_template_kwargs": getattr(args, "chat_template_kwargs", None),
        "messages_delimiter_start": getattr(
            args, "messages_delimiter_start", "<|im_start|>"
        ),
        "messages_delimiter_end": getattr(args, "messages_delimiter_end", "<|im_end|>"),
        "tool_call_parser": getattr(args, "tool_call_parser", "qwen25"),
    }
    return client_template_kwargs


def resolve_rollout_non_think_mode(args: Any) -> tuple[bool, bool]:
    enabled = getattr(args, "non_think_mode", True)
    source = getattr(args, "non_think_mode_source", "prompt").lower()
    if source not in {"prompt", "sglang", "both"}:
        raise ValueError(
            f"Invalid non_think_mode_source={args.non_think_mode_source!r}, "
            "expected one of: 'prompt', 'sglang', 'both'."
        )
    enable_prompt_non_think = enabled and source in {"prompt", "both"}
    enable_sglang_non_think = enabled and source in {"sglang", "both"}
    return enable_prompt_non_think, enable_sglang_non_think


def create_rollout_agent(
    *,
    agent_type: str,
    args: Any,
    tokenizer: Any,
    sampling_params: Dict[str, Any],
    model_type: str,
) -> RolloutAgent:
    client_template_kwargs = resolve_client_template_kwargs(args)
    max_total_tokens = args.rollout_max_context_len
    enable_prompt_non_think, enable_sglang_non_think = resolve_rollout_non_think_mode(
        args
    )
    sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sglang_client = create_sglang_client(
        tokenizer=tokenizer,
        client_template_kwargs=client_template_kwargs,
        max_total_tokens=max_total_tokens,
        sampling_params=sampling_params,
        enable_sglang_non_think=enable_sglang_non_think,
        sglang_url=sglang_url,
    )

    if agent_type == "camel_agent":
        from agent.camel_agent import CamelAgent

        return CamelAgent(
            model_type=model_type,
            sglang_client=sglang_client,
            non_think_mode=enable_prompt_non_think,
            max_total_tokens=max_total_tokens,
        )
    raise ValueError(f"Unsupported agent type: {agent_type!r}. Expected 'camel_agent'.")


def resolve_prm_sglang_url(args: Any) -> str:
    prm_router_ip = args.prm_router_ip
    prm_router_port = args.prm_router_port
    if prm_router_ip and prm_router_port:
        return f"http://{prm_router_ip}:{prm_router_port}/generate"

    prm_sglang_url = getattr(args, "prm_sglang_url", None) or os.getenv(
        "PRM_SGLANG_URL", ""
    )
    if prm_sglang_url:
        return prm_sglang_url

    raise RuntimeError("prm_enable=True but no PRM endpoint")


def create_prm_agent(
    args: Any,
    *,
    tokenizer: Any,
    task_instruction: str,
    log_tag: str,
) -> PRMAgent | None:
    client_template_kwargs = resolve_client_template_kwargs(args)
    max_total_tokens = args.rollout_max_context_len
    sglang_url = resolve_prm_sglang_url(args)

    sampling_params = {
        "temperature": args.prm_temperature,
        "max_new_tokens": args.prm_max_new_tokens,
    }

    sglang_client = create_sglang_client(
        tokenizer=tokenizer,
        client_template_kwargs=client_template_kwargs,
        max_total_tokens=max_total_tokens,
        sampling_params=sampling_params,
        enable_sglang_non_think=True,
        sglang_url=sglang_url,
    )

    from agent.prm_agent import TerminalPRMAgent

    prm_agent = TerminalPRMAgent(
        sglang_client=sglang_client,
        task_instruction=task_instruction,
        history_mode=getattr(args, "prm_history_mode", "head_tail"),
    )
    logger.info(
        "%s PRM enabled: url=%s, temperature=%s, max_new_tokens=%s, history_mode=%s",
        log_tag,
        sglang_url,
        args.prm_temperature,
        args.prm_max_new_tokens,
        args.prm_history_mode,
    )
    return prm_agent
