from __future__ import annotations

import datetime
import json
import logging
import re
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from slime.utils.http_utils import post as async_post

from custom_types import Interaction

logger = logging.getLogger(__name__)


def process_tool_calls(
    text: str,
    tools: list[Any],
    tool_call_parser: str | None,
    finish_reason: str,
    use_responses: bool = False,
) -> tuple[
    list[ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall] | None,
    str,
    str,
]:
    from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
    from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    if use_responses:
        tools = [
            SglTool(
                type=tool["type"],
                function=SglFunction(
                    name=tool.get("name"),
                    description=tool.get("description"),
                    parameters=tool.get("parameters"),
                ),
            )
            for tool in tools
        ]
    else:
        tools = [
            SglTool(type=tool["type"], function=SglFunction(**tool["function"]))
            for tool in tools
        ]

    parser = FunctionCallParser(tools, tool_call_parser)
    if parser.has_tool_call(text):
        if finish_reason == "stop":
            finish_reason = "tool_calls"
        try:
            text, call_info_list = parser.parse_non_stream(text)

            if use_responses:
                tool_calls = [
                    ResponseFunctionToolCall(
                        type="function_call",
                        id=f"fc-{uuid.uuid4().hex[:24]}",
                        call_id=f"call_{uuid.uuid4().hex[:24]}",
                        name=call_info.name,
                        arguments=call_info.parameters,
                        status="completed",
                    )
                    for call_info in call_info_list
                ]
            else:
                tool_calls = [
                    ChatCompletionMessageFunctionToolCall(
                        type="function",
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        function=Function(
                            name=call_info.name, arguments=call_info.parameters
                        ),
                    )
                    for call_info in call_info_list
                ]
            return tool_calls, text, finish_reason
        except Exception as exc:
            logger.error("Tool call parsing error: %s", exc)
            import traceback

            traceback.print_exc()
            return None, text, finish_reason

    return None, text, finish_reason


def _ensure_stop_token_ids(
    tokenizer, sampling_params: Dict[str, Any]
) -> Dict[str, Any]:
    normalized = dict(sampling_params)
    if "stop_token_ids" in normalized:
        return normalized

    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        stop_ids.add(tokenizer.pad_token_id)
    if stop_ids:
        normalized["stop_token_ids"] = list(stop_ids)
    return normalized


def create_sglang_client(
    tokenizer: Any,
    client_template_kwargs: Dict[str, Any],
    max_total_tokens: int,
    sampling_params: Dict[str, Any],
    enable_sglang_non_think: bool,
    sglang_url: str,
    max_retries: int = 30,
) -> "SGLangClient":
    if enable_sglang_non_think:
        raw_chat_template_kwargs = client_template_kwargs.get("chat_template_kwargs")
        if isinstance(raw_chat_template_kwargs, dict):
            merged_chat_template_kwargs = dict(raw_chat_template_kwargs)
        else:
            merged_chat_template_kwargs = {}
        merged_chat_template_kwargs["enable_thinking"] = False
        client_template_kwargs["chat_template_kwargs"] = merged_chat_template_kwargs

    max_new_tokens = sampling_params["max_new_tokens"]
    max_input_tokens = max_total_tokens - max_new_tokens
    logger.info(
        "SGLang client: url=%s, max_input_tokens=%d, max_new_tokens=%d",
        sglang_url,
        max_input_tokens,
        max_new_tokens,
    )

    return SGLangClient(
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        url=sglang_url,
        max_input_tokens=max_input_tokens,
        max_retries=max_retries,
        **client_template_kwargs,
    )


class SGLangClient:
    def __init__(
        self,
        *,
        model_type: str | None = None,
        tokenizer,
        sampling_params: Dict[str, Any],
        url: str,
        chat_template_type: str = "hf",
        chat_template_kwargs: Dict[str, Any] | None = None,
        messages_delimiter_start: str = "<|im_start|>",
        messages_delimiter_end: str = "<|im_end|>",
        tool_call_parser: str | None = None,
        max_input_tokens: int | None = None,
        max_retries: int = 30,
    ) -> None:
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.sampling_params = _ensure_stop_token_ids(tokenizer, sampling_params)
        self.url = url
        self.chat_template_type = chat_template_type
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.messages_delimiter_start = messages_delimiter_start
        self.messages_delimiter_end = messages_delimiter_end
        self.tool_call_parser = tool_call_parser
        self.max_input_tokens = max_input_tokens
        self.max_retries = max_retries

        self.marker = "\n[OMITTED MIDDLE]\n"
        self.sep_ids = self.tokenizer.encode(self.marker, add_special_tokens=False)

    def _truncate_input_ids(self, input_ids: List[int]) -> List[int]:
        max_toks = self.max_input_tokens
        if max_toks is None or len(input_ids) <= max_toks:
            return input_ids

        dropped = len(input_ids) - max_toks
        logger.warning(
            "Prompt is too long for configured budget: input=%d, budget=%d. Truncating %d token(s) from the left and right.",
            len(input_ids),
            max_toks,
            dropped,
        )

        keep_head_ratio = getattr(self, "keep_head_ratio", 0.3)
        head = max(1, int(max_toks * keep_head_ratio))
        tail = max_toks - head - len(self.sep_ids)
        if tail <= 0:
            logger.warning(
                f"tail is not positive: tail={tail}, head={head}, max_toks={max_toks}, len(self.sep_ids)={len(self.sep_ids)}"
            )
            tail = 1

        return input_ids[:head] + self.sep_ids + input_ids[-tail:]

    async def generate(
        self,
        *,
        messages: List[dict[str, Any]],
        tools: List[dict[str, Any]] | None,
        turn_idx: int,
    ) -> tuple[ChatCompletion, Interaction]:
        input_ids = self._apply_chat_template(messages, tools)
        input_ids = self._truncate_input_ids(input_ids)
        payload: Dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": self.sampling_params,
            "return_logprob": True,
        }

        t0 = time.monotonic()
        output = await async_post(self.url, payload, max_retries=self.max_retries)
        latency_ms = (time.monotonic() - t0) * 1000.0

        output_text: str = output["text"]
        raw_output_text = output_text
        meta_info = output["meta_info"]
        finish_reason: str = meta_info["finish_reason"]["type"]

        if "output_token_logprobs" in meta_info:
            raw_logprobs = meta_info["output_token_logprobs"]
            if raw_logprobs and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "output_token_logprobs sample element: %s", raw_logprobs[0]
                )
            output_token_ids: list[int] = [x[1] for x in raw_logprobs]
            output_token_logprobs: list[float] = [x[0] for x in raw_logprobs]
        else:
            output_token_ids = []
            output_token_logprobs = []

        tool_calls = None
        if tools:
            tool_calls, output_text, finish_reason = process_tool_calls(
                output_text,
                tools,
                self.tool_call_parser,
                finish_reason,
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        current_time = int(datetime.datetime.now().timestamp())
        chat_completion = ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason=finish_reason,
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        content=output_text,
                        role="assistant",
                        tool_calls=tool_calls,
                    ),
                )
            ],
            created=current_time,
            model=self.model_type or "unknown",
            object="chat.completion",
            service_tier=None,
            system_fingerprint=None,
            usage=CompletionUsage(
                prompt_tokens=len(input_ids),
                completion_tokens=len(output_token_ids),
                total_tokens=len(input_ids) + len(output_token_ids),
            ),
        )

        interaction = Interaction(
            turn_idx=turn_idx,
            completion=deepcopy(chat_completion),
            input_ids=list(input_ids),
            output_token_ids=output_token_ids,
            output_token_logprobs=output_token_logprobs,
            output_text=raw_output_text,
            finish_reason=finish_reason,
            messages=deepcopy(messages),
            latency_ms=latency_ms,
        )
        return chat_completion, interaction

    def _apply_chat_template(
        self,
        messages: List[dict[str, Any]],
        tools: List[dict[str, Any]] | None,
    ) -> List[int]:
        if self.chat_template_type == "hf":
            try:
                result = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools or None,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.chat_template_kwargs,
                )
            except Exception:
                result = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.chat_template_kwargs,
                )
            # tokenizer may return BatchEncoding instead of list[int]
            if hasattr(result, "input_ids"):
                result = result["input_ids"]
            return result if isinstance(result, list) else list(result)

        if self.chat_template_type == "concat":
            start = self.messages_delimiter_start
            end = self.messages_delimiter_end
            message_strs: List[str] = []
            for msg in messages:
                message_strs.append(f"{start}{msg['role']}\n{msg['content']}{end}\n")
            message_strs.append(f"{start}assistant\n")
            return self.tokenizer.encode("".join(message_strs))

        raise ValueError(f"Unsupported chat_template_type: {self.chat_template_type!r}")
