"""OpenAI-compatible RL proxy for a3s-code agent training.

This is the core RL proxy that sits between an OpenAI-compatible client
(any agent, CLI, or UI that speaks the chat completions API) and the
SGLang policy server.  It collects (prompt, response, logprobs) tuples,
organises them into sessions/turns, scores them with PRM or rule rewards,
and feeds completed samples into the slime async trainer.

Architecture (no SafeClaw dependency):

    Any OpenAI client (curl / a3s-code agent / UI)
      -> code_rl_api_server.py  (this file, OpenAI-compat RL proxy)
        -> SGLang policy server  (LLM inference + logprobs)
        -> slime trainer         (RL training loop)

The proxy is completely framework-agnostic on the client side.  It works
with *any* OpenAI-compatible caller — a3s-code's built-in OpenAI LLM
client, a plain `curl`, or a React UI.
"""

import asyncio
import ast
import collections
import json
import logging
import math
import os
import queue
import re
import threading
import time
from itertools import count
from operator import add, floordiv, mod, mul, neg, pos, pow, sub, truediv
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Function, Tool

from slime.utils.metric_utils import has_repetition
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

_BOXED_RE = re.compile(r"\\boxed\{([-+]?\d)\}")
_NON_STANDARD_BODY_KEYS = {
    "session_id",
    "session_done",
    "turn_type",
    "channel",
}
_DEFAULT_DROPPED_MODEL_TOOLS = {"task", "parallel_task"}
_MATH_EXPR_PATTERNS = [
    re.compile(r"(?:compute|calculate|evaluate|what is|what's)\s+([^=\n\r?]+)", re.IGNORECASE),
    re.compile(r"(?:solve|answer)\s*[:：]?\s*([^=\n\r?]+)", re.IGNORECASE),
]
_FINAL_ANSWER_PATTERNS = [
    re.compile(r"final answer\s*[:：]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"答案\s*[:：]\s*([-+]?\d+(?:\.\d+)?)"),
]
_SESSION_GROUP_RE = re.compile(r"(?:^|-)grp(\d+)(?:-|$)")
_ALLOWED_BINOPS = {
    ast.Add: add,
    ast.Sub: sub,
    ast.Mult: mul,
    ast.Div: truediv,
    ast.FloorDiv: floordiv,
    ast.Mod: mod,
    ast.Pow: pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: pos,
    ast.USub: neg,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_short_repetition(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)
    if len(tokens) < 24:
        return False

    normalized = [token.lower() for token in tokens if token.strip()]
    if not normalized:
        return False

    counts = collections.Counter(normalized)
    most_common = counts.most_common(1)[0][1]
    if most_common / len(normalized) >= 0.45 and len(counts) <= max(6, len(normalized) // 8):
        return True

    for ngram_size in (1, 2, 3):
        if len(normalized) < ngram_size * 8:
            continue
        ngrams = [
            tuple(normalized[i : i + ngram_size])
            for i in range(0, len(normalized) - ngram_size + 1, ngram_size)
        ]
        if ngrams and collections.Counter(ngrams).most_common(1)[0][1] >= 8:
            return True

    return False


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env_set(name: str, default: set[str]) -> set[str]:
    raw = os.getenv(name)
    if raw is None:
        return set(default)
    if raw.strip().lower() in {"", "0", "none", "null", "false", "off", "disabled"}:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {raw!r}") from exc


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


def _normalize_openai_chat_url(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/v1/chat/completions") or cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def _guess_openai_models_url(chat_url: str) -> str:
    cleaned = chat_url.strip()
    if not cleaned:
        return ""
    if cleaned.endswith("/chat/completions"):
        return cleaned[: -len("/chat/completions")] + "/models"
    return cleaned


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return " ".join(parts).strip()
    return str(content) if content is not None else ""


def _normalize_text_for_turn_matching(content: Any) -> str:
    flat = _flatten_message_content(content).strip().lower()
    if not flat:
        return ""
    return " ".join(flat.split())


def _normalize_messages_for_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        item = dict(msg)
        if item.get("role") == "developer":
            item["role"] = "system"
        raw = item.get("content")
        if not isinstance(raw, str) and raw is not None:
            item["content"] = _flatten_message_content(raw)
        if item.get("tool_calls"):
            item["tool_calls"] = [_normalize_tool_call(tc) for tc in item["tool_calls"]]
        out.append(item)
    return out


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    item = dict(tool_call)
    function = item.get("function")
    if isinstance(function, dict):
        function = dict(function)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                parsed_arguments = {}
            function["arguments"] = parsed_arguments if isinstance(parsed_arguments, dict) else {}
        item["function"] = function
    return item


def _normalize_messages_for_matching(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        item: dict[str, Any] = {
            "role": msg.get("role", ""),
            "content": _flatten_message_content(msg.get("content")),
        }
        if "name" in msg:
            item["name"] = msg.get("name")
        if "tool_call_id" in msg:
            item["tool_call_id"] = msg.get("tool_call_id")
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            tool_calls = []
            for tc in msg["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                tool_calls.append(
                    {
                        "id": tc.get("id"),
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments"),
                    }
                )
            if tool_calls:
                item["tool_calls"] = tool_calls
        normalized.append(item)
    return normalized


def _extract_logprobs_from_chat_response(choice: dict[str, Any]) -> list[float]:
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    content = logprobs_obj.get("content")
    if not isinstance(content, list):
        return []
    return [float(item.get("logprob", 0.0)) for item in content if isinstance(item, dict)]


def _is_sglang_chat_batch_error(status_code: int, response_text: str) -> bool:
    return (
        status_code == 400
        and "input_ids should be a list of lists for batch processing" in response_text
    )


def _normalize_finish_reason(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("type") or value.get("reason")
    if value is None:
        return "stop"
    return str(value)


# ---------------------------------------------------------------------------
# PRM (Process Reward Model)
# ---------------------------------------------------------------------------

def _build_prm_judge_prompt(
    response_text: str,
    next_state_text: str,
    next_state_role: str = "user",
) -> list[dict[str, str]]:
    system = (
        "You are a process reward model (PRM) evaluating an AI assistant.\n"
        "You will see the assistant's output and the subsequent next state.\n"
        "Your task: decide whether the assistant's output successfully fulfilled the user's intent "
        "at that step, using the next state as evidence.\n\n"
        "## Understanding the next state's role\n"
        "- role='user': A reply from the user.\n"
        "- role='tool': The return value of a tool the assistant invoked. "
        "This content was NOT available before the assistant's action. "
        "A successful, non-error tool output should usually be scored positively.\n\n"
        "## Scoring rules\n"
        "- \\boxed{1}: The next state shows clear progress or success.\n"
        "- \\boxed{-1}: The next state shows the output was wrong, incomplete, or unwanted.\n"
        "- \\boxed{0}: The next state is ambiguous or insufficient for judgment.\n\n"
        "Think briefly, then give the final score inside \\boxed{}."
    )
    user = (
        f"## Assistant output\n{response_text}\n\n"
        f"## Next state [role: {next_state_role}]\n{next_state_text}\n\n"
        "Now assign \\boxed{1}, \\boxed{-1}, or \\boxed{0}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_prm_score(text: str) -> int | None:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    value = int(matches[-1])
    if value in (1, -1, 0):
        return value
    return None


def _majority_vote(scores: list[int | None]) -> float:
    valid = [score for score in scores if score is not None]
    if not valid:
        return 0.0
    counter = collections.Counter(valid)
    top = counter.most_common(1)[0]
    if list(counter.values()).count(top[1]) > 1:
        return 0.0
    return float(top[0])


# ---------------------------------------------------------------------------
# Rule-based debug math reward
# ---------------------------------------------------------------------------

def _safe_eval_arithmetic(expr: str) -> float:
    expr = expr.strip()
    if not expr or len(expr) > 128:
        raise ValueError("expression too long")
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return float(_ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return float(_ALLOWED_UNARYOPS[type(node.op)](_eval(node.operand)))
        raise ValueError(f"unsupported arithmetic node: {type(node).__name__}")

    return _eval(tree)


def _extract_debug_math_expression(user_text: str) -> str | None:
    text = user_text.strip()
    if not text:
        return None
    text = text.replace("×", "*").replace("÷", "/").replace("^", "**")
    text = re.sub(r"\bmod\b", "%", text, flags=re.IGNORECASE)

    candidates = [text]
    for pattern in _MATH_EXPR_PATTERNS:
        match = pattern.search(text)
        if match:
            candidates.append(match.group(1))

    for candidate in candidates:
        expr = candidate.strip().strip(".!?;:，。？！；：")
        if not expr:
            continue
        if re.fullmatch(r"[\d\s\+\-\*\/%\(\)\.]+", expr):
            return expr
    return None


def _extract_final_answer_number(response_text: str) -> float | None:
    for pattern in _FINAL_ANSWER_PATTERNS:
        match = pattern.search(response_text)
        if match:
            return float(match.group(1))

    numeric_lines: list[str] = []
    for raw_line in response_text.strip().splitlines():
        line = raw_line.strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", line):
            numeric_lines.append(line)
    if numeric_lines:
        return float(numeric_lines[-1])

    all_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", response_text)
    if all_numbers:
        return float(all_numbers[-1])
    return None


def _maybe_rule_reward_debug_math(
    user_text: str,
    response_text: str,
) -> dict[str, Any] | None:
    expr = _extract_debug_math_expression(user_text)
    if expr is None:
        return None

    answer = _extract_final_answer_number(response_text)
    if answer is None:
        return {
            "mode": "debug_math_rule",
            "expression": expr,
            "expected": None,
            "predicted": None,
            "score": 0.0,
            "reason": "answer_not_found",
        }

    try:
        expected = _safe_eval_arithmetic(expr)
    except Exception:
        return None

    if abs(expected - round(expected)) < 1e-9:
        expected = float(round(expected))
    if abs(answer - round(answer)) < 1e-9:
        answer = float(round(answer))

    score = 1.0 if abs(expected - answer) < 1e-6 else -1.0
    return {
        "mode": "debug_math_rule",
        "expression": expr,
        "expected": expected,
        "predicted": answer,
        "score": score,
        "reason": "matched" if score > 0 else "mismatch",
    }


# ---------------------------------------------------------------------------
# slime integration functions (used by train_async.py)
# ---------------------------------------------------------------------------

async def reward_func(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [
            {"score": s.reward.get("score", 0.0) if isinstance(s.reward, dict) else 0.0}
            for s in sample_or_samples
        ]
    sample = sample_or_samples
    return {
        "score": sample.reward.get("score", 0.0) if isinstance(sample.reward, dict) else 0.0
    }


def _normalize_input_ids_for_generate(raw_input_ids) -> list[int]:
    """Convert tokenizer outputs to a plain JSON-serializable token-id list."""
    input_ids = raw_input_ids
    if hasattr(input_ids, "keys") and "input_ids" in input_ids:
        input_ids = input_ids["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, tuple):
        input_ids = list(input_ids)
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    messages = sample.prompt
    if not isinstance(messages, list):
        messages = [{"role": "user", "content": str(sample.prompt)}]

    input_ids = _normalize_input_ids_for_generate(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    ))
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        output = response.json()

    text = output.get("text", "")
    meta = output.get("meta_info", {})
    pairs = meta.get("output_token_logprobs", [])
    if isinstance(pairs, list) and pairs:
        token_ids = [int(pair[1]) for pair in pairs if isinstance(pair, (list, tuple)) and len(pair) >= 2]
        logprobs = [float(pair[0]) for pair in pairs if isinstance(pair, (list, tuple)) and len(pair) >= 2]
    else:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        logprobs = [0.0] * len(token_ids)

    sample.tokens = input_ids + token_ids
    sample.response = text
    sample.response_length = len(token_ids)
    sample.rollout_log_probs = logprobs
    sample.loss_mask = [1] * len(token_ids)
    sample.status = Sample.Status.COMPLETED
    return sample


# ===========================================================================
# CodeRLAPIServer
# ===========================================================================

class CodeRLAPIServer:
    """OpenAI-compatible RL proxy for a3s-code agent training.

    Accepts standard ``/v1/chat/completions`` traffic from any OpenAI client
    (a3s-code's built-in OpenAI LLM client, curl, etc.), forwards to the
    SGLang policy server, collects (prompt, response, logprobs), scores with
    PRM or rule rewards, and feeds samples into the slime async trainer.

    Session tracking:
        1. Header ``X-Session-Id`` or body ``session_id``
        2. Body ``user`` field
        3. Full message-history prefix matching (auto-infer)
    """

    def __init__(self, args, output_queue: queue.Queue, submission_enabled: threading.Event):
        self.args = args
        self.output_queue = output_queue
        self.submission_enabled = submission_enabled
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.sglang_chat_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
        )
        self.sglang_generate_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        )
        self.sglang_health_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/health"
        )
        self.expected_api_key = os.getenv("SGLANG_API_KEY", "")
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "30000"))
        self.served_model_name = os.getenv("SERVED_MODEL_NAME", "qwen3-4b-2507")
        self.tool_call_parser = os.getenv("TOOL_CALL_PARSER", "qwen25")
        matched_context_tokens = os.getenv("CODE_RL_MATCHED_CONTEXT_TOKENS", "16384")
        self.max_train_tokens = int(os.getenv("CODE_RL_MAX_TRAIN_TOKENS", matched_context_tokens))
        self.context_length = int(
            os.getenv(
                "CONTEXT_LENGTH",
                os.getenv("ROLLOUT_MAX_CONTEXT_LEN", matched_context_tokens),
            )
        )
        self.context_safety_margin = int(os.getenv("CODE_RL_CONTEXT_SAFETY_MARGIN", "512"))
        self.max_response_tokens = int(os.getenv("CODE_RL_MAX_RESPONSE_TOKENS", "4096"))
        self.response_trim_margin_tokens = int(
            os.getenv("CODE_RL_RESPONSE_TRIM_MARGIN_TOKENS", "8")
        )
        self.drop_model_tools = _csv_env_set(
            "CODE_RL_DROP_MODEL_TOOLS",
            _DEFAULT_DROPPED_MODEL_TOOLS,
        )
        self.drop_repetitive_samples = _env_flag("CODE_RL_DROP_REPETITIVE_SAMPLES", True)
        self.pause_wait_timeout_sec = _env_float("CODE_RL_PAUSE_WAIT_TIMEOUT_SEC", 1800.0)
        self.pause_drain_timeout_sec = _env_float("CODE_RL_PAUSE_DRAIN_TIMEOUT_SEC", 1800.0)
        self.generation_timeout_sec = _env_float("CODE_RL_GENERATION_TIMEOUT_SEC", 600.0)
        self._inflight_generation_requests = 0
        self._inflight_generation_condition = threading.Condition()
        self._submission_epoch = 0

        self._index_counter = count(0)
        self._group_counter = count(0)
        self._auto_session_counter = count(1)
        self._trace_counter = count(1)
        self._turn_counts: dict[str, int] = {}
        self._pending_records: dict[str, dict[str, Any]] = {}
        self._pending_turn_data: dict[str, dict[int, dict[str, Any]]] = {}
        self._prm_tasks: dict[str, dict[int, asyncio.Task]] = {}
        self._submit_tasks: dict[str, set[asyncio.Task]] = {}
        self._session_effective: dict[str, int] = {}
        self._session_latest_messages: dict[str, list[dict[str, Any]]] = {}
        self._session_last_activity: dict[str, float] = {}
        self._finalizing_sessions: set[str] = set()
        self._overflow_terminated_sessions: set[str] = set()
        self._turn_feedback: dict[str, dict[int, list[dict[str, Any]]]] = {}

        self._submit_side = _env_flag("CODE_RL_SUBMIT_SIDE", False)
        self._train_side = _env_flag("CODE_RL_TRAIN_SIDE", False)
        self._reward_mode = os.getenv("CODE_RL_REWARD_MODE", "hybrid").strip().lower()
        self._require_verifier_feedback = _env_flag("CODE_RL_REQUIRE_VERIFIER_FEEDBACK", False)
        self._idle_flush_sec = int(os.getenv("CODE_RL_SESSION_IDLE_FLUSH_SEC", "30"))

        self._prm_openai_url = _normalize_openai_chat_url(os.getenv("CODE_RL_PRM_OPENAI_URL", ""))
        self._prm_openai_model = os.getenv("CODE_RL_PRM_OPENAI_MODEL_NAME", os.getenv("CODE_RL_PRM_MODEL_NAME", "")).strip()
        self._prm_openai_api_key = os.getenv("CODE_RL_PRM_API_KEY", "")
        self._prm_openai_extra_body = _json_env_dict("CODE_RL_PRM_OPENAI_EXTRA_BODY")
        self._prm_use_openai = bool(self._prm_openai_url and self._prm_openai_model)
        self._prm_enabled = bool(getattr(args, "prm_enable", False) or self._prm_use_openai)
        self._prm_m = int(os.getenv("PRM_M", getattr(args, "prm_m", 3)))
        self._prm_temperature = float(getattr(args, "prm_temperature", 0.6))
        self._prm_max_tokens = int(getattr(args, "prm_max_new_tokens", 8192))
        self._prm_timeout_sec = float(os.getenv("CODE_RL_PRM_TIMEOUT_SEC", "180"))
        prm_ip = getattr(args, "prm_router_ip", None)
        prm_port = getattr(args, "prm_router_port", None)
        self._prm_url = f"http://{prm_ip}:{prm_port}/generate" if getattr(args, "prm_enable", False) and prm_ip and prm_port else ""
        self._prm_health_url = os.getenv("CODE_RL_PRM_HEALTH_URL", "").strip()
        if self._prm_use_openai and not self._prm_health_url:
            self._prm_health_url = _guess_openai_models_url(self._prm_openai_url)
        if self._prm_use_openai:
            self._prm_backend = "external_openai"
        elif self._prm_url:
            self._prm_backend = "local_sglang"
        else:
            self._prm_backend = "disabled"
        self._prm_tokenizer = None
        if getattr(args, "prm_enable", False):
            prm_path = getattr(args, "prm_model_path", None) or args.hf_checkpoint
            self._prm_tokenizer = load_tokenizer(prm_path, trust_remote_code=True)

        self._eval_scores: list[float] = []
        self._eval_scores_lock = threading.Lock()
        self._stats = collections.Counter()
        self._stats_lock = threading.Lock()

        self._record_file = (
            os.getenv("CODE_RL_RECORD_FILE", "")
            if _env_flag("CODE_RL_RECORD_ENABLED", True)
            else ""
        )
        self._prm_record_file = os.getenv("CODE_RL_PRM_RECORD_FILE", "")
        self._feedback_record_file = os.getenv("CODE_RL_FEEDBACK_RECORD_FILE", "")
        self._trace_record_file = (
            os.getenv("CODE_RL_TRACE_RECORD_FILE", "")
            if _env_flag("CODE_RL_TRACE_RECORD_ENABLED", True)
            else ""
        )
        if not self._prm_record_file and self._record_file and self._prm_enabled:
            base, ext = os.path.splitext(self._record_file)
            self._prm_record_file = f"{base}_prm{ext}"
        if not self._feedback_record_file and self._record_file:
            base, ext = os.path.splitext(self._record_file)
            self._feedback_record_file = f"{base}_feedback{ext}"
        if not self._trace_record_file and self._record_file:
            base, ext = os.path.splitext(self._record_file)
            self._trace_record_file = f"{base}_trace{ext}"
        self._purge_record_files_on_pause = _env_flag(
            "CODE_RL_PURGE_RECORD_FILES_ON_PAUSE",
            False,
        )
        for path in [
            self._record_file,
            self._prm_record_file,
            self._feedback_record_file,
            self._trace_record_file,
        ]:
            if path:
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                open(path, "w").close()

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._housekeeping_task: asyncio.Task | None = None
        self.app = self._build_app()

    def _filter_model_tools(self, tools: Any) -> list[dict[str, Any]] | None:
        if not isinstance(tools, list):
            return tools
        if not self.drop_model_tools:
            return tools

        filtered: list[dict[str, Any]] = []
        dropped: list[str] = []
        for tool in tools:
            name = ""
            if isinstance(tool, dict):
                function_def = tool.get("function")
                if isinstance(function_def, dict):
                    name = str(function_def.get("name", "")).strip()
            if name and name in self.drop_model_tools:
                dropped.append(name)
                continue
            filtered.append(tool)

        if dropped:
            self._incr_stat("dropped_model_tools_total", len(dropped))
            for name in dropped:
                self._incr_stat(f"dropped_model_tool_{name}")
            logger.info("[Code-RL] dropped model tools: %s", sorted(set(dropped)))
        return filtered or None

    def _incr_stat(self, key: str, value: int | float = 1) -> None:
        stats = getattr(self, "_stats", None)
        lock = getattr(self, "_stats_lock", None)
        if stats is None:
            return
        if lock is None:
            stats[key] += value
            return
        with lock:
            stats[key] += value

    def snapshot_stats(self) -> dict[str, Any]:
        stats = getattr(self, "_stats", collections.Counter())
        lock = getattr(self, "_stats_lock", None)
        if lock is None:
            counters = dict(stats)
        else:
            with lock:
                counters = dict(stats)

        inflight_condition = getattr(self, "_inflight_generation_condition", None)
        if inflight_condition is None:
            inflight = int(getattr(self, "_inflight_generation_requests", 0))
            submission_epoch = int(getattr(self, "_submission_epoch", 0))
        else:
            with inflight_condition:
                inflight = int(getattr(self, "_inflight_generation_requests", 0))
                submission_epoch = int(getattr(self, "_submission_epoch", 0))

        eval_scores = getattr(self, "_eval_scores", [])
        eval_scores_lock = getattr(self, "_eval_scores_lock", None)
        if eval_scores_lock is None:
            eval_scores_snapshot = list(eval_scores)
        else:
            with eval_scores_lock:
                eval_scores_snapshot = list(eval_scores)

        pending_turn_data = getattr(self, "_pending_turn_data", {})
        prm_tasks = getattr(self, "_prm_tasks", {})
        submit_tasks = getattr(self, "_submit_tasks", {})
        queue_size = None
        try:
            queue_size = self.output_queue.qsize()
        except Exception:
            pass

        return {
            "submission_enabled": bool(getattr(self, "submission_enabled", None) and self.submission_enabled.is_set()),
            "submission_epoch": submission_epoch,
            "inflight_generation_requests": inflight,
            "queue_size": queue_size,
            "counters": counters,
            "pending": {
                "sessions": len(pending_turn_data),
                "turns": sum(len(turns) for turns in pending_turn_data.values()),
                "records": len(getattr(self, "_pending_records", {})),
                "finalizing_sessions": len(getattr(self, "_finalizing_sessions", set())),
                "prm_tasks": sum(len(tasks) for tasks in prm_tasks.values()),
                "live_submit_tasks": sum(
                    1 for tasks in submit_tasks.values() for task in tasks if not task.done()
                ),
            },
            "reward": {
                "mode": getattr(self, "_reward_mode", "unknown"),
                "prm_backend": getattr(self, "_prm_backend", "unknown"),
                "require_verifier_feedback": bool(getattr(self, "_require_verifier_feedback", False)),
                "eval_scores_count": len(eval_scores_snapshot),
                "eval_score_mean": (
                    sum(eval_scores_snapshot) / len(eval_scores_snapshot)
                    if eval_scores_snapshot
                    else None
                ),
            },
        }

    def _build_synthetic_chat_response(
        self,
        *,
        content: str,
        prompt_token_count: int,
        reasoning_content: str = "",
        finish_reason: str = "stop",
    ) -> dict[str, Any]:
        completion_tokens = len(
            self.tokenizer(content or "", add_special_tokens=False)["input_ids"]
        )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        now = int(time.time())
        synthetic_id = f"chatcmpl-synthetic-{now}-{next(self._group_counter)}"
        return {
            "id": synthetic_id,
            "object": "chat.completion",
            "created": now,
            "model": self.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                    "logprobs": {"content": []},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_token_count + completion_tokens,
            },
        }

    def _build_sglang_sampling_params(
        self,
        forward_body: dict[str, Any],
        requested_max_tokens: int | None,
    ) -> dict[str, Any]:
        sampling_params: dict[str, Any] = {
            "temperature": float(forward_body.get("temperature", 0.0) or 0.0),
            "top_p": float(forward_body.get("top_p", 1.0) or 1.0),
            "top_k": int(forward_body.get("top_k", -1) or -1),
            "max_new_tokens": max(1, requested_max_tokens or self.max_response_tokens),
        }
        stop = forward_body.get("stop")
        if stop:
            sampling_params["stop"] = stop
        stop_token_ids = forward_body.get("stop_token_ids")
        if isinstance(stop_token_ids, list) and stop_token_ids:
            sampling_params["stop_token_ids"] = stop_token_ids
        min_new_tokens = forward_body.get("min_new_tokens")
        if min_new_tokens is not None:
            try:
                sampling_params["min_new_tokens"] = int(min_new_tokens)
            except (TypeError, ValueError):
                pass
        return sampling_params

    def _parse_tool_calls_from_generate_text(
        self,
        response_text: str,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        if not isinstance(tools, list) or not tools:
            return response_text, []

        tool_defs: list[Tool] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function_def = tool.get("function")
            if not isinstance(function_def, dict):
                continue
            name = str(function_def.get("name", "")).strip()
            if not name:
                continue
            tool_defs.append(
                Tool(
                    type=str(tool.get("type", "function")),
                    function=Function(
                        name=name,
                        description=str(function_def.get("description", "")),
                        parameters=function_def.get("parameters") or {},
                    ),
                )
            )
        if not tool_defs:
            return response_text, []

        try:
            parser = FunctionCallParser(
                tools=tool_defs,
                tool_call_parser=self.tool_call_parser,
            )
            normal_text, calls = parser.parse_non_stream(response_text)
            return normal_text, [call.model_dump() for call in calls]
        except Exception as exc:
            logger.warning("[Code-RL] tool-call parse fallback failed: %s", exc)
            return response_text, []

    def _wrap_generate_output_as_chat_response(
        self,
        data: dict[str, Any],
        *,
        prompt_token_count: int,
        model_name: str,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        raw_text = _flatten_message_content(data.get("text", ""))
        content, tool_calls = self._parse_tool_calls_from_generate_text(raw_text, tools)
        meta_info = data.get("meta_info") if isinstance(data.get("meta_info"), dict) else {}

        logprob_items: list[dict[str, Any]] = []
        for item in meta_info.get("output_token_logprobs") or []:
            token_id = None
            logprob = 0.0
            if isinstance(item, list):
                if item:
                    try:
                        logprob = float(item[0] or 0.0)
                    except (TypeError, ValueError):
                        logprob = 0.0
                if len(item) > 1:
                    token_id = item[1]
            elif isinstance(item, dict):
                token_id = item.get("token_id")
                try:
                    logprob = float(item.get("logprob", 0.0))
                except (TypeError, ValueError):
                    logprob = 0.0

            token_text = ""
            if token_id is not None:
                try:
                    token_text = self.tokenizer.decode(
                        [int(token_id)],
                        skip_special_tokens=False,
                    )
                except Exception:
                    token_text = ""
            logprob_items.append({"token": token_text, "logprob": logprob})

        completion_tokens = meta_info.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = len(
                self.tokenizer(raw_text or "", add_special_tokens=False)["input_ids"]
            )
        prompt_tokens = meta_info.get("prompt_tokens", prompt_token_count)

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        now = int(time.time())
        return {
            "id": meta_info.get("id") or f"chatcmpl-generate-{now}-{next(self._group_counter)}",
            "object": "chat.completion",
            "created": now,
            "model": model_name or self.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": _normalize_finish_reason(meta_info.get("finish_reason")),
                    "logprobs": {"content": logprob_items},
                }
            ],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens) + int(completion_tokens),
            },
            "_code_rl_raw_response_text": raw_text,
        }

    async def _generate_chat_fallback(
        self,
        client: httpx.AsyncClient,
        *,
        forward_body: dict[str, Any],
        prompt_ids: list[int],
        prompt_token_count: int,
        requested_max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload = {
            "input_ids": prompt_ids,
            "sampling_params": self._build_sglang_sampling_params(
                forward_body,
                requested_max_tokens,
            ),
            "return_logprob": True,
        }
        response = await client.post(self.sglang_generate_url, json=payload)
        if response.status_code != 200:
            logger.error(
                "[Code-RL] SGLang /generate returned %d: %s",
                response.status_code,
                response.text[:1000],
            )
            detail: Any
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:2000]
            raise HTTPException(status_code=response.status_code, detail=detail)
        return self._wrap_generate_output_as_chat_response(
            response.json(),
            prompt_token_count=prompt_token_count,
            model_name=str(forward_body.get("model") or self.served_model_name),
            tools=tools,
        )

    async def _wait_for_submission_enabled(self, *, session_id: str, turn_type: str) -> None:
        if self.submission_enabled.is_set():
            return

        timeout_sec = max(0.0, float(self.pause_wait_timeout_sec))
        logger.info(
            "[Code-RL] waiting for submission resume session=%s turn_type=%s timeout=%.1fs",
            session_id,
            turn_type,
            timeout_sec,
        )
        deadline = asyncio.get_running_loop().time() + timeout_sec
        while True:
            if self.submission_enabled.is_set():
                logger.info(
                    "[Code-RL] submission resumed session=%s turn_type=%s",
                    session_id,
                    turn_type,
                )
                return
            remaining_sec = deadline - asyncio.get_running_loop().time()
            if remaining_sec <= 0:
                break
            await asyncio.sleep(min(1.0, remaining_sec))

        logger.warning(
            "[Code-RL] submission pause wait timed out session=%s turn_type=%s timeout=%.1fs",
            session_id,
            turn_type,
            timeout_sec,
        )
        raise HTTPException(
            status_code=503,
            detail="submission paused for weight update; retry after resume",
        )

    # -----------------------------------------------------------------------
    # FastAPI app
    # -----------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="A3S Code RL Proxy")
        app.state.owner = self

        @app.on_event("startup")
        async def startup():
            owner: CodeRLAPIServer = app.state.owner
            owner._housekeeping_task = asyncio.create_task(owner._housekeeping_loop())

        @app.on_event("shutdown")
        async def shutdown():
            owner: CodeRLAPIServer = app.state.owner
            if owner._housekeeping_task is not None:
                owner._housekeeping_task.cancel()
                try:
                    await owner._housekeeping_task
                except asyncio.CancelledError:
                    pass
                owner._housekeeping_task = None

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.get("/stats")
        async def stats():
            owner: CodeRLAPIServer = app.state.owner
            return owner.snapshot_stats()

        async def _handle_chat(
            request: Request,
            path_session_id: str | None = None,
            authorization: str | None = Header(default=None),
            x_session_id: str | None = Header(default=None),
            x_turn_type: str | None = Header(default=None),
            x_session_done: str | None = Header(default=None),
            x_channel: str | None = Header(default=None),
        ):
            owner: CodeRLAPIServer = request.app.state.owner
            await owner._check_auth(authorization)

            body = await request.json()
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise HTTPException(status_code=400, detail="messages must be a non-empty list")

            explicit_session_id = (
                path_session_id or x_session_id or body.get("session_id") or body.get("user")
            )
            session_id = owner._resolve_session_id(explicit_session_id, messages)
            turn_type = owner._resolve_turn_type(x_turn_type or body.get("turn_type"), messages)
            session_done = _coerce_bool(x_session_done) or _coerce_bool(body.get("session_done"))
            request_meta = {
                "channel": x_channel or body.get("channel") or "api",
            }
            await owner._wait_for_submission_enabled(
                session_id=session_id,
                turn_type=turn_type,
            )

            request_epoch = owner._begin_generation_request(
                session_id=session_id,
                turn_type=turn_type,
            )
            if request_epoch is None:
                raise HTTPException(
                    status_code=503,
                    detail="submission paused for weight update; retry after resume",
                )
            try:
                stream = bool(body.get("stream", False))
                result = await owner._handle_request(
                    body,
                    session_id=session_id,
                    turn_type=turn_type,
                    session_done=session_done,
                    request_meta=request_meta,
                    submission_epoch=request_epoch,
                )
            finally:
                owner._end_generation_request(session_id=session_id, turn_type=turn_type)
            if stream:
                return StreamingResponse(
                    owner._stream_response(result),
                    media_type="text/event-stream",
                )
            return JSONResponse(content=result["response"])

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: str | None = Header(default=None),
            x_session_id: str | None = Header(default=None),
            x_turn_type: str | None = Header(default=None),
            x_session_done: str | None = Header(default=None),
            x_channel: str | None = Header(default=None),
        ):
            return await _handle_chat(
                request,
                authorization=authorization,
                x_session_id=x_session_id,
                x_turn_type=x_turn_type,
                x_session_done=x_session_done,
                x_channel=x_channel,
            )

        @app.post("/session/{path_session_id}/v1/chat/completions")
        async def chat_completions_scoped(
            path_session_id: str,
            request: Request,
            authorization: str | None = Header(default=None),
            x_turn_type: str | None = Header(default=None),
            x_session_done: str | None = Header(default=None),
            x_channel: str | None = Header(default=None),
        ):
            return await _handle_chat(
                request,
                path_session_id=path_session_id,
                authorization=authorization,
                x_turn_type=x_turn_type,
                x_session_done=x_session_done,
                x_channel=x_channel,
            )

        @app.post("/session_done")
        async def session_done_endpoint(
            request: Request,
            authorization: str | None = Header(default=None),
            x_session_id: str | None = Header(default=None),
        ):
            owner: CodeRLAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            body = await request.json()
            session_id = x_session_id or body.get("session_id")
            if not session_id:
                raise HTTPException(status_code=400, detail="session_id is required")
            owner._finalize_session(str(session_id), reason="explicit_done")
            return {"ok": True, "session_id": session_id}

        @app.post("/feedback")
        async def feedback_endpoint(
            request: Request,
            authorization: str | None = Header(default=None),
            x_session_id: str | None = Header(default=None),
        ):
            owner: CodeRLAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            body = await request.json()
            session_id = x_session_id or body.get("session_id")
            if not session_id:
                raise HTTPException(status_code=400, detail="session_id is required")
            turn_id = body.get("turn_id")
            if turn_id is None:
                turn_id = owner._turn_counts.get(str(session_id))
            owner._record_feedback(str(session_id), turn_id, body)
            if str(body.get("event_type", "")) in {"verifier_reward", "task_verifier_reward"}:
                owner._maybe_submit_ready_samples(str(session_id), force_no_prm=True)
            return {"ok": True, "session_id": session_id, "turn_id": turn_id}

        return app

    # -----------------------------------------------------------------------
    # Request drain for weight updates
    # -----------------------------------------------------------------------

    def advance_submission_epoch(self) -> int:
        with self._inflight_generation_condition:
            self._submission_epoch += 1
            epoch = self._submission_epoch
            self._inflight_generation_condition.notify_all()
        logger.info("[Code-RL] advanced submission epoch to %d", epoch)
        return epoch

    def _current_submission_epoch(self) -> int:
        with self._inflight_generation_condition:
            return self._submission_epoch

    def _begin_generation_request(self, *, session_id: str, turn_type: str) -> int | None:
        with self._inflight_generation_condition:
            if not self.submission_enabled.is_set():
                logger.info(
                    "[Code-RL] rejecting generation start while submission is paused "
                    "session=%s turn_type=%s epoch=%d",
                    session_id,
                    turn_type,
                    self._submission_epoch,
                )
                return None
            self._inflight_generation_requests += 1
            current = self._inflight_generation_requests
            epoch = self._submission_epoch
        logger.info(
            "[Code-RL] generation request started session=%s turn_type=%s inflight=%d epoch=%d",
            session_id,
            turn_type,
            current,
            epoch,
        )
        return epoch

    def _end_generation_request(self, *, session_id: str, turn_type: str) -> None:
        with self._inflight_generation_condition:
            self._inflight_generation_requests = max(0, self._inflight_generation_requests - 1)
            current = self._inflight_generation_requests
            self._inflight_generation_condition.notify_all()
        logger.info(
            "[Code-RL] generation request finished session=%s turn_type=%s inflight=%d",
            session_id,
            turn_type,
            current,
        )

    def wait_for_inflight_generation_requests(self, timeout_sec: float | None = None) -> bool:
        timeout = self.pause_drain_timeout_sec if timeout_sec is None else max(0.0, timeout_sec)
        deadline = time.monotonic() + timeout if timeout > 0 else None
        next_log = time.monotonic()

        with self._inflight_generation_condition:
            while self._inflight_generation_requests > 0:
                now = time.monotonic()
                if now >= next_log:
                    logger.info(
                        "[Code-RL] waiting for %d in-flight generation request(s) before pause/update",
                        self._inflight_generation_requests,
                    )
                    next_log = now + 30.0
                if deadline is None:
                    wait_time = 5.0
                else:
                    remaining = deadline - now
                    if remaining <= 0:
                        logger.warning(
                            "[Code-RL] timed out waiting for %d in-flight generation request(s)",
                            self._inflight_generation_requests,
                        )
                        return False
                    wait_time = min(5.0, remaining)
                self._inflight_generation_condition.wait(timeout=wait_time)
            return True

    # -----------------------------------------------------------------------
    # Housekeeping
    # -----------------------------------------------------------------------

    async def _housekeeping_loop(self):
        while True:
            try:
                self.flush_idle_sessions()
            except Exception as exc:
                logger.warning("[Code-RL] housekeeping failed: %s", exc)
            await asyncio.sleep(1.0)

    async def _check_auth(self, authorization: str | None):
        if not self.expected_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self.expected_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    # -----------------------------------------------------------------------
    # Session tracking
    # -----------------------------------------------------------------------

    def _touch_session(self, session_id: str):
        self._session_last_activity[session_id] = time.time()
        self._finalizing_sessions.discard(session_id)

    def _resolve_session_id(
        self,
        explicit_session_id: Any,
        messages: list[dict[str, Any]],
    ) -> str:
        if explicit_session_id:
            session_id = str(explicit_session_id)
            self._touch_session(session_id)
            return session_id

        normalized = _normalize_messages_for_matching(messages)
        best_session_id: str | None = None
        best_prefix_len = -1
        best_activity = -1.0
        for session_id, prefix in self._session_latest_messages.items():
            if len(normalized) < len(prefix):
                continue
            if normalized[: len(prefix)] != prefix:
                continue
            last_activity = self._session_last_activity.get(session_id, 0.0)
            if len(prefix) > best_prefix_len or (
                len(prefix) == best_prefix_len and last_activity > best_activity
            ):
                best_session_id = session_id
                best_prefix_len = len(prefix)
                best_activity = last_activity

        if best_session_id is not None:
            self._touch_session(best_session_id)
            return best_session_id

        session_id = f"code-rl-auto-{next(self._auto_session_counter):08d}"
        self._touch_session(session_id)
        logger.info("[Code-RL] allocated inferred session_id=%s", session_id)
        return session_id

    @staticmethod
    def _resolve_turn_type(explicit_turn_type: Any, messages: list[dict[str, Any]]) -> str:
        if explicit_turn_type:
            return str(explicit_turn_type).strip().lower()
        if not messages:
            return "main"

        last_role = str(messages[-1].get("role", "")).strip().lower()
        if last_role == "tool":
            return "tool_continuation"
        if last_role == "user":
            user_messages = [
                _normalize_text_for_turn_matching(msg.get("content"))
                for msg in messages
                if str(msg.get("role", "")).strip().lower() == "user"
            ]
            if len(user_messages) >= 2 and user_messages[-1] and user_messages[-1] == user_messages[-2]:
                return "continuation_user"
            return "main"
        if last_role == "assistant":
            return "side"
        return "side"

    def _session_has_history(self, session_id: str) -> bool:
        return bool(
            self._turn_counts.get(session_id)
            or self._pending_records.get(session_id)
            or self._pending_turn_data.get(session_id)
            or self._session_latest_messages.get(session_id)
        )

    def _should_synthesize_overflow_terminal(self, session_id: str, turn_type: str) -> bool:
        if self._submit_side:
            return False
        if session_id in self._overflow_terminated_sessions:
            return True
        if turn_type != "main":
            return True
        return self._session_has_history(session_id)

    # -----------------------------------------------------------------------
    # Record / feedback persistence
    # -----------------------------------------------------------------------

    def _append_jsonl(self, path: str, payload: dict[str, Any]):
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("[Code-RL] failed to write %s: %s", path, exc)

    def _append_trace_event(self, event_type: str, payload: dict[str, Any]):
        if not self._trace_record_file:
            return
        event = {
            "event_type": event_type,
            "event_id": next(self._trace_counter),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        event.update(payload)
        self._append_jsonl(self._trace_record_file, event)

    def _buffer_record(
        self,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        messages: list[dict[str, Any]],
    ):
        record = {
            "session_id": session_id,
            "turn": turn_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "turn_type": turn_data["turn_type"],
            "request_meta": {
                "channel": turn_data.get("channel", "api"),
            },
            "messages": messages,
            "tools": turn_data.get("tools") or None,
            "request_settings": turn_data.get("request_settings"),
            "prompt_text": turn_data["prompt_text"],
            "prompt_tokens": turn_data.get(
                "prompt_token_count",
                len(turn_data.get("prompt_ids") or []),
            ),
            "response_text": turn_data["response_text"],
            "response_tokens": turn_data.get(
                "response_token_count",
                len(turn_data.get("response_ids") or []),
            ),
            "tool_calls": turn_data.get("tool_calls") or None,
            "assistant_message": turn_data.get("assistant_message"),
            "reasoning_content": turn_data.get("reasoning_content") or "",
            "response_meta": turn_data.get("response_meta") or {},
            "tracked_for_training": True,
        }
        self._pending_records[session_id] = record

    def _flush_pending_record(self, session_id: str, next_state: dict[str, Any] | None):
        record = self._pending_records.pop(session_id, None)
        if record is None:
            return

        turn_num = int(record["turn"])
        turn_data = self._pending_turn_data.get(session_id, {}).get(turn_num)
        if turn_data is not None:
            if next_state is not None:
                next_state_text = _flatten_message_content(next_state.get("content"))
                next_state_role = str(next_state.get("role", "user"))
                record["next_state"] = {
                    "role": next_state_role,
                    "content": next_state_text,
                }
                turn_data["has_next_state"] = True
                turn_data["next_state_role"] = next_state_role
                turn_data["next_state_text"] = next_state_text
                self._fire_prm_scoring(
                    session_id,
                    turn_num,
                    turn_data["response_text"],
                    next_state,
                )
            else:
                record["next_state"] = None

        self._append_jsonl(self._record_file, record)

    def _record_feedback(
        self,
        session_id: str,
        turn_id: Any,
        feedback: dict[str, Any],
    ):
        turn_num = int(turn_id) if turn_id is not None else self._turn_counts.get(session_id, 0)
        payload = {
            "session_id": session_id,
            "turn_id": turn_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": feedback.get("event_type", "unknown"),
            "severity": feedback.get("severity", "info"),
            "details": feedback.get("details", {}),
        }
        self._turn_feedback.setdefault(session_id, {}).setdefault(turn_num, []).append(payload)
        self._incr_stat("feedback_events_total")
        self._incr_stat(f"feedback_events_{payload['event_type']}")
        self._append_jsonl(self._feedback_record_file, payload)
        self._append_trace_event("feedback", payload)

    def _consume_turn_feedback(self, session_id: str, turn_num: int) -> list[dict[str, Any]]:
        return self._turn_feedback.get(session_id, {}).pop(turn_num, [])

    def _has_verifier_feedback(self, session_id: str, turn_num: int) -> bool:
        feedback_events = self._turn_feedback.get(session_id, {}).get(turn_num, [])
        return any(
            str(event.get("event_type", "unknown")) in {"verifier_reward", "task_verifier_reward"}
            for event in feedback_events
        )

    def _aggregate_feedback(self, feedback_events: list[dict[str, Any]]) -> dict[str, Any]:
        sanitized = False
        redaction_count = 0
        permission_denied = False
        injection_blocked = False
        event_types: list[str] = []
        for event in feedback_events:
            event_type = str(event.get("event_type", "unknown"))
            event_types.append(event_type)
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            if event_type == "output_redaction":
                sanitized = True
                redaction_count += int(details.get("redaction_count", 0) or 0)
            if event_type == "permission_denied":
                permission_denied = True
            if event_type == "injection_blocked":
                injection_blocked = True
        return {
            "sanitized": sanitized,
            "redaction_count": redaction_count,
            "permission_denied": permission_denied,
            "injection_blocked": injection_blocked,
            "event_types": event_types,
        }

    # -----------------------------------------------------------------------
    # PRM scoring
    # -----------------------------------------------------------------------

    async def _query_prm_once(
        self,
        judge_prompt: str,
        messages: list[dict[str, str]],
        vote_id: int,
    ) -> tuple[int | None, str]:
        try:
            async with httpx.AsyncClient(timeout=self._prm_timeout_sec, trust_env=False) as client:
                if self._prm_use_openai:
                    headers = {}
                    if self._prm_openai_api_key:
                        headers["Authorization"] = f"Bearer {self._prm_openai_api_key}"
                    payload = {
                        "model": self._prm_openai_model,
                        "messages": messages,
                        "temperature": self._prm_temperature,
                        "max_tokens": self._prm_max_tokens,
                        "stream": False,
                    }
                    payload.update(self._prm_openai_extra_body)
                    response = await client.post(self._prm_openai_url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    raw = _flatten_message_content(
                        data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    )
                    return _parse_prm_score(str(raw)), str(raw)

                if not self._prm_url:
                    return None, ""
                payload = {
                    "text": judge_prompt,
                    "sampling_params": {
                        "temperature": self._prm_temperature,
                        "top_p": 1.0,
                        "top_k": -1,
                        "max_new_tokens": self._prm_max_tokens,
                        "skip_special_tokens": False,
                        "no_stop_trim": True,
                        "spaces_between_special_tokens": False,
                    },
                    "return_logprob": False,
                }
                response = await client.post(self._prm_url, json=payload)
                response.raise_for_status()
                data = response.json()
            raw = data.get("text", data) if isinstance(data, dict) else str(data)
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            return _parse_prm_score(str(raw)), str(raw)
        except Exception as exc:
            logger.warning("[Code-RL] PRM query failed (vote %d): %s", vote_id, exc)
            return None, ""

    async def _prm_evaluate(
        self,
        session_id: str,
        turn_num: int,
        response_text: str,
        next_state: dict[str, Any],
    ) -> dict[str, Any]:
        next_state_text = _flatten_message_content(next_state.get("content")) if next_state else ""
        next_state_role = str(next_state.get("role", "user")) if next_state else "user"
        messages = _build_prm_judge_prompt(response_text, next_state_text, next_state_role)
        if self._prm_tokenizer:
            judge_prompt = self._prm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            judge_prompt = "\n".join(message["content"] for message in messages)

        results = await asyncio.gather(
            *[self._query_prm_once(judge_prompt, messages, idx) for idx in range(self._prm_m)]
        )
        scores = [result[0] for result in results]
        final = _majority_vote(scores)
        votes = [score if score is not None else "fail" for score in scores]
        representative = ""
        if final != 0.0:
            for score, raw_text in results:
                if score is not None and score == int(final):
                    representative = raw_text
                    break

        logger.info(
            "%s[Code-RL] PRM session=%s turn=%d votes=%s -> score=%s%s",
            _CYAN, session_id, turn_num, votes, final, _RESET,
        )

        record = {
            "session_id": session_id,
            "turn": turn_num,
            "score": final,
            "votes": votes,
            "backend": self._prm_backend,
            "response_text": response_text,
            "next_state_role": next_state_role,
            "next_state_text": next_state_text,
            "judge_prompt": judge_prompt,
            "representative_eval": representative,
        }
        self._append_jsonl(self._prm_record_file, record)
        return {
            "score": final,
            "votes": votes,
            "representative_eval": representative,
            "source": "next_state_prm",
        }

    def _fire_prm_scoring(
        self,
        session_id: str,
        turn_num: int,
        response_text: str,
        next_state: dict[str, Any],
    ):
        if not self._prm_enabled or not next_state:
            return

        task = asyncio.create_task(
            self._prm_evaluate(session_id, turn_num, response_text, next_state)
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _task: self._maybe_submit_ready_samples(session_id))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task

    # -----------------------------------------------------------------------
    # Core request handling
    # -----------------------------------------------------------------------

    async def _handle_request(
        self,
        body: dict[str, Any],
        session_id: str,
        turn_type: str,
        session_done: bool,
        request_meta: dict[str, Any],
        submission_epoch: int,
    ) -> dict[str, Any]:
        self._touch_session(session_id)
        self._incr_stat("chat_requests_total")
        self._incr_stat(f"chat_requests_{turn_type}")
        messages = body["messages"]
        if session_id in self._pending_records and messages:
            self._flush_pending_record(session_id, messages[-1])

        original_tools = body.get("tools")
        tools = self._filter_model_tools(original_tools)
        forward_body = {
            key: value for key, value in body.items() if key not in _NON_STANDARD_BODY_KEYS
        }
        if tools is None:
            forward_body.pop("tools", None)
        else:
            forward_body["tools"] = tools
        forward_body["stream"] = False
        forward_body.pop("stream_options", None)
        forward_body["logprobs"] = True
        forward_body["top_logprobs"] = 1
        if "model" not in forward_body:
            forward_body["model"] = self.served_model_name

        normalized_messages = _normalize_messages_for_template(messages)
        prompt_text = self.tokenizer.apply_chat_template(
            normalized_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_token_count = len(prompt_ids)
        output: dict[str, Any] | None = None
        synthesized_overflow_terminal = False
        overflow_terminal_content = (
            "This session hit the context limit while processing follow-up work. "
            "Stopping here to avoid wasting tokens. Start a fresh turn or session for more edits."
        )
        requested_max_tokens = forward_body.get("max_tokens")
        if requested_max_tokens is None and "max_completion_tokens" in forward_body:
            requested_max_tokens = forward_body.get("max_completion_tokens")
        try:
            requested_max_tokens = int(requested_max_tokens) if requested_max_tokens is not None else None
        except (TypeError, ValueError):
            requested_max_tokens = None

        if session_id in self._overflow_terminated_sessions and not self._submit_side:
            logger.warning(
                "[Code-RL] returning latched overflow terminal response session=%s turn_type=%s",
                session_id,
                turn_type,
            )
            output = self._build_synthetic_chat_response(
                content=overflow_terminal_content,
                prompt_token_count=prompt_token_count,
            )
            synthesized_overflow_terminal = True
            self._incr_stat("overflow_terminal_total")
            if turn_type == "main":
                turn_type = "overflow_terminal"

        if self.context_length > 0 and requested_max_tokens and requested_max_tokens > 0:
            available_completion_tokens = (
                self.context_length - prompt_token_count - self.context_safety_margin
            )
            if output is None and available_completion_tokens <= 0:
                logger.warning(
                    "[Code-RL] rejecting over-context request session=%s turn_type=%s prompt_tokens=%d requested_max_tokens=%d context_limit=%d margin=%d",
                    session_id,
                    turn_type,
                    prompt_token_count,
                    requested_max_tokens,
                    self.context_length,
                    self.context_safety_margin,
                )
                if self._should_synthesize_overflow_terminal(session_id, turn_type):
                    self._overflow_terminated_sessions.add(session_id)
                    logger.warning(
                        "[Code-RL] synthesizing terminal response for over-context follow-up "
                        "session=%s turn_type=%s has_history=%s",
                        session_id,
                        turn_type,
                        self._session_has_history(session_id),
                    )
                    output = self._build_synthetic_chat_response(
                        content=overflow_terminal_content,
                        prompt_token_count=prompt_token_count,
                    )
                    synthesized_overflow_terminal = True
                    self._incr_stat("overflow_terminal_total")
                    if turn_type == "main":
                        turn_type = "overflow_terminal"
                else:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": (
                                "Request prompt already exceeds the usable context window "
                                f"({prompt_token_count} prompt tokens with limit {self.context_length} "
                                f"and safety margin {self.context_safety_margin})."
                            ),
                            "type": "context_length_exceeded",
                        },
                    )
            elif output is None and requested_max_tokens > available_completion_tokens:
                capped_max_tokens = max(1, available_completion_tokens)
                logger.warning(
                    "[Code-RL] capping completion tokens session=%s requested=%d capped=%d prompt_tokens=%d context_limit=%d margin=%d",
                    session_id,
                    requested_max_tokens,
                    capped_max_tokens,
                    prompt_token_count,
                    self.context_length,
                    self.context_safety_margin,
                )
                forward_body["max_tokens"] = capped_max_tokens
                if "max_completion_tokens" in forward_body:
                    forward_body["max_completion_tokens"] = capped_max_tokens

        if output is None:
            request_timeout = (
                None if self.generation_timeout_sec <= 0 else self.generation_timeout_sec
            )
            async with httpx.AsyncClient(timeout=request_timeout, trust_env=False) as client:
                try:
                    response = await client.post(self.sglang_chat_url, json=forward_body)
                except httpx.TimeoutException as exc:
                    logger.warning(
                        "[Code-RL] SGLang generation timed out session=%s turn_type=%s "
                        "timeout=%.1fs",
                        session_id,
                        turn_type,
                        self.generation_timeout_sec,
                    )
                    raise HTTPException(
                        status_code=504,
                        detail={
                            "message": "SGLang generation timed out",
                            "timeout_sec": self.generation_timeout_sec,
                        },
                    ) from exc
                if response.status_code == 200:
                    output = response.json()
                elif _is_sglang_chat_batch_error(response.status_code, response.text):
                    logger.warning(
                        "[Code-RL] SGLang chat request hit known batch-input error; "
                        "falling back to /generate"
                    )
                    output = await self._generate_chat_fallback(
                        client,
                        forward_body=forward_body,
                        prompt_ids=prompt_ids,
                        prompt_token_count=prompt_token_count,
                        requested_max_tokens=requested_max_tokens,
                        tools=tools,
                    )
                else:
                    logger.error(
                        "[Code-RL] SGLang returned %d: %s",
                        response.status_code,
                        response.text[:1000],
                    )
                    detail: Any
                    try:
                        detail = response.json()
                    except Exception:
                        detail = response.text[:2000]
                    raise HTTPException(status_code=response.status_code, detail=detail)

        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {}) or {}
        tool_calls = assistant_msg.get("tool_calls") or []
        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""

        logger.info(
            "%s[Code-RL] [%s] session=%s prompt_msgs=%d%s",
            _YELLOW, turn_type, session_id, len(messages), _RESET,
        )
        logger.info(
            "%s[Code-RL] [%s] session=%s thinking=%d chars, response:%s%s%s",
            _RED, turn_type, session_id, len(reasoning), "\n", content, _RESET,
        )
        if tool_calls:
            logger.info(
                "[Code-RL] tool_calls=%s",
                json.dumps(tool_calls, ensure_ascii=False)[:500],
            )

        turn_num: int | None = None
        response_ids: list[int] = []
        response_text = _flatten_message_content(content)
        should_track_turn = (turn_type == "main" and not synthesized_overflow_terminal) or self._submit_side
        if should_track_turn:
            self._incr_stat("tracked_turns_total")
            response_msg = dict(assistant_msg)
            if response_msg.get("content") is None:
                response_msg["content"] = ""

            fallback_raw_response_text = output.get("_code_rl_raw_response_text")
            if isinstance(fallback_raw_response_text, str):
                response_text = fallback_raw_response_text
            else:
                normalized_response = _normalize_messages_for_template([response_msg])[0]
                full_conversation = normalized_messages + [normalized_response]
                full_text = self.tokenizer.apply_chat_template(
                    full_conversation,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if full_text.startswith(prompt_text):
                    response_text = full_text[len(prompt_text):]
                else:
                    logger.warning(
                        "[Code-RL] prompt_text is not a prefix of full_text for session=%s",
                        session_id,
                    )
                    response_text = full_text

            response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]
            response_logprobs = _extract_logprobs_from_chat_response(choice)
            if len(response_logprobs) > len(response_ids):
                response_logprobs = response_logprobs[: len(response_ids)]
            elif len(response_logprobs) < len(response_ids):
                response_logprobs = response_logprobs + [0.0] * (
                    len(response_ids) - len(response_logprobs)
                )

            self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
            turn_num = self._turn_counts[session_id]
            user_text = ""
            for message in reversed(messages):
                if str(message.get("role", "")).strip().lower() == "user":
                    user_text = _flatten_message_content(message.get("content"))
                    break

            turn_data = {
                "turn_num": turn_num,
                "turn_type": turn_type,
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "tool_calls": tool_calls,
                "tools": tools,
                "channel": request_meta.get("channel", "api"),
                "session_done": session_done,
                "has_next_state": False,
                "last_user_text": user_text,
                "assistant_message": response_msg,
                "reasoning_content": reasoning,
                "response_meta": {
                    "id": output.get("id"),
                    "model": output.get("model"),
                    "created": output.get("created"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": output.get("usage"),
                },
                "request_settings": {
                    "model": forward_body.get("model"),
                    "temperature": forward_body.get("temperature"),
                    "top_p": forward_body.get("top_p"),
                    "max_tokens": requested_max_tokens,
                },
                "prompt_token_count": len(prompt_ids),
                "response_token_count": len(response_ids),
                "synthesized_overflow_terminal": synthesized_overflow_terminal,
                "submission_epoch": submission_epoch,
            }

            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            self._buffer_record(session_id, turn_num, turn_data, messages)
            self._session_latest_messages[session_id] = _normalize_messages_for_matching(
                messages + [response_msg]
            )
            logger.info(
                "[Code-RL] tracked session=%s turn=%d type=%s prompt_tokens=%d response_tokens=%d",
                session_id, turn_num, turn_type, len(prompt_ids), len(response_ids),
            )
            self._maybe_submit_ready_samples(session_id)
        else:
            response_msg = dict(assistant_msg)
            self._session_latest_messages[session_id] = _normalize_messages_for_matching(
                messages + [response_msg]
            )
            logger.info(
                "[Code-RL] skipped training submission for side turn session=%s",
                session_id,
            )

        self._append_trace_event(
            "request_response",
            {
                "session_id": session_id,
                "turn": turn_num,
                "turn_type": turn_type,
                "session_done": session_done,
                "tracked_for_training": should_track_turn,
                "synthesized_overflow_terminal": synthesized_overflow_terminal,
                "request_meta": request_meta,
                "request": {
                    "messages": messages,
                    "tools": tools,
                    "settings": {
                        "model": forward_body.get("model"),
                        "temperature": forward_body.get("temperature"),
                        "top_p": forward_body.get("top_p"),
                        "max_tokens": requested_max_tokens,
                    },
                },
                "response": {
                    "id": output.get("id"),
                    "model": output.get("model"),
                    "created": output.get("created"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": output.get("usage"),
                    "assistant_message": assistant_msg,
                    "tool_calls": tool_calls or None,
                    "reasoning_content": reasoning or "",
                },
                "prompt_text": prompt_text,
                "prompt_tokens": prompt_token_count,
                "response_text": response_text,
                "response_tokens": len(response_ids) if should_track_turn else None,
            },
        )

        if session_done:
            self._finalize_session(session_id, reason="request_marked_done")

        output["session_id"] = session_id
        output["turn_type"] = turn_type
        return {"response": output}

    # -----------------------------------------------------------------------
    # Reward resolution
    # -----------------------------------------------------------------------

    def _resolve_reward(
        self,
        turn_data: dict[str, Any],
        prm_result: dict[str, Any] | None,
        feedback_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        for event in reversed(feedback_events):
            event_type = str(event.get("event_type", "unknown"))
            if event_type not in {"verifier_reward", "task_verifier_reward"}:
                continue
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            raw_score = details.get("score", details.get("reward"))
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            return {
                "score": score,
                "source": "verifier",
                "details": details,
            }

        if prm_result is not None:
            return {
                "score": float(prm_result.get("score", 0.0)),
                "source": "next_state_prm",
                "details": prm_result,
            }

        if self._reward_mode in {"debug_math_rule", "hybrid"}:
            debug_result = _maybe_rule_reward_debug_math(
                turn_data.get("last_user_text", ""),
                turn_data["response_text"],
            )
            if debug_result is not None:
                return {
                    "score": float(debug_result["score"]),
                    "source": "debug_math_rule",
                    "details": debug_result,
                }

        return {
            "score": 0.0,
            "source": "none",
            "details": {},
        }

    # -----------------------------------------------------------------------
    # Metadata & sample submission
    # -----------------------------------------------------------------------

    def _build_metadata(
        self,
        session_id: str,
        turn_data: dict[str, Any],
        reward_info: dict[str, Any],
        feedback_events: list[dict[str, Any]],
        feedback_summary: dict[str, Any],
        prm_result: dict[str, Any] | None,
        exclude_reason: str | None,
        promoted_from_neutral: bool,
    ) -> dict[str, Any]:
        turn_type = turn_data["turn_type"]
        reward_source = reward_info["source"]
        train_split = "debug_math"
        if reward_source != "debug_math_rule":
            if turn_type == "main":
                train_split = "mainline"
            elif turn_type in {"safety", "error_recovery"}:
                train_split = "safety"
            else:
                train_split = "side_branch"

        eligible_for_rl = exclude_reason is None
        metadata: dict[str, Any] = {
            "source": "a3s_code",
            "session_id": session_id,
            "turn_id": turn_data["turn_num"],
            "turn_type": turn_type,
            "channel": turn_data.get("channel", "api"),
            "has_next_state": bool(turn_data.get("has_next_state", False)),
            "next_state_role": turn_data.get("next_state_role"),
            "tool_calls_count": len(turn_data.get("tool_calls") or []),
            "sanitized": feedback_summary["sanitized"],
            "redaction_count": feedback_summary["redaction_count"],
            "permission_denied": feedback_summary["permission_denied"],
            "injection_blocked": feedback_summary["injection_blocked"],
            "feedback_events": feedback_events,
            "session_done": bool(turn_data.get("session_done", False)),
            "reward_source": reward_source,
            "response_has_repetition": bool(turn_data.get("response_has_repetition", False)),
            "train_metadata": {
                "reward_source": reward_source,
                "train_split": train_split,
                "suggested_use": "online_rl" if eligible_for_rl else "analysis_only",
                "eligible_for_rl": eligible_for_rl,
                "exclude_reason": exclude_reason,
            },
        }
        if prm_result is not None:
            metadata["prm_votes"] = prm_result.get("votes")
            metadata["prm_representative_eval"] = prm_result.get("representative_eval")
        if reward_source == "debug_math_rule":
            metadata["debug_math"] = reward_info["details"]
        if promoted_from_neutral:
            metadata["train_metadata"]["promoted_from_neutral"] = True
        return metadata

    def _resolve_sample_group_index(self, session_id: str) -> int:
        match = _SESSION_GROUP_RE.search(session_id)
        if match:
            return int(match.group(1))
        return next(self._group_counter)

    def _append_sample_trace(
        self,
        *,
        session_id: str,
        turn_data: dict[str, Any],
        reward_info: dict[str, Any],
        prm_result: dict[str, Any] | None,
        feedback_summary: dict[str, Any],
        metadata: dict[str, Any],
        decision: str,
        exclude_reason: str | None,
        sample_index: int | None = None,
        group_index: int | None = None,
    ):
        prompt_tokens = len(turn_data.get("prompt_ids") or [])
        response_tokens = len(turn_data.get("response_ids") or [])
        self._append_trace_event(
            "sample_result",
            {
                "session_id": session_id,
                "turn": turn_data.get("turn_num"),
                "turn_type": turn_data.get("turn_type"),
                "decision": decision,
                "exclude_reason": exclude_reason,
                "sample_index": sample_index,
                "group_index": group_index,
                "reward": {
                    "score": float(reward_info.get("score", 0.0)),
                    "source": reward_info.get("source"),
                },
                "has_next_state": bool(turn_data.get("has_next_state", False)),
                "next_state_role": turn_data.get("next_state_role"),
                "next_state_text": turn_data.get("next_state_text"),
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": prompt_tokens + response_tokens,
                "tool_calls_count": len(turn_data.get("tool_calls") or []),
                "feedback_summary": feedback_summary,
                "metadata": metadata,
                "prm_result": prm_result,
            },
        )

    def _maybe_submit_ready_samples(
        self,
        session_id: str,
        force_no_prm: bool = False,
    ):
        prm_tasks = self._prm_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})

        for turn_num in sorted(list(pending.keys())):
            task = prm_tasks.get(turn_num)
            if self._require_verifier_feedback and not self._has_verifier_feedback(session_id, turn_num):
                continue
            if self._prm_enabled:
                if task is not None and not task.done():
                    continue
                if task is None and not force_no_prm:
                    continue

            turn_data = pending.pop(turn_num)
            prm_result = None
            if task is not None and task.done():
                try:
                    prm_result = task.result()
                except Exception:
                    prm_result = None
                prm_tasks.pop(turn_num, None)

            task = self._safe_create_task(
                self._submit_turn_sample(
                    turn_data=turn_data,
                    session_id=session_id,
                    prm_result=prm_result,
                )
            )
            self._submit_tasks.setdefault(session_id, set()).add(task)
            task.add_done_callback(
                lambda done_task, sid=session_id: self._drop_submit_task(sid, done_task)
            )

        self._maybe_cleanup_session(session_id)

    def _drop_pending_turns_without_required_feedback(self, session_id: str, reason: str):
        if not self._require_verifier_feedback:
            return
        pending = self._pending_turn_data.get(session_id)
        if not pending:
            return

        dropped_turns: list[int] = []
        for turn_num in sorted(list(pending.keys())):
            if self._has_verifier_feedback(session_id, turn_num):
                continue
            turn_data = pending.pop(turn_num)
            dropped_turns.append(turn_num)
            self._incr_stat("dropped_samples_total")
            self._incr_stat("dropped_missing_verifier_feedback")
            self._append_sample_trace(
                session_id=session_id,
                turn_data=turn_data,
                reward_info={"score": 0.0, "source": "none"},
                prm_result=None,
                feedback_summary={"events": [], "finalize_reason": reason},
                metadata={
                    "source": "a3s_code",
                    "session_id": session_id,
                    "turn_id": turn_num,
                    "turn_type": turn_data.get("turn_type", "unknown"),
                    "session_done": bool(turn_data.get("session_done", False)),
                    "finalize_reason": reason,
                    "train_metadata": {
                        "reward_source": "none",
                        "suggested_use": "analysis_only",
                        "eligible_for_rl": False,
                        "exclude_reason": "missing_verifier_feedback",
                    },
                },
                decision="dropped_missing_verifier_feedback",
                exclude_reason="missing_verifier_feedback",
            )

        if not pending:
            self._pending_turn_data.pop(session_id, None)
        if dropped_turns:
            logger.info(
                "[Code-RL] dropped pending turns without verifier feedback "
                "session=%s turns=%s reason=%s",
                session_id,
                dropped_turns,
                reason,
            )

    async def _submit_turn_sample(
        self,
        turn_data: dict[str, Any],
        session_id: str,
        prm_result: dict[str, Any] | None,
    ):
        response_text = turn_data["response_text"]
        assistant_message = turn_data.get("assistant_message")
        assistant_content = ""
        if isinstance(assistant_message, dict):
            assistant_content = _flatten_message_content(assistant_message.get("content")).strip()
        tool_call_only = (
            bool(turn_data.get("tool_calls"))
            and not assistant_content
            and not str(turn_data.get("reasoning_content") or "").strip()
        )
        text_repetition = bool(response_text.strip()) and not tool_call_only and (
            has_repetition(response_text) or _has_short_repetition(response_text)
        )
        turn_data["response_has_repetition"] = bool(
            (turn_data.get("response_has_repetition", False) or text_repetition)
            and not tool_call_only
        )
        feedback_events = self._consume_turn_feedback(session_id, turn_data["turn_num"])
        feedback_summary = self._aggregate_feedback(feedback_events)
        reward_info = self._resolve_reward(turn_data, prm_result, feedback_events)
        score = float(reward_info["score"])
        reward_source = reward_info["source"]
        self._incr_stat("resolved_samples_total")
        self._incr_stat(f"reward_source_{reward_source}")

        signal_missing = reward_source == "none"
        neutral_signal = (reward_source != "none") and score == 0.0
        policy_excluded = turn_data["turn_type"] != "main" and not self._train_side
        stale_submission_epoch = (
            int(turn_data.get("submission_epoch", self._current_submission_epoch()))
            != self._current_submission_epoch()
        )

        promoted_from_neutral = False
        exclude_reason: str | None = None
        if stale_submission_epoch:
            exclude_reason = "stale_submission_epoch"
        elif signal_missing:
            exclude_reason = "no_reward_signal"
        elif neutral_signal:
            exclude_reason = "neutral_reward"
        elif policy_excluded:
            exclude_reason = "side_turn_disabled"

        if (
            neutral_signal
            and not stale_submission_epoch
            and not policy_excluded
            and self._session_effective.get(session_id, 0) == 0
        ):
            exclude_reason = None
            promoted_from_neutral = True

        metadata = self._build_metadata(
            session_id=session_id,
            turn_data=turn_data,
            reward_info=reward_info,
            feedback_events=feedback_events,
            feedback_summary=feedback_summary,
            prm_result=prm_result,
            exclude_reason=exclude_reason,
            promoted_from_neutral=promoted_from_neutral,
        )

        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]
        response_text = turn_data["response_text"]
        response_logprobs = turn_data["response_logprobs"]
        response_has_repetition = bool(turn_data["response_has_repetition"])
        verifier_labeled = reward_source == "verifier"
        if response_has_repetition and verifier_labeled:
            metadata["train_metadata"]["repetition_kept_by_verifier"] = True

        if self.drop_repetitive_samples and response_has_repetition and not verifier_labeled:
            logger.warning(
                "[Code-RL] dropping repetitive sample session=%s turn=%d response_tokens=%d",
                session_id,
                turn_data["turn_num"],
                len(response_ids),
            )
            self._incr_stat("dropped_samples_total")
            self._incr_stat("dropped_repetitive")
            self._append_sample_trace(
                session_id=session_id,
                turn_data=turn_data,
                reward_info=reward_info,
                prm_result=prm_result,
                feedback_summary=feedback_summary,
                metadata=metadata,
                decision="dropped_repetitive",
                exclude_reason="repetitive_response",
            )
            self._maybe_cleanup_session(session_id)
            return

        if stale_submission_epoch:
            logger.info(
                "[Code-RL] dropping ready sample from stale submission epoch "
                "session=%s turn=%d sample_epoch=%s current_epoch=%d",
                session_id,
                turn_data["turn_num"],
                turn_data.get("submission_epoch"),
                self._current_submission_epoch(),
            )
            self._incr_stat("dropped_samples_total")
            self._incr_stat("dropped_stale_epoch")
            self._append_sample_trace(
                session_id=session_id,
                turn_data=turn_data,
                reward_info=reward_info,
                prm_result=prm_result,
                feedback_summary=feedback_summary,
                metadata=metadata,
                decision="dropped_stale_epoch",
                exclude_reason="stale_submission_epoch",
            )
            self._maybe_cleanup_session(session_id)
            return

        if exclude_reason is not None:
            logger.info(
                "[Code-RL] skipping excluded sample session=%s turn=%d reason=%s source=%s",
                session_id,
                turn_data["turn_num"],
                exclude_reason,
                reward_source,
            )
            self._incr_stat("excluded_samples_total")
            self._incr_stat(f"excluded_{exclude_reason}")
            self._append_sample_trace(
                session_id=session_id,
                turn_data=turn_data,
                reward_info=reward_info,
                prm_result=prm_result,
                feedback_summary=feedback_summary,
                metadata=metadata,
                decision="excluded",
                exclude_reason=exclude_reason,
            )
            self._maybe_cleanup_session(session_id)
            return

        if self.max_response_tokens > 0 and len(response_ids) > self.max_response_tokens:
            overflow = len(response_ids) - self.max_response_tokens
            if 0 < overflow <= self.response_trim_margin_tokens:
                response_ids = response_ids[: self.max_response_tokens]
                response_logprobs = response_logprobs[: self.max_response_tokens]
                response_text = self.tokenizer.decode(
                    response_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                metadata["train_metadata"]["response_trimmed_tokens"] = overflow
                logger.info(
                    "[Code-RL] trimming near-limit response sample session=%s turn=%d response_tokens=%d -> %d",
                    session_id,
                    turn_data["turn_num"],
                    len(turn_data["response_ids"]),
                    len(response_ids),
                )
            else:
                logger.warning(
                    "[Code-RL] dropping long-response sample session=%s turn=%d response_tokens=%d limit=%d",
                    session_id,
                    turn_data["turn_num"],
                    len(response_ids),
                    self.max_response_tokens,
                )
                self._incr_stat("dropped_samples_total")
                self._incr_stat("dropped_long_response")
                self._append_sample_trace(
                    session_id=session_id,
                    turn_data=turn_data,
                    reward_info=reward_info,
                    prm_result=prm_result,
                    feedback_summary=feedback_summary,
                    metadata=metadata,
                    decision="dropped_long_response",
                    exclude_reason="response_too_long",
                )
                self._maybe_cleanup_session(session_id)
                return

        total_tokens = len(prompt_ids) + len(response_ids)
        if self.max_train_tokens > 0 and total_tokens > self.max_train_tokens:
            logger.warning(
                "[Code-RL] dropping overlong sample session=%s turn=%d total_tokens=%d limit=%d",
                session_id,
                turn_data["turn_num"],
                total_tokens,
                self.max_train_tokens,
            )
            self._incr_stat("dropped_samples_total")
            self._incr_stat("dropped_overlong")
            self._append_sample_trace(
                session_id=session_id,
                turn_data=turn_data,
                reward_info=reward_info,
                prm_result=prm_result,
                feedback_summary=feedback_summary,
                metadata=metadata,
                decision="dropped_overlong",
                exclude_reason="total_tokens_exceeded",
            )
            self._maybe_cleanup_session(session_id)
            return

        if not self.submission_enabled.is_set():
            logger.info(
                "[Code-RL] dropping ready sample because submission is paused "
                "session=%s turn=%d",
                session_id,
                turn_data["turn_num"],
            )
            self._incr_stat("dropped_samples_total")
            self._incr_stat("dropped_paused")
            self._append_sample_trace(
                session_id=session_id,
                turn_data=turn_data,
                reward_info=reward_info,
                prm_result=prm_result,
                feedback_summary=feedback_summary,
                metadata=metadata,
                decision="dropped_paused",
                exclude_reason="submission_paused",
            )
            self._maybe_cleanup_session(session_id)
            return

        with self._eval_scores_lock:
            self._eval_scores.append(score)

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = response_text
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = [0] * len(response_ids) if exclude_reason is not None else [1] * len(response_ids)
        sample.rollout_log_probs = response_logprobs
        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = self._resolve_sample_group_index(session_id)
        sample.reward = {"score": score}
        sample.metadata = metadata

        if exclude_reason is None:
            self._session_effective[session_id] = self._session_effective.get(session_id, 0) + 1
            self._incr_stat("submitted_samples_total")
            self._incr_stat(f"submitted_reward_source_{reward_source}")

        logger.info(
            "[Code-RL] submitted sample session=%s turn=%d index=%d score=%.1f exclude=%s source=%s",
            session_id,
            turn_data["turn_num"],
            sample.index,
            score,
            exclude_reason is not None,
            reward_source,
        )
        self._append_sample_trace(
            session_id=session_id,
            turn_data=turn_data,
            reward_info=reward_info,
            prm_result=prm_result,
            feedback_summary=feedback_summary,
            metadata=metadata,
            decision="submitted",
            exclude_reason=None,
            sample_index=sample.index,
            group_index=sample.group_index,
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))
        self._maybe_cleanup_session(session_id)

    # -----------------------------------------------------------------------
    # Session lifecycle
    # -----------------------------------------------------------------------

    def _drop_submit_task(self, session_id: str, task: asyncio.Task):
        tasks = self._submit_tasks.get(session_id)
        if tasks is not None:
            tasks.discard(task)
            if not tasks:
                self._submit_tasks.pop(session_id, None)
        self._maybe_cleanup_session(session_id)

    def _maybe_cleanup_session(self, session_id: str):
        if session_id not in self._finalizing_sessions:
            return
        pending_records = session_id in self._pending_records
        pending_turns = bool(self._pending_turn_data.get(session_id))
        prm_tasks = self._prm_tasks.get(session_id, {})
        live_tasks = any(not task.done() for task in prm_tasks.values())
        submit_tasks = self._submit_tasks.get(session_id, set())
        live_submit_tasks = any(not task.done() for task in submit_tasks)
        if pending_records or pending_turns or live_tasks or live_submit_tasks:
            return

        self._finalizing_sessions.discard(session_id)
        self._overflow_terminated_sessions.discard(session_id)
        self._pending_turn_data.pop(session_id, None)
        self._prm_tasks.pop(session_id, None)
        self._submit_tasks.pop(session_id, None)
        self._session_latest_messages.pop(session_id, None)
        self._session_last_activity.pop(session_id, None)
        self._turn_feedback.pop(session_id, None)
        self._session_effective.pop(session_id, None)
        self._turn_counts.pop(session_id, None)
        logger.info("[Code-RL] cleaned up session=%s", session_id)

    def _finalize_session(self, session_id: str, reason: str):
        if session_id in self._finalizing_sessions:
            return
        logger.info("[Code-RL] finalizing session=%s reason=%s", session_id, reason)
        self._finalizing_sessions.add(session_id)
        self._flush_pending_record(session_id, None)
        self._drop_pending_turns_without_required_feedback(session_id, reason)
        self._maybe_submit_ready_samples(session_id, force_no_prm=True)
        self._maybe_cleanup_session(session_id)

    def flush_idle_sessions(self):
        if self._idle_flush_sec <= 0:
            return
        now = time.time()
        candidates = [
            session_id
            for session_id, last_activity in list(self._session_last_activity.items())
            if now - last_activity >= self._idle_flush_sec
            and session_id not in self._finalizing_sessions
        ]
        for session_id in candidates:
            self._finalize_session(session_id, reason="idle_timeout")

    # -----------------------------------------------------------------------
    # Eval scores
    # -----------------------------------------------------------------------

    def drain_eval_scores(self) -> list[float]:
        with self._eval_scores_lock:
            scores = list(self._eval_scores)
            self._eval_scores.clear()
            return scores

    def reset_eval_scores(self):
        with self._eval_scores_lock:
            self._eval_scores.clear()

    def purge_record_files(self):
        if not self._purge_record_files_on_pause:
            logger.info(
                "[Code-RL] skipping record purge because CODE_RL_PURGE_RECORD_FILES_ON_PAUSE=0"
            )
            return
        for path in [
            self._record_file,
            self._prm_record_file,
            self._feedback_record_file,
            self._trace_record_file,
        ]:
            if not path:
                continue
            try:
                open(path, "w").close()
                logger.info("[Code-RL] purged record file: %s", path)
            except OSError as exc:
                logger.warning("[Code-RL] failed to purge %s: %s", path, exc)

    # -----------------------------------------------------------------------
    # Streaming
    # -----------------------------------------------------------------------

    async def _stream_response(self, result: dict[str, Any]):
        payload = result["response"]
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        delta: dict[str, Any] = {"role": "assistant", "content": message.get("content", "") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        if message.get("reasoning_content"):
            delta["reasoning_content"] = message["reasoning_content"]

        chunk_base = {
            "id": payload.get("id", ""),
            "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", ""),
            "session_id": payload.get("session_id", ""),
        }
        first = {
            **chunk_base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        final = {
            **chunk_base,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
            ],
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    def _safe_create_task(self, coro):
        task = asyncio.create_task(coro)
        task.add_done_callback(self._task_done_cb)
        return task

    @staticmethod
    def _task_done_cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[Code-RL] background task failed: %s", exc, exc_info=exc)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config=config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        readiness_thread = threading.Thread(
            target=self._wait_for_sglang_ready,
            daemon=True,
        )
        readiness_thread.start()

    def _wait_for_sglang_ready(self):
        while True:
            try:
                response = httpx.get(self.sglang_health_url, timeout=5, trust_env=False)
                if response.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(3)
        logger.info("[Code-RL] policy server ready")

        if self._prm_enabled and self._prm_backend == "external_openai" and self._prm_health_url:
            while True:
                try:
                    response = httpx.get(self._prm_health_url, timeout=5, trust_env=False)
                    if response.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(3)
            logger.info("[Code-RL] external PRM endpoint ready")
        elif self._prm_enabled and self._prm_url:
            prm_health = self._prm_url.rsplit("/", 1)[0] + "/health"
            while True:
                try:
                    response = httpx.get(prm_health, timeout=5, trust_env=False)
                    if response.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(3)
            logger.info("[Code-RL] PRM server ready")

        time.sleep(5)
        banner = (
            f"\n{'=' * 70}\n"
            f"  [Code-RL] proxy ready\n"
            f"  listen: http://{self.host}:{self.port}\n"
            f"  upstream: {self.args.sglang_router_ip}:{self.args.sglang_router_port}\n"
            f"  reward_mode: {self._reward_mode}\n"
            f"  submit_side: {self._submit_side}  train_side: {self._train_side}\n"
            f"  prm_backend: {self._prm_backend}\n"
            f"{'=' * 70}\n"
        )
        logger.info("%s%s%s", _GREEN, banner, _RESET)

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
