"""OpenClaw Fireworks API server.

Replaces the SGLang-backed OPD/Combine server with a Fireworks Deployment
for policy inference and PRM text generation, and uses the Fireworks policy
trainer for teacher log-prob computation.

This server acts as an OpenAI-compatible proxy:
  - Client -> POST /v1/chat/completions -> Fireworks Deployment (policy)
  - PRM hint/eval judges -> same Fireworks Deployment (chat completions)
  - Teacher logprobs -> Fireworks trainer forward() pass
  - Training samples -> output_queue -> training loop
"""

from __future__ import annotations

import asyncio
import collections
import copy
import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from itertools import count
from typing import Any

import httpx
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"
logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([-+]?\d)\}")
_HINT_RE = re.compile(r"\[HINT_START\](.*?)\[HINT_END\]", re.DOTALL)

_NON_STANDARD_BODY_KEYS = {"session_id", "session_done", "turn_type"}


@dataclass
class TrainingSample:
    """Lightweight sample for the Fireworks training loop (no slime dependency)."""

    tokens: list[int] = field(default_factory=list)
    prompt_len: int = 0
    response_len: int = 0
    rollout_log_probs: list[float] = field(default_factory=list)
    teacher_log_probs: list[float] | None = None
    reward: float = 0.0
    group_index: int = 0
    index: int = 0
    prompt_text: str = ""
    response_text: str = ""


def _flatten_message_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    out = []
    for msg in messages:
        m = dict(msg)
        if m.get("role") == "developer":
            m["role"] = "system"
        raw = m.get("content")
        if not isinstance(raw, str) and raw is not None:
            m["content"] = _flatten_message_content(raw)
        if m.get("tool_calls"):
            m["tool_calls"] = [_normalize_tool_call(tc) for tc in m["tool_calls"]]
        out.append(m)
    return out


def _normalize_tool_call(tc: dict) -> dict:
    tc = dict(tc)
    fn = tc.get("function")
    if isinstance(fn, dict):
        fn = dict(fn)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                fn["arguments"] = {}
        tc["function"] = fn
    return tc


def _extract_logprobs_from_chat_response(choice: dict[str, Any]) -> list[float]:
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    content = logprobs_obj.get("content")
    if not isinstance(content, list):
        return []
    return [float(item.get("logprob", 0.0)) for item in content if isinstance(item, dict)]


def _build_hint_judge_messages(
    response_text: str, next_state_text: str, next_state_role: str = "user",
) -> list[dict]:
    system = (
        "You are a process reward model used for hindsight hint extraction.\n"
        "You are given:\n"
        "1) The assistant response at turn t.\n"
        "2) The next state at turn t+1, along with its **role**.\n\n"
        "## Understanding the next state's role\n"
        "- role='user': A reply from the user (follow-up, correction, new request, etc.).\n"
        "- role='tool': The return value of a tool the assistant invoked. "
        "This content was NOT available before the assistant's action -- "
        "it exists BECAUSE the assistant called the tool. "
        "A successful, non-error tool output generally means the assistant's "
        "action was appropriate; do NOT treat it as information the assistant "
        "should have already known.\n\n"
        "Your goal is to decide whether the next state reveals useful hindsight information\n"
        "that could have helped improve the assistant response at turn t.\n\n"
        "Output format rules (strict):\n"
        "- You MUST include exactly one final decision token: \\boxed{1} or \\boxed{-1}.\n"
        "- If and only if decision is \\boxed{1}, provide a concise, information-dense hint in 1-3 sentences,\n"
        "  wrapped between [HINT_START] and [HINT_END].\n"
        "- If decision is \\boxed{-1}, do not provide a hint block.\n"
        "- Hint must be concrete and actionable for improving the previous response."
    )
    user = (
        f"## Assistant response (turn t)\n{response_text}\n\n"
        f"## Next state (turn t+1) [role: {next_state_role}]\n{next_state_text}\n\n"
        "Now output your decision and (if positive) the hint in the required format."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_prm_eval_prompt(
    response_text: str, next_state_text: str, next_state_role: str = "user",
) -> list[dict]:
    system = (
        "You are a process reward model (PRM) evaluating an AI assistant.\n"
        "You will see the assistant's output and the subsequent next state.\n"
        "Your task: decide whether the assistant's output **successfully fulfilled** the user's intent "
        "at that step, using the next state as evidence.\n\n"
        "## Understanding the next state's role\n"
        "- role='user': A reply from the user.\n"
        "- role='tool': The return value of a tool the assistant invoked. "
        "This content was NOT available before the assistant's action -- "
        "it exists BECAUSE the assistant called the tool. "
        "A successful, non-error tool output means the assistant's action worked correctly "
        "and should be scored positively.\n\n"
        "## Scoring rules\n"
        "- \\boxed{1} (good): The next state shows the task progressed as expected.\n"
        "- \\boxed{-1} (bad): The next state signals the assistant's output was wrong, "
        "incomplete, or unwanted.\n"
        "- \\boxed{0} (neutral): The next state is ambiguous.\n\n"
        "## Important\n"
        "A change request IS negative feedback -- it means the previous output did not "
        "meet the user's need. Do NOT treat it as a neutral new instruction.\n\n"
        "Think step-by-step, then give your final score inside \\boxed{}."
    )
    user = (
        f"## Assistant output\n{response_text}\n\n"
        f"## Next state [role: {next_state_role}]\n{next_state_text}\n\n"
        "First, classify the next state: is it (a) positive progression, "
        "(b) a correction / redo / change request, or (c) ambiguous? "
        "Then assign \\boxed{1}, \\boxed{-1}, or \\boxed{0}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_judge_result(text: str) -> tuple[int | None, str]:
    boxed = _BOXED_RE.findall(text)
    score = int(boxed[-1]) if boxed else None
    if score not in (1, -1):
        score = None
    hint_matches = _HINT_RE.findall(text)
    hint = hint_matches[-1].strip() if hint_matches else ""
    return score, hint


def _parse_prm_eval_score(text: str) -> int | None:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    val = int(matches[-1])
    if val in (1, -1, 0):
        return val
    return None


def _prm_eval_majority_vote(scores: list[int | None]) -> float:
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.0
    counter = collections.Counter(valid)
    top = counter.most_common(1)[0]
    if list(counter.values()).count(top[1]) > 1:
        return 0.0
    return float(top[0])


def _select_best_hint(votes: list[dict[str, Any]]) -> dict[str, Any] | None:
    good = [
        v for v in votes
        if v.get("score") == 1 and isinstance(v.get("hint"), str) and len(v["hint"].strip()) > 10
    ]
    if not good:
        return None
    return max(good, key=lambda v: len(v["hint"].strip()))


def _append_hint_to_messages(messages: list[dict], hint: str) -> list[dict]:
    cloned = copy.deepcopy(messages)
    if not cloned:
        return [{"role": "user", "content": f"[user's hint / instruction]\n{hint}"}]
    target_idx = None
    for i in range(len(cloned) - 1, -1, -1):
        if cloned[i].get("role") == "user":
            target_idx = i
            break
    if target_idx is None:
        target_idx = len(cloned) - 1
    content = _flatten_message_content(cloned[target_idx].get("content"))
    suffix = f"\n\n[user's hint / instruction]\n{hint.strip()}"
    cloned[target_idx]["content"] = (content + suffix).strip()
    return cloned


def _is_valid_rl_score(score) -> bool:
    return score in (1, -1, 1.0, -1.0)


class OpenClawFireworksServer:
    """Async proxy that collects training data from user interactions.

    Uses Fireworks Deployment for inference and PRM, and a Fireworks trainer
    for teacher log-prob computation via forward() passes.
    """

    def __init__(
        self,
        *,
        output_queue: queue.Queue,
        submission_enabled: threading.Event,
        tokenizer_model: str,
        deployment_chat_url: str,
        deployment_model: str,
        fw_api_key: str,
        training_client=None,
        prm_enabled: bool = True,
        prm_m: int = 3,
        prm_temperature: float = 0.6,
        prm_max_tokens: int = 4096,
        teacher_lp_max_concurrency: int = 3,
        host: str = "0.0.0.0",
        port: int = 30000,
        served_model_name: str = "default",
        expected_api_key: str = "",
        record_dir: str = "",
    ):
        self.output_queue = output_queue
        self.submission_enabled = submission_enabled
        self.training_client = training_client

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
        self.deployment_chat_url = deployment_chat_url
        self.deployment_model = deployment_model
        self.fw_api_key = fw_api_key
        self.host = host
        self.port = port
        self.served_model_name = served_model_name
        self.expected_api_key = expected_api_key

        self._index_counter = count(0)
        self._group_counter = count(0)
        self._turn_counts: dict[str, int] = {}
        self._pending_turn_data: dict[str, dict[int, dict]] = {}
        self._prm_tasks: dict[str, dict[int, asyncio.Task]] = {}
        self._pending_records: dict[str, dict[str, Any]] = {}

        self._prm_enabled = prm_enabled
        self._prm_m = prm_m
        self._prm_temperature = prm_temperature
        self._prm_max_tokens = prm_max_tokens
        self._teacher_lp_semaphore = asyncio.Semaphore(max(1, teacher_lp_max_concurrency))

        self._eval_mode = os.getenv("OPENCLAW_EVAL_MODE", "0") == "1"
        self._eval_scores: list[float] = []
        self._eval_scores_lock = threading.Lock()

        self._record_dir = record_dir
        self._record_file = ""
        self._prm_record_file = ""
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
            self._record_file = os.path.join(record_dir, "conversations.jsonl")
            self._prm_record_file = os.path.join(record_dir, "prm_results.jsonl")
            for f in (self._record_file, self._prm_record_file):
                open(f, "w").close()

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.app = self._build_app()

    def set_training_client(self, training_client):
        self.training_client = training_client

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="OpenClaw Fireworks Proxy")
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: str | None = Header(default=None),
            x_session_id: str | None = Header(default=None),
            x_turn_type: str | None = Header(default=None),
            x_session_done: str | None = Header(default=None),
        ):
            owner: OpenClawFireworksServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.submission_enabled.is_set():
                raise HTTPException(status_code=503, detail="submission paused for weight update")

            body = await request.json()
            session_id = x_session_id or body.get("session_id") or "unknown"
            turn_type = (x_turn_type or body.get("turn_type") or "side").strip().lower()
            session_done = (
                (x_session_done and x_session_done.strip().lower() in {"1", "true", "yes", "on"})
                or str(body.get("session_done", "")).strip().lower() in {"1", "true", "yes", "on"}
            )

            stream = bool(body.get("stream", False))
            result = await owner._handle_request(
                body, session_id=session_id, turn_type=turn_type, session_done=session_done,
            )
            if stream:
                return StreamingResponse(owner._stream_response(result), media_type="text/event-stream")
            return JSONResponse(content=result["response"])

        return app

    async def _check_auth(self, authorization: str | None):
        if not self.expected_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self.expected_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    # ------------------------------------------------------------------
    # Fireworks Deployment inference (replaces SGLang)
    # ------------------------------------------------------------------

    async def _call_deployment_chat(
        self, body: dict[str, Any], retries: int = 3, backoff: float = 1.0,
        hotload_max_retries: int = 30, hotload_backoff: float = 2.0,
    ) -> dict[str, Any]:
        """Forward a chat completions request to the Fireworks deployment.

        On HTTP 425 (model hot-loading), automatically extends the retry budget
        to ``hotload_max_retries`` with a fixed ``hotload_backoff`` interval,
        giving the deployment time to finish loading the new checkpoint.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.fw_api_key}",
        }
        last_exc = None
        max_attempts = retries
        cur_backoff = backoff
        attempt = 0
        while attempt < max_attempts:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                    resp = await client.post(self.deployment_chat_url, json=body, headers=headers)
                    if resp.status_code == 200:
                        return resp.json()
                    if resp.status_code == 425 and max_attempts < hotload_max_retries:
                        max_attempts = hotload_max_retries
                        cur_backoff = hotload_backoff
                        logger.info(
                            "[Fireworks] Deployment is hot-loading, extending retries "
                            "to %d (attempt %d)",
                            hotload_max_retries, attempt + 1,
                        )
                    else:
                        logger.warning(
                            "[Fireworks] Deployment returned %d (attempt %d/%d): %s",
                            resp.status_code, attempt + 1, max_attempts, resp.text[:500],
                        )
                    last_exc = httpx.HTTPStatusError(
                        f"{resp.status_code}", request=resp.request, response=resp,
                    )
            except Exception as e:
                logger.warning(
                    "[Fireworks] Deployment request failed (attempt %d/%d): %s",
                    attempt + 1, max_attempts, e,
                )
                last_exc = e
            attempt += 1
            if attempt < max_attempts:
                await asyncio.sleep(cur_backoff)
        raise last_exc

    # ------------------------------------------------------------------
    # PRM: hint judge + eval judge (via Fireworks deployment chat API)
    # ------------------------------------------------------------------

    async def _query_judge_once(self, judge_messages: list[dict], vote_id: int) -> dict[str, Any]:
        body = {
            "model": self.deployment_model,
            "messages": judge_messages,
            "temperature": self._prm_temperature,
            "max_tokens": self._prm_max_tokens,
            "stream": False,
        }
        try:
            output = await self._call_deployment_chat(body)
            raw = output.get("choices", [{}])[0].get("message", {}).get("content", "")
            score, hint = _parse_judge_result(raw)
            return {"vote_id": vote_id, "score": score, "hint": hint, "raw": raw}
        except Exception as e:
            logger.warning("[Fireworks] judge query failed (vote %d): %s", vote_id, e)
            return {"vote_id": vote_id, "score": None, "hint": "", "raw": ""}

    async def _query_prm_eval_once(self, eval_messages: list[dict], vote_id: int) -> int | None:
        body = {
            "model": self.deployment_model,
            "messages": eval_messages,
            "temperature": self._prm_temperature,
            "max_tokens": self._prm_max_tokens,
            "stream": False,
        }
        try:
            output = await self._call_deployment_chat(body)
            raw = output.get("choices", [{}])[0].get("message", {}).get("content", "")
            return _parse_prm_eval_score(raw)
        except Exception as e:
            logger.warning("[Fireworks] PRM eval query failed (vote %d): %s", vote_id, e)
            return None

    # ------------------------------------------------------------------
    # Teacher log-probs via Fireworks trainer forward() pass
    # ------------------------------------------------------------------

    async def _compute_teacher_log_probs(
        self, input_ids: list[int], response_len: int,
    ) -> list[float]:
        """Compute teacher log-probs using the Fireworks policy trainer.

        Builds a datum from the hint-augmented full sequence and runs a
        forward-only pass to get per-token log-probs.
        """
        if self.training_client is None:
            return [0.0] * response_len

        import tinker
        from fireworks_loss import build_datum

        prompt_len = max(0, len(input_ids) - response_len)
        datum = build_datum(input_ids, prompt_len)

        try:
            async with self._teacher_lp_semaphore:
                result = await asyncio.to_thread(
                    lambda: self.training_client.forward([datum], "cross_entropy").result()
                )
            lp_data = result.loss_fn_outputs[0]["logprobs"].data
            all_lp = [float(v) for v in lp_data]

            if len(all_lp) >= response_len:
                return all_lp[-response_len:]
            return [0.0] * (response_len - len(all_lp)) + all_lp
        except Exception as e:
            logger.warning("[Fireworks] teacher logprob forward failed: %s", e)
            return [0.0] * response_len

    # ------------------------------------------------------------------
    # OPD evaluation: hint judge + eval judge + teacher logprobs
    # ------------------------------------------------------------------

    async def _opd_evaluate(
        self,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
    ) -> dict[str, Any]:
        next_state_text = _flatten_message_content(next_state.get("content")) if next_state else ""
        next_state_role = next_state.get("role", "user") if next_state else "user"

        judge_msgs = _build_hint_judge_messages(
            turn_data["response_text"], next_state_text, next_state_role,
        )
        votes = await asyncio.gather(
            *[self._query_judge_once(judge_msgs, i) for i in range(self._prm_m)]
        )

        eval_score = None
        if self._eval_mode or self._prm_enabled:
            eval_msgs = _build_prm_eval_prompt(
                turn_data["response_text"], next_state_text, next_state_role,
            )
            eval_raw = await asyncio.gather(
                *[self._query_prm_eval_once(eval_msgs, i) for i in range(self._prm_m)]
            )
            eval_score = _prm_eval_majority_vote(eval_raw)
            logger.info(
                "%s[Fireworks] PRM eval session=%s turn=%d eval_votes=%s -> eval_score=%.1f%s",
                _CYAN, session_id, turn_num,
                [s if s is not None else "fail" for s in eval_raw],
                eval_score, _RESET,
            )

        selected = _select_best_hint(votes)
        votes_display = [v.get("score", "fail") for v in votes]

        if selected is None:
            logger.info(
                "%s[Fireworks] session=%s turn=%d no valid hint (votes=%s)%s",
                _CYAN, session_id, turn_num, votes_display, _RESET,
            )
            self._append_prm_record({
                "session_id": session_id, "turn": turn_num,
                "accepted": False, "hint": "", "votes": votes_display,
            })
            return {
                "accepted": False, "teacher_log_probs": None,
                "hint": "", "votes": votes, "eval_score": eval_score,
            }

        hint = selected["hint"].strip()
        enhanced_messages = _append_hint_to_messages(turn_data["messages"], hint)
        norm_enhanced = _normalize_messages_for_template(enhanced_messages)
        enhanced_prompt_text = self.tokenizer.apply_chat_template(
            norm_enhanced,
            tools=turn_data.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
        )
        enhanced_full_text = enhanced_prompt_text + turn_data["response_text"]
        enhanced_ids = self.tokenizer(enhanced_full_text, add_special_tokens=False)["input_ids"]
        response_len = len(turn_data["response_ids"])
        teacher_log_probs = await self._compute_teacher_log_probs(enhanced_ids, response_len)

        logger.info(
            "%s[Fireworks] session=%s turn=%d accepted hint_len=%d votes=%s%s",
            _CYAN, session_id, turn_num, len(hint), votes_display, _RESET,
        )
        self._append_prm_record({
            "session_id": session_id, "turn": turn_num,
            "accepted": True, "hint": hint, "hint_len": len(hint),
            "votes": votes_display, "teacher_logprob_len": len(teacher_log_probs),
        })
        return {
            "accepted": True, "teacher_log_probs": teacher_log_probs,
            "hint": hint, "votes": votes, "eval_score": eval_score,
        }

    # ------------------------------------------------------------------
    # Sample submission (combined: OPD + RL, or RL-only)
    # ------------------------------------------------------------------

    async def _submit_turn_sample(
        self,
        turn_data: dict[str, Any],
        session_id: str,
        opd_result: dict[str, Any],
        reward: float = 0.0,
    ):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        teacher_log_probs = opd_result.get("teacher_log_probs") or []
        if len(teacher_log_probs) > len(response_ids):
            teacher_log_probs = teacher_log_probs[:len(response_ids)]
        elif len(teacher_log_probs) < len(response_ids):
            teacher_log_probs = teacher_log_probs + [0.0] * (len(response_ids) - len(teacher_log_probs))

        sample = TrainingSample(
            tokens=prompt_ids + response_ids,
            prompt_len=len(prompt_ids),
            response_len=len(response_ids),
            rollout_log_probs=turn_data["response_logprobs"],
            teacher_log_probs=teacher_log_probs,
            reward=reward,
            group_index=next(self._group_counter),
            index=next(self._index_counter),
            prompt_text=turn_data["prompt_text"],
            response_text=turn_data["response_text"],
        )

        tag = "OPD+RL" if reward != 0.0 else "OPD"
        logger.info(
            "[Fireworks] submitted %s sample session=%s index=%d "
            "reward=%.1f prompt_len=%d response_len=%d hint_len=%d",
            tag, session_id, sample.index, reward,
            len(prompt_ids), len(response_ids),
            len(opd_result.get("hint", "")),
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    async def _submit_rl_turn_sample(
        self, turn_data: dict, session_id: str, eval_score: float,
    ):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]
        response_logprobs = turn_data["response_logprobs"]

        if len(response_logprobs) > len(response_ids):
            response_logprobs = response_logprobs[:len(response_ids)]
        elif len(response_logprobs) < len(response_ids):
            response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))

        sample = TrainingSample(
            tokens=prompt_ids + response_ids,
            prompt_len=len(prompt_ids),
            response_len=len(response_ids),
            rollout_log_probs=response_logprobs,
            teacher_log_probs=None,
            reward=float(eval_score),
            group_index=next(self._group_counter),
            index=next(self._index_counter),
            prompt_text=turn_data["prompt_text"],
            response_text=turn_data["response_text"],
        )
        logger.info(
            "[Fireworks] submitted RL sample session=%s index=%d "
            "score=%.1f prompt_len=%d response_len=%d",
            session_id, sample.index, float(eval_score),
            len(prompt_ids), len(response_ids),
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    # ------------------------------------------------------------------
    # Dispatch: one sample per turn, merging OPD + RL when possible
    # ------------------------------------------------------------------

    def _maybe_submit_ready_samples(
        self, session_id: str, force_drop_without_next_state: bool = False,
    ):
        prm_tasks = self._prm_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            td = pending[turn_num]
            task = prm_tasks.get(turn_num)

            if task is None:
                if force_drop_without_next_state:
                    pending.pop(turn_num, None)
                    if self._eval_mode:
                        with self._eval_scores_lock:
                            self._eval_scores.append(0.0)
                    logger.info("[Fireworks] dropped session=%s turn=%d (no next_state)", session_id, turn_num)
                continue
            if not task.done():
                continue

            pending.pop(turn_num, None)
            prm_tasks.pop(turn_num, None)
            try:
                opd_result = task.result()
            except Exception as e:
                logger.warning("[Fireworks] evaluation failed session=%s turn=%d: %s", session_id, turn_num, e)
                if self._eval_mode:
                    with self._eval_scores_lock:
                        self._eval_scores.append(0.0)
                continue

            eval_score = opd_result.get("eval_score")
            if self._eval_mode and eval_score is not None:
                with self._eval_scores_lock:
                    self._eval_scores.append(eval_score)

            opd_accepted = opd_result.get("accepted")
            has_valid_rl = _is_valid_rl_score(eval_score)

            if opd_accepted and has_valid_rl:
                self._safe_create_task(
                    self._submit_turn_sample(td, session_id, opd_result, reward=float(eval_score))
                )
            elif opd_accepted:
                self._safe_create_task(
                    self._submit_turn_sample(td, session_id, opd_result, reward=0.0)
                )
            elif has_valid_rl:
                self._safe_create_task(
                    self._submit_rl_turn_sample(td, session_id, float(eval_score))
                )

    # ------------------------------------------------------------------
    # Main request handler
    # ------------------------------------------------------------------

    async def _handle_request(
        self,
        body: dict[str, Any],
        session_id: str,
        turn_type: str,
        session_done: bool,
    ) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        tools = body.get("tools")
        forward_body = {k: v for k, v in body.items() if k not in _NON_STANDARD_BODY_KEYS}
        forward_body["stream"] = False
        forward_body.pop("stream_options", None)
        forward_body["logprobs"] = True
        forward_body["top_logprobs"] = 1
        forward_body["model"] = self.deployment_model

        output = await self._call_deployment_chat(forward_body)

        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []
        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""
        logger.info(
            "%s[Fireworks] [%s] session=%s prompt_msgs=%d%s",
            _YELLOW, turn_type, session_id, len(messages), _RESET,
        )
        logger.info(
            "%s[Fireworks] [%s] session=%s thinking=%d chars, response:\n%s%s",
            _RED, turn_type, session_id, len(reasoning), content, _RESET,
        )

        if turn_type == "main":
            prev_turn_num = self._turn_counts.get(session_id, 0)
            if prev_turn_num > 0 and messages:
                self._flush_pending_record(session_id, messages[-1])
                prev_turn_data = self._pending_turn_data.get(session_id, {}).get(prev_turn_num)
                if prev_turn_data is not None:
                    self._fire_opd_task(session_id, prev_turn_num, prev_turn_data, messages[-1])

            response_msg = dict(assistant_msg)
            if response_msg.get("content") is None:
                response_msg["content"] = ""
            norm_msgs = _normalize_messages_for_template(messages)
            norm_resp = _normalize_messages_for_template([response_msg])[0]
            full_norm = norm_msgs + [norm_resp]

            prompt_text = self.tokenizer.apply_chat_template(
                norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                full_norm, tools=tools, tokenize=False, add_generation_prompt=False,
            )
            response_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]

            if not response_ids and not response_text.strip():
                logger.info("[Fireworks] MAIN session=%s -> empty response, skipping", session_id)
                output["session_id"] = session_id
                return {"response": output}

            response_logprobs = _extract_logprobs_from_chat_response(choice)
            if len(response_logprobs) > len(response_ids):
                response_logprobs = response_logprobs[:len(response_ids)]
            elif len(response_logprobs) < len(response_ids):
                response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))

            self._turn_counts[session_id] = prev_turn_num + 1
            turn_num = self._turn_counts[session_id]
            turn_data = {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "messages": messages,
                "tools": tools,
                "has_next_state": False,
            }
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)
            logger.info(
                "[Fireworks] MAIN session=%s turn=%d prompt_tokens=%d response_tokens=%d",
                session_id, turn_num, len(prompt_ids), len(response_ids),
            )
            self._maybe_submit_ready_samples(session_id)
        else:
            logger.info("[Fireworks] SIDE session=%s -> skipped (no training data)", session_id)

        if session_done:
            self._flush_pending_record(session_id, None)
            self._maybe_submit_ready_samples(session_id, force_drop_without_next_state=True)
            self._turn_counts.pop(session_id, None)
            logger.info("[Fireworks] session=%s done -> cleaned up", session_id)

        output["session_id"] = session_id
        return {"response": output}

    def _fire_opd_task(
        self, session_id: str, turn_num: int,
        turn_data: dict[str, Any], next_state: dict[str, Any],
    ):
        if not self._prm_enabled or not next_state:
            return
        task = asyncio.create_task(self._opd_evaluate(session_id, turn_num, turn_data, next_state))
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task
        turn_data["has_next_state"] = True

    # ------------------------------------------------------------------
    # Record keeping
    # ------------------------------------------------------------------

    def _buffer_record(self, session_id, turn_num, messages, prompt_text, response_text, tool_calls):
        if not self._record_file:
            return
        self._pending_records[session_id] = {
            "session_id": session_id, "turn": turn_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages, "prompt_text": prompt_text,
            "response_text": response_text, "tool_calls": tool_calls or None,
        }

    def _flush_pending_record(self, session_id, next_state):
        rec = self._pending_records.pop(session_id, None)
        if rec is None:
            return
        rec["next_state"] = next_state
        if self._record_file:
            try:
                with open(self._record_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("[Fireworks] failed to write record: %s", e)

    def _append_prm_record(self, record):
        if not self._prm_record_file:
            return
        try:
            with open(self._prm_record_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("[Fireworks] failed to write PRM record: %s", e)

    def drain_eval_scores(self) -> list[float]:
        with self._eval_scores_lock:
            scores = list(self._eval_scores)
            self._eval_scores.clear()
            return scores

    def reset_eval_scores(self):
        with self._eval_scores_lock:
            self._eval_scores.clear()

    def purge_record_files(self):
        for path in (self._record_file, self._prm_record_file):
            if not path:
                continue
            try:
                open(path, "w").close()
            except OSError:
                pass

    def _safe_create_task(self, coro):
        task = asyncio.create_task(coro)
        task.add_done_callback(self._task_done_cb)

    @staticmethod
    def _task_done_cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[Fireworks] background task failed: %s", exc, exc_info=exc)

    async def _stream_response(self, result: dict[str, Any]):
        payload = result["response"]
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        delta = {"role": "assistant", "content": message.get("content", "") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        chunk_base = {
            "id": payload.get("id", ""),
            "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", ""),
            "session_id": payload.get("session_id", ""),
        }
        first = {**chunk_base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        final = {
            **chunk_base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}],
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config=config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        def _readiness_check():
            while True:
                try:
                    r = httpx.get(self.deployment_chat_url.replace("/chat/completions", "/../models"),
                                  headers={"Authorization": f"Bearer {self.fw_api_key}"}, timeout=10)
                    if r.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(3)
            banner = (
                f"\n{'=' * 70}\n"
                f"  [Fireworks] OpenClaw proxy ready\n"
                f"  proxy {self.host}:{self.port} -> {self.deployment_chat_url}\n"
                f"  PRM enabled: {self._prm_enabled} (m={self._prm_m})\n"
                f"{'=' * 70}\n"
            )
            logger.info(f"{_GREEN}{banner}{_RESET}")

        threading.Thread(target=_readiness_check, daemon=True).start()

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
