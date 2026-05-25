from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from slime.utils.types import Sample

from code_rl_api_server import generate as base_generate


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _normalize_chat_url(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/v1/chat/completions") or cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _judge_backend_config() -> tuple[str, str, str]:
    url = _normalize_chat_url(
        os.getenv("CODE_RL_BENCHMARK_EVAL_JUDGE_URL", "") or os.getenv("CODE_RL_PRM_OPENAI_URL", "")
    )
    model = (
        os.getenv("CODE_RL_BENCHMARK_EVAL_JUDGE_MODEL_NAME", "")
        or os.getenv("CODE_RL_PRM_OPENAI_MODEL_NAME", "")
    ).strip()
    api_key = os.getenv("CODE_RL_BENCHMARK_EVAL_JUDGE_API_KEY", "") or os.getenv("CODE_RL_PRM_API_KEY", "")
    return url, model, api_key


def _heuristic_score(sample: Sample, metadata: dict[str, Any]) -> dict[str, Any]:
    response = (sample.response or "").strip().lower()
    score = 0.1
    matched: list[str] = []

    if len(response.split()) >= 30:
        score += 0.2
        matched.append("substantive")

    if any(token in response for token in ["inspect", "check", "review", "open", "read", "look at", "verify", "run"]):
        score += 0.25
        matched.append("action_plan")

    if any(token in response for token in ["risk", "constraint", "edge case", "backward", "compatib"]):
        score += 0.15
        matched.append("risk_awareness")

    if any(token in response for token in ["test", "verify", "validation", "pytest"]):
        score += 0.15
        matched.append("verification")

    tags = [str(item).lower() for item in metadata.get("tags", [])]
    skills = [str(item).lower() for item in metadata.get("available_skills", [])]
    hint_tokens = tags + skills + [str(metadata.get("category", "")).lower()]
    if any(token and token in response for token in hint_tokens):
        score += 0.15
        matched.append("task_specificity")

    if any(token in response for token in ["already done", "completed it", "finished it"]) and "will" not in response:
        score -= 0.2
        matched.append("premature_completion_penalty")

    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "source": "heuristic",
        "matched_signals": matched,
    }


async def _judge_with_llm(sample: Sample, metadata: dict[str, Any]) -> dict[str, Any] | None:
    url, model, api_key = _judge_backend_config()
    if not url or not model:
        return None

    system = (
        "You are scoring the first assistant response for a benchmark-style coding or coworker-agent task.\n"
        "Judge whether the response shows strong task understanding, concrete first actions, relevant tools or systems, "
        "verification awareness, and no false claim that the task is already done.\n"
        "Return strict JSON only with keys score, strengths, and gaps. score must be between 0 and 1."
    )
    user = (
        f"Benchmark metadata:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        f"Prompt shown to the model:\n{sample.prompt}\n\n"
        f"Model response:\n{sample.response}"
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 400,
        "stream": False,
    }
    timeout = float(os.getenv("CODE_RL_BENCHMARK_EVAL_JUDGE_TIMEOUT_SEC", "90"))
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        content = str(resp.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    parsed = _extract_json_object(content)
    if not parsed:
        return None
    try:
        score = float(parsed.get("score", 0.0))
    except Exception:
        score = 0.0
    parsed["score"] = max(0.0, min(1.0, score))
    parsed["source"] = "llm_judge"
    return parsed


async def generate_with_judge(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    sample = await base_generate(args, sample, sampling_params, evaluation=evaluation)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    judge_result = None
    try:
        judge_result = await _judge_with_llm(sample, metadata)
    except Exception as exc:  # pragma: no cover - best effort external judge
        judge_result = {
            "score": 0.0,
            "source": "llm_judge_error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    if judge_result is None or judge_result.get("source") == "llm_judge_error":
        heuristic = _heuristic_score(sample, metadata)
        if judge_result and judge_result.get("source") == "llm_judge_error":
            heuristic["judge_error"] = judge_result["error"]
        judge_result = heuristic

    sample.reward = {
        "score": float(judge_result.get("score", 0.0)),
        "source": judge_result.get("source", "benchmark_proxy"),
        "benchmark_source": metadata.get("benchmark_source"),
        "task_id": metadata.get("task_id"),
        "details": judge_result,
    }
    return sample
