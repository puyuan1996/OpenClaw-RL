from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


_SCORE_PATTERN = re.compile(r"score\s*=\s*([0-9]+\.?[0-9]*)")


@dataclass
class CSDecision:
    decision: str
    risk_level: str
    composite_score: float
    reason: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CSSummary:
    composite_score: float
    dimensions: dict[str, float]
    raw: dict[str, Any] = field(default_factory=dict)


def _flatten_args(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "shell_exec" and isinstance(args, dict) and "command" in args:
        return str(args["command"])
    if tool_name == "shell_write_content_to_file" and isinstance(args, dict):
        path = args.get("file_path", "")
        content = args.get("content", "")
        return f"write_file:{path}\n{content}"
    if tool_name == "shell_write_to_process" and isinstance(args, dict):
        return str(args.get("command", ""))
    try:
        return json.dumps(args, ensure_ascii=False)[:4096]
    except Exception:
        return str(args)[:4096]


def _parse_composite_score(metadata: dict[str, Any], reason: str) -> float:
    raw = metadata.get("composite_score")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    if reason:
        m = _SCORE_PATTERN.search(reason)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return 0.0


class ClawSentryClient:

    def __init__(
        self,
        base_url: str,
        session_id: str,
        agent_id: str = "openclaw-rl-trainer",
        auth_token: str | None = None,
        timeout: float = 2.0,
        enabled: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.agent_id = agent_id
        self.enabled = enabled
        self._closed = False
        self._calls = 0
        self._errors = 0
        self._decisions: dict[str, int] = {}

        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def pre_action(
        self, tool_name: str, args: dict[str, Any]
    ) -> CSDecision | None:
        if not self.enabled or self._closed:
            return None
        self._calls += 1
        payload_str = _flatten_args(tool_name, args)
        req = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex[:8],
            "method": "ahp/event",
            "params": {
                "event_type": "pre_action",
                "session_id": self.session_id,
                "agent_id": self.agent_id,
                "payload": {"tool": tool_name, "command": payload_str},
            },
        }
        try:
            r = await self._client.post(f"{self.base_url}/ahp/a3s", json=req)
            r.raise_for_status()
            body = r.json()
            result = body.get("result") or {}
            metadata = result.get("metadata") or {}
            decision = str(result.get("decision") or "allow")
            risk_level = str(metadata.get("risk_level") or "low")
            reason = str(result.get("reason") or "")
            composite = _parse_composite_score(metadata, reason)
            self._decisions[decision] = self._decisions.get(decision, 0) + 1
            return CSDecision(
                decision=decision,
                risk_level=risk_level,
                composite_score=composite,
                reason=reason,
                raw=result,
            )
        except (httpx.HTTPError, asyncio.TimeoutError, ValueError, KeyError) as exc:
            self._errors += 1
            logger.warning(
                "ClawSentry pre_action fail-open (session=%s tool=%s): %s",
                self.session_id,
                tool_name,
                exc,
            )
            return None

    async def fetch_summary(self) -> CSSummary | None:
        if not self.enabled or self._closed:
            return None
        try:
            r = await self._client.get(
                f"{self.base_url}/report/session/{self.session_id}/risk"
            )
            r.raise_for_status()
            body = r.json()
            composite = float(body.get("composite_score") or 0.0)
            dims = body.get("dimensions") or {}
            dimensions: dict[str, float] = {}
            for k, v in dims.items():
                try:
                    dimensions[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            return CSSummary(
                composite_score=composite, dimensions=dimensions, raw=body
            )
        except (httpx.HTTPError, asyncio.TimeoutError, ValueError, KeyError) as exc:
            self._errors += 1
            logger.warning(
                "ClawSentry fetch_summary fail-open (session=%s): %s",
                self.session_id,
                exc,
            )
            return None

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._calls,
            "errors": self._errors,
            "decisions": dict(self._decisions),
        }

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._client.aclose()
        except Exception as exc:
            logger.debug("ClawSentry aclose ignored: %s", exc)
