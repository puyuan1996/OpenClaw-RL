"""Async HTTP client for swe_env_pool_server.

Used by generate_with_swe_remote.py (inside the RolloutManager) and standalone
scripts to interact with remote Docker containers via the pool server.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx


class SweEnvClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("SWE_ENV_SERVER_URL", "http://localhost:18090")).rstrip("/")
        self.default_max_retries = int(os.getenv("SWE_ENV_HTTP_MAX_RETRIES", "10"))
        self.evaluate_max_retries = int(os.getenv("SWE_EVALUATE_MAX_RETRIES", "3"))
        max_connections = int(os.getenv("SWE_ENV_HTTP_MAX_CONNECTIONS", "128"))
        trust_env = os.getenv("SWE_ENV_HTTP_TRUST_ENV", "0").lower() in ("1", "true", "yes", "on")
        default_timeout = float(os.getenv("SWE_ENV_HTTP_TIMEOUT", "0"))
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(None) if default_timeout <= 0 else httpx.Timeout(default_timeout),
            trust_env=trust_env,
        )

    async def _post(self, path: str, payload: dict[str, Any], max_retries: int,
                    request_timeout: float | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        retry_count = 0
        req_kwargs: dict[str, Any] = {}
        if request_timeout is not None:
            req_kwargs["timeout"] = request_timeout
        while retry_count < max_retries:
            response = None
            try:
                response = await self._client.post(url, json=payload or {}, **req_kwargs)
                response.raise_for_status()
                content = await response.aread()
                try:
                    output = json.loads(content)
                except json.JSONDecodeError:
                    output = content.decode() if isinstance(content, bytes) else content
                if not isinstance(output, dict):
                    raise RuntimeError(f"SWE request returned non-dict payload: {type(output).__name__}")
                return output
            except Exception:
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                await asyncio.sleep(1)
            finally:
                if response is not None:
                    await response.aclose()

        raise RuntimeError(f"SWE request failed after retries: {url}")

    async def allocate(self, image: str, instance_id: str = "") -> dict[str, Any]:
        out = await self._post("/allocate", {"image": image, "instance_id": instance_id}, self.default_max_retries)
        if not out.get("ok", False):
            raise RuntimeError(f"SWE allocate failed: {out}")
        return out

    async def heartbeat(self, lease_id: str) -> None:
        out = await self._post("/heartbeat", {"lease_id": lease_id}, self.default_max_retries)
        if not out.get("ok", False):
            raise RuntimeError(f"SWE heartbeat failed: {out}")

    async def exec(
        self,
        lease_id: str,
        command: str,
        cwd: str = "/testbed",
        timeout: int = 180,
        env: dict | None = None,
    ) -> dict[str, Any]:
        """Execute a command in the container. Returns {ok, returncode, output}."""
        out = await self._post(
            "/exec",
            {
                "lease_id": lease_id,
                "command": command,
                "cwd": cwd,
                "timeout": timeout,
                "env": env or {},
            },
            self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"SWE exec failed: {out}")
        return out

    async def diff(self, lease_id: str, cwd: str = "/testbed") -> str:
        """Get git diff from the container. Returns the patch string."""
        out = await self._post("/diff", {"lease_id": lease_id, "cwd": cwd}, self.default_max_retries)
        if not out.get("ok", False):
            raise RuntimeError(f"SWE diff failed: {out}")
        return out.get("patch", "")

    async def evaluate(
        self,
        lease_id: str,
        patch: str,
        eval_script: str,
        cwd: str = "/testbed",
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Apply patch + run eval script. Returns {ok, resolved, ...}."""
        http_timeout = timeout + 120
        out = await self._post(
            "/evaluate",
            {
                "lease_id": lease_id,
                "patch": patch,
                "eval_script": eval_script,
                "cwd": cwd,
                "timeout": timeout,
            },
            self.evaluate_max_retries,
            request_timeout=http_timeout,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"SWE evaluate failed: {out}")
        return out

    async def close(self, lease_id: str) -> None:
        out = await self._post("/close", {"lease_id": lease_id}, self.default_max_retries)
        if not out.get("ok", False):
            raise RuntimeError(f"SWE close failed: {out}")
        return None
