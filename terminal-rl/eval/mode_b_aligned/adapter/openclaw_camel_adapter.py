"""Harbor BaseAgent adapter that reuses OpenClaw-RL's CamelAgent + SGLangTurnClient stack.

True training-time alignment: instantiates camel.toolkits.TerminalToolkit with
``use_docker_backend=True, docker_container_name=<harbor compose main container>``
— exactly the same construction used at training time in
``terminal-rl/remote/terminal_env.py``.

The adapter ONLY swaps the LLM/agent driver path. Harbor still owns the TB 2.0
docker-compose lifecycle, task setup, and verifier execution.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# --- ensure OpenClaw-RL terminal-rl is importable ----------------------------
#
# This adapter lives at <repo>/terminal-rl/eval/mode_b_aligned/adapter/, so the
# training-side package root is three levels up. Resolving it from __file__ keeps
# the adapter runnable from any checkout and any cwd; OPENCLAW_TERMINAL_RL_DIR
# stays available to point at a different checkout.

_TERMINAL_RL_MARKER = Path("agent") / "camel_agent.py"


def _resolve_terminal_rl_dir() -> str:
    """Return the terminal-rl package root that provides the training-side modules.

    Raises RuntimeError with an actionable message instead of letting the import
    of `agent.camel_agent` fail later with a bare ModuleNotFoundError.
    """
    override = os.environ.get("OPENCLAW_TERMINAL_RL_DIR")
    candidate = Path(override) if override else Path(__file__).resolve().parents[3]
    if not (candidate / _TERMINAL_RL_MARKER).is_file():
        source = "OPENCLAW_TERMINAL_RL_DIR" if override else "path relative to adapter"
        raise RuntimeError(
            f"terminal-rl package root not found at {candidate} (resolved from {source}); "
            f"expected it to contain {_TERMINAL_RL_MARKER}. "
            "Set OPENCLAW_TERMINAL_RL_DIR to the terminal-rl directory of an OpenClaw-RL checkout."
        )
    return str(candidate)


_OPENCLAW_TERMINAL_RL_DIR = _resolve_terminal_rl_dir()
if _OPENCLAW_TERMINAL_RL_DIR not in sys.path:
    sys.path.insert(0, _OPENCLAW_TERMINAL_RL_DIR)

# --- slime shim: SGLangTurnClient imports `slime.utils.http_utils.post`.  ----
# In the eval-only conda env, the `slime` package may not be importable. Inject
# an httpx-backed stub BEFORE importing inference_client so the import succeeds.


def _install_slime_shim_if_missing() -> None:
    try:
        from slime.utils.http_utils import post as _existing_post  # noqa: F401
        return
    except Exception:
        pass

    import types

    import httpx

    async def _post(
        url: str,
        payload: dict,
        max_retries: int = 60,
        *,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 1.0,
        retry_backoff_factor: float = 1.0,
        retry_jitter: float = 0.0,
        retry_statuses=None,
        non_retry_statuses=None,
        headers: dict | None = None,
    ):
        last_exc: Exception | None = None
        retry_statuses = retry_statuses or {429, 500, 502, 503, 504}
        non_retry_statuses = non_retry_statuses or set()
        delay = retry_base_delay
        async with httpx.AsyncClient(timeout=None) as client:
            for attempt in range(max(1, max_retries)):
                try:
                    response = await client.post(url, json=payload, headers=headers)
                    status = response.status_code
                    if status in non_retry_statuses:
                        response.raise_for_status()
                    if status in retry_statuses:
                        raise httpx.HTTPStatusError(
                            f"retryable status {status}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    return response.json()
                except Exception as exc:
                    last_exc = exc
                    if attempt >= max_retries - 1:
                        break
                    await asyncio.sleep(min(delay, retry_max_delay))
                    delay *= retry_backoff_factor
        assert last_exc is not None
        raise last_exc

    slime_mod = types.ModuleType("slime")
    slime_utils = types.ModuleType("slime.utils")
    slime_http_utils = types.ModuleType("slime.utils.http_utils")
    slime_http_utils.post = _post
    slime_mod.utils = slime_utils
    slime_utils.http_utils = slime_http_utils
    sys.modules.setdefault("slime", slime_mod)
    sys.modules.setdefault("slime.utils", slime_utils)
    sys.modules.setdefault("slime.utils.http_utils", slime_http_utils)


_install_slime_shim_if_missing()


def _ensure_real_slime_http_client_initialized() -> None:
    """Real slime (when imported, not shimmed) expects ``init_http_client(args)``
    to be called by the slime RolloutManager. In a standalone harbor adapter we
    don't have a slime ``args`` namespace, so we directly construct an
    ``httpx.AsyncClient`` and inject it as ``slime.utils.http_utils._http_client``.

    Without this, ``async_post(url, payload, ...)`` -> ``_post(client=None, ...)``
    blows up with ``AttributeError: 'NoneType' object has no attribute 'post'``.
    """
    try:
        from slime.utils import http_utils as slime_http_utils
    except Exception:
        return  # shim path: nothing to do

    import httpx

    if getattr(slime_http_utils, "_http_client", None) is None:
        slime_http_utils._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=64),
            timeout=httpx.Timeout(None),
        )


_ensure_real_slime_http_client_initialized()

# --- OpenClaw-RL imports (must come AFTER sys.path/sliime-shim setup) --------

from agent.camel_agent import CamelAgent  # noqa: E402
from agent.prompts import get_developer_agent_prompt  # noqa: E402
from agent_runner import create_agent_runner  # noqa: E402
from inference_client import SGLangTurnClient  # noqa: E402

# --- harbor imports ----------------------------------------------------------

from camel.toolkits import FunctionTool, TerminalToolkit  # noqa: E402
from harbor.agents.base import BaseAgent  # noqa: E402
from harbor.environments.base import BaseEnvironment  # noqa: E402
from harbor.models.agent.context import AgentContext  # noqa: E402
from harbor.models.trajectories import (  # noqa: E402
    Agent as TrajAgent,
)
from harbor.models.trajectories import (
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.utils.trajectory_utils import format_trajectory_json  # noqa: E402

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class OpenClawCamelAgent(BaseAgent):
    """Harbor BaseAgent that drives OpenClaw-RL's CamelAgent / AgentRunner.

    Behavior mirrors `terminal-rl/generate.py` lines ~3141-3344 (the rollout
    iteration loop) and the training-time tool wiring in
    `remote/terminal_env.py` (4-tool `TerminalToolkit` over docker backend).
    """

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "openclaw-camel-agent"

    def version(self) -> str | None:
        return "0.1.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        extra_env: dict[str, str] | None = None,
        # sglang / model
        sglang_url: str = "http://127.0.0.1:30000",
        sglang_served_name: str | None = None,
        hf_model_dir: str | None = None,
        model_type: str = "Qwen3",
        # harness alignment to /tmp/i271_latest_config/config/
        max_iteration: int = 10,
        max_total_tokens: int = 16384,
        max_parse_errors: int = 3,
        non_think_mode: bool = False,
        tool_call_parser: str = "qwen25",
        # sampling (mirror inference_config.json -> "generation" block)
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_new_tokens: int = 8192,  # rollout_max_response_len
        rollout_seed: int = 42,
        rollout_skip_special_tokens: bool = False,
        # toolkit
        terminal_toolkit_timeout: float = 20.0,
        terminal_toolkit_workdir: str | None = None,
        # safety / robustness
        request_timeout_sec: float | None = 600.0,
        sglang_max_retries: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

        self.extra_env = dict(extra_env or {})
        self.sglang_url = sglang_url.rstrip("/")
        if not self.sglang_url.endswith("/generate"):
            self.sglang_url = self.sglang_url + "/generate"
        # Both identify the checkpoint under evaluation and have no defensible
        # default: a wrong served name silently evaluates a different model, and a
        # wrong tokenizer dir silently changes the chat template. Require them.
        if not sglang_served_name:
            raise ValueError(
                "sglang_served_name is required; pass "
                "--agent-kwarg sglang_served_name=<--served-model-name of the SGLang server>"
            )
        if not hf_model_dir:
            raise ValueError(
                "hf_model_dir is required; pass "
                "--agent-kwarg hf_model_dir=<HF checkpoint dir used for the tokenizer/chat template>"
            )
        self.sglang_served_name = sglang_served_name
        self.hf_model_dir = hf_model_dir
        self.model_type = model_type
        self.max_iteration = int(max_iteration)
        self.max_total_tokens = int(max_total_tokens)
        self.max_parse_errors = int(max_parse_errors)
        self.non_think_mode = bool(non_think_mode)
        self.tool_call_parser = tool_call_parser
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.max_new_tokens = int(max_new_tokens)
        self.rollout_seed = int(rollout_seed)
        self.rollout_skip_special_tokens = bool(rollout_skip_special_tokens)
        self.terminal_toolkit_timeout = float(terminal_toolkit_timeout)
        self.terminal_toolkit_workdir = terminal_toolkit_workdir
        self.request_timeout_sec = request_timeout_sec
        self.sglang_max_retries = int(sglang_max_retries)

        # Lazy-initialized in setup()
        self._tokenizer = None
        self._sglang_client: SGLangTurnClient | None = None
        self._terminal_toolkit: TerminalToolkit | None = None
        self._container_name: str | None = None
        self._tool_schemas: list[dict[str, Any]] = []
        self._tools_by_name: dict[str, Any] = {}

    # -- Tokenizer / sglang client ------------------------------------------

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)

        # transformers 5.x changed apply_chat_template(tokenize=True) to return
        # a BatchEncoding (dict-like) instead of List[int]. OpenClaw-RL's
        # SGLangTurnClient expects List[int] (training was on transformers
        # 4.51). Monkey-patch the bound method to unwrap to the raw token-id
        # list whenever return_tensors is None / unset and tokenize=True.
        _orig_apply = tok.apply_chat_template

        def _apply_chat_template_compat(*args, **kwargs):
            tokenize = kwargs.get("tokenize", True)
            return_tensors = kwargs.get("return_tensors", None)
            out = _orig_apply(*args, **kwargs)
            if not tokenize or return_tensors is not None:
                return out
            # Unwrap BatchEncoding -> List[int] (List[List[int]] if batched)
            if hasattr(out, "get") and "input_ids" in getattr(out, "data", out):
                ids = out["input_ids"]
                return ids
            return out

        tok.apply_chat_template = _apply_chat_template_compat
        return tok

    def _build_sampling_params(self) -> dict[str, Any]:
        # Align to slime rollout knobs at training time. SGLang's /generate API
        # accepts: temperature, top_p, top_k, max_new_tokens, skip_special_tokens,
        # stop, stop_token_ids, seed, ...
        params: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "skip_special_tokens": self.rollout_skip_special_tokens,
        }
        if self.top_k is not None and int(self.top_k) > 0:
            params["top_k"] = int(self.top_k)
        # rollout_stop / rollout_stop_token_ids are null in training; let
        # SGLangTurnClient inject EOS into stop_token_ids automatically.
        return params

    def _build_sglang_client(self) -> SGLangTurnClient:
        sampling_params = self._build_sampling_params()
        client = SGLangTurnClient(
            model_type=self.model_type,
            tokenizer=self._tokenizer,
            sampling_params=sampling_params,
            url=self.sglang_url,
            chat_template_type="hf",
            chat_template_kwargs={},
            tool_call_parser=self.tool_call_parser,
            # Mirror training-time `generate.py::_create_sglang_client`:
            # max_input_tokens = max(1, max_total_tokens - max_new_tokens).
            # The cap is on the PROMPT (input_ids), not the full request.
            max_input_tokens=max(1, self.max_total_tokens - self.max_new_tokens),
            request_timeout=self.request_timeout_sec,
            max_retries=self.sglang_max_retries,
        )
        return client

    # -- Container discovery ------------------------------------------------

    async def _resolve_container_name(self, environment: BaseEnvironment) -> str:
        """Discover the docker container name for harbor's "main" compose service.

        Harbor's DockerEnvironment runs ``docker compose --project-name <sid> up``,
        with a single service called ``main``. The resulting container is named
        ``<sanitized-session-id>-main-<index>``. We resolve it by querying
        ``docker ps`` for the compose project label.
        """
        session_id = getattr(environment, "session_id", None)
        if not session_id:
            raise RuntimeError(
                "environment.session_id missing; cannot resolve docker container name"
            )

        sanitized = _sanitize_docker_compose_project_name(str(session_id))

        # Probe via docker CLI: filter by compose project label.
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={sanitized}",
            "--filter",
            "label=com.docker.compose.service=main",
            "--format",
            "{{.Names}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker ps failed (rc={proc.returncode}): "
                f"{(stderr_bytes or b'').decode(errors='replace')}"
            )
        names = [
            ln.strip()
            for ln in (stdout_bytes or b"").decode(errors="replace").splitlines()
            if ln.strip()
        ]
        if not names:
            raise RuntimeError(
                f"No running main container found for compose project {sanitized!r}"
            )
        # Sort to pick the lowest index deterministically (usually -main-1).
        names.sort()
        return names[0]

    # -- Tool wiring (training-aligned: same 4 tools) -----------------------

    def _build_toolkit(self, container_name: str) -> TerminalToolkit:
        session_logs_dir = Path(self.logs_dir) / "terminal_toolkit_session_logs"
        session_logs_dir.mkdir(parents=True, exist_ok=True)
        # Force TerminalToolkit's own log_dir to live under logs_dir too, since
        # camel uses a fixed cwd-relative default that may be read-only.
        cwd = os.getcwd()
        try:
            os.chdir(str(self.logs_dir))
            toolkit = TerminalToolkit(
                timeout=self.terminal_toolkit_timeout,
                working_directory=self.terminal_toolkit_workdir,
                use_docker_backend=True,
                docker_container_name=container_name,
                session_logs_dir=session_logs_dir,
                safe_mode=False,
            )
        finally:
            os.chdir(cwd)
        return toolkit

    def _extract_4_tool_schemas(
        self, toolkit: TerminalToolkit
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build the same 4-tool schema set used at training (remote/terminal_env.py:1268)."""
        tools_by_name: dict[str, Any] = {
            "shell_exec": toolkit.shell_exec,
            "shell_view": toolkit.shell_view,
            "shell_write_to_process": toolkit.shell_write_to_process,
            "shell_write_content_to_file": toolkit.shell_write_content_to_file,
        }
        function_tools = [FunctionTool(fn) for fn in tools_by_name.values()]
        tool_schemas = [
            func_tool.get_openai_tool_schema() for func_tool in function_tools
        ]
        return tool_schemas, tools_by_name

    # -- BaseAgent: setup ---------------------------------------------------

    async def setup(self, environment: BaseEnvironment) -> None:
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        # Tokenizer and sglang client are independent of the environment.
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
        if self._sglang_client is None:
            self._sglang_client = self._build_sglang_client()

        # Resolve harbor's docker container so TerminalToolkit can attach.
        self._container_name = await self._resolve_container_name(environment)
        self.logger.info("Resolved harbor container: %s", self._container_name)

        self._terminal_toolkit = self._build_toolkit(self._container_name)
        self._tool_schemas, self._tools_by_name = self._extract_4_tool_schemas(
            self._terminal_toolkit
        )
        self.logger.info(
            "Initialized TerminalToolkit with %d tools: %s",
            len(self._tool_schemas),
            list(self._tools_by_name.keys()),
        )

    # -- Tool dispatcher ---------------------------------------------------

    async def _exec_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        if tool_name not in self._tools_by_name:
            return f"[TOOL_ERROR] unknown tool: {tool_name}"
        fn = self._tools_by_name[tool_name]
        # TerminalToolkit functions are synchronous; offload to a thread to
        # avoid blocking the harbor event loop.
        try:
            result = await asyncio.to_thread(fn, **(args or {}))
        except TypeError as exc:
            # Some camel-toolkit funcs accept positional only; fall back to
            # passing args dict positionally is unsafe, so surface a clean
            # error to the agent.
            return f"[TOOL_ERROR] {tool_name}: {exc}"
        except Exception as exc:
            self.logger.exception("Tool %s raised", tool_name)
            return f"[TOOL_ERROR] {tool_name}: {exc!r}"
        if not isinstance(result, str):
            try:
                result = json.dumps(result, ensure_ascii=False)
            except Exception:
                result = str(result)
        return result

    # -- BaseAgent: run ----------------------------------------------------

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if (
            self._sglang_client is None
            or self._terminal_toolkit is None
            or not self._tool_schemas
        ):
            raise RuntimeError(
                "OpenClawCamelAgent.run called before setup completed."
            )

        user_msg = f"Task instruction: {instruction}"

        # Initialize `agent_runner` BEFORE the try/finally so the finally
        # block can always safely dereference it for metadata.
        agent_runner = None
        # Trajectory state
        steps: list[Step] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        reached_iteration_limit = False
        reached_parse_error_limit = False
        terminated_reason: str | None = None
        final_response = None
        final_model_response = None
        run_started = time.monotonic()
        session_id = str(getattr(environment, "session_id", uuid.uuid4().hex))

        # Step 1 = system + user prelude (single composite step for ATIF).
        system_prompt = get_developer_agent_prompt(
            current_date=str(datetime.date.today()),
            system="Linux (in Docker)",
            machine="x86_64",
            is_workforce=False,
            non_think_mode=self.non_think_mode,
        )
        steps.append(
            Step(
                step_id=1,
                timestamp=_utc_now_iso(),
                source="system",
                message=system_prompt,
            )
        )
        steps.append(
            Step(
                step_id=2,
                timestamp=_utc_now_iso(),
                source="user",
                message=user_msg,
            )
        )
        next_step_id = 3

        try:
            agent_runner = create_agent_runner(
                agent_type="camel-agent",
                sglang_client=self._sglang_client,
                model_type=self.model_type,
                tool_schemas=self._tool_schemas,
                non_think_mode=self.non_think_mode,
                max_total_tokens=self.max_total_tokens,
                # env_client/lease/run_ctx/task_meta are only used by slime-side
                # harnesses (a3s-code, claude-code); camel-agent path ignores them.
                env_client=None,
                lease_id=None,
                run_context=None,
                task_meta=None,
            )
            agent_runner.reset(user_msg)
            agent_runner.set_max_parse_errors(self.max_parse_errors)
            agent_runner.set_max_iterations(self.max_iteration)

            while True:
                context_result = await agent_runner.get_turn_context()
                if context_result.terminated_response is not None:
                    self.logger.warning(
                        "Rollout pre-terminated before model turn (context overflow)."
                    )
                    final_response = context_result.terminated_response
                    reasons = (
                        getattr(final_response, "info", {}).get(
                            "termination_reasons", []
                        )
                        if final_response is not None
                        else []
                    )
                    if "max_tokens_exceeded" in (reasons or []):
                        terminated_reason = "max_tokens_exceeded"
                    break
                if context_result.context_messages is None:
                    self.logger.warning("Context is empty; aborting loop.")
                    break

                turn_state = await agent_runner.run_model_turn(
                    context_result.context_messages
                )
                turn_interactions = (
                    getattr(turn_state, "interactions", None)
                    or [turn_state.interaction]
                )
                interaction = turn_interactions[-1]
                turn_idx = int(getattr(interaction, "turn_idx", 0))
                in_tok = len(getattr(interaction, "input_ids", []) or [])
                out_tok = len(getattr(interaction, "output_token_ids", []) or [])
                total_prompt_tokens += in_tok
                total_completion_tokens += out_tok

                assistant_text = getattr(interaction, "output_text", None) or ""
                tool_calls = [
                    ToolCall(
                        tool_call_id=tc.tool_call_id or f"call_{uuid.uuid4().hex[:12]}",
                        function_name=tc.tool_name,
                        arguments=dict(tc.args or {}),
                    )
                    for tc in (turn_state.tool_call_requests or [])
                ]
                # Execute tool calls first, then bundle (assistant message +
                # tool_calls + observation) into a SINGLE ATIF step. ATIF schema
                # requires Observation.source_call_id to reference a tool_call
                # within the SAME step, so we cannot split assistant + obs.
                obs_results: list[ObservationResult] = []
                should_continue = False
                if turn_state.tool_call_requests:
                    for tc in turn_state.tool_call_requests:
                        raw_result = await self._exec_tool(tc.tool_name, tc.args)
                        agent_runner.record_tool_result(tc, raw_result)
                        obs_results.append(
                            ObservationResult(
                                source_call_id=tc.tool_call_id,
                                content=raw_result,
                            )
                        )
                    should_continue = True

                assistant_step = Step(
                    step_id=next_step_id,
                    timestamp=_utc_now_iso(),
                    source="agent",
                    model_name=self.sglang_served_name,
                    message=assistant_text,
                    tool_calls=tool_calls or None,
                    observation=Observation(results=obs_results) if obs_results else None,
                    metrics=Metrics(
                        prompt_tokens=in_tok,
                        completion_tokens=out_tok,
                    ),
                )
                steps.append(assistant_step)
                next_step_id += 1

                if turn_state.terminated_response is not None:
                    final_response = turn_state.terminated_response
                    reasons = getattr(final_response, "info", {}).get(
                        "termination_reasons", []
                    )
                    if "max_tokens_exceeded" in (reasons or []):
                        terminated_reason = "max_tokens_exceeded"
                    break

                if turn_state.model_response is None:
                    self.logger.warning(
                        "Turn %d returned empty model_response.", turn_idx
                    )
                    break

                if turn_state.parse_error_recorded:
                    self.logger.warning("Turn %d: tool-call parse error.", turn_idx)
                    should_continue = True

                if should_continue:
                    if (
                        turn_state.parse_error_recorded
                        and agent_runner.reached_parse_error_limit()
                    ):
                        reached_parse_error_limit = True
                        final_model_response = turn_state.model_response
                        break
                    if agent_runner.reached_iteration_limit():
                        reached_iteration_limit = True
                        final_model_response = turn_state.model_response
                        break
                    continue

                # Agent emitted no tools and no parse error: it's done.
                final_model_response = turn_state.model_response
                break

            if final_response is None and final_model_response is not None:
                final_response = agent_runner.finalize_response(final_model_response)

        except asyncio.CancelledError:
            self.logger.warning("OpenClawCamelAgent.run cancelled; flushing trajectory")
            raise
        except Exception:
            self.logger.exception("OpenClawCamelAgent.run failed")
            raise
        finally:
            # Compute status string for metadata.
            if terminated_reason == "max_tokens_exceeded":
                status = "TRUNCATED"
            elif reached_iteration_limit:
                status = "TRUNCATED"
            elif reached_parse_error_limit:
                status = "FAILED"
            elif final_response is None or not getattr(final_response, "msgs", None):
                status = "ABORTED" if final_response is None else "COMPLETED"
            else:
                status = "COMPLETED"

            elapsed = time.monotonic() - run_started

            # Populate harbor's AgentContext.
            context.n_input_tokens = total_prompt_tokens
            context.n_output_tokens = total_completion_tokens
            context.cost_usd = 0.0  # local sglang, no $$ cost
            context.metadata = {
                "harness_option": "camel-agent",
                "agent_type": "openclaw-camel-agent",
                "model_type": self.model_type,
                "served_model_name": self.sglang_served_name,
                "non_think_mode": self.non_think_mode,
                "tool_call_parser": self.tool_call_parser,
                "max_iteration": self.max_iteration,
                "max_total_tokens": self.max_total_tokens,
                "max_parse_errors": self.max_parse_errors,
                "sampling_params": self._build_sampling_params(),
                "rollout_seed": self.rollout_seed,
                "status": status,
                "model_turn_count": agent_runner.model_turn_count
                if agent_runner is not None
                else 0,
                "parse_error_count": agent_runner.parse_error_count
                if agent_runner is not None
                else 0,
                "elapsed_sec": elapsed,
                "container_name": self._container_name,
                "sglang_url": self.sglang_url,
            }

            # Always write trajectory.json, even on exception.
            try:
                if not steps:
                    steps.append(
                        Step(
                            step_id=1,
                            timestamp=_utc_now_iso(),
                            source="system",
                            message="(no steps recorded)",
                        )
                    )
                trajectory = Trajectory(
                    schema_version="ATIF-v1.6",
                    session_id=session_id,
                    agent=TrajAgent(
                        name=self.name(),
                        version=self.version() or "unknown",
                        model_name=self.sglang_served_name,
                        tool_definitions=list(self._tool_schemas) or None,
                        extra={
                            "harness_option": "camel-agent",
                            "non_think_mode": self.non_think_mode,
                            "tool_call_parser": self.tool_call_parser,
                            "max_iteration": self.max_iteration,
                            "max_total_tokens": self.max_total_tokens,
                            "sampling_params": self._build_sampling_params(),
                        },
                    ),
                    steps=steps,
                    final_metrics=FinalMetrics(
                        total_prompt_tokens=total_prompt_tokens,
                        total_completion_tokens=total_completion_tokens,
                        total_cost_usd=0.0,
                        total_steps=len(steps),
                        extra={"status": status, "elapsed_sec": elapsed},
                    ),
                    notes=f"OpenClaw-RL camel-agent harness over harbor. status={status}.",
                )
                traj_path = Path(self.logs_dir) / "trajectory.json"
                traj_path.write_text(
                    format_trajectory_json(trajectory.to_json_dict(exclude_none=True))
                )
            except Exception:
                self.logger.exception("Failed to write trajectory.json")

            # Close any tmux/toolkit sessions best-effort.
            try:
                close_fn = getattr(self._terminal_toolkit, "cleanup", None)
                if callable(close_fn):
                    await asyncio.to_thread(close_fn)
            except Exception:
                self.logger.exception("TerminalToolkit cleanup failed (non-fatal)")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sanitize_docker_compose_project_name(name: str) -> str:
    """Match harbor.environments.docker.docker._sanitize_docker_compose_project_name."""
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    name = re.sub(r"[^a-z0-9_-]", "-", name)
    return name


__all__ = ["OpenClawCamelAgent"]
