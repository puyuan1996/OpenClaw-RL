#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - script entrypoint path bootstrap
    PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from a3s_code_benchmarks.benchmark_runtime_utils import render_openai_agent_config
from dotenv import load_dotenv


LOGGER = logging.getLogger("clawmark_official_eval_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ClawMark task with a3s-code runtime.")
    parser.add_argument("--clawmark-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--compose-file", type=Path, required=True)
    parser.add_argument("--wheel-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-base-url", type=str, required=True)
    parser.add_argument("--model-api-key", type=str, required=True)
    parser.add_argument("--session-id-header", type=str, default="X-Session-Id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--log-level", type=str, default=os.getenv("A3S_CODE_OFFICIAL_BENCHMARK_LOG_LEVEL", "INFO"))
    return parser.parse_args()


def _load_clawmark_modules(clawmark_root: Path):
    clawmark_src = clawmark_root / "src"
    if str(clawmark_src) not in sys.path:
        sys.path.insert(0, str(clawmark_src))

    from clawmark.main import StageResult  # type: ignore
    from clawmark.orchestrator import Orchestrator  # type: ignore
    from clawmark.sandbox.docker import DockerSandbox  # type: ignore
    from clawmark.sandbox.dry_run import DryRunSandbox  # type: ignore
    from clawmark.state.composite import CompositeStateManager  # type: ignore
    from clawmark.task_loader import load_task_py  # type: ignore

    return {
        "StageResult": StageResult,
        "Orchestrator": Orchestrator,
        "DockerSandbox": DockerSandbox,
        "DryRunSandbox": DryRunSandbox,
        "CompositeStateManager": CompositeStateManager,
        "load_task_py": load_task_py,
    }


class ClawMarkA3SRuntime:
    def __init__(
        self,
        *,
        clawmark_root: Path,
        wheel_path: Path,
        model_name: str,
        model_base_url: str,
        model_api_key: str,
        session_id_header: str,
    ):
        modules = _load_clawmark_modules(clawmark_root)
        self._Orchestrator = modules["Orchestrator"]
        self._DockerSandbox = modules["DockerSandbox"]
        self._DryRunSandbox = modules["DryRunSandbox"]
        self._CompositeStateManager = modules["CompositeStateManager"]
        self._load_task_py = modules["load_task_py"]
        self._StageResult = modules["StageResult"]
        self._wheel_path = wheel_path
        self._model_name = model_name
        self._model_base_url = model_base_url
        self._model_api_key = model_api_key
        self._session_id_header = session_id_header
        self._runner_path = Path(__file__).with_name("clawmark_a3s_code_runner.py")

    def _make_orchestrator_class(self):
        wheel_path = self._wheel_path
        runner_path = self._runner_path
        model_name = self._model_name
        model_base_url = self._model_base_url
        model_api_key = self._model_api_key
        session_id_header = self._session_id_header

        class A3SCodeOrchestrator(self._Orchestrator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._trace_remote_path = f"/root/.a3s/traces/{self.session_id}.jsonl"

            @property
            def trace_remote_path(self) -> str:
                return self._trace_remote_path

            async def _setup_openclaw_config(
                self,
                *,
                model: str,
                api_key: str,
                api_base: str,
                api_format: str = "anthropic",
            ) -> None:
                await self.sandbox.exec("mkdir -p /root/.a3s /root/.a3s/sessions /root/.a3s/traces")

                config_text = render_openai_agent_config(
                    base_url=model_base_url,
                    model_name=model_name,
                    api_key=model_api_key,
                    context_tokens=int(os.getenv("A3S_CODE_CONTEXT_TOKENS", "131072")),
                    output_tokens=int(os.getenv("A3S_CODE_OUTPUT_TOKENS", "8192")),
                    session_id_header=session_id_header,
                )
                with tempfile.TemporaryDirectory(prefix="clawmark-a3s-config-") as tmp:
                    local_config = Path(tmp) / "config.acl"
                    local_install = Path(tmp) / "install_a3s_code.sh"
                    local_config.write_text(config_text, encoding="utf-8")
                    local_install.write_text(
                        """#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! python3 - <<'PY' >/dev/null 2>&1
import venv
PY
then
  apt-get update
  apt-get install -y --no-install-recommends python3 python3-pip python3-venv
  rm -rf /var/lib/apt/lists/*
fi
python3 -m venv /opt/a3s-code-venv
. /opt/a3s-code-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install /root/.a3s/a3s_code.whl
""",
                        encoding="utf-8",
                    )
                    await self.sandbox.upload_file(local_config, "/root/.a3s/config.acl")
                    await self.sandbox.upload_file(local_install, "/root/.a3s/install_a3s_code.sh")

                await self.sandbox.upload_file(wheel_path, "/root/.a3s/a3s_code.whl")
                await self.sandbox.upload_file(runner_path, "/root/.a3s/clawmark_a3s_code_runner.py")
                await self.sandbox.exec("bash /root/.a3s/install_a3s_code.sh", timeout_sec=1800)
                LOGGER.info("Configured a3s-code runtime inside ClawMark sandbox")

            async def _send_to_agent(self, *, message: str, timeout_sec: int):
                env = {
                    "A3S_CODE_CONFIG": "/root/.a3s/config.acl",
                    "A3S_CODE_WORKSPACE": "/workspace",
                    "A3S_CODE_INSTRUCTION": message,
                    "A3S_CODE_SESSION_ID": self.session_id,
                    "A3S_CODE_SESSION_STORE_DIR": "/root/.a3s/sessions",
                    "A3S_CODE_TRACE_PATH": self._trace_remote_path,
                    "A3S_CODE_BUILTIN_SKILLS": os.getenv("A3S_CODE_BUILTIN_SKILLS", "true"),
                    "A3S_CODE_PLANNING": os.getenv("A3S_CODE_PLANNING", "true"),
                    "A3S_CODE_PERMISSIVE": os.getenv("A3S_CODE_PERMISSIVE", "true"),
                    "A3S_CODE_THINKING_BUDGET": os.getenv("A3S_CODE_THINKING_BUDGET", "32000"),
                    "A3S_CODE_MAX_TOOL_ROUNDS": os.getenv("A3S_CODE_MAX_TOOL_ROUNDS", "64"),
                    "A3S_CODE_TOOL_TIMEOUT_MS": os.getenv("A3S_CODE_TOOL_TIMEOUT_MS", "300000"),
                    "A3S_CODE_SKILL_DIRS_JSON": json.dumps(["/root/.openclaw/skills"]),
                }
                cmd = ". /opt/a3s-code-venv/bin/activate && python /root/.a3s/clawmark_a3s_code_runner.py"
                LOGGER.info("Sending to a3s-code session=%s (%d chars)", self.session_id, len(message))
                return await self.sandbox.exec(cmd, timeout_sec=timeout_sec + 60, env=env)

        return A3SCodeOrchestrator

    async def run_task(
        self,
        *,
        task_dir: Path,
        compose_file: Path,
        results_dir: Path,
        dry_run: bool,
    ) -> dict[str, object]:
        task = self._load_task_py(task_dir)

        if dry_run:
            sandbox = self._DryRunSandbox(workspace_dir=results_dir / task.id / "dryrun_workspace")
        else:
            session_id = f"clawmark-{task.id}-{uuid.uuid4().hex[:8]}"
            sandbox = self._DockerSandbox(session_id=session_id, compose_file=compose_file)

        state_manager = self._CompositeStateManager(
            environments=task.environments,
            env_config=task.env_config,
        )
        orchestrator_cls = self._make_orchestrator_class()
        orchestrator = orchestrator_cls(
            sandbox=sandbox,
            state_manager=state_manager,
            openclaw_config_path=None,
        )

        local_workspace = results_dir / task.id / "workspace"
        stage_results: list[object] = []
        started_at = time.time()

        try:
            await sandbox.start()
            await state_manager.setup(sandbox=sandbox)
            ctx = state_manager.create_context(task_dir=task.task_dir, sandbox=sandbox)

            stage_results = await orchestrator.run(
                task=task,
                ctx=ctx,
                model=self._model_name,
                api_key=self._model_api_key,
                api_base=self._model_base_url,
                api_format="openrouter",
            )

            await sandbox.download_dir("/workspace", local_workspace)
            trace_local = results_dir / task.id / "messages.jsonl"
            try:
                await sandbox.download_file(orchestrator.trace_remote_path, trace_local)
            except Exception as exc:
                LOGGER.warning("Could not download a3s-code trace for %s: %s", task.id, exc)
        except Exception as exc:
            LOGGER.error("ClawMark task %s failed: %s", task.id, exc, exc_info=True)
            stage_results.append(self._StageResult(stage_id="FRAMEWORK_ERROR", success=False, error=str(exc)))
        finally:
            try:
                await state_manager.cleanup()
            except Exception as exc:
                LOGGER.warning("Cleanup error for %s: %s", task.id, exc)
            await sandbox.stop(delete=True)

        elapsed = time.time() - started_at
        all_items = [item for stage in stage_results for item in stage.verification]
        total_weight = sum(item.weight for item in all_items)
        passed_weight = sum(item.weight for item in all_items if item.passed)
        score = passed_weight / total_weight if total_weight > 0 else 0.0

        result_json = results_dir / task.id / "result.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "task_id": task.id,
            "score": score,
            "execution_time": elapsed,
            "stages": [
                {
                    "id": stage.stage_id,
                    "success": stage.success,
                    "error": stage.error,
                    "verification_score": stage.verification_score,
                    "verification": [
                        {
                            "id": item.item_id,
                            "passed": item.passed,
                            "weight": item.weight,
                            "detail": item.detail,
                            "method": item.method.value,
                        }
                        for item in stage.verification
                    ],
                }
                for stage in stage_results
            ],
            "rubric": [
                {
                    "id": item.item_id,
                    "passed": item.passed,
                    "weight": item.weight,
                    "detail": item.detail,
                }
                for item in all_items
            ],
        }
        result_json.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        trace_path = results_dir / task.id / "messages.jsonl"
        input_tokens = 0
        output_tokens = 0
        turns = 0
        if trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("kind") != "assistant_result":
                    continue
                turns += 1
                input_tokens += int(record.get("prompt_tokens") or 0)
                output_tokens += int(record.get("completion_tokens") or 0)

        return {
            "task_id": task.id,
            "task_path": str(task_dir),
            "score": score,
            "success": all(stage.success for stage in stage_results if stage.stage_id != "final"),
            "input_tokens": input_tokens or None,
            "output_tokens": output_tokens or None,
            "total_tokens": (input_tokens + output_tokens) or None,
            "turns": turns,
            "execution_time_sec": elapsed,
            "result_payload": result_payload,
        }


def main() -> int:
    args = parse_args()
    load_dotenv(args.clawmark_root / ".env")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    runtime = ClawMarkA3SRuntime(
        clawmark_root=args.clawmark_root,
        wheel_path=args.wheel_path,
        model_name=args.model_name,
        model_base_url=args.model_base_url,
        model_api_key=args.model_api_key,
        session_id_header=args.session_id_header,
    )
    summary = asyncio.run(
        runtime.run_task(
            task_dir=args.task_dir,
            compose_file=args.compose_file,
            results_dir=args.results_dir,
            dry_run=args.dry_run,
        )
    )
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
