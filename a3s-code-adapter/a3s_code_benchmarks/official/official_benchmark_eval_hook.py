from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from slime.utils import logging_utils


LOGGER = logging.getLogger("official_benchmark_eval_hook")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class OfficialBenchmarkEvalHook:
    def __init__(self) -> None:
        self.enabled = _env_flag("A3S_CODE_ENABLE_OFFICIAL_BENCHMARK_EVAL", False)
        self.wait_at_exit = _env_flag("A3S_CODE_OFFICIAL_BENCHMARK_WAIT_AT_EXIT", True)
        self.script_path = Path(os.getenv("A3S_CODE_OFFICIAL_BENCHMARK_SCRIPT", Path(__file__).with_name("official_benchmark_eval.py")))
        self.output_dir = Path(
            os.getenv(
                "A3S_CODE_OFFICIAL_BENCHMARK_OUTPUT_DIR",
                Path(os.getenv("A3S_CODE_RUN_ROOT", ".")) / "official_benchmark_eval",
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._process: subprocess.Popen[str] | None = None
        self._step: int | None = None
        self._summary_path: Path | None = None
        self._stdout_handle = None
        self._stderr_handle = None

    def trigger(self, step: int, args=None) -> bool:
        if not self.enabled:
            return False
        self.poll_and_log(args)
        if self._process is not None and self._process.poll() is None:
            LOGGER.info("Official benchmark eval already running for step %s; skip step %s", self._step, step)
            return False

        step_dir = self.output_dir / f"step_{step:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        self._summary_path = step_dir / "summary.json"
        self._stdout_handle = (step_dir / "launcher_stdout.txt").open("w", encoding="utf-8")
        self._stderr_handle = (step_dir / "launcher_stderr.txt").open("w", encoding="utf-8")
        cmd = [
            sys.executable,
            str(self.script_path),
            "--step",
            str(step),
            "--output-dir",
            str(self.output_dir),
            "--summary-path",
            str(self._summary_path),
        ]
        LOGGER.info("Triggering official benchmark eval: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
            env=os.environ.copy(),
        )
        self._step = step
        return True

    def poll_and_log(self, args) -> bool:
        if self._process is None:
            return False
        return_code = self._process.poll()
        if return_code is None:
            return False

        if self._stdout_handle is not None:
            self._stdout_handle.close()
            self._stdout_handle = None
        if self._stderr_handle is not None:
            self._stderr_handle.close()
            self._stderr_handle = None

        summary_path = self._summary_path
        step = self._step if self._step is not None else 0
        self._process = None
        self._step = None
        self._summary_path = None

        if summary_path is None or not summary_path.exists():
            LOGGER.warning("Official benchmark eval step %s exited with code %s but no summary file was written", step, return_code)
            if args is not None:
                logging_utils.log(
                    args,
                    {
                        "eval/step": step,
                        "eval/official_benchmark_return_code": float(return_code),
                        "eval/official_benchmark_missing_summary": 1.0,
                    },
                    step_key="eval/step",
                )
            return True

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = {
            "eval/step": float(summary.get("step", step)),
            "eval/official_benchmark_return_code": float(return_code),
            "eval/official_benchmark_duration_sec": float(summary.get("duration_sec", 0.0) or 0.0),
        }
        for key, value in (summary.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                metrics[f"eval/{key}"] = float(value)
        LOGGER.info("Official benchmark eval finished for step %s: %s", step, summary_path)
        if args is not None:
            logging_utils.log(args, metrics, step_key="eval/step")
        return True

    def finalize(self, args) -> None:
        if self._process is None:
            return
        if self.wait_at_exit and self._process.poll() is None:
            LOGGER.info("Waiting for official benchmark eval process to finish before exit")
            self._process.wait()
        self.poll_and_log(args)


def maybe_create_official_benchmark_eval_hook() -> OfficialBenchmarkEvalHook | None:
    hook = OfficialBenchmarkEvalHook()
    return hook if hook.enabled else None
