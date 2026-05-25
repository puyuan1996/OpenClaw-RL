#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:  # pragma: no cover - script entrypoint path bootstrap
    PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from a3s_code_benchmarks.benchmark_runtime_utils import ensure_skillsbench_a3s_code_wheel
from a3s_code_benchmarks.model_api_matrix_smoke import clear_proxy_env, load_specs
from a3s_code_benchmarks.official.worker_local_docker import start_worker_local_docker


DEFAULT_TASKS = "fix-build-agentops"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _trial_prefix(output_dir: Path, case_id: str) -> str:
    digest = hashlib.sha1(str(output_dir).encode("utf-8")).hexdigest()[:8]
    return _sanitize_name(f"{digest}-{case_id}")[:80]


def _skillsbench_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics") or {}
    suite = (summary.get("suites") or {}).get("skillsbench") or {}
    records = suite.get("records") or []
    exceptions: list[str] = []
    for record in records:
        metadata = record.get("metadata") or {}
        result = metadata.get("result") or {}
        exception = result.get("exception_info") or {}
        message = exception.get("exception_message")
        if message:
            exceptions.append(str(message))
    successful_runs = int(metrics.get("skillsbench_official_successful_runs") or 0)
    completed_runs = int(metrics.get("skillsbench_official_completed_runs") or 0)
    return {
        "score": metrics.get("skillsbench_official_score"),
        "completed_runs": completed_runs,
        "successful_runs": successful_runs,
        "positive_score_runs": int(metrics.get("skillsbench_official_positive_score_runs") or 0),
        "exception_free_runs": int(metrics.get("skillsbench_official_exception_free_runs") or 0),
        "agent_execution_time_sec_mean": metrics.get("skillsbench_official_agent_execution_time_sec_mean"),
        "agent_timeout_extension_sec_mean": metrics.get("skillsbench_official_agent_timeout_extension_sec_mean"),
        "agent_runtime_over_official_sec_mean": metrics.get("skillsbench_official_agent_runtime_over_official_sec_mean"),
        "agent_runtime_over_official_runs": int(metrics.get("skillsbench_official_agent_runtime_over_official_runs") or 0),
        "exception_count": len(exceptions),
        "exception_tail": exceptions[-1][-1000:] if exceptions else None,
        "trial_ran": completed_runs > 0,
        "mode": suite.get("skillsbench_mode"),
    }


def selected_tasks_dir(output_dir: Path, tasks_root: Path, task_names: list[str]) -> Path | None:
    if not task_names:
        return None
    root = output_dir / "selected_tasks"
    root.mkdir(parents=True, exist_ok=True)
    for name in task_names:
        source = tasks_root / name
        if not source.exists():
            raise FileNotFoundError(f"SkillsBench task not found: {source}")
        target = root / name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source, target_is_directory=True)
    return root


def run_one_model(index: int, spec, args: argparse.Namespace, wheel_path: Path, tasks_dir: Path | None) -> dict[str, Any]:
    case_dir = args.output_dir / "cases" / spec.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    command_log_dir = case_dir / "command_logs"
    command_log_dir.mkdir(parents=True, exist_ok=True)
    step_dir = case_dir / "step_0000000"
    summary_path = step_dir / "summary.json"

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("official_benchmark_eval.py")),
        "--suites",
        "skillsbench",
        "--skillsbench-root",
        str(args.skillsbench_root),
        "--output-dir",
        str(case_dir),
        "--step",
        "0",
        "--skillsbench-max-tasks",
        str(args.skillsbench_max_tasks),
        "--skillsbench-repeats",
        str(args.skillsbench_repeats),
        "--skillsbench-concurrency",
        str(args.skillsbench_concurrency),
        "--skillsbench-mode",
        args.skillsbench_mode,
        "--skillsbench-timeout-sec",
        str(args.skillsbench_timeout_sec),
        "--skillsbench-agent-timeout-sec",
        str(args.skillsbench_agent_timeout_sec),
        "--skillsbench-agent-timeout-multiplier",
        str(args.skillsbench_agent_timeout_multiplier),
        "--no-skillsbench-resume-existing",
        "--model-provider",
        spec.provider,
        "--model-name",
        spec.model,
        "--model-base-url",
        spec.base_url,
        "--model-api-key",
        spec.api_key,
        "--log-level",
        args.log_level,
    ]
    if args.skillsbench_force_allow_internet:
        cmd.append("--skillsbench-force-allow-internet")
    else:
        cmd.append("--no-skillsbench-force-allow-internet")
    if args.skillsbench_keep_images:
        cmd.append("--skillsbench-keep-images")
    else:
        cmd.append("--no-skillsbench-keep-images")
    if tasks_dir is not None:
        cmd.extend(["--skillsbench-tasks-dir", str(tasks_dir)])

    env = os.environ.copy()
    env["A3S_CODE_WHEEL_PATH"] = str(wheel_path)
    env["A3S_CODE_MODEL_PROVIDER"] = spec.provider
    env["A3S_CODE_MODEL_NAME"] = spec.model
    env["A3S_CODE_MODEL_BASE_URL"] = spec.base_url
    env["A3S_CODE_MODEL_API_KEY"] = spec.api_key
    env["A3S_CODE_CONTEXT_TOKENS"] = str(args.context_tokens)
    env["A3S_CODE_OUTPUT_TOKENS"] = str(args.output_tokens)
    env["A3S_CODE_THINKING_BUDGET"] = str(args.thinking_budget)
    env["A3S_CODE_MAX_TOOL_ROUNDS"] = str(args.max_tool_rounds)
    env["A3S_CODE_SKILLSBENCH_TRIAL_PREFIX"] = _trial_prefix(args.output_dir, spec.case_id)
    env["A3S_CODE_AUTO_SAVE_SESSION"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    py_path = env.get("PYTHONPATH", "")
    package_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = f"{package_root}:{py_path}" if py_path else package_root
    if not args.keep_proxy:
        clear_proxy_env(env)

    started = time.time()
    payload: dict[str, Any] = {
        "case": spec.redacted(),
        "command": cmd,
        "summary_path": str(summary_path),
        "started_at": started,
    }
    try:
        completed = subprocess.run(
            cmd,
            env=env,
            text=True,
            capture_output=True,
            timeout=args.per_model_timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        (command_log_dir / "stdout.txt").write_text(exc.stdout or "", encoding="utf-8")
        (command_log_dir / "stderr.txt").write_text(exc.stderr or "", encoding="utf-8")
        payload.update(
            {
                "status": "timeout",
                "passed": False,
                "elapsed_sec": time.time() - started,
                "timeout_sec": args.per_model_timeout_sec,
                "stdout_tail": (exc.stdout or "")[-4000:],
                "stderr_tail": (exc.stderr or "")[-4000:],
            }
        )
        return payload

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    (command_log_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (command_log_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    payload.update(
        {
            "returncode": completed.returncode,
            "elapsed_sec": time.time() - started,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    )
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            payload.update({"status": "summary_decode_error", "passed": False, "error": str(exc)})
        else:
            metrics = summary.get("metrics") or {}
            skillsbench_metrics = _skillsbench_metrics(summary)
            status = "completed" if completed.returncode == 0 else "completed_with_error_code"
            if completed.returncode == 0 and not skillsbench_metrics["trial_ran"]:
                status = "skillsbench_no_completed_trials"
            payload.update(
                {
                    "status": status,
                    "passed": bool(skillsbench_metrics["trial_ran"]),
                    "metrics": metrics,
                    "skillsbench": skillsbench_metrics,
                    "summary": summary,
                }
            )
    else:
        payload.update(
            {
                "status": "missing_summary",
                "passed": False,
                "error": f"summary not found: {summary_path}",
            }
        )
    return payload


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    rows = []
    for result in results:
        case = result.get("case") or {}
        metrics = result.get("metrics") or {}
        skillsbench = result.get("skillsbench") or {}
        rows.append(
            {
                "scenario": case.get("scenario"),
                "provider": case.get("provider"),
                "model": case.get("model"),
                "base_url": case.get("base_url"),
                "status": result.get("status"),
                "trial_ran": skillsbench.get("trial_ran"),
                "returncode": result.get("returncode"),
                "skillsbench_mode": skillsbench.get("mode"),
                "skillsbench_force_allow_internet": (
                    ((result.get("summary") or {}).get("suites") or {}).get("skillsbench") or {}
                ).get("force_allow_internet"),
                "skillsbench_score": metrics.get("skillsbench_official_score"),
                "skillsbench_completed_runs": metrics.get("skillsbench_official_completed_runs"),
                "skillsbench_successful_runs": metrics.get("skillsbench_official_successful_runs"),
                "skillsbench_positive_score_runs": metrics.get("skillsbench_official_positive_score_runs"),
                "skillsbench_exception_free_runs": metrics.get("skillsbench_official_exception_free_runs"),
                "skillsbench_exception_count": skillsbench.get("exception_count"),
                "skillsbench_exception_tail": skillsbench.get("exception_tail"),
                "skillsbench_agent_execution_time_sec_mean": skillsbench.get("agent_execution_time_sec_mean"),
                "skillsbench_agent_timeout_extension_sec_mean": skillsbench.get("agent_timeout_extension_sec_mean"),
                "skillsbench_agent_runtime_over_official_sec_mean": skillsbench.get("agent_runtime_over_official_sec_mean"),
                "skillsbench_agent_runtime_over_official_runs": skillsbench.get("agent_runtime_over_official_runs"),
                "elapsed_sec": result.get("elapsed_sec"),
                "summary_path": result.get("summary_path"),
            }
        )
    return {"total": len(results), "counts": counts, "rows": rows, "results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SkillsBench through a3s-code for models listed in model_api.md.")
    parser.add_argument("--model-api-md", type=Path, default=Path(__file__).resolve().parents[2] / "model_api.md")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".artifacts" / "skillsbench_model_api_matrix" / time.strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--skillsbench-root", type=Path, default=Path(os.getenv("A3S_CODE_SKILLSBENCH_ROOT", Path.home() / "workspace" / "skillsbench")))
    parser.add_argument(
        "--skillsbench-tasks-dir",
        type=Path,
        default=Path(os.environ["A3S_CODE_SKILLSBENCH_MATRIX_TASKS_DIR"]) if os.getenv("A3S_CODE_SKILLSBENCH_MATRIX_TASKS_DIR") else None,
        help="Optional explicit SkillsBench tasks directory, for example a prebuilt-image shadow task tree.",
    )
    parser.add_argument("--task-names", type=str, default=os.getenv("A3S_CODE_SKILLSBENCH_MATRIX_TASKS", DEFAULT_TASKS))
    parser.add_argument("--skillsbench-max-tasks", type=int, default=int(os.getenv("A3S_CODE_SKILLSBENCH_MATRIX_MAX_TASKS", "0")))
    parser.add_argument("--skillsbench-repeats", type=int, default=1)
    parser.add_argument("--skillsbench-concurrency", type=int, default=int(os.getenv("A3S_CODE_SKILLSBENCH_MATRIX_TASK_CONCURRENCY", "1")))
    parser.add_argument(
        "--skillsbench-mode",
        choices=["without-skills", "with-skills"],
        default=os.getenv("A3S_CODE_SKILLSBENCH_MODE", "without-skills"),
        help="SkillsBench condition. without-skills disables task skill dirs; with-skills injects task environment/skills.",
    )
    parser.add_argument("--max-model-workers", type=int, default=int(os.getenv("A3S_CODE_SKILLSBENCH_MATRIX_MODEL_WORKERS", "4")))
    parser.add_argument("--skillsbench-timeout-sec", type=int, default=int(os.getenv("A3S_CODE_SKILLSBENCH_MATRIX_TRIAL_TIMEOUT_SEC", "1800")))
    parser.add_argument(
        "--skillsbench-agent-timeout-sec",
        type=int,
        default=int(os.getenv("A3S_CODE_SKILLSBENCH_MATRIX_AGENT_TIMEOUT_SEC", os.getenv("A3S_CODE_SKILLSBENCH_AGENT_TIMEOUT_SEC", "0"))),
        help="Absolute floor for agent execution timeout. 0 preserves each task's official timeout before applying the multiplier.",
    )
    parser.add_argument(
        "--skillsbench-agent-timeout-multiplier",
        type=float,
        default=float(
            os.getenv(
                "A3S_CODE_SKILLSBENCH_MATRIX_AGENT_TIMEOUT_MULTIPLIER",
                os.getenv("A3S_CODE_SKILLSBENCH_AGENT_TIMEOUT_MULTIPLIER", "1.0"),
            )
        ),
        help="Multiply each task's official [agent].timeout_sec and record the extension/overrun in summary metadata.",
    )
    parser.add_argument(
        "--skillsbench-keep-images",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("A3S_CODE_SKILLSBENCH_MATRIX_KEEP_IMAGES", True),
        help="Keep task images between trials so one model matrix run builds each task image once per Docker daemon.",
    )
    parser.add_argument(
        "--skillsbench-force-allow-internet",
        action=argparse.BooleanOptionalAction,
        default=_env_flag(
            "A3S_CODE_SKILLSBENCH_MATRIX_FORCE_ALLOW_INTERNET",
            _env_flag("A3S_CODE_SKILLSBENCH_FORCE_ALLOW_INTERNET", False),
        ),
        help="Patch a per-trial task copy to set environment.allow_internet=true; not official-leaderboard comparable.",
    )
    parser.add_argument(
        "--worker-local-docker",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("A3S_CODE_WORKER_LOCAL_DOCKER", False),
        help="Start dockerd inside the current privileged worker and override any injected remote DOCKER_HOST.",
    )
    parser.add_argument("--per-model-timeout-sec", type=int, default=int(os.getenv("A3S_CODE_SKILLSBENCH_MATRIX_MODEL_TIMEOUT_SEC", "2400")))
    parser.add_argument("--context-tokens", type=int, default=int(os.getenv("A3S_CODE_MODEL_CONTEXT_TOKENS", "32768")))
    parser.add_argument("--output-tokens", type=int, default=int(os.getenv("A3S_CODE_MODEL_OUTPUT_TOKENS", "4096")))
    parser.add_argument("--thinking-budget", type=int, default=int(os.getenv("A3S_CODE_MODEL_MATRIX_THINKING_BUDGET", "2048")))
    parser.add_argument("--max-tool-rounds", type=int, default=int(os.getenv("A3S_CODE_MODEL_MATRIX_MAX_TOOL_ROUNDS", "64")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scenario-regex", type=str, default=None)
    parser.add_argument("--include-missing-key", action="store_true")
    parser.add_argument("--include-anthropic", action="store_true")
    parser.add_argument("--keep-proxy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wheel-path", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_api_md = args.model_api_md.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.skillsbench_root = args.skillsbench_root.expanduser().resolve()
    args.skillsbench_tasks_dir = args.skillsbench_tasks_dir.expanduser().resolve() if args.skillsbench_tasks_dir else None
    if args.skillsbench_agent_timeout_multiplier <= 0:
        raise ValueError("--skillsbench-agent-timeout-multiplier must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_names = [item.strip() for item in args.task_names.split(",") if item.strip()]
    if task_names:
        tasks_source_root = args.skillsbench_tasks_dir or (args.skillsbench_root / "tasks")
        tasks_dir = selected_tasks_dir(args.output_dir, tasks_source_root, task_names)
    else:
        tasks_dir = args.skillsbench_tasks_dir
    specs = load_specs(
        args.model_api_md,
        include_missing_key=args.include_missing_key,
        include_anthropic=args.include_anthropic,
        scenario_regex=args.scenario_regex,
        limit=args.limit,
    )
    manifest = {
        "model_api_md": str(args.model_api_md),
        "skillsbench_root": str(args.skillsbench_root),
        "tasks_dir": str(tasks_dir) if tasks_dir else None,
        "explicit_tasks_dir": str(args.skillsbench_tasks_dir) if args.skillsbench_tasks_dir else None,
        "task_names": task_names,
        "selected_count": len(specs),
        "skillsbench_keep_images": args.skillsbench_keep_images,
        "skillsbench_mode": args.skillsbench_mode,
        "skillsbench_agent_timeout_sec": args.skillsbench_agent_timeout_sec,
        "skillsbench_agent_timeout_multiplier": args.skillsbench_agent_timeout_multiplier,
        "skillsbench_force_allow_internet": args.skillsbench_force_allow_internet,
        "worker_local_docker": args.worker_local_docker,
        "specs": [spec.redacted() for spec in specs],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if not specs:
        raise RuntimeError("No model specs selected")

    worker_docker = None
    if args.worker_local_docker:
        worker_docker = start_worker_local_docker(log_dir=args.output_dir / "worker_local_docker")

    wheel_path = args.wheel_path.expanduser().resolve() if args.wheel_path else ensure_skillsbench_a3s_code_wheel()
    started = time.time()
    results: list[dict[str, Any]] = []
    max_workers = min(max(1, args.max_model_workers), len(specs))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_one_model, index, spec, args, wheel_path, tasks_dir)
                for index, spec in enumerate(specs)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                case = result.get("case") or {}
                metrics = result.get("metrics") or {}
                print(
                    json.dumps(
                        {
                            "status": result.get("status"),
                            "scenario": case.get("scenario"),
                            "model": case.get("model"),
                            "score": metrics.get("skillsbench_official_score"),
                            "successful_runs": metrics.get("skillsbench_official_successful_runs"),
                            "mode": args.skillsbench_mode,
                            "elapsed_sec": result.get("elapsed_sec"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        if worker_docker is not None:
            worker_docker.stop()

    results.sort(key=lambda item: ((item.get("case") or {}).get("index", 9999)))
    summary = summarize(results)
    summary.update(
        {
            "duration_sec": time.time() - started,
            "wheel_path": str(wheel_path),
            "max_model_workers": max_workers,
            "tasks_dir": str(tasks_dir) if tasks_dir else None,
            "skillsbench_mode": args.skillsbench_mode,
            "skillsbench_agent_timeout_sec": args.skillsbench_agent_timeout_sec,
            "skillsbench_agent_timeout_multiplier": args.skillsbench_agent_timeout_multiplier,
            "skillsbench_force_allow_internet": args.skillsbench_force_allow_internet,
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "results.jsonl").write_text(
        "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results),
        encoding="utf-8",
    )
    print(json.dumps({"summary_path": str(args.output_dir / "summary.json"), "counts": summary["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
