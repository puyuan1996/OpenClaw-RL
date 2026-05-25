#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:  # pragma: no cover - script entrypoint path bootstrap
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from a3s_code_benchmarks.benchmark_runtime_utils import ensure_a3s_code_wheel


TOKEN = "A3S_SMOKE_OK"
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    index: int
    scenario: str
    base_url: str
    model: str
    api_key: str
    api_key_kind: str
    provider: str
    interface: str
    notes: str

    @property
    def case_id(self) -> str:
        return f"{self.index:02d}-{slugify(self.scenario)}"

    def redacted(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload.pop("api_key", None)
        payload["case_id"] = self.case_id
        return payload


def slugify(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
    safe = re.sub(r"-+", "-", safe).strip("-")
    return safe[:80] or "model"


def parse_markdown_table(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = [line.strip() for line in lines if line.strip().startswith("|")]
    if len(table_lines) < 3:
        raise RuntimeError(f"No markdown table found in {path}")

    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def provider_from_interface(interface: str) -> str | None:
    lower = interface.lower()
    if "anthropic-compatible" in lower:
        return "anthropic"
    if "openai-compatible" in lower:
        return "openai"
    return None


def api_key_from_cell(cell: str) -> tuple[str, str]:
    compact = cell.strip()
    lowered = compact.lower()
    if "待确认" in compact:
        return "", "missing"
    if "no-auth" in lowered or compact in {"无", "无（no-auth）", "none", ""}:
        return "", "no-auth"
    return compact, "documented"


def load_specs(
    path: Path,
    *,
    include_missing_key: bool,
    include_anthropic: bool,
    scenario_regex: str | None,
    limit: int,
) -> list[ModelSpec]:
    pattern = re.compile(scenario_regex) if scenario_regex else None
    specs: list[ModelSpec] = []
    for row_index, row in enumerate(parse_markdown_table(path), start=1):
        scenario = row.get("场景", "")
        interface = row.get("接口方式", "")
        provider = provider_from_interface(interface)
        if provider is None:
            continue
        if provider == "anthropic" and not include_anthropic:
            continue
        if pattern and not pattern.search(scenario):
            continue
        api_key, api_key_kind = api_key_from_cell(row.get("密钥", ""))
        if api_key_kind == "missing" and not include_missing_key:
            continue
        specs.append(
            ModelSpec(
                index=row_index,
                scenario=scenario,
                base_url=row.get("调用地址", "").rstrip("/"),
                model=row.get("模型", ""),
                api_key=api_key,
                api_key_kind=api_key_kind,
                provider=provider,
                interface=interface,
                notes=row.get("备注", ""),
            )
        )
        if limit > 0 and len(specs) >= limit:
            break
    return specs


def hcl_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_agent_config(spec: ModelSpec, *, context_tokens: int, output_tokens: int) -> str:
    return (
        f"default_model = {hcl_string(f'{spec.provider}/{spec.model}')}\n\n"
        f"providers {hcl_string(spec.provider)} {{\n"
        f"  apiKey = {hcl_string(spec.api_key)}\n"
        f"  baseUrl = {hcl_string(spec.base_url)}\n"
        f"  models {hcl_string(spec.model)} {{\n"
        f"    name = {hcl_string(spec.model)}\n"
        "    toolCall = true\n"
        "    limit = {\n"
        f"      context = {context_tokens}\n"
        f"      output = {output_tokens}\n"
        "    }\n"
        "  }\n"
        "}\n"
        "\n"
        "storage_backend = \"memory\"\n"
    )


def prompt_for_task(task_mode: str) -> str:
    if task_mode == "respond":
        return (
            f"Reply with exactly this token and no extra text: {TOKEN}. "
            "Do not call tools."
        )
    return (
        f"Create a file named answer.txt in the current workspace containing exactly {TOKEN} "
        f"and then reply with exactly {TOKEN}. Keep the change minimal."
    )


def clear_proxy_env(env: dict[str, str]) -> None:
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"


def import_a3s_code(wheel_path: Path):
    try:
        from a3s_code import Agent, PermissionPolicy, SessionOptions  # type: ignore

        return Agent, SessionOptions, PermissionPolicy
    except ModuleNotFoundError:
        pass

    if wheel_path.suffix == ".whl":
        extracted = Path(tempfile.mkdtemp(prefix="a3s-code-wheel-"))
        with zipfile.ZipFile(wheel_path) as archive:
            archive.extractall(extracted)
        sys.path.insert(0, str(extracted))
    else:
        sys.path.insert(0, str(wheel_path))

        from a3s_code import Agent, PermissionPolicy, SessionOptions  # type: ignore

        return Agent, SessionOptions, PermissionPolicy


def run_one(
    spec: ModelSpec,
    *,
    output_dir: Path,
    wheel_path: Path,
    task_mode: str,
    context_tokens: int,
    output_tokens: int,
    max_tool_rounds: int,
    thinking_budget: int,
    keep_proxy: bool,
) -> dict[str, Any]:
    if not keep_proxy:
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

    Agent, SessionOptions, PermissionPolicy = import_a3s_code(wheel_path)

    case_dir = output_dir / "cases" / spec.case_id
    workspace = case_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "README.md").write_text(
        "Temporary a3s-code model matrix smoke workspace.\n", encoding="utf-8"
    )
    config_path = case_dir / "agent.acl"
    config_path.write_text(
        render_agent_config(spec, context_tokens=context_tokens, output_tokens=output_tokens),
        encoding="utf-8",
    )

    started = time.time()
    payload: dict[str, Any] = {
        "status": "error",
        "case": spec.redacted(),
        "task_mode": task_mode,
        "workspace": str(workspace),
        "config_path": str(config_path),
        "started_at": started,
    }
    try:
        agent = Agent.create(str(config_path))
        opts = SessionOptions()
        opts.builtin_skills = True
        opts.planning = False
        opts.thinking_budget = thinking_budget
        opts.max_tool_rounds = max_tool_rounds
        opts.tool_timeout_ms = 60_000
        opts.permission_policy = PermissionPolicy(default_decision="allow")
        session = agent.session(str(workspace), opts)
        result = session.send(prompt_for_task(task_mode))
        elapsed = time.time() - started
        answer_path = workspace / "answer.txt"
        answer_text = answer_path.read_text(encoding="utf-8", errors="replace") if answer_path.exists() else ""
        response_text = str(getattr(result, "text", "") or "")
        file_pass = TOKEN in answer_text
        text_pass = TOKEN in response_text
        passed = text_pass if task_mode == "respond" else file_pass
        payload.update(
            {
                "status": "passed" if passed else "failed",
                "passed": passed,
                "file_pass": file_pass,
                "text_pass": text_pass,
                "elapsed_sec": elapsed,
                "tool_calls_count": getattr(result, "tool_calls_count", None),
                "prompt_tokens": getattr(result, "prompt_tokens", None),
                "completion_tokens": getattr(result, "completion_tokens", None),
                "total_tokens": getattr(result, "total_tokens", None),
                "response_excerpt": response_text[:1000],
                "answer_txt_excerpt": answer_text[:200],
                "result_class": type(result).__name__,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "error",
                "passed": False,
                "elapsed_sec": time.time() - started,
                "exception_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "traceback_tail": traceback.format_exc()[-4000:],
            }
        )
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def run_child(args: argparse.Namespace) -> int:
    specs = load_specs(
        args.model_api_md,
        include_missing_key=args.include_missing_key,
        include_anthropic=args.include_anthropic,
        scenario_regex=args.scenario_regex,
        limit=args.limit,
    )
    if args.single_index < 0 or args.single_index >= len(specs):
        raise IndexError(f"single index {args.single_index} outside selected specs ({len(specs)})")
    wheel_path = Path(args.wheel_path) if args.wheel_path else ensure_a3s_code_wheel()
    result = run_one(
        specs[args.single_index],
        output_dir=args.output_dir,
        wheel_path=wheel_path,
        task_mode=args.task_mode,
        context_tokens=args.context_tokens,
        output_tokens=args.output_tokens,
        max_tool_rounds=args.max_tool_rounds,
        thinking_budget=args.thinking_budget,
        keep_proxy=args.keep_proxy,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


def run_subprocess_for_spec(index: int, args: argparse.Namespace, wheel_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-api-md",
        str(args.model_api_md),
        "--output-dir",
        str(args.output_dir),
        "--single-index",
        str(index),
        "--wheel-path",
        str(wheel_path),
        "--task-mode",
        args.task_mode,
        "--context-tokens",
        str(args.context_tokens),
        "--output-tokens",
        str(args.output_tokens),
        "--max-tool-rounds",
        str(args.max_tool_rounds),
        "--thinking-budget",
        str(args.thinking_budget),
        "--limit",
        str(args.limit),
    ]
    if args.include_missing_key:
        cmd.append("--include-missing-key")
    if args.include_anthropic:
        cmd.append("--include-anthropic")
    if args.scenario_regex:
        cmd.extend(["--scenario-regex", args.scenario_regex])
    if args.keep_proxy:
        cmd.append("--keep-proxy")

    env = os.environ.copy()
    env["A3S_CODE_WHEEL_PATH"] = str(wheel_path)
    py_path = env.get("PYTHONPATH", "")
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = f"{package_root}:{py_path}" if py_path else package_root
    if not args.keep_proxy:
        clear_proxy_env(env)

    started = time.time()
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
        return {
            "status": "timeout",
            "passed": False,
            "single_index": index,
            "elapsed_sec": time.time() - started,
            "timeout_sec": args.per_model_timeout_sec,
            "stdout_tail": (exc.stdout or "")[-2000:],
            "stderr_tail": (exc.stderr or "")[-2000:],
        }

    stdout = completed.stdout.strip()
    result: dict[str, Any]
    try:
        result = json.loads(stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        result = {
            "status": "harness_error",
            "passed": False,
            "single_index": index,
            "elapsed_sec": time.time() - started,
            "returncode": completed.returncode,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": (completed.stderr or "")[-4000:],
        }
    result["returncode"] = completed.returncode
    result["stderr_tail"] = (completed.stderr or "")[-2000:]
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.get("status", "unknown")] = counts.get(result.get("status", "unknown"), 0) + 1
    return {
        "total": len(results),
        "counts": counts,
        "passed": sum(1 for result in results if result.get("passed") is True),
        "failed_or_error": sum(1 for result in results if result.get("passed") is not True),
        "results": results,
    }


def run_parent(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = load_specs(
        args.model_api_md,
        include_missing_key=args.include_missing_key,
        include_anthropic=args.include_anthropic,
        scenario_regex=args.scenario_regex,
        limit=args.limit,
    )
    spec_manifest = {
        "model_api_md": str(args.model_api_md),
        "selected_count": len(specs),
        "task_mode": args.task_mode,
        "specs": [spec.redacted() for spec in specs],
    }
    (args.output_dir / "model_specs.json").write_text(
        json.dumps(spec_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(json.dumps(spec_manifest, ensure_ascii=False, indent=2))
        return 0
    if not specs:
        raise RuntimeError("No model specs selected")

    wheel_path = Path(args.wheel_path) if args.wheel_path else ensure_a3s_code_wheel()
    results: list[dict[str, Any]] = []
    started = time.time()
    max_workers = min(max(1, args.max_workers), len(specs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(run_subprocess_for_spec, idx, args, wheel_path): idx
            for idx in range(len(specs))
        }
        for future in concurrent.futures.as_completed(future_to_index):
            result = future.result()
            results.append(result)
            case = result.get("case") or {}
            print(
                json.dumps(
                    {
                        "status": result.get("status"),
                        "passed": result.get("passed"),
                        "scenario": case.get("scenario"),
                        "model": case.get("model"),
                        "elapsed_sec": result.get("elapsed_sec"),
                        "tool_calls_count": result.get("tool_calls_count"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    results.sort(key=lambda item: ((item.get("case") or {}).get("index", item.get("single_index", 9999))))
    summary = summarize(results)
    summary.update(
        {
            "started_at": started,
            "duration_sec": time.time() - started,
            "wheel_path": str(wheel_path),
            "max_workers": max_workers,
            "per_model_timeout_sec": args.per_model_timeout_sec,
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "results.jsonl").write_text(
        "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results),
        encoding="utf-8",
    )
    print(json.dumps({"summary_path": str(args.output_dir / "summary.json"), **summary["counts"]}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel a3s-code smoke matrix for models listed in model_api.md.")
    parser.add_argument("--model-api-md", type=Path, default=Path(__file__).resolve().parents[1] / "model_api.md")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".artifacts" / "model_api_matrix_smoke" / time.strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("A3S_CODE_MODEL_MATRIX_WORKERS", "8")))
    parser.add_argument("--per-model-timeout-sec", type=int, default=int(os.getenv("A3S_CODE_MODEL_MATRIX_TIMEOUT_SEC", "180")))
    parser.add_argument("--task-mode", choices=("file", "respond"), default=os.getenv("A3S_CODE_MODEL_MATRIX_TASK_MODE", "file"))
    parser.add_argument("--context-tokens", type=int, default=int(os.getenv("A3S_CODE_MODEL_CONTEXT_TOKENS", "16384")))
    parser.add_argument("--output-tokens", type=int, default=int(os.getenv("A3S_CODE_MODEL_OUTPUT_TOKENS", "2048")))
    parser.add_argument("--max-tool-rounds", type=int, default=int(os.getenv("A3S_CODE_MODEL_MATRIX_MAX_TOOL_ROUNDS", "4")))
    parser.add_argument("--thinking-budget", type=int, default=int(os.getenv("A3S_CODE_MODEL_MATRIX_THINKING_BUDGET", "1024")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scenario-regex", type=str, default=None)
    parser.add_argument("--include-missing-key", action="store_true")
    parser.add_argument("--include-anthropic", action="store_true")
    parser.add_argument("--keep-proxy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--single-index", type=int, default=None)
    parser.add_argument("--wheel-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_api_md = args.model_api_md.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.single_index is not None:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
