from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass(frozen=True)
class EvalRecord:
    dataset_name: str
    task_id: str
    input_text: str
    label: str
    metadata: dict[str, Any]
    group_key: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _list_skill_names(root: Path) -> list[str]:
    skills_dir = root / "environment" / "skills"
    if not skills_dir.exists():
        return []
    return sorted(path.name for path in skills_dir.iterdir() if (path / "SKILL.md").exists())


def _balanced_take(records: list[EvalRecord], limit: int) -> list[EvalRecord]:
    if limit <= 0 or len(records) <= limit:
        return records

    grouped: dict[str, deque[EvalRecord]] = defaultdict(deque)
    for record in records:
        grouped[record.group_key].append(record)

    selected: list[EvalRecord] = []
    keys = deque(sorted(grouped))
    while keys and len(selected) < limit:
        key = keys.popleft()
        bucket = grouped[key]
        if bucket:
            selected.append(bucket.popleft())
        if bucket:
            keys.append(key)
    return selected


def _skillsbench_prompt(
    *,
    task_id: str,
    category: str,
    difficulty: str,
    tags: list[str],
    skill_names: list[str],
    instruction: str,
) -> str:
    return (
        "You are beginning a benchmark-style coding or automation task.\n"
        "Write only the first assistant response you would send before executing tools.\n"
        "Be concrete: summarize the plan, mention the first files/tools/systems you would inspect, "
        "call out the main risks or constraints, and do not claim the task is already done.\n\n"
        f"Benchmark source: SkillsBench\n"
        f"Task id: {task_id}\n"
        f"Category: {category}\n"
        f"Difficulty: {difficulty}\n"
        f"Tags: {json.dumps(tags, ensure_ascii=False)}\n"
        f"Available task-local skills: {json.dumps(skill_names, ensure_ascii=False)}\n\n"
        f"Task brief:\n{instruction}\n\n"
        "Respond in 4-8 sentences."
    )


def collect_skillsbench_records(root: Path, max_tasks: int) -> list[EvalRecord]:
    tasks_root = root / "tasks"
    if not tasks_root.exists():
        return []
    records: list[EvalRecord] = []
    for task_dir in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        instruction_path = task_dir / "instruction.md"
        task_toml_path = task_dir / "task.toml"
        if not instruction_path.exists() or not task_toml_path.exists():
            continue
        task_cfg = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
        metadata = task_cfg.get("metadata", {})
        task_id = task_dir.name
        category = str(metadata.get("category", "unknown"))
        difficulty = str(metadata.get("difficulty", "unknown"))
        tags = [str(tag) for tag in metadata.get("tags", [])]
        skill_names = _list_skill_names(task_dir)
        instruction = _read_text(instruction_path)
        record_metadata = {
            "benchmark_source": "skillsbench",
            "task_id": task_id,
            "category": category,
            "difficulty": difficulty,
            "tags": tags,
            "available_skills": skill_names,
            "task_brief": instruction,
        }
        records.append(
            EvalRecord(
                dataset_name="skillsbench_proxy",
                task_id=task_id,
                input_text=_skillsbench_prompt(
                    task_id=task_id,
                    category=category,
                    difficulty=difficulty,
                    tags=tags,
                    skill_names=skill_names,
                    instruction=instruction,
                ),
                label=instruction,
                metadata=record_metadata,
                group_key=category,
            )
        )
    return _balanced_take(records, max_tasks)


def _extract_module_literal(task_py: Path, name: str) -> Any:
    tree = ast.parse(task_py.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    return None


def _clawmark_prompt(
    *,
    task_id: str,
    category: str,
    environments: list[str],
    tags: list[str],
    summary: str,
    prompt: str,
) -> str:
    return (
        "You are beginning a benchmark-style coworker-agent task.\n"
        "Write only the first assistant response you would send before executing tools.\n"
        "Be concrete about what you will inspect first, which systems or artifacts matter, "
        "and what risks or missing information you will watch for. Do not claim the task is complete.\n\n"
        f"Benchmark source: ClawMark\n"
        f"Task id: {task_id}\n"
        f"Domain: {category}\n"
        f"Environments: {json.dumps(environments, ensure_ascii=False)}\n"
        f"Tags: {json.dumps(tags, ensure_ascii=False)}\n\n"
        f"Task summary:\n{summary}\n\n"
        f"Initial prompt:\n{prompt}\n\n"
        "Respond in 4-8 sentences."
    )


def collect_clawmark_records(root: Path, max_tasks: int) -> list[EvalRecord]:
    tasks_root = root / "tasks"
    if not tasks_root.exists():
        return []
    records: list[EvalRecord] = []
    for task_py in sorted(tasks_root.glob("*/*/task.py")):
        summary_path = task_py.with_name("task_summary.txt")
        if not summary_path.exists():
            continue
        metadata = _extract_module_literal(task_py, "METADATA") or {}
        prompt = str(_extract_module_literal(task_py, "PROMPT") or "").strip()
        task_id = str(metadata.get("id") or f"{task_py.parent.parent.name}_{task_py.parent.name}")
        category = str(metadata.get("category") or task_py.parent.parent.name)
        environments = [str(item) for item in metadata.get("environments", [])]
        tags = [str(item) for item in metadata.get("tags", [])]
        summary = _read_text(summary_path)
        record_metadata = {
            "benchmark_source": "clawmark",
            "task_id": task_id,
            "category": category,
            "environments": environments,
            "tags": tags,
            "task_summary": summary,
            "initial_prompt": prompt,
        }
        records.append(
            EvalRecord(
                dataset_name="clawmark_proxy",
                task_id=task_id,
                input_text=_clawmark_prompt(
                    task_id=task_id,
                    category=category,
                    environments=environments,
                    tags=tags,
                    summary=summary,
                    prompt=prompt,
                ),
                label=summary,
                metadata=record_metadata,
                group_key=category,
            )
        )
    return _balanced_take(records, max_tasks)


def write_jsonl(path: Path, records: list[EvalRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "input": record.input_text,
                "label": record.label,
                "metadata": record.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def build_eval_config(
    *,
    output_dir: Path,
    datasets: list[tuple[str, Path]],
    eval_max_response_len: int,
) -> dict[str, Any]:
    return {
        "eval": {
            "defaults": {
                "input_key": "input",
                "label_key": "label",
                "metadata_key": "metadata",
                "n_samples_per_eval_prompt": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_response_len": eval_max_response_len,
            },
            "datasets": [
                {
                    "name": name,
                    "path": str(path),
                    "custom_generate_function_path": "a3s_code_benchmarks.benchmark_eval.generate_with_judge",
                }
                for name, path in datasets
            ],
        }
    }


def build_artifacts(
    *,
    skillsbench_root: Path,
    clawmark_root: Path,
    output_dir: Path,
    skillsbench_max_tasks: int,
    clawmark_max_tasks: int,
    eval_max_response_len: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    skillsbench_records = collect_skillsbench_records(skillsbench_root, skillsbench_max_tasks)
    clawmark_records = collect_clawmark_records(clawmark_root, clawmark_max_tasks)

    datasets: list[tuple[str, Path]] = []
    summary: dict[str, Any] = {"datasets": {}}

    if skillsbench_records:
        skillsbench_path = write_jsonl(output_dir / "skillsbench_eval.jsonl", skillsbench_records)
        datasets.append(("skillsbench_proxy", skillsbench_path))
        summary["datasets"]["skillsbench_proxy"] = {
            "path": str(skillsbench_path),
            "num_records": len(skillsbench_records),
        }

    if clawmark_records:
        clawmark_path = write_jsonl(output_dir / "clawmark_eval.jsonl", clawmark_records)
        datasets.append(("clawmark_proxy", clawmark_path))
        summary["datasets"]["clawmark_proxy"] = {
            "path": str(clawmark_path),
            "num_records": len(clawmark_records),
        }

    if not datasets:
        raise RuntimeError("No benchmark eval datasets were produced.")

    config = build_eval_config(
        output_dir=output_dir,
        datasets=datasets,
        eval_max_response_len=eval_max_response_len,
    )
    config_path = output_dir / "benchmark_eval_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["config_path"] = str(config_path)

    summary_path = output_dir / "benchmark_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillsbench-root", type=Path, required=True)
    parser.add_argument("--clawmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skillsbench-max-tasks", type=int, default=24)
    parser.add_argument("--clawmark-max-tasks", type=int, default=24)
    parser.add_argument("--eval-max-response-len", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_artifacts(
        skillsbench_root=args.skillsbench_root,
        clawmark_root=args.clawmark_root,
        output_dir=args.output_dir,
        skillsbench_max_tasks=args.skillsbench_max_tasks,
        clawmark_max_tasks=args.clawmark_max_tasks,
        eval_max_response_len=args.eval_max_response_len,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
