from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path


BUILDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "a3s_code_benchmarks"
    / "benchmark_eval_builder.py"
)


def _load_builder_module():
    module_name = f"benchmark_eval_builder_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_skillsbench_records_extracts_metadata(tmp_path: Path) -> None:
    builder = _load_builder_module()
    task_dir = tmp_path / "skillsbench" / "tasks" / "sample-task"
    (task_dir / "environment" / "skills" / "gmail").mkdir(parents=True)
    (task_dir / "environment" / "skills" / "gmail" / "SKILL.md").write_text("# Gmail\n", encoding="utf-8")
    (task_dir / "instruction.md").write_text("Process the inbox and send a reply.", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                'version = "1.0"',
                "",
                "[metadata]",
                'difficulty = "medium"',
                'category = "Scheduling"',
                'tags = ["email", "calendar"]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = builder.collect_skillsbench_records(tmp_path / "skillsbench", max_tasks=10)

    assert len(records) == 1
    record = records[0]
    assert record.task_id == "sample-task"
    assert record.metadata["available_skills"] == ["gmail"]
    assert "Benchmark source: SkillsBench" in record.input_text


def test_collect_clawmark_records_extracts_summary_and_prompt(tmp_path: Path) -> None:
    builder = _load_builder_module()
    task_dir = tmp_path / "ClawMark" / "tasks" / "ops" / "task1"
    task_dir.mkdir(parents=True)
    (task_dir / "task_summary.txt").write_text("Investigate the queue drift and send a summary.", encoding="utf-8")
    (task_dir / "task.py").write_text(
        "\n".join(
            [
                'METADATA = {"id": "ops_task1", "category": "ops", "environments": ["filesystem", "email"], "tags": ["triage"]}',
                'PROMPT = "Check your email and workspace for new evidence."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = builder.collect_clawmark_records(tmp_path / "ClawMark", max_tasks=10)

    assert len(records) == 1
    record = records[0]
    assert record.task_id == "ops_task1"
    assert record.metadata["environments"] == ["filesystem", "email"]
    assert "Initial prompt:" in record.input_text


def test_build_artifacts_writes_eval_config(tmp_path: Path) -> None:
    builder = _load_builder_module()

    skills_task = tmp_path / "skillsbench" / "tasks" / "task-a"
    (skills_task / "instruction.md").parent.mkdir(parents=True)
    (skills_task / "instruction.md").write_text("Summarize the spreadsheet and draft the reply.", encoding="utf-8")
    (skills_task / "task.toml").write_text(
        "\n".join(
            [
                'version = "1.0"',
                "",
                "[metadata]",
                'difficulty = "easy"',
                'category = "analysis"',
                'tags = ["sheet"]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    claw_task = tmp_path / "ClawMark" / "tasks" / "ops" / "task2"
    claw_task.mkdir(parents=True)
    (claw_task / "task_summary.txt").write_text("Review the incident queue.", encoding="utf-8")
    (claw_task / "task.py").write_text(
        "\n".join(
            [
                'METADATA = {"id": "ops_task2", "category": "ops", "environments": ["filesystem"], "tags": ["ops"]}',
                'PROMPT = "Start with the workspace and recent notes."',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = builder.build_artifacts(
        skillsbench_root=tmp_path / "skillsbench",
        clawmark_root=tmp_path / "ClawMark",
        output_dir=tmp_path / "out",
        skillsbench_max_tasks=10,
        clawmark_max_tasks=10,
        eval_max_response_len=1536,
    )

    config = json.loads((tmp_path / "out" / "benchmark_eval_config.json").read_text(encoding="utf-8"))
    dataset_names = [item["name"] for item in config["eval"]["datasets"]]

    assert summary["datasets"]["skillsbench_proxy"]["num_records"] == 1
    assert summary["datasets"]["clawmark_proxy"]["num_records"] == 1
    assert dataset_names == ["skillsbench_proxy", "clawmark_proxy"]
    assert config["eval"]["defaults"]["max_response_len"] == 1536
