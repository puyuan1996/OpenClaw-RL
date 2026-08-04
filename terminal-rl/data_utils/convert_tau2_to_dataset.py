#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any


def _install_deepdiff_stub() -> None:
    if "deepdiff" in sys.modules:
        return
    try:
        import deepdiff  # noqa: F401
        return
    except ImportError:
        pass

    module = types.ModuleType("deepdiff")

    class DeepDiff(dict):
        def __init__(self, left: Any, right: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            if left != right:
                self["values_changed"] = {"root": {"old_value": left, "new_value": right}}

    module.DeepDiff = DeepDiff
    sys.modules["deepdiff"] = module


def _install_addict_stub() -> None:
    if "addict" in sys.modules:
        return
    try:
        import addict  # noqa: F401
        return
    except ImportError:
        pass

    module = types.ModuleType("addict")

    class Dict(dict):
        def __getattr__(self, key: str) -> Any:
            try:
                value = self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc
            if isinstance(value, dict) and not isinstance(value, Dict):
                value = Dict(value)
                self[key] = value
            return value

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

        def update(self, other: Any = None, **kwargs: Any) -> None:
            if other is None:
                other = {}
            items = dict(other)
            items.update(kwargs)
            for key, value in items.items():
                if key in self and isinstance(self[key], dict) and isinstance(value, dict):
                    nested = self[key]
                    if not isinstance(nested, Dict):
                        nested = Dict(nested)
                    nested.update(value)
                    self[key] = nested
                else:
                    self[key] = Dict(value) if isinstance(value, dict) else value

        def to_dict(self) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in self.items():
                if isinstance(value, Dict):
                    result[key] = value.to_dict()
                elif isinstance(value, dict):
                    result[key] = Dict(value).to_dict()
                else:
                    result[key] = value
            return result

    module.Dict = Dict
    sys.modules["addict"] = module


def _install_toml_stub() -> None:
    if "toml" in sys.modules:
        return
    try:
        import toml  # noqa: F401
        return
    except ImportError:
        pass

    import tomllib

    module = types.ModuleType("toml")

    def load(fp: Any) -> Any:
        if hasattr(fp, "read"):
            return tomllib.loads(fp.read())
        return tomllib.loads(Path(fp).read_text(encoding="utf-8"))

    def loads(text: str) -> Any:
        return tomllib.loads(text)

    module.load = load
    module.loads = loads
    sys.modules["toml"] = module


def ensure_tau2_importable(root: Path) -> None:
    _install_deepdiff_stub()
    _install_addict_stub()
    _install_toml_stub()

    src_dir = root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"tau2 src dir not found: {src_dir}")
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
    os.environ.setdefault("TAU2_DATA_DIR", str(root / "data"))


def _structured_instruction_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()

    lines: list[str] = []
    for label, attr in (
        ("Domain", "domain"),
        ("Reason", "reason_for_call"),
        ("Known info", "known_info"),
        ("Unknown info", "unknown_info"),
        ("Task instructions", "task_instructions"),
    ):
        raw = getattr(value, attr, None)
        if raw:
            lines.append(f"{label}: {raw}")
    return "\n".join(lines).strip()


def task_instruction(task: Any) -> str:
    ticket = getattr(task, "ticket", None)
    if ticket:
        return str(ticket).strip()

    user_scenario = getattr(task, "user_scenario", None)
    if user_scenario is not None:
        instructions = getattr(user_scenario, "instructions", None)
        structured = _structured_instruction_text(instructions)
        if structured:
            return structured
        if instructions is not None:
            return str(instructions).strip()

    description = getattr(task, "description", None)
    if description is not None:
        for attr in ("notes", "purpose"):
            raw = getattr(description, attr, None)
            if raw:
                return str(raw).strip()

    return str(getattr(task, "id", "unknown")).strip()


def convert_task(task: Any, *, domain: str, task_split: str | None, policy_type: str) -> dict[str, Any]:
    instruction = task_instruction(task)
    metadata = {
        "task_name": f"tau2_{domain}_{task.id}",
        "task_path": f"tau2/{domain}/{task.id}",
        "instruction": instruction,
        "data_source": "tau2",
        "tau2_domain": domain,
        "tau2_task_id": str(task.id),
        "tau2_task_split": task_split or "",
        "tau2_policy_type": policy_type,
        "tau2_ticket": str(getattr(task, "ticket", "") or ""),
        "tau2_has_ticket": bool(getattr(task, "ticket", None)),
        "tau2_solo_mode": 1,
    }
    return {
        "task": [{"role": "user", "content": instruction}],
        "metadata": metadata,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert tau2 solo-compatible tasks to terminal-rl JSONL."
    )
    parser.add_argument("--tau2-root", type=Path, default=Path("tau2-bench"))
    parser.add_argument("--domain", choices=["mock", "telecom"], default="telecom")
    parser.add_argument("--task-split", default="train")
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        default=None,
        help="Optional specific task id; may be repeated",
    )
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument(
        "--policy-type",
        choices=["manual", "workflow"],
        default="manual",
        help="Only meaningful for telecom.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("terminal-rl/dataset/tau2_telecom_train_solo"),
    )
    args = parser.parse_args()

    ensure_tau2_importable(args.tau2_root.resolve())
    from tau2.runner.helpers import get_tasks

    tasks = get_tasks(
        task_set_name=args.domain,
        task_split_name=args.task_split or None,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
    )
    if not tasks:
        raise ValueError(
            f"No tau2 tasks loaded for domain={args.domain} split={args.task_split}"
        )

    records = [
        convert_task(
            task,
            domain=args.domain,
            task_split=args.task_split or None,
            policy_type=args.policy_type,
        )
        for task in tasks
    ]

    write_jsonl(args.output_dir / "train.jsonl", records)
    write_jsonl(args.output_dir / "val.jsonl", [])

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "domain": args.domain,
                "task_split": args.task_split,
                "policy_type": args.policy_type,
                "count": len(records),
                "sample_task_ids": [
                    record["metadata"]["tau2_task_id"] for record in records[:20]
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
