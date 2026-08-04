"""Centralized path management for a single training run.

Usage in shell script:
    RUN_ID=$(python3 -m terminal-rl.run_paths init --runs-root ./runs --ckpt-root /mnt/.../ckpt)
    # prints JSON with all paths to stdout

Usage in Python (generate.py etc.):
    from run_paths import RunPaths
    rp = RunPaths.from_env()  # reads RUN_DIR env var
    traj_dir = rp.trajectories_dir
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunPaths:
    def __init__(self, run_id: str, runs_root: Path, ckpt_root: Path):
        self.run_id = run_id
        self.run_dir = runs_root / run_id
        self.config_dir = self.run_dir / "config"
        self.logs_dir = self.run_dir / "logs"
        self.trajectories_dir = self.run_dir / "trajectories"
        self.metrics_dir = self.run_dir / "metrics"
        self.wandb_dir = self.metrics_dir / "wandb"
        self.analysis_dir = self.metrics_dir / "analysis"
        self.ckpt_real = ckpt_root / run_id
        self.ckpt_link = self.run_dir / "ckpt"
        self.meta_file = self.run_dir / "meta.json"

    @classmethod
    def from_env(cls) -> "RunPaths":
        run_dir = os.getenv("RUN_DIR", "")
        if not run_dir:
            return None
        run_dir = Path(run_dir)
        run_id = run_dir.name
        runs_root = run_dir.parent
        ckpt_root = Path(os.getenv(
            "CKPT_ROOT",

            "/mnt/shared-storage-gpfs2/narmodel/agenticrl/ckpt"

        ))
        return cls(run_id, runs_root, ckpt_root)

    def create_all(self) -> None:
        for d in [
            self.config_dir, self.logs_dir, self.trajectories_dir,
            self.metrics_dir, self.wandb_dir, self.analysis_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        if os.getenv("MAX_CKPT_KEEP", "0") != "0":
            self.ckpt_real.mkdir(parents=True, exist_ok=True)
            if not self.ckpt_link.exists():
                try:
                    self.ckpt_link.symlink_to(self.ckpt_real)
                except OSError:
                    pass

    def write_meta(self, extra: dict[str, Any] | None = None) -> Path:
        meta = {
            "run_id": self.run_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "hostname": os.uname().nodename,
            "git_commit": _git_commit(),
            "git_branch": _git_branch(),
            "command": " ".join(sys.argv),
            "paths": {
                "run_dir": str(self.run_dir),
                "logs": str(self.logs_dir),
                "trajectories": str(self.trajectories_dir),
                "metrics": str(self.metrics_dir),
                "wandb": str(self.wandb_dir),
                "ckpt_real": str(self.ckpt_real),
            },
        }
        if extra:
            meta.update(extra)
        self.meta_file.write_text(json.dumps(meta, indent=2, default=str))
        return self.meta_file

    def print_summary(self) -> None:
        print("=" * 60)
        print(f"  Run ID:        {self.run_id}")
        print(f"  Run dir:       {self.run_dir}")
        print(f"  Logs:          {self.logs_dir}")
        print(f"  Trajectories:  {self.trajectories_dir}")
        print(f"  Metrics/wandb: {self.wandb_dir}")
        print(f"  Ckpt (real):   {self.ckpt_real}")
        print(f"  Ckpt (link):   {self.ckpt_link}")
        print(f"  Meta:          {self.meta_file}")
        print("=" * 60)

    def to_env_dict(self) -> dict[str, str]:
        return {
            "RUN_ID": self.run_id,
            "RUN_DIR": str(self.run_dir),
            "RUN_LOG_DIR": str(self.logs_dir),
            "TERMINAL_SAVE_TRAJ_DIR": str(self.trajectories_dir),
            "WANDB_DIR": str(self.wandb_dir),
            "SAVE_CKPT": str(self.ckpt_real),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_env_dict(), indent=2)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def generate_run_id(model: str = "qwen3-8b", method: str = "grpo") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{model}_{method}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    init_p = sub.add_parser("init")
    init_p.add_argument("--runs-root", default="./runs")

    init_p.add_argument("--ckpt-root", default="/mnt/shared-storage-gpfs2/narmodel/agenticrl/ckpt")

    init_p.add_argument("--model", default="qwen3-8b")
    init_p.add_argument("--method", default="grpo")
    init_p.add_argument("--run-id", default="")

    args = parser.parse_args()
    if args.cmd == "init":
        run_id = args.run_id or generate_run_id(args.model, args.method)
        rp = RunPaths(run_id, Path(args.runs_root), Path(args.ckpt_root))
        rp.create_all()
        rp.write_meta()
        rp.print_summary()
        print(rp.to_json())
    else:
        parser.print_help()
