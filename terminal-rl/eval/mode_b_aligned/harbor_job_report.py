#!/usr/bin/env python3
"""Summarise or watch a Harbor eval job directory.

Harbor writes one result.json per trial plus an aggregate result.json at the job
root. Reading those by hand is where two mistakes keep happening, so both are
handled here once:

- The score denominator. The job's own n_total_trials is the reporting
  denominator. Averaging only over trials that carry a reward field silently
  drops the ones that errored before the verifier ran, which inflates the score.
  Both numbers are printed, clearly labelled.
- Deciding whether a running job is stuck. A task-level timeout is an eval
  result, not an infrastructure failure. --watch reports progress and lets the
  operator judge; it never kills anything.

    python harbor_job_report.py <job-dir>
    python harbor_job_report.py <job-dir> --watch --interval 300
    python harbor_job_report.py <job-dir> --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Trial:
    name: str
    reward: float | None
    exception_type: str | None
    exception_message: str


@dataclass
class JobReport:
    job_dir: Path
    started_at: str | None
    finished_at: str | None
    n_total_trials: int | None
    trials: list[Trial] = field(default_factory=list)

    @property
    def is_finished(self) -> bool:
        return bool(self.finished_at)

    @property
    def reward_sum(self) -> float:
        return sum(t.reward or 0.0 for t in self.trials)

    @property
    def rewarded_trials(self) -> list[Trial]:
        return [t for t in self.trials if t.reward is not None]

    @property
    def denominator(self) -> int:
        """Harbor's reporting denominator: every trial the job intended to run."""
        return self.n_total_trials or len(self.trials)

    @property
    def score(self) -> float | None:
        return self.reward_sum / self.denominator if self.denominator else None

    @property
    def score_over_rewarded_only(self) -> float | None:
        """The inflated variant, reported so the gap is visible rather than hidden."""
        rewarded = self.rewarded_trials
        return self.reward_sum / len(rewarded) if rewarded else None

    @property
    def solved(self) -> list[Trial]:
        return [t for t in self.trials if (t.reward or 0.0) > 0]

    @property
    def error_counts(self) -> dict[str, int]:
        return dict(Counter(t.exception_type for t in self.trials if t.exception_type))

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_dir": str(self.job_dir),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "is_finished": self.is_finished,
            "n_total_trials": self.n_total_trials,
            "n_result_files": len(self.trials),
            "n_rewarded_trials": len(self.rewarded_trials),
            "reward_sum": self.reward_sum,
            "score": self.score,
            "score_over_rewarded_only": self.score_over_rewarded_only,
            "solved_tasks": sorted(t.name for t in self.solved),
            "error_counts": self.error_counts,
        }


def _as_reward(value: Any) -> float | None:
    """A malformed reward must not abort a report the way a malformed file must not."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # NaN would propagate into reward_sum and annihilate every other trial's
    # score, and json.dumps emits it as a bare NaN token that is not valid JSON.
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A trial result being written while we poll must not abort a watch loop.
        return {}


def read_job(job_dir: Path) -> JobReport:
    if not job_dir.is_dir():
        raise FileNotFoundError(f"{job_dir} is not a directory")

    aggregate = _read_json(job_dir / "result.json")
    trials: list[Trial] = []
    for trial_result in sorted(job_dir.glob("*/result.json")):
        data = _read_json(trial_result)
        exception = data.get("exception_info") or {}
        rewards = (data.get("verifier_result") or {}).get("rewards") or {}
        trials.append(
            Trial(
                name=trial_result.parent.name,
                reward=_as_reward(rewards.get("reward")),
                exception_type=exception.get("exception_type"),
                exception_message=str(exception.get("exception_message") or ""),
            )
        )

    return JobReport(
        job_dir=job_dir,
        started_at=aggregate.get("started_at"),
        finished_at=aggregate.get("finished_at"),
        n_total_trials=aggregate.get("n_total_trials"),
        trials=trials,
    )


def format_report(report: JobReport, *, show_errors: bool = True) -> str:
    lines = [
        f"job_dir          {report.job_dir}",
        f"started_at       {report.started_at}",
        f"finished_at      {report.finished_at}",
        f"progress         {len(report.trials)} / {report.n_total_trials} trial results on disk",
        f"reward_sum       {report.reward_sum}",
    ]
    if report.score is not None:
        lines.append(
            f"score            {report.reward_sum} / {report.denominator} = {report.score:.10f}"
            "   <- report this one"
        )
    if report.score_over_rewarded_only is not None and len(report.rewarded_trials) != report.denominator:
        lines.append(
            f"  (over the {len(report.rewarded_trials)} trials that reached the verifier: "
            f"{report.score_over_rewarded_only:.10f} -- not the reporting number)"
        )
    lines.append(f"error_counts     {report.error_counts or '{}'}")
    lines.append(f"solved_tasks     {sorted(t.name for t in report.solved) or '[]'}")
    if show_errors and report.error_counts:
        lines.append("errored trials")
        for trial in report.trials:
            if trial.exception_type:
                message = trial.exception_message[:120].replace("\n", " ")
                lines.append(f"  {trial.name}  {trial.exception_type}  {message}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("job_dir", type=Path, help="Harbor job directory")
    parser.add_argument("--watch", action="store_true", help="poll until finished_at is set")
    parser.add_argument(
        "--interval", type=float, default=300.0,
        help="seconds between polls; floored at 0.1 so a typo cannot busy-loop the job dir",
    )
    parser.add_argument("--max-polls", type=int, help="stop after this many polls (for testing)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    args = parser.parse_args(argv)

    polls = 0
    while True:
        report = read_job(args.job_dir)
        polls += 1
        if args.as_json:
            # One document per line while watching, so the stream stays parseable.
            print(json.dumps(report.to_dict(), ensure_ascii=False,
                             indent=None if args.watch else 2))
        else:
            print(f"=== poll {polls} {time.strftime('%F %T')} ===" if args.watch else "", end="")
            print("\n" if args.watch else "", end="")
            print(format_report(report, show_errors=not args.watch))
        if not args.watch or report.is_finished:
            break
        if args.max_polls is not None and polls >= args.max_polls:
            break
        time.sleep(max(args.interval, 0.1))

    return 0 if report.is_finished or not args.watch else 1


if __name__ == "__main__":
    sys.exit(main())
