#!/usr/bin/env python3
"""Compare SETA-env eval runs with a two-proportion test, not by eyeballing bars.

Filling a benchmark table one checkpoint at a time invites reading 21.61% next
to 23.60% as an improvement. exact_pass is a binomial count over the dataset, so
whether two checkpoints differ has an actual answer.

    python terminal-rl/scripts/compare_seta_env_evals.py \\
        baseline=runs/<run-a>/final_analysis/summary.json \\
        rl-iter499=runs/<run-b>/final_analysis/summary.json

Each run gets a Wilson 95% interval, which describes where that run's own rate
sits. Whether two runs differ is a separate question and is answered by a
two-proportion z-test, because comparing the intervals by eye does not answer it:
non-overlap does imply significance, but overlap does NOT imply its absence. On
1356 samples, 293 vs 352 exact passes have overlapping Wilson intervals and yet
p = 0.008. Reading that pair off the intervals would throw away a real effect,
which is the exact mistake this tool exists to prevent, so the reported verdict
comes from the test and the intervals are shown as description only.

Two runs over the same dataset are paired data, and the two-proportion z-test
assumes independent samples. Pass ``per_sample.csv`` instead of ``summary.json``
and the tool joins on ``sample_index`` and uses an exact McNemar test on the
discordant items, which is the right test and the more powerful one: the
unpaired test is conservative under the positive correlation that same-item
measurement produces, so it can report "no evidence" for a real effect. The
unpaired test remains for summary-only input, labelled as such.

The z-test is a normal approximation (appropriate when expected counts are
comfortably above 5, as they are at this dataset size) and additionally assumes
independence, which paired input does not satisfy. With k runs the tool makes
k(k-1)/2 comparisons at nominal alpha, so p values close to 0.05 in a wide table
should be read with that in mind.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval; reproduces the intervals published in issues #27-#29."""
    if trials <= 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    return (centre - margin) / denominator, (centre + margin) / denominator


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test; returns (z, two-sided p).

    Normal approximation. Returns (0.0, 1.0) when it does not apply, so a
    degenerate run is reported as "no evidence" rather than as a difference.
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0
    pooled = (k1 + k2) / (n1 + n2)
    variance = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    if variance <= 0:
        return 0.0, 1.0
    z = (k2 / n2 - k1 / n1) / math.sqrt(variance)
    return z, math.erfc(abs(z) / math.sqrt(2))


def mcnemar_exact(only_left: int, only_right: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs.

    Concordant items carry no information about a difference, so the test is a
    binomial sign test on the ``only_left`` / ``only_right`` split.
    """
    discordant = only_left + only_right
    if discordant == 0:
        return 1.0
    smaller = min(only_left, only_right)
    # Summed in log space: math.comb returns an exact int that overflows float
    # before the 0.5**n scaling can bring it back, from 1025 discordant pairs up.
    log_half = discordant * math.log(0.5)
    tail = math.fsum(
        math.exp(
            math.lgamma(discordant + 1)
            - math.lgamma(i + 1)
            - math.lgamma(discordant - i + 1)
            + log_half
        )
        for i in range(smaller + 1)
    )
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class Run:
    label: str
    total: int
    exact_pass: int
    raw_score_mean: float | None
    missing: int
    # sample_index -> exact_pass, present only when loaded from per_sample.csv
    per_sample: dict[int, bool] | None = None

    @property
    def exact_pass_rate(self) -> float:
        return self.exact_pass / self.total if self.total else 0.0

    @property
    def exact_pass_interval(self) -> tuple[float, float]:
        return wilson_interval(self.exact_pass, self.total)


@dataclass(frozen=True)
class Pair:
    left: Run
    right: Run
    z: float | None
    p_value: float
    test: str                      # "mcnemar-exact" (paired) or "two-proportion-z"
    discordant: tuple[int, int] | None = None   # (only_left, only_right)

    @property
    def delta_pp(self) -> float:
        return (self.right.exact_pass_rate - self.left.exact_pass_rate) * 100

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def intervals_overlap(self) -> bool:
        low_l, high_l = self.left.exact_pass_interval
        low_r, high_r = self.right.exact_pass_interval
        return low_l <= high_r and low_r <= high_l


REQUIRED_KEYS = (
    "dataset_total",
    "exact_pass_count",
    "raw_score_mean_all_dataset_missing_as_zero",
    "missing_count",
)


def load_per_sample(label: str, csv_path: Path) -> Run:
    """Load a per_sample.csv, keeping the per-item exact_pass needed for pairing."""
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    per_sample, scores, missing = {}, [], 0
    for row in rows:
        index = int(row["sample_index"])
        if index in per_sample:
            raise ValueError(
                f"{csv_path} repeats sample_index {index}; "
                "the denominator and the per-item join would disagree"
            )
        raw = row.get("raw_score")
        if raw in ("", None):
            per_sample[index] = False
            missing += 1
            continue
        score = float(raw)
        per_sample[index] = score == 1.0
        scores.append(score)
    return Run(
        label=label,
        total=len(rows),
        exact_pass=sum(1 for passed in per_sample.values() if passed),
        raw_score_mean=sum(scores) / len(rows),
        missing=missing,
        per_sample=per_sample,
    )


def load_run(label: str, summary_path: Path) -> Run:
    if summary_path.suffix.lower() == ".csv":
        return load_per_sample(label, summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    absent = [key for key in REQUIRED_KEYS if key not in summary]
    if absent:
        raise KeyError(
            f"{summary_path} is missing {', '.join(absent)}; "
            "expected a summary.json written by analyze_seta_env_eval.py"
        )
    return Run(
        label=label,
        total=int(summary["dataset_total"]),
        exact_pass=int(summary["exact_pass_count"]),
        # The conservative denominator, matching what the docs report. Genuinely
        # None on an empty dataset; rendered as n/a rather than silently 0.00%.
        raw_score_mean=(
            None if summary["raw_score_mean_all_dataset_missing_as_zero"] is None
            else float(summary["raw_score_mean_all_dataset_missing_as_zero"])
        ),
        missing=int(summary["missing_count"]),
    )


def compare_pairs(runs: Sequence[Run]) -> list[Pair]:
    """Every unordered pair, tested. Paired when both sides carry per-sample data."""
    pairs = []
    for index, left in enumerate(runs):
        for right in runs[index + 1:]:
            shared = (
                sorted(set(left.per_sample) & set(right.per_sample))
                if left.per_sample is not None and right.per_sample is not None
                else []
            )
            if shared:
                only_left = sum(
                    1 for i in shared if left.per_sample[i] and not right.per_sample[i]
                )
                only_right = sum(
                    1 for i in shared if right.per_sample[i] and not left.per_sample[i]
                )
                pairs.append(Pair(
                    left=left, right=right, z=None,
                    p_value=mcnemar_exact(only_left, only_right),
                    test="mcnemar-exact", discordant=(only_left, only_right),
                ))
                continue
            z, p_value = two_proportion_test(
                left.exact_pass, left.total, right.exact_pass, right.total
            )
            degenerate = left.total <= 0 or right.total <= 0
            pairs.append(Pair(
                left=left, right=right, z=None if degenerate else z,
                p_value=p_value, test="two-proportion-z",
            ))
    return pairs


def format_comparison(runs: Sequence[Run]) -> str:
    width = max(len(run.label) for run in runs)
    lines = [
        f"{'run'.ljust(width)}  {'n':>5}  {'miss':>4}  {'raw_score':>9}  "
        f"{'exact_pass':>10}  {'rate':>7}  {'Wilson 95%':>17}",
    ]
    for run in runs:
        low, high = run.exact_pass_interval
        mean = "     n/a" if run.raw_score_mean is None else f"{run.raw_score_mean * 100:8.2f}%"
        interval = "            n/a" if run.total <= 0 else \
            f"{low * 100:6.2f}% - {high * 100:6.2f}%"
        lines.append(
            f"{run.label.ljust(width)}  {run.total:5d}  {run.missing:4d}  "
            f"{mean}  {run.exact_pass:10d}  {run.exact_pass_rate * 100:6.2f}%  {interval}"
        )

    if any(run.total <= 0 for run in runs):
        lines.append("")
        lines.append("WARNING: a run has dataset_total = 0; it cannot be compared.")

    if len(runs) < 2:
        return "\n".join(lines)

    pairs = compare_pairs(runs)
    lines.append("")
    lines.append("exact_pass, per pair:")
    for pair in pairs:
        verdict = "differ (p < 0.05)" if pair.is_significant else "no evidence of a difference"
        if pair.test == "mcnemar-exact":
            only_left, only_right = pair.discordant or (0, 0)
            statistic = f"McNemar exact   discordant {only_left}/{only_right}"
        else:
            statistic = "two-proportion z (unpaired)   z " + (
                "  n/a" if pair.z is None else f"{pair.z:+.3f}"
            )
        note = ""
        degenerate = pair.left.total <= 0 or pair.right.total <= 0
        if not degenerate and pair.is_significant != (not pair.intervals_overlap):
            overlap = "overlap" if pair.intervals_overlap else "do not overlap"
            note = f"   [Wilson intervals {overlap}; the test, not the intervals, decides]"
        lines.append(
            f"  {pair.left.label} vs {pair.right.label}   delta {pair.delta_pp:+.2f} pp   "
            f"{statistic}   p {pair.p_value:.4f}   {verdict}{note}"
        )
    if any(pair.test == "two-proportion-z" for pair in pairs):
        lines.append(
            "The unpaired test assumes independent samples. Two runs over the same dataset are"
            " paired; pass per_sample.csv instead of summary.json for an exact McNemar test,"
            " which has more power here."
        )
    lines.append(
        "Comparing the Wilson intervals by eye does not answer this: overlap does not imply"
        " absence of a difference."
    )
    lines.append(
        "Note: raw_score is average partial credit, not a solve rate; exact_pass is the solve rate."
    )
    return "\n".join(lines)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected label=path/to/summary.json, got {value!r}")
    label, _, path = value.partition("=")
    if not label:
        raise argparse.ArgumentTypeError(f"label must be non-empty, got {value!r}")
    return label, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs", nargs="+", type=_parse_run, metavar="LABEL=PATH",
        help="summary.json, or per_sample.csv to enable the paired McNemar test",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    args = parser.parse_args(argv)

    runs = [load_run(label, path) for label, path in args.runs]
    if args.as_json:
        print(json.dumps(
            {
                "runs": [
                    {
                        "label": run.label,
                        "dataset_total": run.total,
                        "missing_count": run.missing,
                        "raw_score_mean": run.raw_score_mean,
                        "exact_pass_count": run.exact_pass,
                        "exact_pass_rate": run.exact_pass_rate,
                        "exact_pass_wilson95": list(run.exact_pass_interval),
                    }
                    for run in runs
                ],
                "pairs": [
                    {
                        "left": pair.left.label,
                        "right": pair.right.label,
                        "delta_pp": pair.delta_pp,
                        "test": pair.test,
                        "z": pair.z,
                        "discordant": list(pair.discordant) if pair.discordant else None,
                        "p_value": pair.p_value,
                        "significant_at_0_05": pair.is_significant,
                        "wilson_intervals_overlap": pair.intervals_overlap,
                    }
                    for pair in compare_pairs(runs)
                ],
            },
            indent=2,
        ))
    else:
        print(format_comparison(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
