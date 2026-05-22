#!/usr/bin/env python3
"""Parse <run_dir>/logs/train.log and plot core training curves.

Generates the same figures previously produced by the inline analyzer in
run-specific notebooks:
  overview.png  reward_curve.png  response_length.png
  loss_curve.png  grad_norm.png  kl_entropy.png
  summary_stats.json

Reusable across runs.

Usage:
  python terminal-rl/scripts/plot_training_metrics.py --run-dir runs/<run_id>

Optional:
  --log-file PATH  Override (default <run_dir>/logs/train.log)
  --out-dir DIR    Override output (default <run_dir>/metrics/analysis)
  --no-figs        Skip image generation, only emit summary_stats.json

Exits 0 on success, 1 if log not found, 2 if no parsed rollouts.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROLLOUT_RE = re.compile(r"data\.py:\d+ - rollout (\d+): (\{.+\})")
TRAIN_RE = re.compile(r"model\.py:\d+ - step (\d+): (\{.+\})")
PERF_RE = re.compile(r"rollout\.py:\d+ - perf (\d+): (\{.+\})")
TRAJ_RE = re.compile(
    r"\[task=(\S+) uid=(\S+) group_idx=(\d+) sample_idx=(\d+)\] "
    r"Rollout finished: status=(\S+) turns=(\d+) parse_errors=(\d+)"
)
CLAW_RE = re.compile(r"ClawSentry pre_action fail-open.*?'(\d+) ([^']+)'")
RESET500_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*Server error '500 .*?/reset'"
)


def _parse_log(log_path: Path) -> dict[str, Any]:
    rollout_metrics: dict[int, dict] = {}
    train_metrics: dict[int, dict] = {}
    perf_metrics: dict[int, dict] = {}
    clawsentry_errs: Counter = Counter()
    status_counts: Counter = Counter()
    turn_counts: list[int] = []
    parse_errs: list[int] = []
    reset500_per_min: Counter = Counter()

    print(f"[+] parsing {log_path}")
    with log_path.open(errors="replace") as f:
        for line in f:
            m = ROLLOUT_RE.search(line)
            if m:
                try:
                    rollout_metrics[int(m.group(1))] = ast.literal_eval(m.group(2))
                except Exception:
                    pass
                continue
            m = TRAIN_RE.search(line)
            if m:
                try:
                    train_metrics[int(m.group(1))] = ast.literal_eval(m.group(2))
                except Exception:
                    pass
                continue
            m = PERF_RE.search(line)
            if m:
                try:
                    perf_metrics[int(m.group(1))] = ast.literal_eval(m.group(2))
                except Exception:
                    pass
                continue
            m = TRAJ_RE.search(line)
            if m:
                st = m.group(5).split(".")[-1]
                status_counts[st] += 1
                turn_counts.append(int(m.group(6)))
                parse_errs.append(int(m.group(7)))
                continue
            m = CLAW_RE.search(line)
            if m:
                clawsentry_errs[f"{m.group(1)} {m.group(2)}"] += 1
                continue
            m = RESET500_RE.search(line)
            if m:
                # bucket by minute
                reset500_per_min[m.group(1)[:16]] += 1

    return dict(
        rollout_metrics=rollout_metrics,
        train_metrics=train_metrics,
        perf_metrics=perf_metrics,
        clawsentry_errs=clawsentry_errs,
        status_counts=status_counts,
        turn_counts=turn_counts,
        parse_errs=parse_errs,
        reset500_per_min=reset500_per_min,
    )


def _stats(arr: list[float], label: str) -> dict[str, float]:
    import math
    nums = [x for x in arr if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not nums:
        return {}
    nums = [float(x) for x in nums]
    n = len(nums)
    head = nums[:10] if n >= 10 else nums
    tail = nums[-10:] if n >= 10 else nums
    return {
        f"{label}_mean": sum(nums) / n,
        f"{label}_first10_mean": sum(head) / len(head),
        f"{label}_last10_mean": sum(tail) / len(tail),
        f"{label}_max": max(nums),
        f"{label}_min": min(nums),
    }


def _detect_collapse(
    r_ids: list[int], resp_len: list[float | None], threshold: float = 5.0
) -> int | None:
    """Return rollout id where mean response length first collapses below threshold."""
    for i, (rid, rl) in enumerate(zip(r_ids, resp_len)):
        if rl is not None and rl < threshold and i > 5:
            return rid
    return None


def _get_series(d: dict, ids: list[int], key: str) -> list[Any]:
    return [d[i].get(key) for i in ids]


def _filter_positive(xs: list[int], ys: list[Any]) -> tuple[list[int], list[float]]:
    out_x: list[int] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys):
        try:
            v = float(y)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out_x.append(x)
            out_y.append(v)
    return out_x, out_y


def _plot_all(
    parsed: dict[str, Any],
    out_dir: Path,
    collapse: int | None,
    reset500_total: int,
    clawsentry_total: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    rollout_metrics = parsed["rollout_metrics"]
    train_metrics = parsed["train_metrics"]
    perf_metrics = parsed["perf_metrics"]
    status_counts = parsed["status_counts"]
    turn_counts = parsed["turn_counts"]

    r_ids = sorted(rollout_metrics)
    t_ids = sorted(train_metrics)
    p_ids = sorted(perf_metrics)

    raw_rew = _get_series(rollout_metrics, r_ids, "rollout/raw_reward")
    rew = _get_series(rollout_metrics, r_ids, "rollout/rewards")
    trunc = _get_series(rollout_metrics, r_ids, "rollout/truncated")
    resp_len = _get_series(rollout_metrics, r_ids, "rollout/response_lengths")

    pg_loss = _get_series(train_metrics, t_ids, "train/pg_loss")
    kl_loss = _get_series(train_metrics, t_ids, "train/kl_loss")
    ent = _get_series(train_metrics, t_ids, "train/entropy_loss")
    gnorm = _get_series(train_metrics, t_ids, "train/grad_norm")

    rl_med = _get_series(perf_metrics, p_ids, "rollout/response_len/median") if p_ids else []
    rl_max = _get_series(perf_metrics, p_ids, "rollout/response_len/max") if p_ids else []

    def fig_save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figs_dir / name, dpi=120)
        plt.close()

    # reward_curve
    print("[+] plotting reward_curve.png")
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(r_ids, raw_rew, ".-", label="raw_reward (outcome)")
    ax[0].plot(r_ids, rew, ".-", alpha=0.6, label="reward (after norm)")
    ax[0].axhline(0, color="gray", ls=":", lw=0.8)
    if collapse is not None:
        ax[0].axvline(collapse, color="red", ls="--", alpha=0.5, label=f"collapse@{collapse}")
    ax[0].set_ylabel("reward")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.3)
    ax[0].set_title("Reward curve — raw_reward = 2·acc - 1 (outcome only)")
    ax[1].plot(r_ids, [t for t in trunc], ".-", label="truncated_frac")
    if collapse is not None:
        ax[1].axvline(collapse, color="red", ls="--", alpha=0.5)
    ax[1].set_xlabel("rollout")
    ax[1].set_ylabel("truncated frac")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig_save("reward_curve.png")

    # response_length
    print("[+] plotting response_length.png")
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    xs, ys = _filter_positive(r_ids, resp_len)
    if ys:
        ax.semilogy(xs, ys, ".-", label="mean response_length")
    if rl_med:
        xs2, ys2 = _filter_positive(p_ids, rl_med)
        if ys2:
            ax.semilogy(xs2, ys2, ".-", alpha=0.5, label="median (perf)")
    if rl_max:
        xs3, ys3 = _filter_positive(p_ids, rl_max)
        if ys3:
            ax.semilogy(xs3, ys3, ".-", alpha=0.4, label="max (perf)")
    if collapse is not None:
        ax.axvline(collapse, color="red", ls="--", alpha=0.5, label=f"collapse@{collapse}")
    ax.set_xlabel("rollout")
    ax.set_ylabel("response length (tokens, log)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    title = "Response length"
    if collapse is not None:
        title += f" — collapse @ rollout {collapse}"
    ax.set_title(title)
    fig_save("response_length.png")

    # loss_curve
    print("[+] plotting loss_curve.png")
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.plot(t_ids, pg_loss, ".-", label="pg_loss")
    ax.plot(t_ids, kl_loss, ".-", alpha=0.7, label="kl_loss")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("train step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Loss curves")
    fig_save("loss_curve.png")

    # grad_norm
    print("[+] plotting grad_norm.png")
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.plot(t_ids, gnorm, ".-", label="grad_norm")
    ax.set_xlabel("train step")
    ax.set_ylabel("grad_norm")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("grad_norm")
    fig_save("grad_norm.png")

    # kl_entropy
    print("[+] plotting kl_entropy.png")
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.plot(t_ids, ent, ".-", label="entropy_loss")
    ax.plot(t_ids, kl_loss, ".-", alpha=0.7, label="kl_loss")
    ax.set_xlabel("train step")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Entropy & KL")
    fig_save("kl_entropy.png")

    # overview
    print("[+] plotting overview.png")
    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    axs = axes.flatten()
    axs[0].plot(r_ids, raw_rew, ".-")
    axs[0].axhline(0, color="gray", ls=":")
    axs[0].set_title("raw_reward")
    axs[0].grid(alpha=0.3)
    xs, ys = _filter_positive(r_ids, resp_len)
    if ys:
        axs[1].semilogy(xs, ys, ".-")
    axs[1].set_title("response_length (log)")
    axs[1].grid(alpha=0.3, which="both")
    axs[2].plot(r_ids, trunc, ".-")
    axs[2].set_title("truncated_frac")
    axs[2].grid(alpha=0.3)
    axs[3].plot(t_ids, pg_loss, ".-")
    axs[3].set_title("pg_loss")
    axs[3].grid(alpha=0.3)
    axs[4].plot(t_ids, gnorm, ".-")
    axs[4].set_title("grad_norm")
    axs[4].grid(alpha=0.3)
    axs[5].plot(t_ids, ent, ".-")
    axs[5].set_title("entropy_loss")
    axs[5].grid(alpha=0.3)
    axs[6].plot(t_ids, kl_loss, ".-")
    axs[6].set_title("kl_loss")
    axs[6].grid(alpha=0.3)
    if status_counts:
        labels, sizes = zip(*[(k, v) for k, v in status_counts.items() if v > 0])
        axs[7].pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        axs[7].set_title(f"trajectory status (n={sum(sizes)})")
    if turn_counts:
        axs[8].hist(turn_counts, bins=range(0, max(turn_counts) + 2), alpha=0.7)
        mean_turns = sum(turn_counts) / len(turn_counts)
        axs[8].set_title(f"turns/trajectory (mean={mean_turns:.1f})")
        axs[8].set_xlabel("turns")
        axs[8].grid(alpha=0.3)
    if collapse is not None:
        for a in axs[:3]:
            a.axvline(collapse, color="red", ls="--", alpha=0.4)
    suptitle_parts = []
    if collapse is not None:
        suptitle_parts.append(f"collapse @ rollout {collapse}")
    if reset500_total:
        suptitle_parts.append(f"/reset 500: {reset500_total}")
    if clawsentry_total:
        suptitle_parts.append(f"ClawSentry errors: {clawsentry_total}")
    if suptitle_parts:
        fig.suptitle("Run overview — " + " | ".join(suptitle_parts), fontsize=13)
    fig_save("overview.png")


def _build_summary(
    parsed: dict[str, Any], collapse: int | None, run_name: str
) -> dict[str, Any]:
    rollout_metrics = parsed["rollout_metrics"]
    train_metrics = parsed["train_metrics"]
    clawsentry_errs = parsed["clawsentry_errs"]
    status_counts = parsed["status_counts"]
    turn_counts = parsed["turn_counts"]
    parse_errs = parsed["parse_errs"]
    reset500_per_min = parsed["reset500_per_min"]

    r_ids = sorted(rollout_metrics)
    t_ids = sorted(train_metrics)

    raw_rew = _get_series(rollout_metrics, r_ids, "rollout/raw_reward")
    rew = _get_series(rollout_metrics, r_ids, "rollout/rewards")
    trunc = _get_series(rollout_metrics, r_ids, "rollout/truncated")
    resp_len = _get_series(rollout_metrics, r_ids, "rollout/response_lengths")
    pg_loss = _get_series(train_metrics, t_ids, "train/pg_loss")
    kl_loss = _get_series(train_metrics, t_ids, "train/kl_loss")
    ent = _get_series(train_metrics, t_ids, "train/entropy_loss")
    gnorm = _get_series(train_metrics, t_ids, "train/grad_norm")
    lr = _get_series(train_metrics, t_ids, "train/lr-pg_0")

    trunc_nums = [t for t in trunc if isinstance(t, (int, float))]
    trunc_mean = sum(trunc_nums) / len(trunc_nums) if trunc_nums else None

    cs_total = sum(clawsentry_errs.values())
    if any("429" in k for k in clawsentry_errs):
        cs_status = "ALIVE_BUT_RATE_LIMITED"
    elif clawsentry_errs:
        cs_status = "OFFLINE"
    else:
        cs_status = "OK"

    summary = {
        "run_name": run_name,
        "n_rollouts_logged": len(r_ids),
        "max_rollout_id": int(max(r_ids)) if r_ids else None,
        "n_train_steps": len(t_ids),
        "max_train_step": int(max(t_ids)) if t_ids else None,
        "collapse_rollout": collapse,
        "trajectories_logged": sum(status_counts.values()),
        "status_counts": dict(status_counts),
        "raw_reward": _stats(raw_rew, "raw_rew"),
        "rewards_norm": _stats(rew, "rew"),
        "response_lengths": _stats(resp_len, "resp_len"),
        "truncated_frac_mean": trunc_mean,
        "train": {
            "pg_loss": _stats(pg_loss, "pg_loss"),
            "grad_norm": _stats(gnorm, "gnorm"),
            "kl_loss": _stats(kl_loss, "kl"),
            "entropy_loss": _stats(ent, "ent"),
            "lr_first": float(lr[0]) if lr and lr[0] is not None else None,
            "lr_last": float(lr[-1]) if lr and lr[-1] is not None else None,
        },
        "clawsentry": {
            "total_errors": cs_total,
            "error_breakdown": dict(clawsentry_errs),
            "status": cs_status,
        },
        "reset500": {
            "total": sum(reset500_per_min.values()),
            "max_per_minute": max(reset500_per_min.values()) if reset500_per_min else 0,
        },
        "turn_count_stats": (
            {
                "mean": sum(turn_counts) / len(turn_counts),
                "max": max(turn_counts),
                "median": sorted(turn_counts)[len(turn_counts) // 2],
            }
            if turn_counts
            else None
        ),
        "parse_error_total": int(sum(parse_errs)) if parse_errs else 0,
    }
    return summary


def plot_run(
    run_dir: Path,
    log_file: Path | None = None,
    out_dir: Path | None = None,
    no_figs: bool = False,
) -> dict[str, Any]:
    log_file = log_file or (run_dir / "logs" / "train.log")
    out_dir = out_dir or (run_dir / "metrics" / "analysis")
    if not log_file.is_file():
        raise FileNotFoundError(f"train log not found: {log_file}")
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = _parse_log(log_file)
    rollout_metrics = parsed["rollout_metrics"]
    train_metrics = parsed["train_metrics"]

    if not rollout_metrics and not train_metrics:
        print("[!] no rollouts or train steps parsed — empty log?")
        return {}

    print(
        f"  rollouts: {len(rollout_metrics)} "
        f"(max id: {max(rollout_metrics) if rollout_metrics else 'n/a'})"
    )
    print(
        f"  train steps: {len(train_metrics)} "
        f"(max id: {max(train_metrics) if train_metrics else 'n/a'})"
    )
    print(f"  trajectories logged: {sum(parsed['status_counts'].values())}")
    print(f"  status: {dict(parsed['status_counts'])}")
    print(f"  ClawSentry errors: {sum(parsed['clawsentry_errs'].values())}")
    print(f"  /reset 500 events:  {sum(parsed['reset500_per_min'].values())}")

    r_ids = sorted(rollout_metrics)
    resp_len = _get_series(rollout_metrics, r_ids, "rollout/response_lengths")
    collapse = _detect_collapse(r_ids, resp_len)
    print(f"  collapse rollout: {collapse}")

    summary = _build_summary(parsed, collapse, run_name=run_dir.name)
    json_path = out_dir / "summary_stats.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[+] wrote {json_path}")

    if not no_figs:
        _plot_all(
            parsed,
            out_dir=out_dir,
            collapse=collapse,
            reset500_total=sum(parsed["reset500_per_min"].values()),
            clawsentry_total=sum(parsed["clawsentry_errs"].values()),
        )

    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Run root, e.g. runs/<run_id>")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Override train log (default: <run_dir>/logs/train.log)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output dir (default: <run_dir>/metrics/analysis)")
    p.add_argument("--no-figs", action="store_true",
                   help="Only emit summary_stats.json, skip image generation")
    args = p.parse_args(argv)

    try:
        s = plot_run(
            run_dir=args.run_dir.resolve(),
            log_file=args.log_file.resolve() if args.log_file else None,
            out_dir=args.out_dir.resolve() if args.out_dir else None,
            no_figs=args.no_figs,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    if not s:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
