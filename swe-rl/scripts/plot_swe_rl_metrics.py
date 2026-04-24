#!/usr/bin/env python3
"""Extract core metrics from SWE-RL train.log and plot training curves.

SWE-RL logs have two metric sources:
  - train data:   model.py:<line> - step <N>: {dict}
  - rollout data:  data.py:<line> - rollout <N>: {dict}
  - perf data:    rollout.py:<line> - perf <N>: {dict}

Usage:
    python plot_swe_rl_metrics.py /path/to/train.log
    python plot_swe_rl_metrics.py /path/to/train.log -o /path/to/output.png
    python plot_swe_rl_metrics.py /path/to/train.log --title "My Custom Title"
"""

import argparse
import ast
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Regex patterns ────────────────────────────────────────────────
TRAIN_RE = re.compile(r"model\.py:\d+ - step \d+: (\{.+\})")
ROLLOUT_DATA_RE = re.compile(r"data\.py:\d+ - rollout \d+: (\{.+\})")
PERF_RE = re.compile(r"rollout\.py:\d+ - perf \d+: (\{.+\})")
RUN_NAME_RE = re.compile(r"SWE-RL Run:\s*(\S+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract core metrics from SWE-RL train.log and plot training curves."
    )
    parser.add_argument(
        "log_path",
        help="Path to the train.log file",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PNG path (default: training_metrics.png next to log file)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom chart title (default: auto-extracted from log header)",
    )
    return parser.parse_args()


def extract_run_name(log_path):
    """Try to extract run name from the log file header (first 20 lines)."""
    try:
        with open(log_path) as f:
            for _ in range(20):
                line = f.readline()
                m = RUN_NAME_RE.search(line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return None


def parse_log(log_path):
    """Parse train, rollout, and perf metrics from a SWE-RL train.log file."""
    # Train metrics
    train_steps, losses, pg_losses, kl_losses = [], [], [], []
    ppo_kls, grad_norms, clipfracs, entropy_losses = [], [], [], []

    # Rollout metrics (from data.py)
    rollout_ids, raw_rewards, advantages, returns = [], [], [], []
    resp_lengths, log_probs, ref_log_probs, truncated = [], [], [], []

    # Perf metrics (from rollout.py)
    perf_ids, resp_len_means, rollout_times = [], [], []
    tokens_per_gpu, truncated_ratios = [], []

    with open(log_path) as f:
        for line in f:
            # Train step
            m = TRAIN_RE.search(line)
            if m:
                d = ast.literal_eval(m.group(1))
                train_steps.append(d["train/step"])
                losses.append(d["train/loss"])
                pg_losses.append(d["train/pg_loss"])
                kl_losses.append(d["train/kl_loss"])
                ppo_kls.append(d["train/ppo_kl"])
                grad_norms.append(d["train/grad_norm"])
                clipfracs.append(d["train/pg_clipfrac"])
                entropy_losses.append(d["train/entropy_loss"])
                continue

            # Rollout data
            m = ROLLOUT_DATA_RE.search(line)
            if m:
                d = ast.literal_eval(m.group(1))
                rollout_ids.append(len(rollout_ids))
                raw_rewards.append(d.get("rollout/raw_reward", 0))
                advantages.append(d.get("rollout/advantages", 0))
                returns.append(d.get("rollout/returns", 0))
                resp_lengths.append(d.get("rollout/response_lengths", 0))
                log_probs.append(d.get("rollout/log_probs", 0))
                ref_log_probs.append(d.get("rollout/ref_log_probs", 0))
                truncated.append(d.get("rollout/truncated", 0))
                continue

            # Perf data
            m = PERF_RE.search(line)
            if m:
                d = ast.literal_eval(m.group(1))
                perf_ids.append(len(perf_ids))
                resp_len_means.append(d.get("rollout/response_len/mean", 0))
                rollout_times.append(d.get("perf/rollout_time", 0))
                tokens_per_gpu.append(d.get("perf/tokens_per_gpu_per_sec", 0))
                truncated_ratios.append(d.get("rollout/truncated_ratio", 0))
                continue

    train_data = dict(
        steps=train_steps, losses=losses, pg_losses=pg_losses,
        kl_losses=kl_losses, ppo_kls=ppo_kls, grad_norms=grad_norms,
        clipfracs=clipfracs, entropy_losses=entropy_losses,
    )
    rollout_data = dict(
        ids=rollout_ids, raw_rewards=raw_rewards, advantages=advantages,
        returns=returns, resp_lengths=resp_lengths,
        log_probs=log_probs, ref_log_probs=ref_log_probs,
        truncated=truncated,
    )
    perf_data = dict(
        ids=perf_ids, resp_len_means=resp_len_means,
        rollout_times=rollout_times, tokens_per_gpu=tokens_per_gpu,
        truncated_ratios=truncated_ratios,
    )
    return train_data, rollout_data, perf_data


def build_title(log_path, train_data, rollout_data, custom_title=None):
    """Build the suptitle for the figure."""
    if custom_title:
        return custom_title
    run_name = extract_run_name(log_path)
    n_train = len(train_data["steps"])
    n_rollout = len(rollout_data["ids"])
    if run_name:
        return f"SWE-RL Training Metrics — {run_name}\n({n_train} train steps, {n_rollout} rollouts)"
    return f"SWE-RL Training Metrics  ({n_train} train steps, {n_rollout} rollouts)"


def plot_metrics(train_data, rollout_data, perf_data, title, out_path):
    """Create 4x2 = 8-panel training metrics figure and save to disk."""
    t = train_data
    r = rollout_data
    p = perf_data

    fig, axes = plt.subplots(4, 2, figsize=(16, 18), dpi=120)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # ── Row 0 ────────────────────────────────────────────────────
    # (0,0) Raw Reward
    ax = axes[0, 0]
    ax.plot(r["ids"], r["raw_rewards"], color="tab:blue", linewidth=1.2, marker="o", markersize=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.set_title("Rollout Raw Reward")
    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("raw_reward")
    ax.grid(True, alpha=0.3)

    # (0,1) Train Loss + PG Loss
    ax = axes[0, 1]
    ax.plot(t["steps"], t["losses"], label="loss", linewidth=1.2)
    ax.plot(t["steps"], t["pg_losses"], label="pg_loss", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.set_title("Train Loss & PG Loss")
    ax.set_xlabel("Train Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 1 ────────────────────────────────────────────────────
    # (1,0) PPO KL Divergence
    ax = axes[1, 0]
    ax.plot(t["steps"], t["ppo_kls"], color="tab:red", linewidth=1.2)
    ax.set_title("PPO KL Divergence")
    ax.set_xlabel("Train Step")
    ax.set_ylabel("ppo_kl")
    ax.grid(True, alpha=0.3)

    # (1,1) KL Loss (weighted)
    ax = axes[1, 1]
    ax.plot(t["steps"], t["kl_losses"], color="tab:orange", linewidth=1.2)
    ax.set_title("KL Loss (weighted)")
    ax.set_xlabel("Train Step")
    ax.set_ylabel("kl_loss")
    ax.grid(True, alpha=0.3)

    # ── Row 2 ────────────────────────────────────────────────────
    # (2,0) PG Clip Fraction
    ax = axes[2, 0]
    ax.plot(t["steps"], t["clipfracs"], color="tab:purple", linewidth=1.2)
    ax.set_title("PG Clip Fraction")
    ax.set_xlabel("Train Step")
    ax.set_ylabel("pg_clipfrac")
    ax.grid(True, alpha=0.3)

    # (2,1) Entropy Loss
    ax = axes[2, 1]
    ax.plot(t["steps"], t["entropy_losses"], color="tab:cyan", linewidth=1.2)
    ax.set_title("Entropy Loss")
    ax.set_xlabel("Train Step")
    ax.set_ylabel("entropy_loss")
    ax.grid(True, alpha=0.3)

    # ── Row 3 ────────────────────────────────────────────────────
    # (3,0) Response Length & Grad Norm (dual Y-axis)
    ax = axes[3, 0]
    ax2 = ax.twinx()
    ln1 = ax.plot(r["ids"], r["resp_lengths"], color="tab:green", linewidth=1.2, label="resp_len (rollout)")
    ln2 = ax2.plot(t["steps"], t["grad_norms"], color="tab:brown", linewidth=0.8, alpha=0.7, label="grad_norm")
    ax.set_title("Response Length & Grad Norm")
    ax.set_xlabel("Step")
    ax.set_ylabel("Response Length", color="tab:green")
    ax2.set_ylabel("Grad Norm", color="tab:brown")
    lns = ln1 + ln2
    ax.legend(lns, [l.get_label() for l in lns], fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # (3,1) Log Probs vs Ref Log Probs
    ax = axes[3, 1]
    ax.plot(r["ids"], r["log_probs"], label="log_probs", linewidth=1.2, color="tab:blue")
    ax.plot(r["ids"], r["ref_log_probs"], label="ref_log_probs", linewidth=1.0, alpha=0.7, color="tab:red")
    ax.set_title("Log Probs vs Ref Log Probs")
    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("Log Probability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


def main():
    args = parse_args()

    log_path = args.log_path
    if not os.path.isfile(log_path):
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(os.path.dirname(log_path), "training_metrics.png")
        out_path = os.path.normpath(out_path)

    # Parse
    train_data, rollout_data, perf_data = parse_log(log_path)
    print(f"Parsed {len(train_data['steps'])} train steps, "
          f"{len(rollout_data['ids'])} rollout steps, "
          f"{len(perf_data['ids'])} perf entries")

    if not train_data["steps"] and not rollout_data["ids"]:
        print("Error: no metrics found in log file.", file=sys.stderr)
        sys.exit(1)

    # Build title & plot
    title = build_title(log_path, train_data, rollout_data, args.title)
    plot_metrics(train_data, rollout_data, perf_data, title, out_path)


if __name__ == "__main__":
    main()
