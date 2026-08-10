#!/usr/bin/env python3
"""Plot the score and status distribution of a SETA-env eval run.

Reads the summary.json written by analyze_seta_env_eval.py and writes a light and
a dark PNG, so the figure can be regenerated for any run rather than being a
one-off image checked into the docs.

    python terminal-rl/scripts/plot_seta_env_eval.py \\
        --summary runs/<run>/final_analysis/summary.json \\
        --out terminal-rl/docs/assets/seta_env_eval \\
        --prefix seta_env_baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette; see terminal-rl/docs/SETA_ENV_EVAL_zh.md.
THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#8a887f", grid="#e3e2dc", accent="#2a78d6", highlight="#eb6834"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#8a887f", grid="#33322f", accent="#3987e5", highlight="#d95926"),
}

STATUS_ORDER = ["COMPLETED", "TRUNCATED", "FAILED", "MISSING"]


def render(summary: dict[str, Any], mode: str, out_path: Path) -> None:
    theme = THEME[mode]
    fig, (ax_scores, ax_status) = plt.subplots(
        1, 2, figsize=(11.2, 4.8), dpi=200, gridspec_kw={"width_ratios": [1.75, 1]}
    )
    fig.patch.set_facecolor(theme["surface"])

    distribution = summary["raw_score_distribution"]
    keys = sorted(distribution, key=float)
    counts = [distribution[key] for key in keys]
    scored = sum(counts)
    zero = distribution.get("0.0", 0)
    exact = distribution.get("1.0", 0)
    # missing_count is authoritative; it counts samples with no result at all,
    # which is not always dataset_total minus the scored histogram.
    missing = summary["missing_count"]

    ax_scores.set_facecolor(theme["surface"])
    bars = ax_scores.bar(range(len(keys)), counts, color=theme["accent"], width=0.72, zorder=2)
    # Highlight the two ends by value, not by position: a run whose distribution
    # happens to lack a 0.0 or 1.0 bucket must not get its lowest partial-credit
    # bar painted as "no credit".
    for bar, key in zip(bars, keys):
        if float(key) == 0.0:
            bar.set_color(theme["muted"])
        elif float(key) == 1.0:
            bar.set_color(theme["highlight"])
    for position, count in enumerate(counts):
        if count >= 40:
            ax_scores.text(position, count + 8, str(count), ha="center", va="bottom",
                           fontsize=8.5, color=theme["secondary"])
    ax_scores.set_xticks(range(len(keys)))
    ax_scores.set_xticklabels([f"{float(key):.2f}" for key in keys], fontsize=8,
                              color=theme["secondary"], rotation=45, ha="right")
    ax_scores.set_ylabel("samples", fontsize=9, color=theme["secondary"])
    ax_scores.set_title(f"raw_score distribution over the {scored} scored samples",
                        fontsize=12, color=theme["primary"], pad=30, loc="left", fontweight="600")
    ax_scores.text(
        0, 1.045,
        f"{zero} got no credit, {exact} passed every check, "
        f"{scored - zero - exact} landed in between; {missing} never produced a result.",
        transform=ax_scores.transAxes, fontsize=9, color=theme["secondary"], va="bottom",
    )

    status = summary["status_counts"]
    order = [name for name in STATUS_ORDER if name in status]
    order += sorted(name for name in status if name not in STATUS_ORDER)
    values = [status[name] for name in order]
    positions = list(range(len(order)))[::-1]

    ax_status.set_facecolor(theme["surface"])
    ax_status.barh(positions, values, color=theme["accent"], height=0.62, zorder=2)
    for position, value in zip(positions, values):
        ax_status.text(value + max(values) * 0.015, position, str(value), va="center",
                       ha="left", fontsize=9, color=theme["secondary"])
    ax_status.set_yticks(positions)
    ax_status.set_yticklabels(order, fontsize=9, color=theme["primary"])
    ax_status.set_xlim(0, max(values) * 1.22)
    ax_status.set_title("terminal status", fontsize=12, color=theme["primary"], pad=30,
                        loc="left", fontweight="600")
    ax_status.text(0, 1.045, "TRUNCATED is not failure: some score 1.0.",
                   transform=ax_status.transAxes, fontsize=9, color=theme["secondary"], va="bottom")

    for axis, grid_axis in ((ax_scores, "y"), (ax_status, "x")):
        axis.tick_params(colors=theme["secondary"], labelsize=8.5, length=0)
        axis.grid(axis=grid_axis, color=theme["grid"], lw=1, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            axis.spines[side].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True,
                        help="summary.json from analyze_seta_env_eval.py")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--prefix", default="seta_env_eval", help="output filename prefix")
    args = parser.parse_args(argv)

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    for mode in ("light", "dark"):
        path = args.out / f"{args.prefix}_{mode}.png"
        render(summary, mode, path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
