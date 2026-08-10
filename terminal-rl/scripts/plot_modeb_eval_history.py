#!/usr/bin/env python3
"""Plot recorded Terminal-Bench pass@1 with Wilson 95% intervals.

The point of the figure is the overlap. Reporting eight point estimates in a
table invites reading a 1.12% and a 3.00% as different; at n=267 with single
digit success counts they are not separable, and drawing the intervals is the
cheapest way to keep that in view.

    python terminal-rl/scripts/plot_modeb_eval_history.py \\
        --out terminal-rl/docs/assets/harbor_camel_mode_b

The eval table lives in EVALS below; add a row when a new full eval lands, and
keep it in step with the history table in docs/HARBOR_CAMEL_MODE_B_zh.md.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Two-hue categorical palette, checked for colorblind separation and for >=3:1
# contrast against each surface it is paired with.
THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  grid="#e3e2dc", mode_a="#eb6834", mode_b="#2a78d6"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 grid="#33322f", mode_a="#d95926", mode_b="#3987e5"),
}

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Eval:
    issue: str
    checkpoint: str
    dataset: str
    harness: str  # "A" for terminus-2, "B" for the camel adapter
    successes: int
    trials: int


# Every number below is taken from the corresponding issue body. TB 2.0 and
# TB 2.1 are different task sets; the figure labels each row with its dataset
# rather than pretending the rows are one series.
EVALS = [
    Eval("#21", "seta-agent57-i271", "TB 2.0", "A", 3, 267),
    Eval("#22", "Qwen3-8B base", "TB 2.0", "A", 8, 267),
    Eval("#24", "seta-agent57-i271", "TB 2.0", "B", 6, 267),
    Eval("#25", "Qwen3-8B base", "TB 2.0", "B", 3, 267),
    Eval("#27", "Qwen3-8B base", "TB 2.1", "B", 5, 267),
    Eval("#28", "RL outcome_gate i299", "TB 2.1", "B", 6, 267),
    Eval("#29", "SETA-DAPO mt10 i899", "TB 2.1", "B", 5, 267),
    Eval("#31", "SETA-DAPO mt10 i1099", "TB 2.1", "B", 3, 267),
]


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval. Reproduces the intervals published in #27/#28/#29."""
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    return (centre - margin) / denominator, (centre + margin) / denominator


def render(evals: list[Eval], mode: str, out_path: Path) -> None:
    theme = THEME[mode]
    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    rows = list(reversed(evals))  # first eval at the top
    upper_bounds = []
    for position, row in enumerate(rows):
        low, high = wilson_interval(row.successes, row.trials)
        upper_bounds.append(high * 100)
        colour = theme["mode_b"] if row.harness == "B" else theme["mode_a"]
        ax.plot([low * 100, high * 100], [position, position], color=colour, lw=2,
                solid_capstyle="round", zorder=2, alpha=0.85)
        ax.plot([row.successes / row.trials * 100], [position], "o", ms=9, color=colour,
                zorder=3, markeredgecolor=theme["surface"], markeredgewidth=2)
        ax.text(high * 100 + 0.35, position,
                f"{row.successes / row.trials * 100:.2f}%  ({row.successes}/{row.trials})",
                va="center", ha="left", fontsize=9, color=theme["secondary"])

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r.issue}  {r.checkpoint}\n{r.dataset}" for r in rows],
                       fontsize=9, color=theme["primary"], linespacing=1.5)
    ax.set_xlim(0, max(upper_bounds) + 2.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    ax.tick_params(axis="x", colors=theme["secondary"], labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=theme["grid"], lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    ax.set_title("Terminal-Bench pass@1, with Wilson 95% intervals", fontsize=13,
                 color=theme["primary"], pad=34, loc="left", fontweight="600")
    ax.text(0, 1.035,
            f"Every interval overlaps every other: no checkpoint is separable at n={rows[0].trials}.",
            transform=ax.transAxes, fontsize=9.5, color=theme["secondary"], va="bottom")

    handles = [
        plt.Line2D([], [], color=theme["mode_a"], lw=2, marker="o", ms=8,
                   markeredgecolor=theme["surface"], markeredgewidth=2,
                   label="mode A  (terminus-2)"),
        plt.Line2D([], [], color=theme["mode_b"], lw=2, marker="o", ms=8,
                   markeredgecolor=theme["surface"], markeredgewidth=2,
                   label="mode B  (camel adapter)"),
    ]
    legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09),
                       ncol=2, frameon=False, fontsize=9, columnspacing=3.0)
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--prefix", default="modeb_pass_at_1", help="output filename prefix")
    args = parser.parse_args(argv)

    for mode in ("light", "dark"):
        path = args.out / f"{args.prefix}_{mode}.png"
        render(EVALS, mode, path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
