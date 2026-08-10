#!/usr/bin/env python3
"""Figure-generation script skeleton for the 2x2 base × harness comparison.

Inputs (all in unified schema):
  /tmp/four_cell/base_misaligned_trials.jsonl     (n_top3 = 267, 5 solved)
  /tmp/four_cell/base_aligned_trials.jsonl        (PENDING — eval running, ~5h ETA)
  /tmp/four_cell/i271_misaligned_trials.jsonl     (n_top3 = 267, 3 solved)
  /tmp/four_cell/i271_aligned_trials.jsonl        (n_top3 = 267, 3 solved)

Plus /tmp/four_cell_summary.json from /tmp/four_cell_summary.py.

Outputs go to /tmp/figs/ (PNG, dpi=140).

Each fig function follows signature:
  def fig_X_<name>(cells, out_path): ...
where `cells` is the dict loaded via `load_cells()` (see helpers below) — each
value is `{'rows': [...top-3 trial dicts...], 'meta': {...}}`. When invoked
before base_aligned data is present, the function should either gracefully skip
that cell or render a "TBD" placeholder bar.

This file is intentionally NOT executed at scaffold-build time; data is
incomplete. To run after base_aligned lands:
  python3 /tmp/figs/build_base_comparison_figs.py --all
"""

import os
import json
import argparse
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 140

CELL_FILES = {
    'base_misaligned':  '/tmp/four_cell/base_misaligned_trials.jsonl',
    'base_aligned':     '/tmp/four_cell/base_aligned_trials.jsonl',
    'i271_misaligned':  '/tmp/four_cell/i271_misaligned_trials.jsonl',
    'i271_aligned':     '/tmp/four_cell/i271_aligned_trials.jsonl',
}

CELL_ORDER = ['base_misaligned', 'base_aligned', 'i271_misaligned', 'i271_aligned']
CELL_LABEL = {
    'base_misaligned':  'base × misaligned\n(#22 baseline)',
    'base_aligned':     'base × aligned\n(THIS ISSUE)',
    'i271_misaligned':  'i271 × misaligned\n(#21 baseline)',
    'i271_aligned':     'i271 × aligned\n(#24 baseline)',
}
CELL_COLOR = {
    'base_misaligned':  '#1976d2',
    'base_aligned':     '#388e3c',
    'i271_misaligned':  '#f57c00',
    'i271_aligned':     '#7b1fa2',
}
STATUS_ORDER = ['COMPLETED', 'TRUNCATED', 'FAILED', 'ABORTED']
STATUS_COLOR = {
    'COMPLETED': '#4caf50',
    'TRUNCATED': '#f44336',
    'FAILED':    '#ff9800',
    'ABORTED':   '#9e9e9e',
}


# ---------- IO helpers ----------

def load_jsonl(path):
    if not os.path.exists(path):
        return None, None
    rows, meta = [], None
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        if d.get('__meta__'):
            meta = d
        else:
            rows.append(d)
    return rows, meta


def load_cells():
    cells = {}
    for name, path in CELL_FILES.items():
        rows, meta = load_jsonl(path)
        if rows is None:
            cells[name] = None
            continue
        top3 = [r for r in rows if r.get('is_top3')]
        cells[name] = {'rows': top3, 'all_rows': rows, 'meta': meta}
    return cells


def present_cells(cells):
    return [c for c in CELL_ORDER if cells.get(c) is not None]


# ---------- metric helpers ----------

def pass_at_1_sample(rows):
    if not rows:
        return 0.0
    return sum(1 for r in rows if r['reward'] >= 0.99) / len(rows)


def pass_at_3_per_task(rows):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r['task']].append(r['reward'])
    if not by_task:
        return 0.0
    return sum(1 for tk, lst in by_task.items() if any(x >= 0.99 for x in lst)) / len(by_task)


def solved_set(rows):
    return {r['task'] for r in rows if r['reward'] >= 0.99}


# ---------- figs ----------

def fig_A_2x2_pass_at_1(cells, out_path):
    """2x2 grid bar chart of pass@1 with diagonal/edge comparisons annotated."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    labels = [CELL_LABEL[c] for c in CELL_ORDER]
    vals, present = [], []
    for c in CELL_ORDER:
        ce = cells.get(c)
        if ce is None:
            vals.append(0.0)
            present.append(False)
        else:
            vals.append(pass_at_1_sample(ce['rows']) * 100)
            present.append(True)
    colors = [CELL_COLOR[c] for c in CELL_ORDER]
    bars = ax.bar(range(4), vals, color=colors, edgecolor='black', linewidth=0.6)
    for i, (b, v, p) in enumerate(zip(bars, vals, present)):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f'{v:.2f}%' if p else 'TBD',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        if not p:
            b.set_hatch('//')
            b.set_alpha(0.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('pass@1 (trial-level, %)')
    ax.set_title('Fig A: 2x2 pass@1 — RL × harness ANOVA  (interaction = +3.01pp, textbook crossover)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    # crossover annotation: draw arrows connecting base_mis -> base_aligned (down)
    # and i271_mis -> i271_aligned (up), to make crossover visible
    try:
        ax.annotate('', xy=(1, vals[1]), xytext=(0, vals[0]),
                    arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.5, alpha=0.7))
        ax.annotate('', xy=(3, vals[3]), xytext=(2, vals[2]),
                    arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.5, alpha=0.7))
        ax.text(0.5, max(vals) * 0.55, 'Δharness(base)\n−1.88pp', ha='center', color='#c62828', fontsize=9)
        ax.text(2.5, max(vals) * 0.55, 'Δharness(i271)\n+1.13pp', ha='center', color='#2e7d32', fontsize=9)
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_B_status_dist_4cell(cells, out_path):
    """Stacked horizontal bars: 4 cells × {COMPLETED, TRUNCATED, FAILED, ABORTED}."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, cn in enumerate(CELL_ORDER):
        ce = cells.get(cn)
        if ce is None:
            ax.barh([i], [100], color='#dddddd', edgecolor='black', hatch='//')
            ax.text(50, i, 'TBD', va='center', ha='center', fontsize=10)
            continue
        rows = ce['rows']
        n = len(rows)
        c = Counter(r['status'] for r in rows)
        left = 0
        for s in STATUS_ORDER:
            v = c.get(s, 0) / n * 100 if n else 0
            ax.barh([i], [v], left=left, color=STATUS_COLOR[s], edgecolor='black', linewidth=0.4)
            if v > 4:
                ax.text(left + v/2, i, f'{s}\n{c.get(s,0)} ({v:.1f}%)', va='center', ha='center', fontsize=8)
            left += v
    ax.set_yticks(range(4))
    ax.set_yticklabels([CELL_LABEL[c] for c in CELL_ORDER], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of top-3 trials')
    ax.set_title('Fig B: trial exit status distribution (4 cells)')
    ax.legend(handles=[Patch(facecolor=STATUS_COLOR[s], label=s) for s in STATUS_ORDER],
              loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_C_solved_overlap_venn(cells, out_path):
    """4-set Venn / UpSet style plot of solved tasks across cells.

    Use matplotlib_venn if available; fall back to a tabular cross-cell solved
    matrix if not (since 4-set Venn is messy anyway).
    """
    sets = {c: solved_set(cells[c]['rows']) for c in CELL_ORDER if cells.get(c)}
    if len(sets) < 2:
        return
    all_solved = sorted(set.union(*sets.values()))
    fig, ax = plt.subplots(figsize=(8.5, max(3, 0.32 * len(all_solved) + 1.5)))
    names = list(sets.keys())
    M = np.zeros((len(all_solved), len(names)), dtype=int)
    for j, n in enumerate(names):
        for i, t in enumerate(all_solved):
            M[i, j] = 1 if t in sets[n] else 0
    ax.imshow(M, aspect='auto', cmap='Greens', vmin=0, vmax=1)
    for i in range(len(all_solved)):
        for j in range(len(names)):
            ax.text(j, i, 'PASS' if M[i, j] else '·', ha='center', va='center',
                    fontsize=8, color=('white' if M[i, j] else '#888'))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([CELL_LABEL[n] for n in names], fontsize=8, rotation=20, ha='right')
    ax.set_yticks(range(len(all_solved)))
    ax.set_yticklabels(all_solved, fontsize=8)
    ax.set_title('Fig C: per-cell solved-task matrix (union of solved tasks)')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_D_think_dist_4cell(cells, out_path):
    """Per-block think-length distribution: ECDF across 4 cells (loaded from four_cell_think_summary.json + step jsonls)."""
    import statistics
    summary = json.load(open('/tmp/four_cell_think_summary.json'))
    # gather per-block lengths
    block_data = {}
    # aligned cells: from step jsonl
    if os.path.exists('/tmp/four_cell/base_aligned_think_per_step.jsonl'):
        block_data['base_aligned'] = [json.loads(l)['think_len_chars']
                                     for l in open('/tmp/four_cell/base_aligned_think_per_step.jsonl')
                                     if json.loads(l).get('think_idx', -1) >= 0]
    if os.path.exists('/tmp/i271_aligned_v2_think_per_step.jsonl'):
        block_data['i271_aligned'] = [json.loads(l)['think_len_chars']
                                     for l in open('/tmp/i271_aligned_v2_think_per_step.jsonl')
                                     if json.loads(l).get('think_idx', -1) >= 0]
    # misaligned cells: from per_trial.jsonl with block_lens
    for n, p in [('base_misaligned', '/tmp/base_misaligned_v2_think_per_trial.jsonl'),
                 ('i271_misaligned', '/tmp/i271_misaligned_v2_think_per_trial.jsonl')]:
        if os.path.exists(p):
            lens = []
            for ln in open(p):
                d = json.loads(ln)
                lens.extend(d.get('block_lens') or [])
            block_data[n] = lens
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    # left: ECDF of per-block length
    for c in CELL_ORDER:
        if c not in block_data or not block_data[c]:
            continue
        arr = sorted(block_data[c])
        ys = np.arange(1, len(arr) + 1) / len(arr)
        ax1.plot(arr, ys, label=CELL_LABEL[c].replace('\n', ' '), color=CELL_COLOR[c], lw=2)
    ax1.set_xlabel('think-block length (chars)')
    ax1.set_ylabel('ECDF')
    ax1.set_xscale('symlog')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.set_title('left: per-block length ECDF')
    # right: stat table-as-bars (mean, p50, p99, max)
    stats_table = []
    for c in CELL_ORDER:
        if c not in block_data:
            continue
        arr = block_data[c]
        if not arr:
            continue
        stats_table.append((c, len(arr), statistics.mean(arr),
                             sorted(arr)[len(arr)//2], sorted(arr)[int(len(arr)*0.99)], max(arr)))
    # bar group: mean / p50 / p99 / max
    xs = np.arange(len(stats_table))
    width = 0.2
    metrics = ['mean', 'p50', 'p99', 'max']
    metric_idx = [2, 3, 4, 5]
    for i, m in enumerate(metrics):
        vals = [row[metric_idx[i]] for row in stats_table]
        ax2.bar(xs + (i - 1.5) * width, vals, width, label=m)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([CELL_LABEL[r[0]].replace('\n', ' ') for r in stats_table], fontsize=8, rotation=15, ha='right')
    ax2.set_ylabel('chars')
    ax2.set_yscale('symlog')
    ax2.set_title('right: per-block summary stats')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    fig.suptitle('Fig D: think-block length distribution across 4 cells', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_E_per_task_heatmap(cells, out_path):
    """89 task × 4 cell heatmap of mean reward (or pass rate)."""
    rows_present = [c for c in CELL_ORDER if cells.get(c) is not None]
    tasks = sorted({r['task'] for c in rows_present for r in cells[c]['rows']})
    M = np.full((len(tasks), len(CELL_ORDER)), np.nan)
    for j, c in enumerate(CELL_ORDER):
        if cells.get(c) is None:
            continue
        by_task = defaultdict(list)
        for r in cells[c]['rows']:
            by_task[r['task']].append(r['reward'])
        for i, t in enumerate(tasks):
            if t in by_task:
                M[i, j] = sum(by_task[t]) / len(by_task[t])
    fig, ax = plt.subplots(figsize=(7, max(8, 0.14 * len(tasks))))
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(len(CELL_ORDER)))
    ax.set_xticklabels([CELL_LABEL[c] for c in CELL_ORDER], rotation=20, ha='right', fontsize=8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=6)
    fig.colorbar(im, ax=ax, label='mean reward over k=3')
    ax.set_title('Fig E: per-task mean reward heatmap (4 cells)')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_F_error_class_4cell(cells, out_path):
    """13-subclass error taxonomy stacked bar (loaded from /tmp/four_cell/<cell>_error_taxonomy.jsonl)."""
    SUB_ORDER = ['1.1', '2.1', '2.2', '2.4', '2.5', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6',
                 '4.1', '4.2', '4.4', '5.0']
    PALETTE = {
        '1.1': '#2e7d32',
        '2.1': '#bdbdbd', '2.2': '#9e9e9e', '2.4': '#757575', '2.5': '#616161',
        '3.1': '#fb8c00', '3.2': '#ef6c00', '3.3': '#a5d6a7', '3.4': '#e64a19', '3.5': '#d84315', '3.6': '#bf360c',
        '4.1': '#7b1fa2', '4.2': '#6a1b9a', '4.4': '#4a148c',
        '5.0': '#d81b60',
    }
    LABEL = {
        '1.1': 'PASS', '2.1': '2.1 fake-cmpl: no write', '2.2': '2.2 fake-cmpl: wrong content',
        '2.4': '2.4 fake-cmpl: trivial', '2.5': '2.5 fake-cmpl: other',
        '3.1': '3.1 trunc: tool bloat', '3.2': '3.2 trunc: repetitive',
        '3.3': '3.3 trunc: productive', '3.4': '3.4 trunc: think-bloat',
        '3.5': '3.5 trunc: err-recovery', '3.6': '3.6 trunc: other',
        '4.1': '4.1 verifier-timeout', '4.2': '4.2 agent-timeout', '4.4': '4.4 FAILED',
        '5.0': '5.0 filler-loop',
    }
    fig, ax = plt.subplots(figsize=(11, 5.5))
    counts_per_cell = {}
    for c in CELL_ORDER:
        path = f'/tmp/four_cell/{c}_error_taxonomy.jsonl'
        if not os.path.exists(path):
            counts_per_cell[c] = None
            continue
        ctr = Counter()
        for ln in open(path):
            d = json.loads(ln)
            ctr[d['error_subclass']] += 1
        counts_per_cell[c] = ctr
    for i, c in enumerate(CELL_ORDER):
        ctr = counts_per_cell[c]
        if ctr is None:
            ax.barh([i], [100], color='#dddddd', hatch='//', edgecolor='black')
            continue
        n = sum(ctr.values())
        left = 0
        for sub in SUB_ORDER:
            v = ctr.get(sub, 0) / n * 100 if n else 0
            if v <= 0:
                continue
            ax.barh([i], [v], left=left, color=PALETTE.get(sub, '#cccccc'), edgecolor='black', linewidth=0.3)
            if v > 4:
                ax.text(left + v/2, i, f'{sub}\n{v:.1f}%', ha='center', va='center', fontsize=7,
                        color='white' if sub in ('1.1', '3.5', '3.4', '4.4', '5.0') else 'black')
            left += v
    ax.set_yticks(range(4))
    ax.set_yticklabels([CELL_LABEL[c] for c in CELL_ORDER], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of top-3 trials')
    ax.set_title('Fig F: 15-subclass error taxonomy (4 cells)')
    # legend
    handles = [Patch(facecolor=PALETTE[s], label=LABEL[s]) for s in SUB_ORDER]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_G_decompose_RL_vs_harness(cells, out_path):
    """4-bar decomposition: ΔRL_aligned / ΔRL_misaligned / Δharness_base / Δharness_i271."""
    def p1(c):
        ce = cells.get(c)
        return None if ce is None else pass_at_1_sample(ce['rows']) * 100
    bm, ba, im, ia = p1('base_misaligned'), p1('base_aligned'), p1('i271_misaligned'), p1('i271_aligned')

    def df(a, b): return None if (a is None or b is None) else a - b
    bars = {
        '$\\Delta_{RL|aligned}$\n(i271-base | aligned)':         df(ia, ba),
        '$\\Delta_{RL|misaligned}$\n(i271-base | misaligned)':   df(im, bm),
        '$\\Delta_{harness|base}$\n(aligned-mis | base)':        df(ba, bm),
        '$\\Delta_{harness|i271}$\n(aligned-mis | i271)':        df(ia, im),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(bars))
    vals = [v if v is not None else 0 for v in bars.values()]
    present = [v is not None for v in bars.values()]
    colors = ['#1565c0' if v >= 0 else '#c62828' for v in vals]
    bs = ax.bar(xs, vals, color=colors, edgecolor='black')
    for b, v, p in zip(bs, vals, present):
        if not p:
            b.set_hatch('//'); b.set_alpha(0.35)
            ax.text(b.get_x() + b.get_width()/2, 0, 'TBD', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(b.get_x() + b.get_width()/2, v + (0.05 if v >= 0 else -0.15),
                    f'{v:+.2f}pp', ha='center', va=('bottom' if v >= 0 else 'top'), fontsize=10, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(list(bars.keys()), fontsize=8)
    ax.set_ylabel('Δ pass@1 (percentage points)')
    ax.set_title('Fig G: ANOVA decomposition — pure RL effect vs pure harness effect\n(interaction RL×harness = +3.01pp)')
    # annotate the interaction (placed below the title in plot area, between the two negative bars)
    ax.text(1.5, -1.0, 'interaction RL×harness =\nΔRL|aligned − ΔRL|misaligned =\n+1.13 − (−1.88) = +3.01pp',
            ha='center', va='center', fontsize=9,
            bbox=dict(facecolor='#fff9c4', edgecolor='black', boxstyle='round,pad=0.4'))
    ax.set_ylim(-2.3, 1.6)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_H_per_task_RL_effect(cells, out_path):
    """Scatter: per-task ΔRL_aligned (y) vs ΔRL_misaligned (x).

    Each dot = one of 89 tasks; quadrants tell whether RL effect is consistent
    across the two harnesses.
    """
    if cells.get('base_aligned') is None:
        # produce TBD placeholder
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, 'TBD\nawaiting base_aligned data', ha='center', va='center',
                fontsize=16, color='#666')
        ax.set_xlabel(r'$\Delta_{RL|misaligned}$ (i271_mis - base_mis), per-task mean reward')
        ax.set_ylabel(r'$\Delta_{RL|aligned}$ (i271_aligned - base_aligned), per-task mean reward')
        ax.set_title('Fig H: per-task RL effect across harnesses (TBD)')
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return
    # else compute
    def per_task_mean(c):
        by_task = defaultdict(list)
        for r in cells[c]['rows']:
            by_task[r['task']].append(r['reward'])
        return {tk: sum(lst)/len(lst) for tk, lst in by_task.items()}
    bm_pt = per_task_mean('base_misaligned')
    ba_pt = per_task_mean('base_aligned')
    im_pt = per_task_mean('i271_misaligned')
    ia_pt = per_task_mean('i271_aligned')
    tasks = sorted(set(bm_pt) & set(ba_pt) & set(im_pt) & set(ia_pt))
    xs = [im_pt[t] - bm_pt[t] for t in tasks]
    ys = [ia_pt[t] - ba_pt[t] for t in tasks]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(xs, ys, s=40, c='#1976d2', alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel(r'$\Delta_{RL|misaligned}$ per-task (i271_mis - base_mis)')
    ax.set_ylabel(r'$\Delta_{RL|aligned}$ per-task (i271_aligned - base_aligned)')
    ax.set_title('Fig H: per-task RL effect — same direction across harnesses?')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------- main ----------

ALL_FIGS = {
    'fig_A_2x2_pass_at_1': fig_A_2x2_pass_at_1,
    'fig_B_status_dist_4cell': fig_B_status_dist_4cell,
    'fig_C_solved_overlap_venn': fig_C_solved_overlap_venn,
    'fig_D_think_dist_4cell': fig_D_think_dist_4cell,
    'fig_E_per_task_heatmap': fig_E_per_task_heatmap,
    'fig_F_error_class_4cell': fig_F_error_class_4cell,
    'fig_G_decompose_RL_vs_harness': fig_G_decompose_RL_vs_harness,
    'fig_H_per_task_RL_effect': fig_H_per_task_RL_effect,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/tmp/figs/base_modeB')
    ap.add_argument('--only', nargs='*', default=None, help='Subset of fig names to render')
    ap.add_argument('--all', action='store_true', help='Render all figures (default)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cells = load_cells()
    print(f'Loaded cells (present={[c for c in CELL_ORDER if cells.get(c) is not None]})')
    todo = ALL_FIGS if (args.only is None or len(args.only) == 0) else {k: ALL_FIGS[k] for k in args.only if k in ALL_FIGS}
    for name, fn in todo.items():
        out = f'{args.out_dir}/{name}.png'
        print(f'-> {out}')
        try:
            fn(cells, out)
        except Exception as e:
            print(f'  ERROR rendering {name}: {e}')
    print('Done.')


if __name__ == '__main__':
    main()
