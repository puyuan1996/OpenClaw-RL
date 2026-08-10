#!/usr/bin/env python3
"""Comparison figures for i271 × TB 2.0 mode B aligned eval vs prior misaligned eval (issue #21).

Inputs:
  aligned_jsonl   = mode B aligned eval JSONL (same schema as
                    /tmp/i271_misaligned_v2_trials.jsonl). To be produced by
                    Phase 4/5 after the camel-agent eval finishes.
  misaligned_jsonl = /tmp/i271_misaligned_v2_trials.jsonl (already built).

Schema for each line: {task, k_idx, is_top3, reward, status, n_steps, n_filler,
first_filler_step, critical_prompt_tokens, trajectory_path, ...}; final line has
__meta__=True with aggregate stats.

All figures use is_top3=True rows only, so denominator is 267 per arm.
Each function signature: fn(aligned_path: str, misaligned_path: str, out_path: str) -> None.
"""

import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 140


# --- IO ---

def load_trials(path):
    rows = []
    meta = None
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            if d.get('__meta__'):
                meta = d
            else:
                rows.append(d)
    return rows, meta


def top3(rows):
    return [r for r in rows if r.get('is_top3')]


# --- metrics ---

def compute_pass_at_k(rows, k=1):
    """Sample-level pass@1 = (#trials with reward>=1) / n_trials. pass@3 = #tasks with >=1 success / n_tasks."""
    t = top3(rows)
    if not t:
        return 0.0
    if k == 1:
        return sum(1 for r in t if r['reward'] >= 1.0) / len(t)
    by_task = defaultdict(list)
    for r in t:
        by_task[r['task']].append(r['reward'])
    return sum(1 for tk, lst in by_task.items() if any(x >= 1.0 for x in lst)) / len(by_task)


STATUS_ORDER = ['COMPLETED', 'TRUNCATED', 'FAILED', 'ABORTED']
STATUS_COLOR = {
    'COMPLETED': '#4caf50',
    'TRUNCATED': '#f44336',
    'FAILED': '#ff9800',
    'ABORTED': '#9e9e9e',
}


def status_counts(rows):
    c = Counter(r['status'] for r in top3(rows))
    return {s: c.get(s, 0) for s in STATUS_ORDER}


def status_pct(rows):
    sc = status_counts(rows)
    total = sum(sc.values())
    if total == 0:
        return {s: 0.0 for s in STATUS_ORDER}
    return {s: sc[s] / total * 100.0 for s in STATUS_ORDER}


def solved_tasks(rows):
    out = set()
    for r in top3(rows):
        if r['reward'] >= 1.0:
            out.add(r['task'])
    return out


# --- figures ---

def fig_a_status_dist_aligned_vs_misaligned(aligned_path, misaligned_path, out_path):
    """Stacked or side-by-side bars of {COMPLETED,TRUNCATED,FAILED,ABORTED} counts/% for the two arms."""
    a, _ = load_trials(aligned_path)
    m, _ = load_trials(misaligned_path)
    a_pct = status_pct(a)
    m_pct = status_pct(m)
    a_cnt = status_counts(a)
    m_cnt = status_counts(m)

    labels = STATUS_ORDER
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars_m = ax.bar(x - width / 2, [m_pct[s] for s in labels], width, label='misaligned (terminus-2)',
                    color=[STATUS_COLOR[s] for s in labels], edgecolor='black', linewidth=0.5)
    bars_a = ax.bar(x + width / 2, [a_pct[s] for s in labels], width, label='mode B aligned (camel-agent)',
                    color=[STATUS_COLOR[s] for s in labels], edgecolor='black', linewidth=0.5, hatch='//')
    for b, s in zip(bars_m, labels):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.5, f"{m_cnt[s]}\n({h:.1f}%)", ha='center', va='bottom', fontsize=8)
    for b, s in zip(bars_a, labels):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.5, f"{a_cnt[s]}\n({h:.1f}%)", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('% of top-3 trials')
    ax.set_title('Status distribution: misaligned vs mode B aligned (n=267 top-3 each arm)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(m_pct.values()), max(a_pct.values())) * 1.25 + 5)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_b_pass_at_k_per_task_scatter(aligned_path, misaligned_path, out_path):
    """Scatter of per-task mean reward in misaligned (x) vs aligned (y), jittered, with diagonal."""
    a, _ = load_trials(aligned_path)
    m, _ = load_trials(misaligned_path)
    a_by = defaultdict(list)
    m_by = defaultdict(list)
    for r in top3(a): a_by[r['task']].append(r['reward'])
    for r in top3(m): m_by[r['task']].append(r['reward'])
    tasks = sorted(set(a_by) | set(m_by))
    rng = np.random.default_rng(0)
    xs, ys, names = [], [], []
    raw_xs, raw_ys = [], []
    for tk in tasks:
        mm = float(np.mean(m_by.get(tk, [0.0]))) if m_by.get(tk) else 0.0
        aa = float(np.mean(a_by.get(tk, [0.0]))) if a_by.get(tk) else 0.0
        raw_xs.append(mm)
        raw_ys.append(aa)
        xs.append(mm + rng.uniform(-0.005, 0.005))
        ys.append(aa + rng.uniform(-0.005, 0.005))
        names.append(tk)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    raw_xs = np.asarray(raw_xs)
    raw_ys = np.asarray(raw_ys)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # Color from un-jittered means so shared (y=x) tasks render as neutral grey
    colors = ['#4caf50' if ry > rx else ('#f44336' if ry < rx else '#9e9e9e') for rx, ry in zip(raw_xs, raw_ys)]
    ax.scatter(xs, ys, c=colors, alpha=0.75, s=42, edgecolor='black', linewidth=0.5)
    ax.plot([-0.05, 1.05], [-0.05, 1.05], 'k--', alpha=0.4)
    # annotate points where either arm > 0
    for x, y, n in zip(xs, ys, names):
        if x > 0.1 or y > 0.1:
            ax.annotate(n, (x, y), fontsize=7, alpha=0.85, xytext=(3, 3), textcoords='offset points')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('mean reward (misaligned, k=3)')
    ax.set_ylabel('mean reward (mode B aligned, k=3)')
    ax.set_title('Per-task mean reward: misaligned vs mode B aligned\n(green=aligned higher, red=misaligned higher)')
    ax.grid(linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_c_critical_token_distribution(aligned_path, misaligned_path, out_path):
    """Histogram of max critical_prompt_tokens per trial; the misaligned eval often
    spirals past 14-16k due to litellm context softcap fallback loops; aligned
    should be bounded near max_input_tokens=8192."""
    a, _ = load_trials(aligned_path)
    m, _ = load_trials(misaligned_path)
    a_pt = [r.get('critical_prompt_tokens') or 0 for r in top3(a)]
    m_pt = [r.get('critical_prompt_tokens') or 0 for r in top3(m)]
    bins = np.linspace(0, max(max(a_pt, default=0), max(m_pt, default=0), 16384) + 1, 60)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.hist(m_pt, bins=bins, alpha=0.55, color='#f44336', label='misaligned (terminus-2)', edgecolor='black', linewidth=0.3)
    ax.hist(a_pt, bins=bins, alpha=0.55, color='#4caf50', label='mode B aligned (camel-agent)', edgecolor='black', linewidth=0.3, hatch='//')
    ax.axvline(8192, color='black', linestyle='--', alpha=0.7, label='max_input_tokens=8192')
    ax.axvline(16384, color='black', linestyle=':', alpha=0.7, label='max_total_tokens=16384')
    ax.set_xlabel('peak per-step prompt_tokens (max across trial)')
    ax.set_ylabel('# trials')
    ax.set_title('Peak prompt_tokens per trial — aligned should be bounded')
    ax.legend()
    ax.grid(linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_d_status_transition_sankey(aligned_path, misaligned_path, out_path):
    """Per-task migration of "best top-3 status" from misaligned -> aligned.

    For each task: misaligned_best_status (best of 3) -> aligned_best_status.
    Visualized as a 4x4 contingency heatmap with annotated counts.
    "Best" priority: COMPLETED+reward>=1 > COMPLETED+reward=0 > TRUNCATED > FAILED > ABORTED.
    """
    a, _ = load_trials(aligned_path)
    m, _ = load_trials(misaligned_path)

    def best_label(rows):
        # Any reward >= 1.0 is PASS regardless of status (configure-git-webserver
        # has misaligned reward=1.0 with status=TRUNCATED on the lucky trial).
        if any(r['reward'] >= 1.0 for r in rows):
            return 'PASS'
        s = [r['status'] for r in rows]
        for cand in ['COMPLETED', 'TRUNCATED', 'FAILED', 'ABORTED']:
            if cand in s:
                return cand
        return 'ABORTED'

    a_by = defaultdict(list)
    m_by = defaultdict(list)
    for r in top3(a): a_by[r['task']].append(r)
    for r in top3(m): m_by[r['task']].append(r)
    tasks = sorted(set(a_by) | set(m_by))
    labels = ['PASS', 'COMPLETED', 'TRUNCATED', 'FAILED', 'ABORTED']
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for tk in tasks:
        mlbl = best_label(m_by.get(tk, []))
        albl = best_label(a_by.get(tk, []))
        mat[labels.index(mlbl), labels.index(albl)] += 1
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, cmap='Blues', aspect='auto')
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mat[i, j]
            if v:
                ax.text(j, i, str(v), ha='center', va='center',
                        color='white' if v > mat.max() * 0.5 else 'black', fontsize=10)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel('mode B aligned best-of-3 status')
    ax.set_ylabel('misaligned best-of-3 status')
    ax.set_title('Per-task best-status migration: misaligned -> aligned')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_e_solved_task_overlap(aligned_path, misaligned_path, out_path):
    """Venn-style 2-set diagram of solved tasks (any reward>=1 in top-3) under each arm."""
    a, _ = load_trials(aligned_path)
    m, _ = load_trials(misaligned_path)
    sa = solved_tasks(a)
    sm = solved_tasks(m)
    only_a = sa - sm
    only_m = sm - sa
    both = sa & sm

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # crude two-circle overlap
    from matplotlib.patches import Circle
    c1 = Circle((-0.6, 0), 1.0, alpha=0.35, color='#f44336', label='misaligned')
    c2 = Circle((0.6, 0), 1.0, alpha=0.35, color='#4caf50', label='mode B aligned')
    ax.add_patch(c1); ax.add_patch(c2)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
    ax.axis('off')
    ax.text(-1.2, 1.05, f'misaligned only ({len(only_m)})', ha='center', fontsize=11, weight='bold')
    ax.text(1.2, 1.05, f'aligned only ({len(only_a)})', ha='center', fontsize=11, weight='bold')
    ax.text(0, -1.3, f'shared ({len(both)})', ha='center', fontsize=11, weight='bold')
    # list overlap members
    ax.text(-1.2, -0.05, '\n'.join(sorted(only_m)[:8]) or '(none)', ha='center', va='center', fontsize=8)
    ax.text(1.2, -0.05, '\n'.join(sorted(only_a)[:8]) or '(none)', ha='center', va='center', fontsize=8)
    ax.text(0, 0.4, '\n'.join(sorted(both)[:6]) or '(none)', ha='center', va='center', fontsize=8, weight='bold')
    ax.set_title(f'Solved-task overlap (top-3, |aligned|={len(sa)} |misaligned|={len(sm)})')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_f_per_task_status_heatmap(aligned_path, misaligned_path, out_path):
    """89 × {3 misaligned slots, 3 aligned slots} = 89×6 heatmap, color by status.

    Each cell = top-3 chronological roll for that arm. Useful to spot per-task
    consistency / which arm migrates the truncation."""
    a, _ = load_trials(aligned_path)
    m, _ = load_trials(misaligned_path)
    a_by = defaultdict(list)
    m_by = defaultdict(list)
    for r in top3(a): a_by[r['task']].append(r)
    for r in top3(m): m_by[r['task']].append(r)
    for tk, lst in list(a_by.items()) + list(m_by.items()):
        lst.sort(key=lambda x: x.get('k_idx', 0))
    tasks = sorted(set(a_by) | set(m_by))

    # Encoding
    code_map = {'PASS': 4, 'COMPLETED': 3, 'TRUNCATED': 1, 'FAILED': 2, 'ABORTED': 0}
    color_map = {4: '#1b5e20', 3: '#4caf50', 1: '#f44336', 2: '#ff9800', 0: '#9e9e9e'}
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([color_map[k] for k in sorted(color_map)])

    def encode(rows):
        out = []
        for r in rows[:3]:
            if r['status'] == 'COMPLETED' and r['reward'] >= 1.0:
                out.append(code_map['PASS'])
            else:
                out.append(code_map.get(r['status'], 0))
        while len(out) < 3:
            out.append(0)
        return out

    n = len(tasks)
    grid = np.zeros((n, 6), dtype=int)
    for i, tk in enumerate(tasks):
        grid[i, :3] = encode(m_by.get(tk, []))
        grid[i, 3:] = encode(a_by.get(tk, []))
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.5, max(11.0, n * 0.20)))
    ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(6))
    ax.set_xticklabels(['M-1', 'M-2', 'M-3', 'A-1', 'A-2', 'A-3'])
    ax.set_yticks(range(n)); ax.set_yticklabels(tasks, fontsize=8)
    ax.axvline(2.5, color='black', linewidth=1.0)
    ax.set_title('Per-task status grid: misaligned vs aligned (k=3 each)')
    # legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color=color_map[4], label='PASS (reward=1)'),
        Patch(color=color_map[3], label='COMPLETED (reward=0)'),
        Patch(color=color_map[1], label='TRUNCATED'),
        Patch(color=color_map[2], label='FAILED'),
        Patch(color=color_map[0], label='ABORTED / missing'),
    ]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.45, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # Skeleton — actual paths filled in by parent agent in Phase 5
    # Example:
    # aligned   = '/tmp/i271_modeB_aligned_v2_trials.jsonl'
    # misalign  = '/tmp/i271_misaligned_v2_trials.jsonl'
    # outdir    = '/tmp/figs'
    # fig_a_status_dist_aligned_vs_misaligned(aligned, misalign, f'{outdir}/fig_a_status_dist.png')
    # fig_b_pass_at_k_per_task_scatter(aligned, misalign, f'{outdir}/fig_b_per_task_scatter.png')
    # fig_c_critical_token_distribution(aligned, misalign, f'{outdir}/fig_c_critical_tokens.png')
    # fig_d_status_transition_sankey(aligned, misalign, f'{outdir}/fig_d_status_transition.png')
    # fig_e_solved_task_overlap(aligned, misalign, f'{outdir}/fig_e_solved_overlap.png')
    # fig_f_per_task_status_heatmap(aligned, misalign, f'{outdir}/fig_f_status_heatmap.png')
    pass
