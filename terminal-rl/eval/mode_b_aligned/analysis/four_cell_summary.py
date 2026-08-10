#!/usr/bin/env python3
"""Aggregate the 2x2 four-cell ANOVA summary.

Inputs:
  /tmp/four_cell/base_misaligned_trials.jsonl
  /tmp/four_cell/base_aligned_trials.jsonl     (may be absent if eval still running)
  /tmp/four_cell/i271_misaligned_trials.jsonl
  /tmp/four_cell/i271_aligned_trials.jsonl

Output:
  /tmp/four_cell_summary.json

Per-cell stats: n_trials, pass@1 (sample-level: trial reward>=0.99 / n_top3),
pass@1_per_task (mean over 89 tasks of any-of-k-pass), pass@3_per_task (same),
status distribution, mean_reward, solved tasks list, filler rate, think summary.

Cross-cell decomposition (NULL when base_aligned absent):
  delta_RL_aligned     = i271_aligned - base_aligned
  delta_RL_misaligned  = i271_misaligned - base_misaligned
  delta_harness_base   = base_aligned - base_misaligned
  delta_harness_i271   = i271_aligned - i271_misaligned
  interaction          = delta_harness_i271 - delta_harness_base
"""

import os
import json
from collections import Counter, defaultdict
import statistics

CELL_FILES = {
    'base_misaligned':  '/tmp/four_cell/base_misaligned_trials.jsonl',
    'base_aligned':     '/tmp/four_cell/base_aligned_trials.jsonl',
    'i271_misaligned':  '/tmp/four_cell/i271_misaligned_trials.jsonl',
    'i271_aligned':     '/tmp/four_cell/i271_aligned_trials.jsonl',
}
OUT = '/tmp/four_cell_summary.json'


def load_trials(path):
    if not os.path.exists(path):
        return None, None
    rows = []
    meta = None
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


def percentile(arr, q):
    if not arr:
        return None
    s = sorted(arr)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def summarize_cell(rows, meta):
    if rows is None:
        return {'present': False}
    top3 = [r for r in rows if r.get('is_top3')]
    n = len(top3)
    pos = sum(1 for r in top3 if r['reward'] >= 0.99)
    by_task = defaultdict(list)
    for r in top3:
        by_task[r['task']].append(r)
    # per-task pass@1: mean over tasks of (#pass / #trials_in_task)
    per_task_pass1 = []
    for tk, lst in by_task.items():
        per_task_pass1.append(sum(1 for r in lst if r['reward'] >= 0.99) / max(len(lst), 1))
    # per-task pass@k=3: 1 if any pass else 0, mean over tasks (== pass@3 empirical)
    per_task_passk = []
    solved_set = set()
    for tk, lst in by_task.items():
        any_pass = any(r['reward'] >= 0.99 for r in lst)
        per_task_passk.append(1 if any_pass else 0)
        if any_pass:
            solved_set.add(tk)
    n_tasks = len(by_task)
    status_dist = Counter(r['status'] for r in top3)
    mean_reward = statistics.mean([r['reward'] for r in top3]) if top3 else 0.0
    # filler
    filler_trial = [r for r in top3 if (r.get('n_filler') or 0) >= 20]
    # think
    think_chars = [r.get('max_think_chars') or 0 for r in top3]
    n_with_think = sum(1 for r in top3 if (r.get('n_think_blocks') or 0) > 0)
    max_block = max(think_chars) if think_chars else 0
    p50_block_with_think = percentile(
        [r['max_think_chars'] for r in top3 if (r.get('n_think_blocks') or 0) > 0],
        0.5,
    )
    # n_agent_steps stats
    n_steps = [r.get('n_agent_steps') or 0 for r in top3]
    # input tokens stats
    in_toks = [r.get('n_input_tokens') or 0 for r in top3]
    out_toks = [r.get('n_output_tokens') or 0 for r in top3]
    crit_toks = [r.get('critical_prompt_tokens') or 0 for r in top3]
    return {
        'present': True,
        'cell': meta.get('cell') if meta else None,
        'model': meta.get('model') if meta else None,
        'harness': meta.get('harness') if meta else None,
        'n_trials_all': len(rows),
        'n_trials_top3': n,
        'n_tasks': n_tasks,
        'pass_at_1_sample': pos / n if n else 0.0,        # trial-level pass@1
        'pass_at_1_per_task': statistics.mean(per_task_pass1) if per_task_pass1 else 0.0,
        'pass_at_3_per_task': statistics.mean(per_task_passk) if per_task_passk else 0.0,
        'n_positive_trials': pos,
        'n_solved_tasks': len(solved_set),
        'solved_tasks': sorted(solved_set),
        'status_distribution': dict(status_dist),
        'status_distribution_pct': {k: v / n * 100.0 for k, v in status_dist.items()} if n else {},
        'mean_reward': mean_reward,
        'filler_trial_count_ge20': len(filler_trial),
        'filler_trial_rate_ge20': len(filler_trial) / n if n else 0.0,
        'think': {
            'n_trials_with_any_think': n_with_think,
            'pct_trials_with_any_think': n_with_think / n * 100.0 if n else 0.0,
            'max_block_chars_overall': max_block,
            'p50_block_chars_among_trials_with_think': p50_block_with_think,
        },
        'n_agent_steps': {
            'mean': statistics.mean(n_steps) if n_steps else 0,
            'median': statistics.median(n_steps) if n_steps else 0,
            'max': max(n_steps) if n_steps else 0,
        },
        'tokens': {
            'sum_input': sum(in_toks),
            'sum_output': sum(out_toks),
            'mean_input': statistics.mean(in_toks) if in_toks else 0,
            'mean_output': statistics.mean(out_toks) if out_toks else 0,
            'critical_prompt_max': max(crit_toks) if crit_toks else 0,
            'critical_prompt_p50': percentile(crit_toks, 0.5),
        },
    }


def decompose(cells):
    def pa1(c):
        if c is None or not c.get('present'):
            return None
        return c['pass_at_1_sample']
    bm = pa1(cells.get('base_misaligned'))
    ba = pa1(cells.get('base_aligned'))
    im = pa1(cells.get('i271_misaligned'))
    ia = pa1(cells.get('i271_aligned'))
    def diff(a, b):
        if a is None or b is None:
            return None
        return a - b
    out = {
        'delta_RL_aligned_pass1': diff(ia, ba),       # i271 - base under aligned
        'delta_RL_misaligned_pass1': diff(im, bm),
        'delta_harness_base_pass1': diff(ba, bm),
        'delta_harness_i271_pass1': diff(ia, im),
    }
    out['interaction_pass1'] = diff(out['delta_harness_i271_pass1'], out['delta_harness_base_pass1'])
    # pass@3 same decomposition
    def pa3(c):
        if c is None or not c.get('present'):
            return None
        return c['pass_at_3_per_task']
    bm3, ba3, im3, ia3 = pa3(cells.get('base_misaligned')), pa3(cells.get('base_aligned')), pa3(cells.get('i271_misaligned')), pa3(cells.get('i271_aligned'))
    out['delta_RL_aligned_pass3'] = diff(ia3, ba3)
    out['delta_RL_misaligned_pass3'] = diff(im3, bm3)
    out['delta_harness_base_pass3'] = diff(ba3, bm3)
    out['delta_harness_i271_pass3'] = diff(ia3, im3)
    out['interaction_pass3'] = diff(out['delta_harness_i271_pass3'], out['delta_harness_base_pass3'])
    return out


def overlap_table(cells):
    """4-way intersection / per-pair intersection of solved task sets."""
    sets = {}
    for name, c in cells.items():
        if c and c.get('present'):
            sets[name] = set(c.get('solved_tasks') or [])
    out = {'sets': {k: sorted(v) for k, v in sets.items()}, 'pairwise': {}, 'all_4': None, 'any_cell': None}
    names = sorted(sets.keys())
    for i, a in enumerate(names):
        for b in names[i+1:]:
            out['pairwise'][f'{a}__AND__{b}'] = sorted(sets[a] & sets[b])
    if sets:
        inter = set.intersection(*sets.values()) if len(sets) > 1 else next(iter(sets.values()))
        out['all_4'] = sorted(inter)
        out['any_cell'] = sorted(set.union(*sets.values()))
    return out


def main():
    cells = {}
    for name, path in CELL_FILES.items():
        rows, meta = load_trials(path)
        cells[name] = summarize_cell(rows, meta)
    decomp = decompose(cells)
    overlaps = overlap_table(cells)
    summary = {
        'design': {
            'rows': ['Qwen3-8B base', 'Qwen3-8B i271 (RL)'],
            'cols': ['misaligned (terminus-2)', 'aligned (camel-agent mode B)'],
            'tasks': 89, 'k': 3,
        },
        'cells': cells,
        'cross_cell': decomp,
        'solved_overlap': overlaps,
        'notes': {
            'pass_at_1_sample_def': 'sample-level: (# top-3 trials with reward>=0.99) / n_top3_trials',
            'pass_at_3_per_task_def': 'task-level pass@k=3: 1 if any of 3 trials passes else 0, mean over 89 tasks',
            'status_semantics': 'terminus-2: derive from exception_info; camel-agent: derive from agent_result.metadata.status when exception_info is None',
            'reward_source': 'verifier_result.rewards.reward',
        },
    }
    with open(OUT, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {OUT}')
    for name, c in cells.items():
        if not c.get('present'):
            print(f'  [{name}] absent / pending')
            continue
        print(f'  [{name}] pass@1_sample={c["pass_at_1_sample"]:.4f}  pass@3_per_task={c["pass_at_3_per_task"]:.4f}  '
              f'solved={c["n_solved_tasks"]}  status={c["status_distribution"]}')
    print(f'  delta_RL_aligned (pass@1): {decomp["delta_RL_aligned_pass1"]}')
    print(f'  delta_RL_misaligned (pass@1): {decomp["delta_RL_misaligned_pass1"]}')
    print(f'  delta_harness_base (pass@1): {decomp["delta_harness_base_pass1"]}')
    print(f'  delta_harness_i271 (pass@1): {decomp["delta_harness_i271_pass1"]}')
    print(f'  interaction (pass@1): {decomp["interaction_pass1"]}')


if __name__ == '__main__':
    main()
