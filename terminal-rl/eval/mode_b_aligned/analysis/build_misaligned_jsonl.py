#!/usr/bin/env python3
"""Walk halfA / halfB / rebalance shards for the i271 misaligned TB 2.0 eval and
produce a per-trial JSONL summary suitable for downstream pass@k / status / filler
analysis.

Convention follows issue #21:
- 280 trial total (halfA 132 + halfB 135 + rebalance 13)
- pass@k uses "first 3 attempts per task chronological"  (k=3)
- the 13 rebalance ones are extra rolls (k=4 for those tasks); they go into the
  JSONL with `is_top3 = false`, downstream pass@k uses only is_top3=true rolls.

Dedup rule: trial dirs across halfA/halfB/rebalance are unique by hash; group
by task; sort by started_at; first 3 trials per task = is_top3=true (k_idx 0..2);
remaining trials = is_top3=false.

The `main` shard `bench_v2_seta-agent57-i271_2026-06-23/` is an aborted
partial-init run (only 3/9 trial dirs have result.json) and is EXCLUDED.

Status semantics (terminus-2):
- exception_info.exception_type == "AgentTimeoutError" -> TRUNCATED
- exception_info == None                                -> COMPLETED
- exception_info.exception_type == "RuntimeError"       -> FAILED
- exception_info.exception_type == "VerifierTimeoutError" -> ABORTED
- other                                                 -> FAILED (catch-all)
"""

import os
import json
import sys
from collections import defaultdict

SHARDS = [
    ('/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_halfA_2026-06-23', 'halfA'),
    ('/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_halfB_2026-06-23', 'halfB'),
    ('/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_rebalance_2026-06-23', 'rebalance'),
]

OUT_JSONL = '/tmp/i271_misaligned_v2_trials.jsonl'
OUT_SUMMARY = '/tmp/i271_misaligned_v2_summary.json'


def status_from_exc(exc):
    if exc is None:
        return 'COMPLETED'
    t = exc.get('exception_type') if isinstance(exc, dict) else None
    if t == 'AgentTimeoutError':
        return 'TRUNCATED'
    if t == 'VerifierTimeoutError':
        return 'ABORTED'
    if t == 'RuntimeError':
        return 'FAILED'
    return 'FAILED'


def scan_trajectory_for_filler(traj_path):
    """Return (n_filler, first_filler_step, max_prompt_tokens)."""
    if not os.path.exists(traj_path):
        return 0, None, None
    try:
        d = json.load(open(traj_path))
    except Exception:
        return 0, None, None
    steps = d.get('steps', [])
    n_filler = 0
    first_filler_step = None
    max_pt = 0
    for s in steps:
        msg = s.get('message', '') or ''
        # Look for the "Technical difficulties" filler observation appearing as
        # a user/tool-observation message (case-insensitive)
        if 'Technical difficulties' in msg or 'technical difficulties' in msg.lower():
            n_filler += 1
            if first_filler_step is None:
                first_filler_step = s.get('step_id')
        # also check observation field
        obs = s.get('observation')
        if isinstance(obs, dict):
            res = obs.get('results')
            if isinstance(res, str) and 'echnical difficulties' in res:
                n_filler += 1
                if first_filler_step is None:
                    first_filler_step = s.get('step_id')
            elif isinstance(res, list):
                for r in res:
                    rs = json.dumps(r) if not isinstance(r, str) else r
                    if 'echnical difficulties' in rs:
                        n_filler += 1
                        if first_filler_step is None:
                            first_filler_step = s.get('step_id')
                        break
        m = s.get('metrics')
        if isinstance(m, dict):
            pt = m.get('prompt_tokens') or 0
            if pt > max_pt:
                max_pt = pt
    return n_filler, first_filler_step, max_pt


def main():
    # 1st pass: collect all rows
    rows = []
    for base, shard in SHARDS:
        for d in sorted(os.listdir(base)):
            p = f'{base}/{d}'
            if not os.path.isdir(p):
                continue
            rj = f'{p}/result.json'
            if not os.path.exists(rj):
                continue
            try:
                r = json.load(open(rj))
            except Exception as e:
                print(f'WARN parse {rj}: {e}', file=sys.stderr)
                continue
            task = d.split('__')[0]
            trial_hash = d.split('__')[-1] if '__' in d else ''
            vr = r.get('verifier_result') or {}
            rwd = (vr.get('rewards') or {}).get('reward', 0.0)
            try:
                reward = float(rwd) if rwd is not None else 0.0
            except Exception:
                reward = 0.0
            exc = r.get('exception_info')
            status = status_from_exc(exc)
            ar = r.get('agent_result') or {}
            md = ar.get('metadata') or {}
            n_steps = md.get('n_episodes')
            n_in = ar.get('n_input_tokens')
            n_out = ar.get('n_output_tokens')
            started_at = r.get('started_at')

            traj_path = f'{p}/agent/trajectory.json'
            n_filler, first_filler_step, max_pt = scan_trajectory_for_filler(traj_path)

            rows.append({
                'task': task,
                'trial_dir': d,
                'shard': shard,
                'trial_hash': trial_hash,
                'started_at': started_at,
                'reward': reward,
                'status': status,
                'exception_type': (exc or {}).get('exception_type') if isinstance(exc, dict) else None,
                'n_steps': n_steps,
                'n_input_tokens': n_in,
                'n_output_tokens': n_out,
                'n_filler': n_filler,
                'first_filler_step': first_filler_step,
                'critical_prompt_tokens': max_pt,
                'trajectory_path': traj_path,
            })

    # 2nd: dedup-by-task, chronological, assign k_idx in 0..2 (is_top3=True) and >=3
    by_task = defaultdict(list)
    for row in rows:
        by_task[row['task']].append(row)
    for tk, lst in by_task.items():
        lst.sort(key=lambda x: x['started_at'] or '')
        for i, r in enumerate(lst):
            r['k_idx'] = i
            r['is_top3'] = i < 3

    # Sort final rows for stability
    final_rows = []
    for tk in sorted(by_task.keys()):
        for r in by_task[tk]:
            final_rows.append(r)

    # write JSONL
    with open(OUT_JSONL, 'w') as f:
        for r in final_rows:
            f.write(json.dumps(r) + '\n')
        # meta line
        n_top3 = sum(1 for r in final_rows if r['is_top3'])
        n_tasks = len(by_task)
        # pass@1: at least one of first-3 trials reward >= 1.0 ?  But pass@1 here
        # = mean(reward_first_trial) over tasks? The issue uses
        #   pass@1 = (sum of all top-3 rewards) / total top-3 trials, NOT per-task.
        # Let's match issue #21 which reports 1.12% = 3 / 267.
        n_pos_top3 = sum(1 for r in final_rows if r['is_top3'] and r['reward'] >= 1.0)
        pass_at_1 = n_pos_top3 / n_top3 if n_top3 else 0.0
        # pass@3 empirical = fraction of tasks with any reward>=1 in top-3
        solved_tasks = sorted({r['task'] for r in final_rows if r['is_top3'] and r['reward'] >= 1.0})
        pass_at_3 = len(solved_tasks) / n_tasks if n_tasks else 0.0
        # status dist
        from collections import Counter
        status_top3 = Counter(r['status'] for r in final_rows if r['is_top3'])
        meta = {
            '__meta__': True,
            'n_trials_all': len(final_rows),
            'n_trials_top3': n_top3,
            'n_tasks': n_tasks,
            'pass_at_1_top3': pass_at_1,
            'pass_at_3_top3': pass_at_3,
            'n_solved_tasks': len(solved_tasks),
            'solved_tasks_top3': solved_tasks,
            'status_distribution_top3': dict(status_top3),
        }
        f.write(json.dumps(meta) + '\n')

    # 3rd: write summary JSON
    from collections import Counter
    status_top3 = Counter(r['status'] for r in final_rows if r['is_top3'])
    fillers_top3 = [r for r in final_rows if r['is_top3'] and r['n_filler'] > 0]
    fillers_top3_n = len(fillers_top3)
    if fillers_top3:
        import statistics
        med = statistics.median([r['n_filler'] for r in fillers_top3])
        mean_f = statistics.mean([r['n_filler'] for r in fillers_top3])
        max_f = max(r['n_filler'] for r in fillers_top3)
    else:
        med = mean_f = max_f = 0
    # per-task aggregate reward
    per_task_reward = {tk: sum(r['reward'] for r in by_task[tk] if r['is_top3']) / 3.0 for tk in by_task}

    # solved tasks list
    n_top3 = sum(1 for r in final_rows if r['is_top3'])
    n_pos_top3 = sum(1 for r in final_rows if r['is_top3'] and r['reward'] >= 1.0)
    solved_tasks = sorted({r['task'] for r in final_rows if r['is_top3'] and r['reward'] >= 1.0})
    pass_at_1 = n_pos_top3 / n_top3 if n_top3 else 0.0
    pass_at_3 = len(solved_tasks) / len(by_task) if by_task else 0.0

    summary = {
        'n_trials_all': len(final_rows),
        'n_trials_top3': n_top3,
        'n_tasks': len(by_task),
        'k': 3,
        'pass_at_1_top3': pass_at_1,
        'pass_at_3_top3': pass_at_3,
        'n_solved_tasks': len(solved_tasks),
        'solved_tasks_top3': solved_tasks,
        'status_distribution_top3': dict(status_top3),
        'status_distribution_top3_pct': {k: v / n_top3 * 100.0 for k, v in status_top3.items()},
        'filler_trigger_top3_n': fillers_top3_n,
        'filler_trigger_top3_rate': fillers_top3_n / n_top3 if n_top3 else 0.0,
        'filler_depth_top3_median': med,
        'filler_depth_top3_mean': mean_f,
        'filler_depth_top3_max': max_f,
        'per_task_mean_reward_top3': per_task_reward,
        'shards_used': [s for _, s in SHARDS],
        'shards_excluded': ['main (bench_v2_seta-agent57-i271_2026-06-23: aborted partial-init)'],
        'reward_source': 'result.json::verifier_result.rewards.reward',
        'dedup_rule': 'per task, chronological by started_at, first 3 -> is_top3=true; remainder -> is_top3=false',
    }
    with open(OUT_SUMMARY, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print quick stats
    print(f"Wrote {OUT_JSONL} ({len(final_rows)} rows + 1 meta)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"n_trials_top3: {n_top3}")
    print(f"n_tasks: {len(by_task)}")
    print(f"pass@1 (top3): {pass_at_1:.4f}")
    print(f"pass@3 (top3): {pass_at_3:.4f}")
    print(f"status dist top3: {dict(status_top3)}")
    print(f"filler trigger rate top3: {fillers_top3_n}/{n_top3} = {fillers_top3_n/n_top3:.3f}")
    print(f"filler depth top3 median={med} mean={mean_f:.1f} max={max_f}")
    print(f"solved tasks top3 ({len(solved_tasks)}): {solved_tasks}")


if __name__ == '__main__':
    main()
