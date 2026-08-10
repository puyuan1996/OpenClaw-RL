#!/usr/bin/env python3
"""Build unified-schema JSONL for i271 misaligned and i271 aligned cells.

Re-scans trajectories so every cell shares an identical schema for 2×2 ANOVA.

Source dirs:
  misaligned: /nfs/eval_results/jobs/bench_v2_seta-agent57-i271_{halfA,halfB,rebalance}_2026-06-23
  aligned   : /nfs/eval_results/jobs/bench_v2_seta-agent57-i271_modeB_2026-06-25

Outputs:
  /tmp/four_cell/i271_misaligned_trials.jsonl
  /tmp/four_cell/i271_aligned_trials.jsonl
"""

import os
import re
import json
import sys
from collections import defaultdict, Counter

MISALIGNED_SHARDS = [
    ('/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_halfA_2026-06-23', 'halfA'),
    ('/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_halfB_2026-06-23', 'halfB'),
    ('/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_rebalance_2026-06-23', 'rebalance'),
]
ALIGNED_SHARDS = [
    ('/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_modeB_2026-06-25', 'modeB'),
]

THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def status_from_exc(exc, metadata=None):
    """Unified status for both terminus-2 (misaligned) and camel-agent (aligned).

    For terminus-2 the truncation surfaces as AgentTimeoutError exception.
    For camel-agent the metadata.status field carries TRUNCATED / COMPLETED
    even when exception_info is None (turn-based truncation, not exception-based).
    """
    if exc is None:
        # Check camel-agent metadata first
        if isinstance(metadata, dict):
            ms = metadata.get('status')
            if ms in ('TRUNCATED', 'COMPLETED', 'ABORTED', 'FAILED'):
                return ms, None
        return 'COMPLETED', None
    t = exc.get('exception_type') if isinstance(exc, dict) else None
    if t == 'AgentTimeoutError':
        return 'TRUNCATED', t
    if t == 'VerifierTimeoutError':
        return 'ABORTED', t
    if t == 'RuntimeError':
        return 'FAILED', t
    return 'FAILED', t


def scan_trajectory(traj_path):
    out = {
        'n_agent_steps': 0,
        'total_tool_calls': 0,
        'max_tool_calls_per_step': 0,
        'max_think_chars': 0,
        'n_think_blocks': 0,
        'n_filler': 0,
        'first_filler_step': None,
        'critical_prompt_tokens': 0,
    }
    if not os.path.exists(traj_path):
        return out
    try:
        d = json.load(open(traj_path))
    except Exception:
        return out
    steps = d.get('steps', []) or []
    for s in steps:
        src = s.get('source')
        if src == 'agent':
            out['n_agent_steps'] += 1
        tc = s.get('tool_calls')
        if isinstance(tc, list):
            out['total_tool_calls'] += len(tc)
            if len(tc) > out['max_tool_calls_per_step']:
                out['max_tool_calls_per_step'] = len(tc)
        msg = s.get('message') or ''
        if msg:
            blocks = THINK_RE.findall(msg)
            if blocks:
                out['n_think_blocks'] += len(blocks)
                mb = max(len(b) for b in blocks)
                if mb > out['max_think_chars']:
                    out['max_think_chars'] = mb
        if msg and 'echnical difficulties' in msg:
            out['n_filler'] += 1
            if out['first_filler_step'] is None:
                out['first_filler_step'] = s.get('step_id')
        obs = s.get('observation')
        if isinstance(obs, dict):
            res = obs.get('results')
            text = ''
            if isinstance(res, str):
                text = res
            elif isinstance(res, list):
                try:
                    text = json.dumps(res)
                except Exception:
                    text = str(res)
            if text and 'echnical difficulties' in text:
                out['n_filler'] += 1
                if out['first_filler_step'] is None:
                    out['first_filler_step'] = s.get('step_id')
        m = s.get('metrics')
        if isinstance(m, dict):
            pt = m.get('prompt_tokens') or 0
            if pt > out['critical_prompt_tokens']:
                out['critical_prompt_tokens'] = pt
    return out


def build_one(shards, cell, model, harness, out_path):
    rows = []
    n_missing = 0
    for base, shard in shards:
        if not os.path.isdir(base):
            print(f'WARN: missing shard dir {base}', file=sys.stderr)
            continue
        for d in sorted(os.listdir(base)):
            p = f'{base}/{d}'
            if not os.path.isdir(p):
                continue
            rj = f'{p}/result.json'
            if not os.path.exists(rj):
                n_missing += 1
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
            ar = r.get('agent_result') or {}
            md = ar.get('metadata') or {}
            status, exc_type = status_from_exc(exc, md)
            # n_steps: terminus-2 has n_episodes; camel-agent has model_turn_count
            n_steps = md.get('n_episodes') if 'n_episodes' in md else md.get('model_turn_count')
            n_in = ar.get('n_input_tokens')
            n_out = ar.get('n_output_tokens')
            started_at = r.get('started_at')
            agent_exc = None
            verifier_exc = None
            if isinstance(exc, dict):
                t = exc.get('exception_type')
                if t == 'VerifierTimeoutError':
                    verifier_exc = t
                else:
                    agent_exc = t

            traj_path = f'{p}/agent/trajectory.json'
            sc = scan_trajectory(traj_path)

            rows.append({
                'cell': cell,
                'model': model,
                'harness': harness,
                'task': task,
                'trial_dir': d,
                'trial_hash': trial_hash,
                'shard': shard,
                'started_at': started_at,
                'reward': reward,
                'status': status,
                'exception_type': exc_type,
                'has_exception': exc is not None,
                'agent_exception': agent_exc,
                'verifier_exception': verifier_exc,
                'n_steps': n_steps,
                'n_agent_steps': sc['n_agent_steps'],
                'n_input_tokens': n_in,
                'n_output_tokens': n_out,
                'total_tool_calls': sc['total_tool_calls'],
                'max_tool_calls_per_step': sc['max_tool_calls_per_step'],
                'max_think_chars': sc['max_think_chars'],
                'n_think_blocks': sc['n_think_blocks'],
                'n_filler': sc['n_filler'],
                'first_filler_step': sc['first_filler_step'],
                'critical_prompt_tokens': sc['critical_prompt_tokens'],
                'trajectory_path': traj_path,
            })

    by_task = defaultdict(list)
    for row in rows:
        by_task[row['task']].append(row)
    for tk, lst in by_task.items():
        lst.sort(key=lambda x: x['started_at'] or '')
        for i, r in enumerate(lst):
            r['k_idx'] = i
            r['is_top3'] = i < 3

    final = []
    for tk in sorted(by_task.keys()):
        for r in by_task[tk]:
            final.append(r)

    with open(out_path, 'w') as f:
        for r in final:
            f.write(json.dumps(r) + '\n')
        top3 = [r for r in final if r['is_top3']]
        n = len(top3)
        pos = sum(1 for r in top3 if r['reward'] >= 0.99)
        solved = sorted({r['task'] for r in top3 if r['reward'] >= 0.99})
        status_top3 = Counter(r['status'] for r in top3)
        meta = {
            '__meta__': True,
            'cell': cell, 'model': model, 'harness': harness,
            'src_job_dirs': [s[0] for s in shards],
            'n_missing_result_json': n_missing,
            'n_trials_all': len(final),
            'n_trials_top3': n,
            'n_tasks': len(by_task),
            'k': 3,
            'pass_at_1_top3': pos / n if n else 0.0,
            'pass_at_3_top3': len(solved) / len(by_task) if by_task else 0.0,
            'n_solved_tasks': len(solved),
            'solved_tasks_top3': solved,
            'status_distribution_top3': dict(status_top3),
            'status_distribution_top3_pct': {k: v / n * 100.0 for k, v in status_top3.items()} if n else {},
        }
        f.write(json.dumps(meta) + '\n')

    print(f'[{cell}] wrote {out_path}')
    print(f'  all={len(final)} top3={n} tasks={len(by_task)} pass@1={pos/n if n else 0:.4f} ({pos}/{n}) pass@3={len(solved)/len(by_task) if by_task else 0:.4f}')
    print(f'  status_top3={dict(status_top3)}  solved={solved}')


def main():
    build_one(MISALIGNED_SHARDS, 'i271_misaligned',
              'qwen3-8b-seta-agent57-i271', 'terminus-2',
              '/tmp/four_cell/i271_misaligned_trials.jsonl')
    build_one(ALIGNED_SHARDS, 'i271_aligned',
              'qwen3-8b-seta-agent57-i271', 'camel-agent-modeB',
              '/tmp/four_cell/i271_aligned_trials.jsonl')


if __name__ == '__main__':
    main()
