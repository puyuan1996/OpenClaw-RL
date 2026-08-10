#!/usr/bin/env python3
"""Build per-trial JSONL for base Qwen3-8B × TB 2.0 misaligned (terminus-2) eval.

Source: /nfs/eval_results/jobs/bench_v2_qwen3-8b-base_2026-06-24/
(per #22: single job, single sglang TP=8, no shard split, all 270 dirs of which
267 are k<=3 attempts; 3 are k=4 extra rolls if any exist).

Output schema (same 4-cell unified schema for ANOVA work):
{
  cell, model, harness, task, trial_dir, trial_hash, shard, started_at, k_idx, is_top3,
  status, exception_type, reward,
  n_agent_steps, n_steps, n_input_tokens, n_output_tokens,
  total_tool_calls, max_tool_calls_per_step,
  max_think_chars, n_think_blocks,
  n_filler, first_filler_step,
  critical_prompt_tokens,
  has_exception, agent_exception, verifier_exception,
  trajectory_path
}

Status semantics (terminus-2, same as #21/#22):
  exception_info==None                                    -> COMPLETED
  exception_info.type == AgentTimeoutError                -> TRUNCATED
  exception_info.type == VerifierTimeoutError             -> ABORTED
  other / RuntimeError                                    -> FAILED
"""

import os
import re
import json
import sys
from collections import defaultdict, Counter

SRC = '/nfs/eval_results/jobs/bench_v2_qwen3-8b-base_2026-06-24'
OUT = '/tmp/four_cell/base_misaligned_trials.jsonl'

CELL = 'base_misaligned'
MODEL = 'qwen3-8b-base'
HARNESS = 'terminus-2'

THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def status_from_exc(exc, metadata=None):
    """Unified status: terminus-2 via exception, camel-agent via metadata.status fallback."""
    if exc is None:
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
    """Return dict of scan info for a trajectory.json."""
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
        # tool calls (only agent steps have these usually)
        tc = s.get('tool_calls')
        if isinstance(tc, list):
            out['total_tool_calls'] += len(tc)
            if len(tc) > out['max_tool_calls_per_step']:
                out['max_tool_calls_per_step'] = len(tc)
        # think blocks in message
        msg = s.get('message') or ''
        if msg:
            blocks = THINK_RE.findall(msg)
            if blocks:
                out['n_think_blocks'] += len(blocks)
                mb = max(len(b) for b in blocks)
                if mb > out['max_think_chars']:
                    out['max_think_chars'] = mb
        # filler ("Technical difficulties" / "echnical difficulties")
        for hay in (msg,):
            if 'echnical difficulties' in hay:
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
        # critical prompt tokens
        m = s.get('metrics')
        if isinstance(m, dict):
            pt = m.get('prompt_tokens') or 0
            if pt > out['critical_prompt_tokens']:
                out['critical_prompt_tokens'] = pt
    return out


def main():
    rows = []
    n_missing_result = 0
    for d in sorted(os.listdir(SRC)):
        p = f'{SRC}/{d}'
        if not os.path.isdir(p):
            continue
        rj = f'{p}/result.json'
        if not os.path.exists(rj):
            n_missing_result += 1
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
        n_steps = md.get('n_episodes') if 'n_episodes' in md else md.get('model_turn_count')
        n_in = ar.get('n_input_tokens')
        n_out = ar.get('n_output_tokens')
        started_at = r.get('started_at')
        # exception split: agent vs verifier
        agent_exc = None
        verifier_exc = None
        if isinstance(exc, dict):
            t = exc.get('exception_type')
            if t in ('AgentTimeoutError', 'RuntimeError', 'ContextLengthExceededError'):
                agent_exc = t
            elif t == 'VerifierTimeoutError':
                verifier_exc = t
            else:
                agent_exc = t

        traj_path = f'{p}/agent/trajectory.json'
        sc = scan_trajectory(traj_path)

        rows.append({
            'cell': CELL,
            'model': MODEL,
            'harness': HARNESS,
            'task': task,
            'trial_dir': d,
            'trial_hash': trial_hash,
            'shard': 'main',
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

    # Dedup per task: chronological order; first 3 -> is_top3 = True
    by_task = defaultdict(list)
    for row in rows:
        by_task[row['task']].append(row)
    for tk, lst in by_task.items():
        lst.sort(key=lambda x: x['started_at'] or '')
        for i, r in enumerate(lst):
            r['k_idx'] = i
            r['is_top3'] = i < 3

    # Final sorted writing
    final = []
    for tk in sorted(by_task.keys()):
        for r in by_task[tk]:
            final.append(r)

    with open(OUT, 'w') as f:
        for r in final:
            f.write(json.dumps(r) + '\n')
        # meta
        top3 = [r for r in final if r['is_top3']]
        n = len(top3)
        pos = sum(1 for r in top3 if r['reward'] >= 0.99)
        solved = sorted({r['task'] for r in top3 if r['reward'] >= 0.99})
        status_top3 = Counter(r['status'] for r in top3)
        meta = {
            '__meta__': True,
            'cell': CELL,
            'model': MODEL,
            'harness': HARNESS,
            'src_job_dir': SRC,
            'n_trial_dirs_total': len(rows) + n_missing_result,
            'n_missing_result_json': n_missing_result,
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

    print(f'Wrote {OUT}')
    print(f'  n_trials_all: {len(final)}, n_top3: {n}, n_tasks: {len(by_task)}')
    print(f'  pass@1: {pos/n if n else 0:.4f} ({pos}/{n})')
    print(f'  pass@3: {len(solved)/len(by_task) if by_task else 0:.4f} ({len(solved)}/{len(by_task)})')
    print(f'  status_top3: {dict(status_top3)}')
    print(f'  solved: {solved}')


if __name__ == '__main__':
    main()
