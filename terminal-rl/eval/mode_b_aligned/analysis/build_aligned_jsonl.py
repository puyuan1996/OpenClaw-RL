#!/usr/bin/env python3
"""Walk the modeB aligned shard for the i271 TB 2.0 eval and produce a per-trial
JSONL summary suitable for downstream pass@k / status / token analysis.

The aligned eval used:
- OpenClaw-RL camel-agent harness (openclaw_camel_adapter)
- ATIF v1.6 trajectory schema
- harbor over-the-CLI runner
- 89 task × k=3 (plus 1 extra retry for 1 task = 270 trial dirs, but some failed
  to produce result.json — we walk all that have it)

Status semantics (camel-agent adapter):
- agent_result.metadata.status field stored directly (COMPLETED/TRUNCATED/FAILED/ABORTED)
- exception_info may be present too for harbor-level exceptions (VerifierTimeoutError,
  AgentTimeoutError) — these override the metadata status.

Key differences from misaligned trajectory walker:
- step.source ('agent' / 'user' / 'system' / 'tool') vs misaligned ('message'/'observation')
- step.message is a string (raw text) in ATIF v1.6; tool_calls / observation are separate
- step.metrics holds {prompt_tokens, completion_tokens} per agent step
- final_metrics.total_prompt_tokens / total_completion_tokens / extra.status at top
- No "Technical difficulties" filler is expected — that's harbor terminus-2 specific
- max_input_tokens=8192 cliff is the analogue of misaligned's 16k cliff
"""

import os
import json
import sys
from collections import defaultdict, Counter
import statistics

BASE = '/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_modeB_2026-06-25'
OUT_JSONL = '/tmp/i271_aligned_v2_trials.jsonl'
OUT_SUMMARY = '/tmp/i271_aligned_v2_summary.json'


def status_from_result(r):
    """Derive a unified status string from a result.json dict.

    Priority:
    1. exception_info != None -> map exception_type to status
       - AgentTimeoutError -> TRUNCATED (harness-level timeout)
       - VerifierTimeoutError -> ABORTED (verifier timeout, env trouble)
       - RuntimeError -> FAILED
       - other -> FAILED
    2. agent_result.metadata.status -> use as-is
    3. fall through -> UNKNOWN
    """
    exc = r.get('exception_info')
    if isinstance(exc, dict):
        et = exc.get('exception_type')
        if et == 'AgentTimeoutError':
            return 'TRUNCATED', et
        if et == 'VerifierTimeoutError':
            return 'ABORTED', et
        if et == 'RuntimeError':
            return 'FAILED', et
        return 'FAILED', et
    ar = r.get('agent_result') or {}
    md = ar.get('metadata') or {}
    s = md.get('status')
    if s:
        return s, None
    return 'UNKNOWN', None


def scan_trajectory(traj_path):
    """Return dict of trajectory stats:
       n_steps, n_agent_steps, prompt_tokens_per_step (list), completion_tokens_per_step (list),
       max_prompt_tokens, last_agent_prompt_tokens, n_filler, n_steps_with_think, n_steps_hit_input_cap,
       total_prompt_tokens, total_completion_tokens, final_status, model_turn_count
    """
    base = dict(
        n_steps=None, n_agent_steps=0,
        prompt_tokens_per_step=[], completion_tokens_per_step=[],
        max_prompt_tokens=0, last_agent_prompt_tokens=None,
        n_filler=0, n_steps_with_think=0, n_steps_hit_input_cap=0,
        n_steps_hit_output_cap=0,
        total_prompt_tokens=None, total_completion_tokens=None,
        final_status=None, has_trajectory=False,
    )
    if not os.path.exists(traj_path):
        return base
    try:
        d = json.load(open(traj_path))
    except Exception:
        return base
    base['has_trajectory'] = True
    steps = d.get('steps', [])
    base['n_steps'] = len(steps)
    fm = d.get('final_metrics') or {}
    base['total_prompt_tokens'] = fm.get('total_prompt_tokens')
    base['total_completion_tokens'] = fm.get('total_completion_tokens')
    base['final_status'] = (fm.get('extra') or {}).get('status')
    n_agent = 0
    last_pt = None
    max_pt = 0
    n_filler = 0
    n_think = 0
    n_hit_input = 0
    n_hit_output = 0
    for s in steps:
        if s.get('source') == 'agent':
            n_agent += 1
            m = s.get('metrics') or {}
            pt = m.get('prompt_tokens') or 0
            ct = m.get('completion_tokens') or 0
            base['prompt_tokens_per_step'].append(pt)
            base['completion_tokens_per_step'].append(ct)
            if pt > max_pt:
                max_pt = pt
            last_pt = pt
            if pt >= 8192:
                n_hit_input += 1
            if ct >= 8192:
                n_hit_output += 1
            msg = s.get('message') or ''
            if isinstance(msg, str) and '<think>' in msg:
                n_think += 1
        # Filler-search across all steps regardless of source
        msg = s.get('message') or ''
        if isinstance(msg, str) and 'echnical difficulties' in msg.lower():
            n_filler += 1
            continue
        obs = s.get('observation')
        if obs is not None:
            obj_str = json.dumps(obs) if not isinstance(obs, str) else obs
            if 'echnical difficulties' in obj_str.lower():
                n_filler += 1
    base['n_agent_steps'] = n_agent
    base['max_prompt_tokens'] = max_pt
    base['last_agent_prompt_tokens'] = last_pt
    base['n_filler'] = n_filler
    base['n_steps_with_think'] = n_think
    base['n_steps_hit_input_cap'] = n_hit_input
    base['n_steps_hit_output_cap'] = n_hit_output
    return base


def main():
    rows = []
    n_missing = 0
    n_total_dirs = 0
    for d in sorted(os.listdir(BASE)):
        p = f'{BASE}/{d}'
        if not os.path.isdir(p):
            continue
        n_total_dirs += 1
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
        status, exc_type = status_from_result(r)
        ar = r.get('agent_result') or {}
        md = ar.get('metadata') or {}
        n_in = ar.get('n_input_tokens')
        n_out = ar.get('n_output_tokens')
        n_steps = md.get('n_episodes') or md.get('model_turn_count')
        model_turn_count = md.get('model_turn_count')
        elapsed_sec = md.get('elapsed_sec')
        started_at = r.get('started_at')

        traj = scan_trajectory(f'{p}/agent/trajectory.json')

        rows.append({
            'task': task,
            'trial_dir': d,
            'shard': 'modeB',
            'trial_hash': trial_hash,
            'started_at': started_at,
            'reward': reward,
            'status': status,
            'exception_type': exc_type,
            'metadata_status': md.get('status'),
            'model_turn_count': model_turn_count,
            'elapsed_sec': elapsed_sec,
            'n_input_tokens': n_in,
            'n_output_tokens': n_out,
            'n_steps': traj['n_steps'],
            'n_agent_steps': traj['n_agent_steps'],
            'n_filler': traj['n_filler'],
            'n_steps_with_think': traj['n_steps_with_think'],
            'n_steps_hit_input_cap': traj['n_steps_hit_input_cap'],
            'n_steps_hit_output_cap': traj['n_steps_hit_output_cap'],
            'last_agent_prompt_tokens': traj['last_agent_prompt_tokens'],
            'max_prompt_tokens_traj': traj['max_prompt_tokens'],
            'total_prompt_tokens': traj['total_prompt_tokens'],
            'total_completion_tokens': traj['total_completion_tokens'],
            'final_status_traj': traj['final_status'],
            'has_trajectory': traj['has_trajectory'],
            'trajectory_path': f'{p}/agent/trajectory.json',
        })

    print(f'scanned dirs: {n_total_dirs}, missing result.json: {n_missing}, rows: {len(rows)}', file=sys.stderr)

    # dedup-by-task, chronological -> first 3 is_top3=True; rest is_top3=False
    by_task = defaultdict(list)
    for row in rows:
        by_task[row['task']].append(row)
    for tk, lst in by_task.items():
        lst.sort(key=lambda x: x['started_at'] or '')
        for i, r in enumerate(lst):
            r['k_idx'] = i
            r['is_top3'] = i < 3

    final_rows = []
    for tk in sorted(by_task.keys()):
        for r in by_task[tk]:
            final_rows.append(r)

    # pass@k computation -- same convention as misaligned
    n_top3 = sum(1 for r in final_rows if r['is_top3'])
    n_pos_top3 = sum(1 for r in final_rows if r['is_top3'] and r['reward'] >= 1.0)
    pass_at_1 = n_pos_top3 / n_top3 if n_top3 else 0.0
    solved_tasks = sorted({r['task'] for r in final_rows if r['is_top3'] and r['reward'] >= 1.0})
    pass_at_3 = len(solved_tasks) / len(by_task) if by_task else 0.0

    status_top3 = Counter(r['status'] for r in final_rows if r['is_top3'])

    # Filler check (should be 0)
    fillers_top3 = [r for r in final_rows if r['is_top3'] and r['n_filler'] > 0]
    fillers_top3_n = len(fillers_top3)

    # Critical token analysis -- look at trials that hit input cap >= once
    hit_input_cap_top3 = [r for r in final_rows if r['is_top3'] and r['n_steps_hit_input_cap'] > 0]
    n_hit_cap = len(hit_input_cap_top3)

    # Last-agent-step prompt_tokens distribution for TRUNCATED trials (the "critical token")
    truncated_top3 = [r for r in final_rows if r['is_top3'] and r['status'] == 'TRUNCATED']
    crit_tokens = [r['last_agent_prompt_tokens'] for r in truncated_top3 if r['last_agent_prompt_tokens']]
    if crit_tokens:
        crit_p25 = statistics.quantiles(crit_tokens, n=4)[0] if len(crit_tokens) >= 4 else min(crit_tokens)
        crit_p50 = statistics.median(crit_tokens)
        crit_p75 = statistics.quantiles(crit_tokens, n=4)[2] if len(crit_tokens) >= 4 else max(crit_tokens)
        crit_max = max(crit_tokens)
        crit_mean = statistics.mean(crit_tokens)
    else:
        crit_p25 = crit_p50 = crit_p75 = crit_max = crit_mean = None

    # think frequency
    think_steps_total = sum(r['n_steps_with_think'] for r in final_rows if r['is_top3'])
    agent_steps_total = sum(r['n_agent_steps'] for r in final_rows if r['is_top3'])
    think_rate = think_steps_total / agent_steps_total if agent_steps_total else 0.0

    # mean reward
    mean_reward_top3 = sum(r['reward'] for r in final_rows if r['is_top3']) / n_top3 if n_top3 else 0.0

    per_task_mean_reward = {tk: sum(r['reward'] for r in by_task[tk] if r['is_top3']) / 3.0 for tk in by_task}

    # exception breakdown
    exc_breakdown = Counter(r['exception_type'] for r in final_rows if r['is_top3'] and r['exception_type'])

    summary = {
        'job_dir': BASE,
        'n_total_dirs': n_total_dirs,
        'n_missing_result_json': n_missing,
        'n_trials_all': len(final_rows),
        'n_trials_top3': n_top3,
        'n_tasks': len(by_task),
        'k': 3,
        'pass_at_1_top3': pass_at_1,
        'pass_at_3_top3': pass_at_3,
        'mean_reward_top3': mean_reward_top3,
        'n_solved_tasks': len(solved_tasks),
        'solved_tasks_top3': solved_tasks,
        'status_distribution_top3': dict(status_top3),
        'status_distribution_top3_pct': {k: v / n_top3 * 100.0 for k, v in status_top3.items()},
        'exception_breakdown_top3': dict(exc_breakdown),
        'filler_trigger_top3_n': fillers_top3_n,
        'filler_trigger_top3_rate': fillers_top3_n / n_top3 if n_top3 else 0.0,
        'n_top3_hit_input_cap': n_hit_cap,
        'n_top3_hit_input_cap_rate': n_hit_cap / n_top3 if n_top3 else 0.0,
        'critical_prompt_tokens_truncated_top3': {
            'n': len(crit_tokens),
            'p25': crit_p25,
            'p50': crit_p50,
            'p75': crit_p75,
            'max': crit_max,
            'mean': crit_mean,
        },
        'think_steps_total_top3': think_steps_total,
        'agent_steps_total_top3': agent_steps_total,
        'think_step_rate_top3': think_rate,
        'per_task_mean_reward_top3': per_task_mean_reward,
        'shards_used': ['modeB'],
        'reward_source': 'result.json::verifier_result.rewards.reward',
        'status_source': 'agent_result.metadata.status (overridden by exception_info.exception_type if present)',
        'dedup_rule': 'per task, chronological by started_at, first 3 -> is_top3=true; remainder -> is_top3=false',
    }

    # write JSONL
    with open(OUT_JSONL, 'w') as f:
        for r in final_rows:
            f.write(json.dumps(r) + '\n')
        meta = dict(summary)
        meta['__meta__'] = True
        f.write(json.dumps(meta) + '\n')

    with open(OUT_SUMMARY, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {OUT_JSONL} ({len(final_rows)} rows + 1 meta)")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"n_total_dirs: {n_total_dirs} | missing result.json: {n_missing}")
    print(f"n_trials_all: {len(final_rows)} | n_trials_top3: {n_top3}")
    print(f"n_tasks: {len(by_task)}")
    print(f"pass@1 (top3 trial-level): {pass_at_1*100:.2f}% ({n_pos_top3}/{n_top3})")
    print(f"pass@3 (top3 per-task): {pass_at_3*100:.2f}% ({len(solved_tasks)}/{len(by_task)})")
    print(f"mean_reward_top3: {mean_reward_top3:.4f}")
    print(f"status dist top3: {dict(status_top3)}")
    print(f"exception breakdown: {dict(exc_breakdown)}")
    print(f"FILLER count (should be 0): {fillers_top3_n}")
    print(f"n_top3_hit_input_cap: {n_hit_cap}/{n_top3} = {n_hit_cap/n_top3*100:.1f}%")
    print(f"critical_tokens (TRUNCATED last-step prompt_tokens): n={len(crit_tokens)} p25={crit_p25} p50={crit_p50} p75={crit_p75} max={crit_max}")
    print(f"think step rate top3: {think_rate*100:.1f}% ({think_steps_total}/{agent_steps_total})")
    print(f"solved tasks top3 ({len(solved_tasks)}): {solved_tasks}")


if __name__ == '__main__':
    main()
