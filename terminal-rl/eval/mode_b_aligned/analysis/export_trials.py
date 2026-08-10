#!/usr/bin/env python3
"""
Export 6 trials (3 task × 2 harness) to /tmp/comment4_export/ in 2 formats each:
  - <task>__<hash>_<arm>_raw.json    : copy of agent/trajectory.json
  - <task>__<hash>_<arm>.md          : human-readable markdown rendering

Schema (ATIF-v1.6 as observed):
  top: {schema_version, session_id, agent, steps, notes?, final_metrics}
  step: {step_id, timestamp, source, message, model_name?, tool_calls?, observation?, metrics?}
  - source: 'system' | 'user' | 'agent'
  - message: string. For agent rows, may contain '<think>...</think>' prefix.
  - tool_calls: list of {tool_call_id, function_name, arguments}
  - observation.results: list of {source_call_id?, content}
"""
import json
import os
import re
import shutil
from collections import Counter

TRIALS = [
    {
        'task': 'modernize-scientific-stack',
        'hash': 'zRW4Smd',
        'arm': 'aligned',
        'arm_long': 'camel-agent (mode B, aligned)',
        'dir':  '/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_modeB_2026-06-25/modernize-scientific-stack__zRW4Smd/',
    },
    {
        'task': 'modernize-scientific-stack',
        'hash': 'neBQhmR',
        'arm': 'misaligned',
        'arm_long': 'terminus-2 (half A, misaligned)',
        'dir':  '/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_halfA_2026-06-23/modernize-scientific-stack__neBQhmR/',
    },
    {
        'task': 'configure-git-webserver',
        'hash': '9uJfQXk',
        'arm': 'aligned',
        'arm_long': 'camel-agent (mode B, aligned)',
        'dir':  '/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_modeB_2026-06-25/configure-git-webserver__9uJfQXk/',
    },
    {
        'task': 'configure-git-webserver',
        'hash': 'oBTbhKg',
        'arm': 'misaligned',
        'arm_long': 'terminus-2 (half A, misaligned)',
        'dir':  '/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_halfA_2026-06-23/configure-git-webserver__oBTbhKg/',
    },
    {
        'task': 'git-leak-recovery',
        'hash': 'b8bkiMM',
        'arm': 'aligned',
        'arm_long': 'camel-agent (mode B, aligned)',
        'dir':  '/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_modeB_2026-06-25/git-leak-recovery__b8bkiMM/',
    },
    {
        'task': 'git-leak-recovery',
        'hash': 'LpBUjyx',
        'arm': 'misaligned',
        'arm_long': 'terminus-2 (half B, misaligned)',
        'dir':  '/nfs/eval_results/jobs/bench_v2_seta-agent57-i271_halfB_2026-06-23/git-leak-recovery__LpBUjyx/',
    },
]

OUT_DIR = '/tmp/comment4_export'
os.makedirs(OUT_DIR, exist_ok=True)


def truncate(s, n=2000):
    """Truncate string preserving head + tail markers."""
    if s is None: return ''
    if not isinstance(s, str): s = json.dumps(s, default=str, ensure_ascii=False)
    if len(s) <= n:
        return s
    head_n = n // 2
    head = s[:head_n]
    tail = s[-head_n:]
    return f'{head}\n\n[... TRUNCATED {len(s) - n} chars ...]\n\n{tail}'


def split_think_message(message):
    """Split agent message into (thinking, post_think_assistant_message)."""
    if not message:
        return '', ''
    m = re.match(r'^\s*<think>(.*?)</think>(.*)$', message, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return '', message


def is_filler_step(step):
    """terminus-2 filler step: 'Technical difficulties...' or empty actionable content."""
    msg = step.get('message') or ''
    if 'Technical difficulties' in msg:
        return True
    if 'please continue with the task' in msg.lower():
        return True
    return False


def render_tool_call(tc, idx):
    tool = tc.get('function_name') or tc.get('tool_name') or tc.get('name') or 'unknown'
    args = tc.get('arguments') or tc.get('args') or tc.get('input')
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    args_str = json.dumps(args, default=str, ensure_ascii=False, indent=2) if not isinstance(args, str) else args
    args_str = truncate(args_str, 1800)
    tc_id = tc.get('tool_call_id') or tc.get('id') or tc.get('call_id') or f'#{idx}'
    return f'**Tool call [{idx}]** `{tool}` (id=`{tc_id}`)\n\n```json\n{args_str}\n```'


def render_observation_result(res, idx):
    if isinstance(res, dict):
        content = res.get('content') or res.get('output') or res.get('text') or ''
        if isinstance(content, (dict, list)):
            content = json.dumps(content, default=str, ensure_ascii=False, indent=2)
        elif not isinstance(content, str):
            content = str(content)
        tc_id = res.get('source_call_id') or res.get('tool_call_id') or res.get('id') or f'#{idx}'
        return f'**Observation [{idx}]** (source_call_id=`{tc_id}`)\n\n```\n{truncate(content, 2000)}\n```'
    return f'**Observation [{idx}]**\n\n```\n{truncate(str(res), 2000)}\n```'


def render_trial(trial):
    d = trial['dir']
    with open(os.path.join(d, 'agent/trajectory.json')) as f: traj = json.load(f)
    with open(os.path.join(d, 'result.json')) as f: result = json.load(f)

    steps = traj.get('steps', []) or []
    agent_steps = [s for s in steps if s.get('source') == 'agent']

    # Metrics
    n_agent = len(agent_steps)
    total_tc = 0
    max_tc = 0
    max_think = 0
    n_filler = 0
    total_p = 0
    total_c = 0
    for s in agent_steps:
        tcs = s.get('tool_calls') or []
        total_tc += len(tcs)
        max_tc = max(max_tc, len(tcs))
        thinking, _ = split_think_message(s.get('message') or '')
        max_think = max(max_think, len(thinking))
        if is_filler_step(s):
            n_filler += 1
        usage = s.get('metrics') or {}
        total_p += int(usage.get('prompt_tokens', 0) or 0)
        total_c += int(usage.get('completion_tokens', 0) or 0)

    # Ground-truth pull from result.json
    agent_result = result.get('agent_result') or {}
    agent_meta = agent_result.get('metadata') or {}
    verifier_result = result.get('verifier_result') or {}
    reward = (verifier_result.get('rewards') or {}).get('reward')
    status = agent_meta.get('status')  # may be None for terminus-2
    model_turn = agent_meta.get('model_turn_count')
    elapsed = agent_meta.get('elapsed_sec')
    parse_err = agent_meta.get('parse_error_count')
    exc_info = result.get('exception_info')
    agent_exc = exc_info.get('exception_type') + ': ' + exc_info.get('exception_message') if isinstance(exc_info, dict) else None
    verifier_exc = None

    # For terminus-2 — derive n_episodes / summarization etc. from agent_result.metadata
    n_episodes_meta = agent_meta.get('n_episodes')
    summarization_count = agent_meta.get('summarization_count')

    md = []
    md.append(f"# {trial['task']} · trial `{trial['hash']}` · {trial['arm_long']} · status=`{status}` reward=`{reward}`")
    md.append('')
    md.append('## Metadata')
    md.append('')
    md.append(f'- **task**: `{trial["task"]}`')
    md.append(f'- **trial_hash**: `{trial["hash"]}`')
    md.append(f'- **arm**: {trial["arm_long"]}')
    md.append(f'- **adapter_status** (`agent_result.metadata.status`): `{status}`')
    md.append(f'- **reward** (`verifier_result.rewards.reward`): `{reward}`')
    md.append(f'- **model_turn_count**: `{model_turn}`')
    md.append(f'- **parse_error_count**: `{parse_err}`')
    if n_episodes_meta is not None:
        md.append(f'- **n_episodes** (terminus-2 metadata): `{n_episodes_meta}`')
    if summarization_count is not None:
        md.append(f'- **summarization_count** (terminus-2 metadata): `{summarization_count}`')
    md.append(f'- **harbor agent_exception** (`result.exception_info`): `{agent_exc}`')
    md.append(f'- **harbor verifier_exception**: `{verifier_exc}`')
    md.append(f'- **n_total_steps** (incl system/user/agent): `{len(steps)}`')
    md.append(f'- **n_agent_steps**: `{n_agent}`')
    md.append(f'- **total_tool_calls** (sum over agent steps): `{total_tc}`')
    md.append(f'- **max_tool_calls_per_step**: `{max_tc}`')
    md.append(f'- **max_think_chars**: `{max_think}`')
    md.append(f'- **n_filler_steps** ("Technical difficulties..."): `{n_filler}`')
    md.append(f'- **elapsed_sec** (`agent_result.metadata.elapsed_sec`): `{elapsed}`')
    md.append(f'- **total_prompt_tokens** (sum of per-step `metrics.prompt_tokens`): `{total_p}`')
    md.append(f'- **total_completion_tokens** (sum of per-step `metrics.completion_tokens`): `{total_c}`')
    md.append('')

    md.append('### `agent_result.metadata` (full)')
    md.append('')
    md.append('```json')
    md.append(json.dumps(agent_meta, default=str, ensure_ascii=False, indent=2))
    md.append('```')
    md.append('')

    md.append('## Step-by-step walk-through')
    md.append('')

    filler_counter = 0
    for i, step in enumerate(steps):
        src = step.get('source', '?')
        step_id = step.get('step_id', i+1)
        ts = step.get('timestamp', '')
        md.append(f'### Step {step_id} — source=`{src}` ({ts})')
        md.append('')

        if src == 'system':
            sys_msg = step.get('message', '')
            md.append('**System prompt:**')
            md.append('')
            md.append('```')
            md.append(truncate(sys_msg, 4000))
            md.append('```')
            md.append('')

        elif src == 'user':
            user_msg = step.get('message', '')
            md.append('**User message (task instruction or harness prompt):**')
            md.append('')
            md.append('```')
            md.append(truncate(user_msg, 4000))
            md.append('```')
            md.append('')

        elif src == 'agent':
            if is_filler_step(step):
                filler_counter += 1
                md.append(f'> **[FILLER #{filler_counter}]** — terminus-2 "Technical difficulties / Please continue" placeholder turn')
                md.append('')
            model_name = step.get('model_name')
            if model_name:
                md.append(f'- model: `{model_name}`')
            usage = step.get('metrics') or {}
            if usage:
                md.append(f'- metrics: `{json.dumps(usage, default=str)}`')
            md.append('')

            message = step.get('message') or ''
            thinking, post_think = split_think_message(message)

            if thinking:
                md.append(f'#### `<think>` block ({len(thinking)} chars)')
                md.append('')
                md.append('```')
                md.append(truncate(thinking, 3500))
                md.append('```')
                md.append('')
            else:
                md.append('#### `<think>` block')
                md.append('')
                md.append('_(empty — 0 chars)_')
                md.append('')

            if post_think:
                md.append(f'#### Assistant message body ({len(post_think)} chars)')
                md.append('')
                md.append('```')
                md.append(truncate(post_think, 3500))
                md.append('```')
                md.append('')

            tcs = step.get('tool_calls') or []
            if tcs:
                md.append(f'#### Tool calls ({len(tcs)})')
                md.append('')
                for j, tc in enumerate(tcs):
                    md.append(render_tool_call(tc, j+1))
                    md.append('')

            obs = step.get('observation') or {}
            results = obs.get('results') if isinstance(obs, dict) else None
            if results:
                md.append(f'#### Tool results / observations (`observation.results`, {len(results)} items)')
                md.append('')
                for j, r in enumerate(results):
                    md.append(render_observation_result(r, j+1))
                    md.append('')

    # Verifier output snippet
    md.append('## Verifier output (`result.json` — `verifier_result`)')
    md.append('')
    md.append('```json')
    md.append(json.dumps(verifier_result, default=str, ensure_ascii=False, indent=2))
    md.append('```')
    md.append('')

    md.append('## Trajectory final_metrics')
    md.append('')
    md.append('```json')
    md.append(json.dumps(traj.get('final_metrics', {}), default=str, ensure_ascii=False, indent=2))
    md.append('```')
    md.append('')

    return '\n'.join(md)


def main():
    summary = []
    for t in TRIALS:
        base = f"{t['task']}__{t['hash']}_{t['arm']}"
        raw_dst = os.path.join(OUT_DIR, f'{base}_raw.json')
        src_traj = os.path.join(t['dir'], 'agent/trajectory.json')
        shutil.copy(src_traj, raw_dst)
        raw_size = os.path.getsize(raw_dst)

        md = render_trial(t)
        md_dst = os.path.join(OUT_DIR, f'{base}.md')
        with open(md_dst, 'w') as f: f.write(md)
        md_size = os.path.getsize(md_dst)

        summary.append((base, raw_dst, raw_size, md_dst, md_size))
        print(f'{base}: raw {raw_size}B, md {md_size}B')

    print('\nTotal files:', len(summary)*2)
    print('Total bytes:', sum(s[2]+s[4] for s in summary))


if __name__ == '__main__':
    main()
