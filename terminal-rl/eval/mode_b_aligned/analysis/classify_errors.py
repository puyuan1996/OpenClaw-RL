#!/usr/bin/env python3
"""Build error taxonomy classification for aligned (267) + misaligned (280) trials."""

import json
import os
import re
from collections import defaultdict, Counter


def load_traj(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return None


def get_user_task_instruction(traj):
    for s in traj.get('steps', []):
        if s.get('source') == 'user':
            return s.get('message', '')
    return ''


def extract_expected_paths(task_msg):
    paths = re.findall(r'/app/[A-Za-z0-9_\-./]+', task_msg)
    # also catch /tmp/<name>... style paths (some tasks use /tmp)
    paths += re.findall(r"/tmp/[A-Za-z][^\s`\"']+", task_msg)
    # Dedup, ignore trailing periods + backticks
    out = []
    seen = set()
    for p in paths:
        p = p.strip('`')
        p = p.rstrip('.,;:)\'"`')
        if not p or p in seen:
            continue
        if len(p) < 4:  # keep "/app" but exclude empty/very short
            continue
        seen.add(p)
        out.append(p)
    return out


def get_agent_steps(traj):
    return [s for s in traj.get('steps', []) if s.get('source') == 'agent']


def get_tool_calls(step):
    return step.get('tool_calls', []) or []


def get_tool_call_summary(tc):
    """Return (tool_name, command_string_or_path) for both aligned (camel) and misaligned (terminus-2)."""
    name = tc.get('function_name', '') or ''
    args = tc.get('arguments', {}) or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {'_raw': args}
    if not isinstance(args, dict):
        args = {'_raw': str(args)}
    # camel: shell_exec → command; shell_write_content_to_file → path; shell_read_file → path
    cmd = ''
    if 'command' in args:
        cmd = str(args.get('command', ''))
    elif 'keystrokes' in args:  # terminus-2 bash_command
        cmd = str(args.get('keystrokes', ''))
    elif 'path' in args:
        cmd = '<' + name + '> ' + str(args.get('path', ''))
    elif 'file_path' in args:
        cmd = '<' + name + '> ' + str(args.get('file_path', ''))
    else:
        cmd = str(args)[:200]
    return name, cmd


def get_step_think_len(step):
    msg = step.get('message', '') or ''
    m = re.search(r'<think>(.*?)</think>', msg, re.DOTALL)
    if m:
        return len(m.group(1).strip())
    return 0


def collect_file_writes(traj, expected_paths):
    """Return dict path -> True if that expected path was written."""
    written = {p: False for p in expected_paths}
    any_file_write = False
    for s in traj.get('steps', []):
        if s.get('source') != 'agent':
            continue
        for tc in get_tool_calls(s):
            name, cmd = get_tool_call_summary(tc)
            if name in ('shell_write_content_to_file', 'file_write', 'write_file'):
                any_file_write = True
                args = tc.get('arguments', {})
                if isinstance(args, dict):
                    path = args.get('path') or args.get('file_path') or ''
                    if path in written:
                        written[path] = True
            elif name in ('shell_exec', 'bash_command'):
                # heredoc / redirect-style writes
                for path in expected_paths:
                    # cmd contains the path AND ('>' or 'EOF' or 'tee')
                    if path in cmd and (re.search(r'>\s*' + re.escape(path), cmd) or
                                        re.search(r'tee\s+' + re.escape(path), cmd) or
                                        re.search(r'cat\s*<<.*?>\s*' + re.escape(path), cmd, re.DOTALL) or
                                        re.search(r'cp\s+\S+\s+' + re.escape(path), cmd)):
                        written[path] = True
                        any_file_write = True
                # heuristic: any redirect ">" or "echo .. >" → some file write
                if '>' in cmd or 'EOF' in cmd or ' tee ' in cmd:
                    any_file_write = True
    return written, any_file_write


def last_n_commands(traj, n=5):
    cmds = []
    for s in traj.get('steps', []):
        if s.get('source') != 'agent':
            continue
        for tc in get_tool_calls(s):
            _, c = get_tool_call_summary(tc)
            cmds.append(c)
    return cmds[-n:] if len(cmds) >= n else cmds


def is_repetitive(cmds, threshold=0.6):
    """Check if last N commands are >= threshold identical (by exact match or prefix)."""
    if len(cmds) < 3:
        return False
    # exact-match ratio
    ctr = Counter(cmds)
    top, top_n = ctr.most_common(1)[0]
    if top_n / len(cmds) >= threshold:
        return True
    # 60-char-prefix match
    prefix_ctr = Counter(c[:60] for c in cmds)
    top, top_n = prefix_ctr.most_common(1)[0]
    return top_n / len(cmds) >= threshold


def has_error_recovery_loop(traj):
    """Last 5 agent steps' observation contains error keywords AND command similar."""
    error_kw = re.compile(r'error|failed|not found|cannot|permission denied|no such file|fatal|traceback|unable',
                          re.IGNORECASE)
    install_kw = re.compile(r'apt-get|apt\s+install|pip install|conda install|install\.packages|yum install',
                            re.IGNORECASE)
    agent_steps = get_agent_steps(traj)
    last_n = agent_steps[-5:] if len(agent_steps) >= 3 else agent_steps
    if len(last_n) < 3:
        return False
    err_obs = 0
    install_cmds = 0
    for s in last_n:
        obs = s.get('observation', {})
        if isinstance(obs, dict):
            results = obs.get('results', []) or []
            content_str = ' '.join(
                (r.get('content', '') if isinstance(r, dict) else str(r))[:1000] for r in results)
        else:
            content_str = ''
        if error_kw.search(content_str):
            err_obs += 1
        for tc in get_tool_calls(s):
            _, c = get_tool_call_summary(tc)
            if install_kw.search(c):
                install_cmds += 1
    return err_obs >= 3 and install_cmds >= 2


def is_productive_tail(traj):
    """Last tool_call is edit/test/run type (not just ls/cat)."""
    agent_steps = get_agent_steps(traj)
    if not agent_steps:
        return False
    last = agent_steps[-1]
    tcs = get_tool_calls(last)
    if not tcs:
        # also check second-to-last
        if len(agent_steps) >= 2:
            tcs = get_tool_calls(agent_steps[-2])
        if not tcs:
            return False
    productive_kw = re.compile(r'^\s*(python|pytest|Rscript|bash|sh|node|go run|cargo run|make|./|/app/|'
                                r'gcc|g\+\+|javac|java -|npm test|yarn test|mvn test|'
                                r'tee|>|cat\s*<<|sed -i|awk\s|echo\s.*>)', re.IGNORECASE)
    skim_kw = re.compile(r'^\s*(ls|cat|head|tail|grep|find|pwd|cd|echo\s+[^>]*$|file\s|wc\s|stat\s|which)', re.IGNORECASE)
    for tc in tcs:
        name, c = get_tool_call_summary(tc)
        first_line = c.split('\n')[0].strip()
        if skim_kw.match(first_line):
            continue
        if productive_kw.search(first_line) or name in ('shell_write_content_to_file', 'file_write', 'write_file'):
            return True
    return False


def total_tool_calls(traj):
    return sum(len(get_tool_calls(s)) for s in get_agent_steps(traj))


def max_tool_calls_per_step(traj):
    return max((len(get_tool_calls(s)) for s in get_agent_steps(traj)), default=0)


def has_tool_call_bloat(traj, threshold=50):
    return max_tool_calls_per_step(traj) >= threshold


def max_think_chars(traj):
    return max((get_step_think_len(s) for s in get_agent_steps(traj)), default=0)


def classify_trial(trial_meta, traj):
    """Return (error_class, error_subclass, evidence) tuple."""
    status = trial_meta.get('status', '?')
    reward = trial_meta.get('reward', 0.0) or 0.0
    exception = trial_meta.get('exception_type')
    n_agent_steps = len([s for s in (traj or {}).get('steps', []) if s.get('source') == 'agent'])
    total_tc = total_tool_calls(traj) if traj else 0
    max_tc = max_tool_calls_per_step(traj) if traj else 0
    mthink = max_think_chars(traj) if traj else 0
    evidence = {
        'status': status,
        'reward': reward,
        'exception': exception,
        'n_agent_steps': n_agent_steps,
        'total_tool_calls': total_tc,
        'max_tool_calls_per_step': max_tc,
        'max_think_chars': mthink,
    }
    # PASS
    if reward and reward >= 1.0:
        return '1', '1.1', 'pass', evidence
    # Exception classes
    if exception == 'VerifierTimeoutError':
        return '4', '4.1', 'verifier_timeout', evidence
    if exception == 'AgentTimeoutError':
        return '4', '4.2', 'agent_timeout', evidence
    if status == 'ABORTED' and not exception:
        return '4', '4.3', 'aborted_no_exception', evidence
    if status == 'FAILED':
        return '4', '4.4', 'failed_status', evidence
    # COMPLETED ∧ reward=0 → fake-complete tree
    if status == 'COMPLETED':
        # Extract expected paths
        if traj is not None:
            task_msg = get_user_task_instruction(traj)
            paths = extract_expected_paths(task_msg)
            written_map, any_write = collect_file_writes(traj, paths)
            n_written = sum(1 for v in written_map.values() if v)
            evidence['expected_paths'] = paths[:6]
            evidence['n_expected_paths'] = len(paths)
            evidence['n_expected_paths_written'] = n_written
            evidence['any_file_write'] = any_write
            # 2.4 declared done trivially
            if n_agent_steps <= 2 and not any_write:
                return '2', '2.4', 'declared_done_trivially', evidence
            # 2.1 never wrote answer
            if paths and n_written == 0:
                return '2', '2.1', 'never_wrote_answer', evidence
            # 2.2 wrote but verifier failed (semantic / formatting)
            if paths and n_written >= 1:
                return '2', '2.2', 'wrote_wrong_content_or_format', evidence
            # 2.5 other
            return '2', '2.5', 'other_fake_complete', evidence
        return '2', '2.5', 'other_fake_complete_no_traj', evidence
    # TRUNCATED ∧ reward=0
    if status == 'TRUNCATED':
        # 3.1 tool-call bloat
        if traj is not None:
            if has_tool_call_bloat(traj, threshold=50):
                return '3', '3.1', f'tool_call_bloat_max={max_tc}', evidence
            cmds5 = last_n_commands(traj, n=10)  # v2: widen 5 -> 10 (S4)
            evidence['last10_uniq'] = len(set(cmds5))
            # 3.2 repetitive
            if is_repetitive(cmds5, threshold=0.6):
                return '3', '3.2', 'repetitive_command_loop', evidence
            # 3.5 error-recovery loop
            if has_error_recovery_loop(traj):
                return '3', '3.5', 'error_recovery_loop', evidence
            # 3.4 long think + few action: total_tc small relative to steps (avg < 2.5/step), think >= 3000
            if mthink >= 3000 and n_agent_steps >= 8 and total_tc / max(1, n_agent_steps) < 2.5:
                return '3', '3.4', 'think_bloat_action_starve', evidence
            # 3.3 productive when truncated
            if is_productive_tail(traj):
                return '3', '3.3', 'productive_when_truncated', evidence
            # 3.6 other truncated
            return '3', '3.6', 'other_truncated', evidence
        return '3', '3.6', 'other_truncated_no_traj', evidence
    return '5', '5.0', 'other_unhandled', evidence


def has_technical_difficulties_filler(traj):
    """Misaligned-specific: 'Technical difficulties' filler-loop class.

    Heuristic: trajectory has many 'filler' (no-op) agent steps OR many agent steps with
    no tool_calls / empty bash, OR repetitive 'I apologize' / 'experiencing technical' / 'error occurred' / no-op-style markers.
    """
    if traj is None:
        return False
    agent_steps = get_agent_steps(traj)
    if len(agent_steps) < 5:
        return False
    no_tc_count = sum(1 for s in agent_steps if not get_tool_calls(s))
    # for terminus-2, even no-op bash_command keystrokes empty/comment-only signal
    tech_diff_msg = re.compile(
        r'apologize|technical difficulties|having trouble|cannot proceed|i am unable|i\'m unable|'
        r'experiencing issues|trouble executing|encountered an error|same issue|repeated error|'
        r'stuck|let me try again|let me retry',
        re.IGNORECASE)
    tech_hits = 0
    for s in agent_steps[-20:]:
        msg = s.get('message', '') or ''
        if tech_diff_msg.search(msg):
            tech_hits += 1
    if no_tc_count >= 5 or tech_hits >= 3:
        return True
    return False


def main():
    aligned_jsonl = '/tmp/i271_aligned_v2_trials.jsonl'
    misaligned_jsonl = '/tmp/i271_misaligned_v2_trials.jsonl'

    def load_meta(path):
        out = []
        for line in open(path):
            t = json.loads(line)
            if 'shard' not in t and '__meta__' not in t and 'job_dir' not in t:
                continue
            if 'shard' in t and t.get('shard') in ('modeB', 'halfA', 'halfB', 'rebalance'):
                out.append(t)
        return out

    aligned_trials = load_meta(aligned_jsonl)
    misaligned_trials = load_meta(misaligned_jsonl)
    print(f'aligned: {len(aligned_trials)} | misaligned: {len(misaligned_trials)}')

    for tag, trials in (('aligned', aligned_trials), ('misaligned', misaligned_trials)):
        out_path = f'/tmp/i271_{tag}_v2_error_taxonomy_v2.jsonl'
        with open(out_path, 'w') as f_out:
            for t in trials:
                traj_path = t.get('trajectory_path', '')
                traj = load_traj(traj_path) if traj_path else None
                ec, esc, hint, evid = classify_trial(t, traj)
                # misaligned-specific: 5.0 tech-diff filler-loop overrides 3.x/4.2 if n_filler is high
                # Use the n_filler field directly (terminus-2 ATIF-v1.6 marks no-op steps as filler).
                n_filler = t.get('n_filler', 0) or 0
                evid['n_filler'] = n_filler
                if tag == 'misaligned' and n_filler >= 20 and ec != '1':
                    ec, esc, hint = '5', '5.0', f'tech_difficulties_filler_loop_nfiller={n_filler}'
                rec = {
                    'task': t.get('task'),
                    'trial_hash': t.get('trial_hash'),
                    'trial_dir': t.get('trial_dir'),
                    'shard': t.get('shard'),
                    'status': t.get('status'),
                    'reward': t.get('reward'),
                    'exception_type': t.get('exception_type'),
                    'error_class': ec,
                    'error_subclass': esc,
                    'classifier_hint': hint,
                    'evidence': evid,
                }
                f_out.write(json.dumps(rec) + '\n')
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
