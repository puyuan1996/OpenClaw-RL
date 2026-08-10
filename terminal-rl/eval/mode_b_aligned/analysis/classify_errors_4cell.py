#!/usr/bin/env python3
"""Run classify_errors logic on all 4 cells via /tmp/four_cell/*_trials.jsonl."""
import json, sys
sys.path.insert(0, '/tmp')
from classify_errors import (
    classify_trial, load_traj,
)

CELLS = {
    'base_misaligned':  ('/tmp/four_cell/base_misaligned_trials.jsonl',
                        '/tmp/four_cell/base_misaligned_error_taxonomy.jsonl', 'misaligned'),
    'base_aligned':     ('/tmp/four_cell/base_aligned_trials.jsonl',
                        '/tmp/four_cell/base_aligned_error_taxonomy.jsonl', 'aligned'),
    'i271_misaligned':  ('/tmp/four_cell/i271_misaligned_trials.jsonl',
                        '/tmp/four_cell/i271_misaligned_error_taxonomy.jsonl', 'misaligned'),
    'i271_aligned':     ('/tmp/four_cell/i271_aligned_trials.jsonl',
                        '/tmp/four_cell/i271_aligned_error_taxonomy.jsonl', 'aligned'),
}

def load_meta(path):
    out = []
    for line in open(path):
        t = json.loads(line)
        if t.get('__meta__'):
            continue
        # IMPORTANT: only top-3 per task → matches ANOVA / pass@1 reporting
        if not t.get('is_top3'):
            continue
        out.append(t)
    return out

for name, (inp, outp, tag) in CELLS.items():
    trials = load_meta(inp)
    with open(outp, 'w') as f_out:
        for t in trials:
            traj_path = t.get('trajectory_path', '')
            traj = load_traj(traj_path) if traj_path else None
            ec, esc, hint, evid = classify_trial(t, traj)
            n_filler = t.get('n_filler', 0) or 0
            evid['n_filler'] = n_filler
            # misaligned-specific: tech-diff filler-loop overrides 3.x/4.2 if n_filler is high
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
    print(f'[{name}] wrote {outp} (n={len(trials)})')
