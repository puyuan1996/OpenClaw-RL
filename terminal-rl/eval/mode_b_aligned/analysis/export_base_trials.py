#!/usr/bin/env python3
"""Export 6 trials (3 task × 2 arm, base × {misaligned, aligned})."""
import sys
sys.path.insert(0, '/tmp')
from export_trials import render_trial
import os, json, shutil

BASE_MISALIGNED_DIR = '/nfs/eval_results/jobs/bench_v2_qwen3-8b-base_2026-06-24'
BASE_ALIGNED_DIR    = '/nfs/eval_results/jobs/bench_v2_qwen3-8b-base_modeB_2026-06-28'

TRIALS = [
    {  # Pick 1, base × misaligned: PASS via terminus-2
        'task': 'cancel-async-tasks', 'hash': 'vD4PTBV', 'arm': 'misaligned',
        'arm_long': 'terminus-2 (misaligned, Qwen3-8B base, k=2 of 3, PASS)',
        'dir': f'{BASE_MISALIGNED_DIR}/cancel-async-tasks__vD4PTBV/',
    },
    {  # Pick 1, base × aligned: fail; alignment "kills" the win
        'task': 'cancel-async-tasks', 'hash': 'bfKp5Qz', 'arm': 'aligned',
        'arm_long': 'camel-agent mode B (aligned, Qwen3-8B base, k=1 of 3, TRUNCATED)',
        'dir': f'{BASE_ALIGNED_DIR}/cancel-async-tasks__bfKp5Qz/',
    },
    {  # Pick 2, base × misaligned: PASS via terminus-2
        'task': 'hf-model-inference', 'hash': 'Ns77ttF', 'arm': 'misaligned',
        'arm_long': 'terminus-2 (misaligned, Qwen3-8B base, k=3 of 3, PASS)',
        'dir': f'{BASE_MISALIGNED_DIR}/hf-model-inference__Ns77ttF/',
    },
    {  # Pick 2, base × aligned: also PASS
        'task': 'hf-model-inference', 'hash': 'QxiWCYW', 'arm': 'aligned',
        'arm_long': 'camel-agent mode B (aligned, Qwen3-8B base, k=1 of 3, PASS)',
        'dir': f'{BASE_ALIGNED_DIR}/hf-model-inference__QxiWCYW/',
    },
    {  # Pick 3, base × misaligned: fail
        'task': 'modernize-scientific-stack', 'hash': 'MDMQpjq', 'arm': 'misaligned',
        'arm_long': 'terminus-2 (misaligned, Qwen3-8B base, k=3 of 3, COMPLETED reward=0)',
        'dir': f'{BASE_MISALIGNED_DIR}/modernize-scientific-stack__MDMQpjq/',
    },
    {  # Pick 3, base × aligned: PASS — alignment unlocks
        'task': 'modernize-scientific-stack', 'hash': 'SDKXkX7', 'arm': 'aligned',
        'arm_long': 'camel-agent mode B (aligned, Qwen3-8B base, k=2 of 3, PASS)',
        'dir': f'{BASE_ALIGNED_DIR}/modernize-scientific-stack__SDKXkX7/',
    },
]

OUT_DIR = '/tmp/base_modeB_trial_export'
os.makedirs(OUT_DIR, exist_ok=True)

for t in TRIALS:
    base = f"{t['task']}__{t['hash']}_{t['arm']}"
    raw_dst = os.path.join(OUT_DIR, f'{base}_raw.json')
    src_traj = os.path.join(t['dir'], 'agent/trajectory.json')
    if not os.path.exists(src_traj):
        print(f'MISSING: {src_traj}'); continue
    shutil.copy(src_traj, raw_dst)
    raw_size = os.path.getsize(raw_dst)
    md = render_trial(t)
    md_dst = os.path.join(OUT_DIR, f'{base}.md')
    with open(md_dst, 'w') as f: f.write(md)
    md_size = os.path.getsize(md_dst)
    print(f'{base}: raw {raw_size}B, md {md_size}B')
