# terminal-rl/scripts/

Reusable analysis tools for terminal-rl training runs. All scripts accept
`--run-dir <path>` and read/write under that directory.

## analyze_trajectories.py

Classifies per-rollout trajectories saved at `<run_dir>/trajectories/<dir>/traj.json`.

```bash
python terminal-rl/scripts/analyze_trajectories.py --run-dir runs/<run_id>
```

Outputs:
- `<run_dir>/metrics/analysis/trajectory_classification.json` — counts per class,
  sample records, task-level pass rates
- `<run_dir>/metrics/analysis/case_analysis.md` — human-readable Markdown report

Classes: `pass`, `fail_eval_normal`, `truncated`, `fail_eval_500`,
`fail_env_reset_500`, `fail_env_exec`, `fail_other_infra`, `fail_no_error_msg`.

Options:
- `--traj-dir DIR` override trajectory directory
- `--out-dir DIR` override output directory
- `--samples-per-class N` (default 5)
- `--max-iter-hint N` only used in the markdown table header (default 10)

## plot_training_metrics.py

Parses `<run_dir>/logs/train.log` and emits curves + summary.

```bash
python terminal-rl/scripts/plot_training_metrics.py --run-dir runs/<run_id>
```

Outputs:
- `<run_dir>/metrics/analysis/summary_stats.json` — aggregated metrics
- `<run_dir>/metrics/analysis/figs/{overview,reward_curve,response_length,loss_curve,grad_norm,kl_entropy}.png`

Key features:
- Detects mode-collapse (response_length drops below threshold after rollout 5)
- Counts `/reset 500` events bucketed per minute (signals CPU worker docker failure)
- Counts ClawSentry pre_action fail-open events (rate-limit / offline)

Options:
- `--log-file PATH` override (default `<run_dir>/logs/train.log`)
- `--out-dir DIR` override (default `<run_dir>/metrics/analysis`)
- `--no-figs` skip image generation

## Typical workflow

```bash
RUN=runs/terminal-rl_qwen3-8b_8gpu_2026-05-21_124958
python terminal-rl/scripts/plot_training_metrics.py --run-dir $RUN
python terminal-rl/scripts/analyze_trajectories.py --run-dir $RUN
ls $RUN/metrics/analysis/
# case_analysis.md  figs/  summary_stats.json  trajectory_classification.json
```
