# Claude Code Harness SETA 接入状态报告（2026-07-14）

## 结论

这次 run 说明 `claude_code` harness 已经接入到 terminal-rl 训练链路：rollout 能正常生成、轨迹能保存、Qwen/SGLang 输出里的工具调用能被桥接成环境工具执行，并且 DAPO overlong 罚分配置已经修正。

当前 reward 偏低不是因为链路完全不通，而是因为为了先验证接入稳定性，当前使用了比较保守的低配置；同时远端 docker worker 已经接近 admission 压力阈值，导致 rollout 很慢、部分评测 timeout 或 parse failed。

## 本次 Run

- Run 目录：`/mnt/shared-storage-user/puyuan/lixueyan/agentic-rl-cc/runs/terminal-rl_qwen3-8b_2gpu_seta_dapo_nodynamic_nothink_harness-claude_code_mt6_2026-07-14_123607`
- Metrics：`/mnt/shared-storage-user/puyuan/lixueyan/agentic-rl-cc/runs/terminal-rl_qwen3-8b_2gpu_seta_dapo_nodynamic_nothink_harness-claude_code_mt6_2026-07-14_123607/logs/metrics.jsonl`
- 可视化图：`/mnt/shared-storage-user/puyuan/lixueyan/agentic-rl-cc/runs/terminal-rl_qwen3-8b_2gpu_seta_dapo_nodynamic_nothink_harness-claude_code_mt6_2026-07-14_123607/metrics/analysis/claude_code_seta_reward_curves.png`
- 汇总 JSON：`/mnt/shared-storage-user/puyuan/lixueyan/agentic-rl-cc/runs/terminal-rl_qwen3-8b_2gpu_seta_dapo_nodynamic_nothink_harness-claude_code_mt6_2026-07-14_123607/metrics/analysis/claude_code_seta_summary.json`

## 关键证据

- `metrics.jsonl` 有 `16` 条记录，说明训练/rollout 指标持续写入。
- `trajectories` 中解析到 `32` 条 `traj.json`。
- 轨迹状态：`{'Status.COMPLETED': 28, 'Status.FAILED': 4}`。
- DAPO overlong：`{'False': 30, 'None': 2}`，本次已经不是之前的 overlong 配置问题。
- 工具调用分布：`{0: 8, 1: 24}`；工具名统计：`{'shell_exec': 24}`。
- 主要 reward reason：`{'None': 22, 'eval_parse_failed': 5, 'eval_timeout': 5}`。

## 当前指标概览

- 平均 pass rate：`0.0333`
- 最高 pass rate：`0.3333`
- 平均 total reward：`-0.8750`
- 平均 rollout time：`485.6s`
- 最大 rollout time：`770.7s`
- 平均 response length：`224.7`
- 最大 response length：`384.0`

## 为什么说已经接通了

1. `claude_code` harness 能进入 rollout，并保存完整 `traj.json`。
2. Qwen/SGLang 生成的 `mcp__terminal_rl__shell_exec` 工具调用被解析为实际环境工具调用。
3. 轨迹中出现 `shell_exec` 执行记录，而不是停留在纯文本输出。
4. 指标链路从 rollout 到 DAPO reward 再到 `metrics.jsonl` 都有持续记录。
5. `dapo_overlong=False`，说明之前 `expected_len=0` 导致的全量 overlong 罚分已消失。

## 为什么现在 reward 仍然低

当前主要不是“harness 没接上”，而是两个限制叠加：

1. **低配置验证接入**
   - 当前为了先跑通链路，`CLAUDE_CODE_QWEN_MAX_NEW_TOKENS` 配得较低。
   - 多数轨迹 `finish_reason=length`，说明模型输出经常被截断。
   - 被截断后，模型往往只执行一条浅层 `shell_exec`，没有完成完整脚本/文件实现。

2. **远端 worker 压力较高**
   - 日志中出现 `WORKER_SHIM_PRESSURE >= allocate threshold 160`。
   - worker `/readyz` 显示仍可用，但 active/in-flight runs 较多，后续 allocate 会重试或变慢。
   - 这会放大 `eval_timeout`，也会拉长 rollout time。

## 推荐下一步

短期目标如果是继续验证链路：

```bash
cd /mnt/shared-storage-user/puyuan/lixueyan/agentic-rl-cc

WORKER_URLS="http://100.96.26.133:18081" CLAUDE_CODE_QWEN_MAX_NEW_TOKENS=768 N_SAMPLES=2 ROLLOUT_BATCH_SIZE=1 bash terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_claude_code_pu.sh
```

如果目标是提升 SETA reward：

- 换一个更空的 worker，或先清理当前 worker 的 stale/active runs。
- 把 `CLAUDE_CODE_QWEN_MAX_NEW_TOKENS` 从当前低值提高到 `768` 或 `1024`。
- 维持 `ROLLOUT_BATCH_SIZE=1`、`N_SAMPLES=2` 做小规模稳定性验证。
- 如果 worker 稳定后仍大量 `finish_reason=length`，再继续提高输出预算或优化 Claude Code/Qwen gateway prompt。

## 当前判断

这次结果可以作为“Claude Code harness 已经接入 terminal-rl + Qwen/SGLang 训练链路”的证据，但还不适合作为最终能力评测。现在的低 reward 更像是低输出预算和 worker 压力造成的保守 smoke-test 结果，而不是接入失败。
