# Gen-Verse main 合并说明

形成日期：2026-06-02

## 1. 合并范围

本次将 `genverse/main` 合并到本地 `dev-glm` 分支。

合并前本地分支 HEAD：

```text
f77c65a5d7479f2ea2129be7b69f25b8ccacd566
```

合并来源：

```text
genverse/main
f48ac358adf9873b5cb2210f1cb234a52ed8a8a3
```

## 2. 主要新增与更新

本次上游主要变化集中在以下方向：

- `slime`：新增/更新 Qwen3.5、Qwen3.5-VL 相关模型脚本、Megatron/FSDP LoRA 支持、Qwen3.5 bridge/SGlang 插件，以及 PRM teacher / top-k 选择相关训练路径。
- `openclaw-combine`：新增 hybrid top-k OPD 训练与选择逻辑，包括 `hint_opd_loss.py`、`hint_opd_select_loss.py`、`openclaw_topk_select_loss.py`、select API server/rollout 和对应启动脚本。
- `openclaw-rl/oel`：新增 OEL 数据、蒸馏 loss、API server、rollout、评测和在线训练脚本。
- `gui-rl`：新增 Qwen3.5 agent、AgentNet 生成入口和多份 Qwen3.5 GUI RL 启动脚本。
- `openclaw-fireworks` / `openclaw-tinker` / `openclaw-test`：新增 Fireworks、Tinker、三阶段测试相关模块和示例脚本。
- `toolcall-rl`：新增 Qwen3.5 retool / PRM RL 启动脚本，并更新 retool 生成逻辑。
- 文档与资源：更新顶层 `README.md`、部分模块 README、assets 图片和 `extensions/rl-training-headers` 插件目录。

## 3. 冲突处理

### terminal-rl

`genverse/main` 会删除或大幅改写当前分支中的 `terminal-rl` 训练链路，包括：

- `terminal-rl/terminal-rl_qwen3-8b_pu.sh`
- `terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh`
- `terminal-rl/remote/setup_new_worker.sh`
- `terminal-rl/remote/run_pool_server_pu_v2.sh`
- `terminal-rl/generate.py`
- `terminal-rl/rollout_log.py`
- `terminal-rl/run_paths.py`
- `terminal-rl/dataset/seta_env*`

为了不影响当前 Qwen3-8B terminal-rl 训练流程，本次冲突处理中将 `terminal-rl` 目录恢复为合并前本地 `dev-glm` 版本。也就是说，当前 Qwen 训练入口、remote worker、GPU-side router、generate/reward/trajectory 逻辑和已有数据集路径保持不变。

### slime

`slime/slime/backends/megatron_utils/actor.py` 和 `slime/train_async.py` 存在功能性冲突。

处理方式：

- 保留本地分支的 pending metrics / wandb relay / `flush_pending_metrics()` 逻辑。
- 保留上游 `genverse/main` 的 PRM teacher、student top-k payload merge、partitioned rollout refs merge 和 top-k OPD 训练调度逻辑。

### swe-rl

`swe-rl` 不属于当前 Qwen terminal-rl 训练链路，因此冲突按 `genverse/main` 版本解决，并接受上游对 `swe-rl/generate_with_swe_remote.py` 的删除。

## 4. Qwen 训练流程保护点

本次合并后需要保持以下当前训练流程不变：

```text
terminal-rl_qwen3-8b_pu.sh
  -> Ray
  -> slime/train_async.py
  -> Megatron actor + SGLang rollout
  -> GPU-side router
  -> remote CPU worker pool
  -> terminal env reward
```

保护策略：

- `terminal-rl` 保持本地版本，不采用 `genverse/main` 对该目录的删除式覆盖。
- 当前 stash 中的本地改动在 merge 提交完成后再恢复到工作区。
- 两个本地未跟踪数据文件不纳入本次 merge 提交：
  - `terminal-rl/dataset/mixed_seta_safety.filtered.jsonl`
  - `terminal-rl/dataset/mixed_seta_safety.jsonl`

## 5. 后续检查建议

合并后建议至少执行：

```bash
bash -n terminal-rl/terminal-rl_qwen3-8b_pu.sh
bash -n terminal-rl/remote/setup_new_worker.sh
bash -n terminal-rl/remote/run_pool_server_pu_v2.sh
python -m py_compile slime/train_async.py slime/slime/backends/megatron_utils/actor.py
```

如果要正式运行 Qwen 训练，仍按原流程先启动 CPU worker pool，再在 GPU 节点设置 `WORKER_URLS` 并启动 `terminal-rl/terminal-rl_qwen3-8b_pu.sh`。
