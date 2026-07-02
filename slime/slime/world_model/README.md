# OpenClaw Text Latent World Model v1

这是默认关闭的 JEPA-style text latent world model 插拔实现。

## 设计边界

- v1 不在线抓 Megatron middle-layer hidden，避免影响 PP/TP/CP/SP 训练路径。
- rollout 只在 `--world-model-enable` 打开时附加轻量 `metadata["world_model"]` / `train_metadata["world_model"]`。
- policy/value loss 不被替换；`--world-model-loss-coef > 0` 时才追加 optional loss hook。
- 大 hidden tensor 不进入 `Sample.metadata`，离线 probe 使用 cached/frozen encoder 产物。

## 主要入口

- `metadata.py`：terminal rollout 的 action / next observation / reward 轻量记录。
- `modules.py`：`TextLatentWorldModel`，包含 controlled projector、action-conditioned predictor、SIGReg、value/uncertainty heads。
- `loss_hook.py`：在线训练的可选 hook 边界，默认只在提供 `wm_pred_latents` 和 `wm_target_latents` 时计算 MSE。
- `build_dataset.py`：从 debug rollout samples 抽取 world-model JSONL。
- `cache_text_hidden.py`：把 world-model JSONL 转成 `state_hidden/action_hidden/target_hidden` cache；默认 `hash` encoder 只用于 smoke，正式训练应切换到 frozen HF/LLM encoder。
- `train_probe.py`：用预计算 hidden tensors 训练离线 probe。
- `evaluate_probe.py`：对 probe 做 Stage-A action ablation / collapse / reward calibration 诊断。
- `rank_candidates.py`：用已训练 probe 对候选 action 排序。

## 典型离线路径

```bash
PYTHONPATH=slime python -m slime.world_model.build_dataset \
  --input runs/debug_rollout.pt \
  --output runs/world_model/records.jsonl

PYTHONPATH=slime python -m slime.world_model.cache_text_hidden \
  --input runs/world_model/records.jsonl \
  --output runs/world_model/cached_hidden.pt \
  --encoder hf \
  --hf-model /path/to/frozen/text/encoder

PYTHONPATH=slime python -m slime.world_model.train_probe \
  --input runs/world_model/cached_hidden.pt \
  --output runs/world_model/probe.pt

PYTHONPATH=slime python -m slime.world_model.evaluate_probe \
  --checkpoint runs/world_model/probe.pt \
  --input runs/world_model/cached_hidden.pt \
  --output runs/world_model/eval_summary.json
```

`cached_hidden.pt` 需要包含：

- `state_hidden`
- `action_hidden`
- `target_hidden`
- 可选 `reward`
- 可选 `reward_mask`，用于标记哪些 reward 是真实标签；缺失 reward 不会被当作 `0.0` 训练 value head。

这些 tensor 第一维必须一致。`cache_text_hidden.py` 生成的 payload 还会追加 `metadata` 和 `record_metadata`，但旧版只含核心 tensor 的 cache/checkpoint 仍可被 `train_probe.py` / `evaluate_probe.py` 读取。

快速 smoke 可以使用 deterministic hash encoder：

```bash
terminal-rl/scripts/run_world_model_offline_probe_smoke.sh
```

这只验证数据接口、projector/predictor/loss/ranking 能闭环，不代表语义 latent 已可用。正式实验应使用 frozen LLM hidden 或专门的 text encoder hidden 生成 `cached_hidden.pt`。

如果需要先采集 SETA rollout metadata，可运行：

```bash
terminal-rl/scripts/run_world_model_seta_smoke.sh
```

该脚本默认复用 `terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh`，也可通过 `WM_TRAIN_SCRIPT=/path/to/train_wrapper.sh` 覆盖。

多 rollout 批量路径：

```bash
WM_ENCODER=hf \
WM_ALLOW_HF=1 \
WM_HF_MODEL=/path/to/Qwen3-8B \
WM_INPUT_GLOB="runs/world_model_smoke/*/metadata/rollout_*.pt" \
WM_OUT_DIR="runs/world_model_probe_batch/qwen3_8b" \
terminal-rl/scripts/run_world_model_batch_probe.sh
```

该脚本会：

1. 从多个 rollout `.pt` 抽取并合并 `records.jsonl`。
2. 用 `cache_text_hidden.py` 生成 hidden cache。
3. 训练 `TextLatentWorldModel` probe，并在样本数足够时打印 `val_loss`。
4. 生成 `rankings.jsonl` 作为候选 action 排序接口 smoke。

## Stage-A 验收口径

`train_probe.py` 的 `loss/val_loss` 是训练 smoke 指标，包含 prediction、SIGReg、contrast、可选 value loss，不能单独作为 world model 有效性的科学证据。进入 online auxiliary 或 U2 前，至少需要 `evaluate_probe.py` 输出的门控指标支持：

- `pred_mse_real`：真实 action 的 prediction MSE。
- `shuffle_gap_mse_shuffled_minus_real`：shuffled action loss 减真实 action loss，正值才是 action-conditioned predictor 的弱证据。
- `zero_action_gap_mse_zero_minus_real`：zero action 对照。
- `action_delta`：替换 action 后 prediction latent 的移动量；只能说明模型对 action 敏感，不能单独说明方向正确。
- `latents.*.effective_rank/variance_mean`：collapse 诊断。
- `value_reward.spearman`：仅在有有效 `reward_mask` 且 reward 非常数时有意义；否则输出 `null` 和 reason。

`rank_candidates.py` 仍只是全局 ranking interface smoke，不是 U2 ranking eval。真正 U2 需要同一 state 下多个真实执行候选 action 及对应 reward/status 标签。

一键 Stage-A 评估：

```bash
# 默认 hash encoder，只验证离线链路和指标 schema，不代表语义 latent。
terminal-rl/scripts/run_world_model_stage_a_eval.sh

# Qwen/HF full/clean/tool-only bucket 并行：
WM_ENCODER=hf \
WM_ALLOW_HF=1 \
WM_HF_MODEL=/path/to/Qwen3-8B \
WM_DUAL_GPU_IDS=0,1 \
WM_OUT_DIR="runs/world_model_stage_a_eval/qwen_seta" \
terminal-rl/scripts/run_world_model_stage_a_eval.sh
```

脚本默认 `WM_FILTERS=full,clean,tool_only`、`WM_REQUIRED_FILTERS=full,clean`。它默认 `WM_ENCODER=hash` 以避免误跑大 HF cache；使用 HF 时必须显式 `WM_ALLOW_HF=1`，且默认 `WM_HF_LOCAL_FILES_ONLY=1`。它会生成每个 bucket 的 `records_summary.json`、`cached_hidden.pt`、`probe.pt`、`eval_summary.json`，并聚合到顶层 `summary.json`。
