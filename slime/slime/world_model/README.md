# OpenClaw Text Latent World Model v1

这是默认关闭的 JEPA-style text latent world model 插拔实现。

完整 motivation、方法、实验设置、阶段结果与能力边界见
[`rl_doc/openclaw_latent_world_model_value_head_talk_20260714.md`](../../../rl_doc/openclaw_latent_world_model_value_head_talk_20260714.md)。

## 设计边界

- v1 不在线抓 Megatron middle-layer hidden，避免影响 PP/TP/CP/SP 训练路径。
- rollout 只在 `--world-model-enable` 打开时附加轻量 `metadata["world_model"]` / `train_metadata["world_model"]`。
- policy/value loss 不被替换；`--world-model-loss-coef > 0` 时才追加 optional loss hook。
- 当前 optional loss hook 只支持 sample-normalized、`context_parallel_size=1` 的路径；per-token loss 或 CP>1 会 fail closed，尚不声称已完成在线分布式训练适配。
- 大 hidden tensor 不进入 `Sample.metadata`，离线 probe 使用 cached/frozen encoder 产物。
- 中间 turn 的 prediction target 不注入未来 terminal status/reward/eval；常见 credential 在 metadata 落盘前被替换为 `[REDACTED]`。
- HF encoder 默认 local-files-only 且不信任 remote code；下载和 `trust_remote_code` 都必须显式开启。

Credential redaction 是 best-effort safety net，不是完整 DLP。rollout/records 仍可能包含私有代码和业务内容，应按敏感训练数据管理。

## 主要入口

- `metadata.py`：terminal rollout 的因果对齐 action/observation 与独立 per-turn return/step reward label 轻量记录，以及 credential redaction。
- `modules.py`：`TextLatentWorldModel`，包含 controlled projector、action-conditioned predictor、SIGReg、value/uncertainty heads。
- `loss_hook.py`：在线训练的可选 hook 边界；mean loss 会先按样本数转为 sample-sum，再进入 Megatron global-batch scaling。
- `build_dataset.py`：从 debug rollout samples 抽取 world-model JSONL。
- `cache_text_hidden.py`：把 world-model JSONL 转成 `state_hidden/action_hidden/target_hidden` cache；默认 `hash` encoder 只用于 smoke，正式训练应切换到 frozen HF/LLM encoder。
- `train_probe.py`：用预计算 hidden tensors 做可复现、优先按 `context_hash` 分组的离线训练。
- `evaluate_probe.py`：默认在同 cache 的 group-heldout validation rows 上做 Stage-A action ablation / collapse / reward diagnostic。
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
  --hf-model /path/to/frozen/text/encoder \
  --hf-local-files-only

PYTHONPATH=slime python -m slime.world_model.train_probe \
  --input runs/world_model/cached_hidden.pt \
  --output runs/world_model/probe.pt \
  --val-ratio 0.25 \
  --split-group-key context_hash

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

这些 tensor 第一维必须一致。`cache_text_hidden.py` 生成的 payload 还会追加 records digest、固定 canary 的 encoder behavioral fingerprint、hidden tensor digest、完整 cache fingerprint、reward-label contract、`metadata` 和 `record_metadata`。旧版核心 tensor 仍可用于训练 smoke，但缺少当前 fingerprint 的 legacy cache/checkpoint 不能进入严格 eval 或 target-free ranking，需要重建。

encoder contract fingerprint 绑定 model path、pooling、max length、hidden dim 和固定 canary 的量化输出，因此同一路径替换权重通常会直接 mismatch；它仍不是大型 HF 权重文件的完整内容 digest。跨 cache 实验应使用不可变 model revision 并在实验 manifest 中记录 revision；严格同-cache heldout 会重新计算并核对实际 hidden tensor 与 metadata digest。

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
3. 训练 `TextLatentWorldModel` probe；固定 seed，并优先按 `context_hash` 生成 group-disjoint train/val split。
4. 默认生成 `oracle_pred_error_diagnostic.jsonl` 作为 target-aware pipeline smoke；只有显式训练 value head 并选择 `value` mode 才写 target-free `rankings.jsonl`。只有 reward-label contract 明确验证为 execution outcome 时，后者才可标记为 execution-eligible。

## Stage-A 验收口径

`train_probe.py` 的 `loss/val_loss` 是训练 smoke 指标，包含 prediction、SIGReg、contrast、可选 value loss，不能单独作为 world model 有效性的科学证据。进入 online auxiliary 或 U2 前，至少需要 `evaluate_probe.py` 输出的门控指标支持：

默认 `--split auto` 只接受训练时的精确 hidden cache，并要求 checkpoint 保存了非空 `group_holdout` validation indices；缺 group、record-level fallback、空 validation 或 cache fingerprint 不一致都会 fail closed。抽取阶段会按最终 encoder 输入的 `context_text` 重算 canonical `context_hash`，避免混合 v1/v2 hash 造成 split 泄漏。外部 cache 必须显式使用 `--split all`，并标记为 `external_cache_unverified_disjointness`，不能当作 heldout 证据。`candidate_set_eval.py` 还会核对 records SHA256、encoder/cache fingerprint，并要求 heldout split key 与 candidate group key 相同。

- `pred_mse_real`：真实 action 的 prediction MSE。
- `shuffle_gap_mse_shuffled_minus_real`：shuffled action loss 减真实 action loss，正值才是 action-conditioned predictor 的弱证据。
- `zero_action_gap_mse_zero_minus_real`：zero action 对照。
- `action_delta`：替换 action 后 prediction latent 的移动量；只能说明模型对 action 敏感，不能单独说明方向正确。
- `latents.*.effective_rank/variance_mean`：collapse 诊断。
- `value_reward.spearman`：仅在有有效 `reward_mask` 且 reward 非常数时有意义；否则输出 `null` 和 reason。

`rank_candidates.py` 的 `auto/value` mode 会校验 checkpoint 是否有正 optimizer/value update steps，且 train split 中存在受正 `value_coef` 监督的 reward labels；同时拒绝 encoder fingerprint 不匹配和 NaN/Inf score。输出保留 reward-label contract；默认 `sample.reward.score` 仅是语义未验证的 replay training label，因此可用于 target-free reward ranking，但不能标记为 execution outcome。`pred_error` 明确依赖 target，只能用于 oracle diagnostic。真正 U2 需要同一 state 下多个真实执行候选 action 及结构化 reward/status 标签。

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

脚本默认 `WM_FILTERS=full,clean,tool_only`、`WM_REQUIRED_FILTERS=full,clean`。它默认 `WM_ENCODER=hash` 以避免误跑大 HF cache；使用 HF 时必须显式 `WM_ALLOW_HF=1`，且默认 `WM_HF_LOCAL_FILES_ONLY=1`、`WM_HF_TRUST_REMOTE_CODE=0`。它会生成每个 bucket 的 `records_summary.json`、`cached_hidden.pt`、`probe.pt`、`eval_summary.json`，并聚合到顶层 `summary.json`。

严格 gate 默认要求 `evaluation_split.scope=group_heldout`：

```bash
PYTHONPATH=slime python -m slime.world_model.summarize_stage_a \
  --input runs/world_model_stage_a_eval/qwen_seta \
  --output runs/world_model_stage_a_eval/qwen_seta/gate_summary.json
```

只有显式 `--allow-non-group-heldout` 才允许 diagnostic split 通过该项检查；这类结果不能作为 heldout 证据。

P2 不能直接复用默认 `value_coef=0` 的 Stage-A checkpoint。需要显式训练 value head：

```bash
P2_ROOT="runs/world_model_stage_a_eval/p2_value"
WM_OUT_DIR="${P2_ROOT}" \
WM_ENCODER=hf \
WM_ALLOW_HF=1 \
WM_HF_MODEL=/path/to/Qwen3-8B \
WM_VALUE_COEF=0.05 \
WM_VAL_RATIO=0.25 \
terminal-rl/scripts/run_world_model_stage_a_eval.sh

WM_P2_BASE_EXP="${P2_ROOT}" \
terminal-rl/scripts/run_world_model_p2_candidate_set_eval.sh
```

P2 preflight 会验证 value optimizer updates 与 train reward labels；随后 evaluator 严格验证 records、encoder/cache provenance 和 group-heldout split。不满足时给出可执行错误并退出。
