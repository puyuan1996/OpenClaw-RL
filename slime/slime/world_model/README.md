# OpenClaw Text Latent World Model v1

这是默认关闭的 JEPA-style text latent world model 插拔实现。

完整 motivation、方法、实验设置、验证状态与能力边界见
[`rl_doc/openclaw_latent_world_model_value_head_talk_20260714.md`](../../../rl_doc/openclaw_latent_world_model_value_head_talk_20260714.md)。

## 设计边界

- v1 不在线抓 Megatron middle-layer hidden，避免影响 PP/TP/CP/SP 训练路径。
- 通用 metadata smoke 默认使用 `camel-agent`。A3S/Claude Code 可能在一个 outer turn 中返回多个内部 interactions；v1 会标记 `world_model_skipped` 并拒绝生成 WM record，接入前需要 harness-specific adapter。
- rollout 只在 `--world-model-enable` 打开时附加轻量 `metadata["world_model"]` / `train_metadata["world_model"]`。
- policy/value loss 不被替换；`--world-model-loss-coef > 0` 时才追加 optional loss hook。
- 当前 auxiliary path 只支持 sample-level objective 和 `context_parallel_size=1`。内置 hook 会把 mean loss 按样本数转成 sample-sum；正系数下缺 latent、loss 已 detach、per-token loss 或 CP>1 都会拒绝运行。自定义 graph-connected hook 可能更新 policy，尚未完成 PP/CP/SP 组合验证。
- 大 hidden tensor 不进入 `Sample.metadata`，离线 probe 使用 cached/frozen encoder 产物。
- 中间 turn 的 prediction target 不注入未来 terminal status/reward/eval；长 tool result 采用定长 head-tail 截断，保留结果尾部；常见 credential 在 metadata 落盘前被替换为 `[REDACTED]`。
- HF encoder 默认 local-files-only 且不信任 remote code；下载和 `trust_remote_code` 都必须显式开启。

Credential redaction 是 best-effort safety net，不是完整 DLP。rollout/records 仍可能包含私有代码和业务内容，应按敏感训练数据管理。所有 `.pt` 和 checkpoint 都通过 PyTorch pickle loader 读取，只能使用可信文件。

## 主要入口

- `metadata.py`：terminal rollout 的因果对齐 action/observation 与独立 per-turn return/step reward label 轻量记录，以及 credential redaction。
- `modules.py`：`TextLatentWorldModel`，包含 controlled projector、action-conditioned predictor、SIGReg、value/uncertainty heads。
- `loss_hook.py`：在线训练的可选 hook 边界；mean loss 会先按样本数转为 sample-sum，再进入 Megatron global-batch scaling。
- `build_dataset.py`：从 debug rollout samples 抽取 world-model JSONL。
- `cache_text_hidden.py`：把 world-model JSONL 转成 `state_hidden/action_hidden/target_hidden` cache；HF `last` pooling 兼容 left/right padding。默认 `hash` encoder 只用于 smoke，正式训练应切换到 frozen HF/LLM encoder。
- `train_probe.py`：用预计算 hidden tensors 做可复现、优先按 `context_hash` 分组的离线训练。
- `evaluate_probe.py`：默认在同 cache 的 group-heldout validation rows 上做 Stage-A action ablation / collapse / reward diagnostic。
- `rank_candidates.py`：用已训练 probe 对候选 action 排序。
- `candidate_set_eval.py`：在已经执行并带 reward 的同 context 候选记录上做离线排序评估。
- `metrics.py`：有限数检查和 action sensitivity 等共享指标。
- `checkpoint.py` / `summarize_stage_a.py`：split、provenance、value 训练证据和 Stage-A gate。

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
- 可选 `reward_mask`，用于标记哪些 reward 是有限监督值；缺失 reward 不会被当作 `0.0` 训练 value head。

这些 tensor 第一维必须一致。`cache_text_hidden.py` 还会记录 records、encoder、hidden、逐样本 metadata、reward 和 mask 的一致性 fingerprint。它们用于发现产物错配或未同步修改，不是数据签名，也不能证明输入记录真实可信。缺少当前 fingerprint 的 legacy cache/checkpoint 只能用于无 value、无 heldout 的训练 smoke。

encoder contract fingerprint 绑定 model path、pooling、max length、hidden dim 和固定 canary 的量化输出，因此同一路径替换权重通常会直接 mismatch；它仍不是大型 HF 权重文件的完整内容 digest。跨 cache 实验应使用不可变 model revision 并在实验 manifest 中记录 revision；严格同-cache heldout 会重新计算并核对实际 hidden tensor 与 metadata digest。

快速 smoke 可以使用 deterministic hash encoder：

```bash
terminal-rl/scripts/run_world_model_offline_probe_smoke.sh
```

这只验证数据接口、projector、predictor、loss 和 ranking 可以运行，不代表语义 latent 已可用。正式实验应使用 frozen HF text encoder hidden 生成 `cached_hidden.pt`。

如果需要先采集 SETA rollout metadata，可运行：

```bash
terminal-rl/scripts/run_world_model_seta_smoke.sh
```

该脚本默认复用仓库现有的 SETA DAPO training wrapper，也可通过 `WM_TRAIN_SCRIPT=/path/to/train_wrapper.sh` 覆盖。

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
4. 默认生成 `oracle_pred_error_diagnostic.jsonl` 作为 target-aware pipeline smoke；显式训练 value head 并选择 `auto` 或 `value` mode 才写 target-free `rankings.jsonl`。只有可信 adapter 明确声明 execution-outcome label 时，后者才可标记为 execution-eligible；代码不会独立核验该声明的真实性。

## Stage-A 验收口径

`train_probe.py` 的 `loss/val_loss` 是训练 smoke 指标，包含 prediction、SIGReg、contrast、可选 value loss，不能单独作为 world model 有效性的科学证据。进入 online auxiliary 或 U2 前，至少需要 `evaluate_probe.py` 输出的门控指标支持：

默认 `--split auto` 只接受训练时的精确 hidden cache，并要求 checkpoint 保存非空、互斥且完整覆盖 cache 的 `group_holdout` train/val indices。`context_hash` 根据 tokenizer 之前的 canonical `context_text` 重算，可统一 v1/v2 哈希口径；它不能保证 HF `max_length` 截断后的 token 序列严格互斥。显式 `--split all` 会标记为 in-sample 或 provenance 未验证，不能冒充 heldout 证据。`candidate_set_eval.py` 还会核对 records/cache 一致性，并要求 split key 与 candidate group key 相同。

- `pred_mse_real`：真实 action 的 prediction MSE。
- `shuffle_gap_mse_shuffled_minus_real`：shuffled action loss 减真实 action loss，正值才是 action-conditioned predictor 的弱证据。
- `zero_action_gap_mse_zero_minus_real`：zero action 对照。
- `action_delta`：替换 action 后 prediction latent 的移动量；只能说明模型对 action 敏感，不能单独说明方向正确。
- `latents.*.effective_rank/variance_mean`：collapse 诊断。
- `value_reward.spearman`：仅在有有效 `reward_mask` 且 reward 非常数时有意义；否则输出 `null` 和 reason。

`rank_candidates.py` 的 `auto/value` mode 会校验 value update steps、train reward labels、heldout split、encoder fingerprint 和有限 score。默认 `sample.reward.score` 语义未验证，只能用于 gate-ineligible diagnostic；`pred_error` 依赖 target，只能用于 oracle diagnostic。当前 P2 评估的对象是 replay 中已经执行过的候选，不是在线 pre-execution branching。

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

脚本默认 `WM_FILTERS=full,clean,tool_only`、`WM_REQUIRED_FILTERS=full,clean`。它默认 `WM_ENCODER=hash` 以避免误跑大 HF cache；使用 HF 时必须显式 `WM_ALLOW_HF=1`，且默认 `WM_HF_LOCAL_FILES_ONLY=1`、`WM_HF_TRUST_REMOTE_CODE=0`。它会生成每个 bucket 的 `records_summary.json`、`cached_hidden.pt`、`probe.pt`、`eval_summary.json`，并聚合到顶层 `summary.json`。复用 cache/checkpoint 时会校验 records、配置和 world-model 源码签名；截断统计缺失或非法时 gate 拒绝通过。

严格 gate 默认要求 `evaluation_split.scope=group_heldout`：

```bash
PYTHONPATH=slime python -m slime.world_model.summarize_stage_a \
  --input runs/world_model_stage_a_eval/qwen_seta \
  --output runs/world_model_stage_a_eval/qwen_seta/gate_summary.json
```

只有显式 `--allow-non-group-heldout` 才允许 diagnostic split 通过该项检查；这类结果不能作为 heldout 证据。

P2 不能直接复用默认 `value_coef=0` 的 Stage-A checkpoint。输入还必须在 heldout context 中包含至少两个 action 不同、reward 有限的候选，默认要求 reward 不同；重复 action 不会被当作多个候选。脚本不会生成或执行候选，普通单动作 rollout 可能因没有 candidate group 而退出。满足数据条件后，再显式训练 value head：

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

P2 preflight 会验证 value update、train labels、records/cache 一致性、group-heldout split，以及记录声明的 reward contract。默认 collector 的 `training_reward_unspecified` 不能作为 execution eval。仅做开发诊断时可设置 `WM_P2_ALLOW_UNVERIFIED_REWARD_LABELS=1`；输出会标记 `diagnostic_only=true`、`execution_outcome_eligible=false`。只有 execution label 合格且 split 为 `group_heldout` 时，`gate_eligible` 才为 true；显式 `--split all` 始终是 diagnostic。即使 adapter 声明 `reward_label_is_execution_outcome=true`，正式结论仍依赖可信的数据采集与真实执行审计。
