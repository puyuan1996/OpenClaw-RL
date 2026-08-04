# SWE-smith × terminal-rl 使用说明

本文用于复现 OpenClaw-RL 中的 SWE-smith 训练。所有命令均在 repo root 执行。

**结论：** 已完成 59,136 条全量训练 task 的可复现转换、独立 Docker worker、binary reward 和现有 DAPO 训练入口适配；历史 4-GPU 实验已跑通 `data -> Docker -> evaluate -> DAPO update` 主链路。该训练 reward 用于 RL，不等同于官方 SWE benchmark 分数。

参考入口：[SWE-smith 官网](https://swesmith.com/)；[Hugging Face 数据集](https://huggingface.co/datasets/SWE-bench/SWE-smith)。

## 1. 实现概览

本适配复用现有 `terminal-rl` remote env：

```text
SWE-smith snapshot
  -> terminal-rl JSONL + Terminal-Bench task dirs
  -> namespaced Docker worker (:18082)
  -> env router -> rollout -> binary reward -> DAPO update
```

主要能力：

- 根据每条样本的 `instance_id` checkout 对应 Git ref；
- 支持 `pytest`、Go、Mypy、uv/pytest 及官方特殊 command profile；
- 执行 `FAIL_TO_PASS`（F2P）和 `PASS_TO_PASS`（P2P），防止奖励破坏原有测试的补丁；
- 固定 dataset revision、Parquet 指纹、artifact SHA 和 task-dir fingerprint；
- Docker object 按 `TERMINAL_RL_POOL_NAMESPACE` 隔离，可与同机 SETA `:18081` 共存；
- reset 超时后先隔离 lease，等待 reset/Docker thread 实际退出再清理；周期性清理使用统一 wall-clock deadline；
- 通过现有训练入口设置 `DATASET=swesmith`，不修改 DAPO 或 off-policy 算法。

本 PR 不包含 SWE-bench Verified eval，也不实现 Replay Buffer、SPEAR 等算法。

## 2. 正式数据契约

正式训练固定到以下 snapshot：

| 字段 | 值 |
|---|---|
| dataset / split | `SWE-bench/SWE-smith` / `train` |
| revision | `ea6d7173829c7ec8fa16c22055699ff2e9188091` |
| task 数 | 59,136 |
| runner 分布 | `pytest=49,883`、`go=8,212`、`command=346`、`mypy=19`、`pytest_uv=676` |
| 训练测试上限 | 每个 task 最多 `50 F2P + 200 P2P` |
| artifact | 1,039,459,893 bytes；SHA-256 `4f9c34bb6b2b268b2b5952d6f67725c73b572bbddab1e47d5375d0558a6309eb` |

这里的“全量”指 59,136 个 task 全部保留。单个 task 的 test ID 采用固定上限，避免超长命令和数百次 pytest 启动；因此训练 reward 不是官方 benchmark 分数，正式评测应使用独立的官方 SWE eval。

## 3. 生成数据

先生成默认 64 条 smoke 数据。它写入独立的 `dataset/swesmith_smoke/`，不会覆盖已发布的 full 环境：

```bash
bash terminal-rl/data_utils/download_swesmith.sh
```

确认磁盘和网络预算后生成正式全量数据：

```bash
MODE=full ALLOW_FULL=1 \
  bash terminal-rl/data_utils/download_swesmith.sh
```

输出目录不提交到 Git：

```text
terminal-rl/dataset/swesmith_convert/train.jsonl
terminal-rl/dataset/swesmith_convert/convert_stats.json
terminal-rl/dataset/swesmith_env/<instance_id>/
```

downloader 使用 exclusive `flock` 先生成完整 generation dir，再依次替换 env、JSONL 和 manifest；普通失败/信号会回滚，worker 和 trainer 持 shared lock。`SIGKILL` 或宿主机故障仍可能中断多文件替换，但 consumer 会因 manifest/fingerprint 不一致而 fail closed；此时重新运行 downloader 即可。重新转换前应先停止正在运行的 SWE-smith worker 和训练。
trainer 的 preflight 严格只读；发现 task dir 缺失或 fingerprint 过期时会直接退出，必须通过 downloader 在 exclusive lock 下重新发布，训练进程不会现场改写数据。

若只测试 smoke worker，启动 launcher 时额外设置 `DATASET_DIR="$PWD/terminal-rl/dataset/swesmith_smoke" SWESMITH_WORKER_REQUIRE_FULL_DATA=0`，并把 smoke client 的 `--dataset` 指向 `terminal-rl/dataset/swesmith_smoke/swesmith_convert/smoke.jsonl`。

代理由用户预先设置；脚本不会直接 `source` 远程脚本。也可通过仅包含白名单 proxy key 的本地文件加载：

```bash
AUTO_PROXY=1 PROXY_ENV_FILE=/path/to/proxy.env \
MODE=full ALLOW_FULL=1 \
  bash terminal-rl/data_utils/download_swesmith.sh
```

## 4. 启动 Docker worker

worker 需要 Docker Compose 和 Python >= 3.12。首次部署先安装 pinned direct dependency：

```bash
python3.12 -m venv .venv-swesmith-worker
.venv-swesmith-worker/bin/python -m pip install \
  -r terminal-rl/remote/requirements-swesmith-worker.txt
```

启动独立的 `:18082` 服务：

```bash
POOL_SERVER_PYTHON="$PWD/.venv-swesmith-worker/bin/python" \
ENV_SERVER_PORT=18082 \
WORKER_MAX_TASKS=8 WORKER_MAX_RUNS_PER_TASK=4 \
WORKER_MAX_CONCURRENT_BUILDS=1 \
WORKER_MIN_DOCKER_FREE_GB=80 \
CONTAINER_MEMORY_LIMIT=16g CONTAINER_PIDS_LIMIT=256 \
  bash terminal-rl/remote/run_pool_server_swesmith_pu.sh
```

正式 worker 默认校验 pinned dependency，并强制 `SWESMITH_RUN_PASS_TO_PASS=1`。namespace 默认为 `swesmith`；host-wide prune 默认关闭，只清理本 namespace 拥有的 Docker object。`default` namespace 继续识别历史 SETA unlabeled task，non-default namespace 则必须精确匹配 label。

```bash
curl -fsS --noproxy '*' http://<worker-ip>:18082/healthz
```

## 5. Worker smoke

正式训练前验证 `healthz -> allocate -> reset -> evaluate -> close`：

```bash
python3 terminal-rl/scripts/smoke_swesmith_worker.py \
  --worker-url http://<worker-ip>:18082 \
  --dataset terminal-rl/dataset/swesmith_convert/train.jsonl \
  --index 0 \
  --expect-score 0 \
  --expect-reason test_exit_nonzero
```

未修复样本得到 `score=0` 是正确负对照；原因必须是测试失败，而不是 timeout、parser error 或零测试。建议再使用已知正确 patch 验证 `score=1`。

## 6. 4-GPU DAPO 验收

下面运行 10 rollout 流程验证，路径按集群实际位置填写：

```bash
LIGHTRFT_PY312_BIN=/path/to/lightrft_py312/bin \
HF_CKPT=/path/to/Qwen3-8B \
REF_LOAD=/path/to/Qwen3-8B_torch_dist \
EXPORT_ROOT=/path/to/training-output \
WORKER_URLS="http://<worker-ip>:18082" \
DATASET=swesmith ALGO=dapo \
NUM_GPUS=4 ACTOR_GPUS=2 ROLLOUT_GPUS=2 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 TP_SIZE=2 \
NUM_ROLLOUT=10 ROLLOUT_BATCH_SIZE=4 N_SAMPLES=4 \
MAX_CKPT_KEEP=0 MAX_TURN=10 \
CUSTOM_CONFIG_PATH=terminal-rl/configs/rollout_qwen3_think.yaml \
  bash terminal-rl/terminal-rl_qwen3-8b_seta_dapo_nodynamic_pu.sh
```

入口文件名保留 `seta` 是兼容历史调用；实际由 `DATASET=swesmith` 选择数据和 remote env。`swe-smith`、`swe_smith`、`swemith` 也会归一化为 `swesmith`。

训练预检会验证 59,136 行、artifact SHA、runner profile 和 task-dir fingerprint，并用只读 hard link 将已验证 JSONL 绑定到本次 run。`run_config.json` 记录 dataset revision、artifact SHA 和 converter SHA。

上面是短流程验收，因此关闭 checkpoint。正式训练应增大 `NUM_ROLLOUT`，并设置 `MAX_CKPT_KEEP>=2`。

## 7. Reward 与验证边界

- `reward=1`：选中的 F2P/P2P 全部通过、runner exit code 为 0，且确实解析到测试结果。
- `reward=0`：任一测试失败、timeout、parser error、零测试或环境异常。
- grader 在 agent 获得 shell 前捕获 task/bug Git SHA，并恢复 protected tests；但 agent 与 grader 仍在同一 root container，这不是恶意代码安全边界。
- artifact lock 与 task-dir fingerprint 用于一致性和误操作防护，不构成对同一 host/same-UID 管理者的安全边界。
- 官方 `image_name` 仍是 registry tag；生产部署如需完全供应链固定，应额外记录实际 image digest。

当前候选代码的本地验证包括：固定 snapshot 独立流式审计、focused converter/Docker lifecycle tests、完整 `terminal-rl/tests`、Python compile、`bash -n`、`git diff --check`、SWE-smith DAPO `DRY_RUN` 和 SETA regression `DRY_RUN`。最终测试数字以本 PR 最新 commit 的说明为准。

本次 clean staged tree 的最终检查结果为：Python 3.10.12 focused tests `90 passed`；Python 3.12 完整 suite `155 passed, 2 failed`。两项失败均来自 PR base 的 `router_server_readyz` test fake 缺少 `maybe_reload_workers()`，已在未修改的 base commit 上独立复现。本 PR 新增测试全部通过，pinned worker requirements 的 `pip --dry-run` 解析成功。

历史真实 4-GPU run `terminal-rl_qwen3-8b_4gpu_swesmith_dapo_nodynamic_think_mt10_2026-07-14_031857` 使用 59,136-row 数据完成 3 个 rollout batch 和 4 次 actor update（`train/step=0..3`），随后进入下一轮生成阶段；日志无 Python traceback、CUDA OOM 或显式 `Run failed`。这证明 data -> Docker -> evaluate -> DAPO update 主链路可运行。该 run 的 commit 为 `fc872aa8`，使用较早 task format/reward，不能表述为当前精确 commit 已完成 10 rollout；当前 hardening commit 已完成上述 CPU tests 与 DAPO/SETA `DRY_RUN`，合并前仍建议按第 6 节命令做最终 GPU 验收。
