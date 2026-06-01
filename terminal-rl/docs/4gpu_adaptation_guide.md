# Terminal-RL 4卡训练适配指南

本文档记录将 `terminal-rl_qwen3-8b.sh`（8卡）适配为 `terminal-rl_qwen3-8b_pu.sh`（4卡）的完整过程，包括环境要求、部署步骤、已知问题和解决方案。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│  GPU Worker (4× GPU)                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Ray Head    │  │ SGLang Engine│  │ Megatron Actor│  │
│  │ + Router    │  │ (rollout)    │  │ (training)    │  │
│  │ :18080      │  │ :15000       │  │               │  │
│  └──────┬──────┘  └──────────────┘  └───────────────┘  │
│         │ HTTP forward                                   │
└─────────┼───────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  CPU Worker (Docker host)                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │ pool_server :18081                               │    │
│  │   └── docker compose build/up/exec per task     │    │
│  └─────────────────────────────────────────────────┘    │
│  Docker data-root: /data (1TB+)                          │
└─────────────────────────────────────────────────────────┘
```

## 环境要求

### GPU Worker

| 项目 | 要求 |
|---|---|
| GPU | 4× (H100/A100/H200) |
| Python | lightrft_py312 (`/mnt/shared-storage-user/puyuan/conda_envs/lightrft_py312/bin`) |
| 关键包 | transformer_engine 2.14.1, torch, sglang, ray |
| Megatron | 通过 PYTHONPATH 注入本仓库的 `Megatron-LM/` |

### CPU Worker

| 项目 | 要求 |
|---|---|
| 磁盘 | 200GB+ 可用（docker images 累积） |
| Docker | 26.x + Compose V2 plugin |
| Docker 代理 | daemon 级 + build 级（`~/.docker/config.json` proxies） |
| Python | 3.12 (.venv)，装 terminal-bench, fastapi, camel-ai |
| 网络 | GPU worker 能访问 :18081 |

## 部署步骤

### 1. CPU Worker 部署（新机器）

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

# 一键部署（Docker + Compose V2 + 代理 + base image + Python env）
DOCKER_ROOT=/data bash terminal-rl/remote/setup_new_worker.sh

# 一站式 4 层代理注入（含 base image apt 代理 wrap，watchdog-aware）
sudo bash terminal-rl/remote/fix_dockerd_and_proxy.sh

# 启动 pool_server（终端实时显示 + 日志落盘）
set -a; . /etc/seta_build_proxy.env; set +a
source .venv/bin/activate
bash terminal-rl/remote/run_pool_server_pu_v2.sh
```

> 详细从零配置 / pre-flight / 故障回退见 `terminal-rl/docs/cpu_worker_docker_ops.md` §0–§2。

### 2. GPU Worker 启动训练

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL

export ROLLOUT_PROMPT_DATA="/mnt/shared-storage-user/puyuan/code/OpenClaw-RL/terminal-rl/dataset/seta_env_convert/train.jsonl"
export WORKER_URLS="http://<cpu-worker-ip>:18081"

# 默认不存 ckpt（验证/debug）
bash terminal-rl/terminal-rl_qwen3-8b_pu.sh

# 存 ckpt（正式训练）
MAX_CKPT_KEEP=2 bash terminal-rl/terminal-rl_qwen3-8b_pu.sh
```

## 关键配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `NUM_GPUS` | 自动检测 | GPU 总数 |
| `ACTOR_GPUS` | 2 | 训练用 GPU |
| `ROLLOUT_GPUS` | 2 | 推理用 GPU |
| `TP_SIZE` | 2 | Tensor Parallel |
| `ROLLOUT_BATCH_SIZE` | 8 | 每轮采样 task 数 |
| `N_SAMPLES` | 4 | 每 task 生成 sample 数 |
| `NUM_ROLLOUT` | 2000 | 总 rollout 轮数 |
| `MAX_CKPT_KEEP` | 0 | 保留 ckpt 数（0=不存） |
| `SAVE_INTERVAL` | 8 | 每 N 个 rollout 存一次 |
| `EXPORT_ROOT` | `/mnt/.../narmodel/agenticrl` | ckpt 存储路径 |
| `USE_BLACKLIST` | 1 | 是否过滤已知坏 task |
| `DEBUG_MODE` | 0 | 调试模式（小 batch，不存 ckpt） |

## 日志系统

训练自动将日志写入 `tmp_doc_latest/`（symlink 到当前 run 的 `tmp_doc_<timestamp>/`）：

| 文件 | 来源 | 内容 |
|---|---|---|
| `gpu_run.log` | GPU worker | 全量 stdout/stderr |
| `gpu_err.log` | GPU worker | 失败时自动过滤的错误行 |
| `gpu_tail.log` | GPU worker | 失败时最后 300 行 |
| `cpu_pool.log` | CPU worker | pool_server 全量输出 |
| `cpu_err.log` | CPU worker | 每 30s 自动刷新的错误过滤 |

## 已知问题和解决方案

### 1. terminal-bench 版本不匹配

**症状**：`Terminal.start() got an unexpected keyword argument 'timeout'` 或 `DockerComposeManager.build() got an unexpected keyword argument 'timeout'`

**原因**：仓库代码期望新版 terminal-bench API，但 .venv 装的是旧版。

**解决**：已在 `terminal_env.py` 和 `docker_compose_utils.py` 加了 `inspect.signature` 兼容检查。

### 2. ghcr.io 不可达

**症状**：`dial tcp 20.205.243.164:443: i/o timeout`

**原因**：公司代理封锁 ghcr.io。

**解决**：
- 方案 A：`daemon.json` 加 `registry-mirrors`（如 `docker.1ms.run`）—— `setup_new_worker.sh` 默认已配
- 方案 B：从能访问 ghcr.io 的机器 `docker save` 后 `docker load`
- 历史归档：`archive/remote/build_base_images.sh` 曾用于本地构建等价 image，现已不需要

### 3. docker build 容器内无网络

**症状**：`apt-get update` 在 Dockerfile RUN 步骤中失败 / `Could not connect to archive.ubuntu.com`

**原因**：apt 不读 HTTP_PROXY env，只读 `/etc/apt/apt.conf.d/*`。

**解决**：`sudo bash terminal-rl/remote/fix_dockerd_and_proxy.sh`（Phase 5.5 会自动调 `prebuild_proxied_base_images.sh` wrap 4 个高频 base image，把 `Acquire::http::Proxy` 写到 `apt.conf.d/95proxies`）。详见 `cpu_worker_docker_ops.md` §4.6。

### 4. 共享存储满

**症状**：`CheckpointException` + `unexpected pos 704 vs 598`

**原因**：`/mnt/shared-storage-user/puyuan` 3TB 已满。

**解决**：`EXPORT_ROOT` 改到 `/mnt/shared-storage-user/narmodel/agenticrl`；默认 `MAX_CKPT_KEEP=0` 不存 ckpt。

### 5. Dataset 黑名单

以下 task 已知会导致训练卡顿或 100% 失败，默认被过滤：

```
786,96,90,456,856,210,999,305,25,684,345,553,962,916,1264,282,324,768,46,996
```

来源：[HansBug/OpenClaw-RL#3](https://github.com/HansBug/OpenClaw-RL/issues/3)

### 6. APEX 缺失

**症状**：`fused_weight_gradient_mlp_cuda module is not found`

**解决**：`--no-gradient-accumulation-fusion`（已加入 MISC_ARGS）

## 与原始 8 卡脚本的差异

| 项目 | 原始 (8 GPU) | 适配 (4 GPU) |
|---|---|---|
| actor / rollout | 4 / 4 | 2 / 2 |
| TP size | 4 | 2 |
| rollout_batch_size | 16 | 8 |
| n_samples | 8 | 4 |
| conda env | 需手动设 | 硬编码 lightrft_py312 |
| HF_CKPT / REF_LOAD | 需手动设 | 硬编码 puyuan 路径 |
| --no-gradient-accumulation-fusion | 无 | 有（APEX 缺失） |
| --clip-grad | 无 | 1.0（防 NaN） |
| dataset 黑名单 | 无 | 20 个 task |
| 日志自动落盘 | 无 | tmp_doc_latest/ |
| ckpt 自动清理 | 无 | MAX_CKPT_KEEP |
| router timeout/retry | 默认 | 900s / 3次 / 1s backoff |
| NO_PROXY | 无 | 自动从 WORKER_URLS 提取 |
