# a3s-code-adapter

`a3s-code-adapter` 将真实 `a3s-code` Agent 会话接入 OpenClaw-RL / slime 在线强化学习流程。它不依赖离线 SFT 指令数据，而是让 `a3s-code` 在隔离 workspace 中完成真实代码任务，并把可训练 turn、工具反馈、verifier 结果和可选 PRM 分数组织成 slime `Sample`。

核心目标是支持完整 agent-RL，而不是 smoke test：

- 用 OpenAI-compatible RL proxy 拦截 `a3s-code` 对策略模型的请求。
- 用 traffic driver 按任务 seed 持续生成 agent session。
- 用 verifier/PRM/debug reward 生成奖励信号。
- 用 GRPO 分组样本训练策略模型，支持多回答组内归一化。
- 支持长上下文、Docker workspace、LoRA/FSDP 和可选 benchmark eval。

## 目录

| 路径 | 作用 |
| --- | --- |
| `code_rl_api_server.py` | OpenAI-compatible RL proxy；负责会话跟踪、转发策略模型、记录 turn、接收 verifier/PRM 反馈并提交样本。 |
| `code_rl_rollout.py` | slime rollout 入口；从 proxy 等待已标注样本并返回训练 batch。 |
| `a3s_code_agent_traffic_driver.py` | 启动真实 `a3s-code` session，复制任务模板，运行工具/Verifier，并向 proxy 回传反馈。 |
| `run_a3s_code_rl.sh` | 通用训练入口；启动 Ray、slime、SGLang rollout engine 和 RL proxy。 |
| `run_a3s_code_agent_traffic.sh` | traffic driver 入口；可与训练脚本分进程或分机器运行。 |
| `seed_data/` | 内置任务 seed。 |
| `task_templates/` | 供 agent 操作的任务 workspace 模板。 |
| `a3s_code_benchmarks/` | 可选 SkillsBench / ClawMark 评测 runtime。默认训练不会自动启用。 |
| `tests/` | adapter 单元测试。 |

运行产物默认写入 `RUN_ROOT` 或 `ARTIFACT_ROOT` 下，包括 `a3s_workspaces/`、`a3s_configs/`、`a3s_results/`、`logs/` 和 `launch_info.json`。

## 环境准备

在训练环境中安装依赖，并确保 `a3s-code` SDK 可直接从 Python 导入：

```bash
python -m pip install -r requirements.txt
python -m pip install 'a3s-code>=3.0.0'
python -c "import a3s_code; print(a3s_code.__file__)"
```

本 adapter 的长期使用方式是安装发布包，例如 `pip install a3s-code`。只有在对应 SDK 能力尚未发布时，才临时通过 wheel 覆盖当前环境；不要依赖本地源码目录作为常态运行方式。

当前训练链路要求 `a3s-code` 至少支持：

- OpenAI provider 的 `sessionIdHeader`，用于共享配置下按 session 路由请求。
- Skill `allowed-tools` 的 YAML list、legacy 空格分隔和空值语义。
- ACL 中 canonical token limit 写法：`limit = { context = ..., output = ... }`。

## 启动训练

先启动训练服务。下面示例使用 4 卡 Qwen3.5-4B，路径和资源都可以通过环境变量覆盖：

```bash
export HF_CKPT=/path/to/Qwen3.5-4B
export ARTIFACT_ROOT=/path/to/openclaw-rl-artifacts
export NUM_GPUS=4
export ACTOR_GPUS=2
export ROLLOUT_GPUS=2
export ROLLOUT_BATCH_SIZE=8
export N_SAMPLES_PER_PROMPT=4
export CODE_RL_REWARD_MODE=verifier
export CODE_RL_REQUIRE_VERIFIER_FEEDBACK=1

bash a3s-code-adapter/run_a3s_code_rl.sh
```

模型选择通过 `A3S_CODE_MODEL_VARIANT` 控制。当前脚本内置这些 variant，默认权重路径面向本地集群镜像，可用 `HF_CKPT` 覆盖：

- `qwen3.5-4b`
- `qwen3-4b`
- `qwen3.6-35b-a3b`
- `qwen3.5-122b-a10b`
- `qwen3.5-122b-a10b-fp8`
- `qwen3-next-80b-a3b-instruct`
- `glm4.7-flash`

启动后确认 proxy 健康：

```bash
curl http://127.0.0.1:30000/healthz
curl http://127.0.0.1:30000/stats
```

## 注入 agent traffic

训练服务就绪后，在另一个进程启动真实 `a3s-code` 会话：

```bash
export RL_BASE_URL=http://127.0.0.1:30000
export A3S_MODEL_NAME=qwen3.5-4b
export A3S_API_KEY="${SGLANG_API_KEY:-apiKey}"
export A3S_CODE_TRAFFIC_SESSION_LIMIT=32
export A3S_CODE_TRAFFIC_CONCURRENCY=4
export A3S_CODE_SESSION_GROUP_SIZE="${N_SAMPLES_PER_PROMPT:-4}"

bash a3s-code-adapter/run_a3s_code_agent_traffic.sh
```

如果 workspace/verifier 需要 Docker 隔离，启用 Docker backend：

```bash
export A3S_CODE_AGENT_ENV_BACKEND=docker
export A3S_CODE_WORKER_LOCAL_DOCKER=1
export A3S_CODE_AGENT_DOCKER_IMAGE=<image-with-python-and-a3s-code>
export A3S_CODE_AGENT_DOCKER_NETWORK=host
```

在多机部署时，GPU 节点只需运行 `run_a3s_code_rl.sh`，CPU/Docker 节点只需拿到 `RL_BASE_URL`、任务数据路径和相同的 `a3s-code` Python 环境后运行 `run_a3s_code_agent_traffic.sh`。

## 奖励信号

推荐正式 agent 任务优先使用 verifier reward：

```bash
export CODE_RL_REWARD_MODE=verifier
export CODE_RL_REQUIRE_VERIFIER_FEEDBACK=1
export A3S_CODE_ENABLE_TASK_VERIFIER_REWARD=1
```

traffic driver 会按模板的 verifier command 运行检查，并向 proxy 回传 `task_verifier_reward`。Proxy 只提交带有效 verifier feedback 的训练样本，避免无奖励 turn 污染 GRPO。

可选模式：

- `verifier`：只使用任务 verifier，适合 ClawGym / 单测类任务。
- `prm`：使用外部或本地 PRM，对没有 hidden verifier 的多轮 next-state 打分。
- `hybrid`：优先 verifier，缺失时使用 PRM 或规则信号。
- `debug`：用于链路调试，不适合作为正式训练奖励。

## GRPO 分组

GRPO 需要同一 prompt 的多个回答进行组内归一化。traffic driver 通过 session id 中的 `grpXXXXXX-repYY` 组织回答组。关键变量：

```bash
export ROLLOUT_BATCH_SIZE=8
export N_SAMPLES_PER_PROMPT=4
export A3S_CODE_SESSION_GROUP_SIZE=4
export NUM_ROLLOUT=200
```

总 session 预算约为：

```text
(NUM_ROLLOUT - START_ROLLOUT_ID) * ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT
```

如果只生成 1 个回答，GRPO 的组内优势会退化。资源允许时应保持 `N_SAMPLES_PER_PROMPT >= 4`。

## 验证

运行核心单测：

```bash
PYTHONPATH=a3s-code-adapter python -m pytest \
  a3s-code-adapter/tests/test_runtime_guards.py \
  a3s-code-adapter/tests/test_driver_config_modes.py \
  a3s-code-adapter/tests/test_benchmark_eval_builder.py \
  a3s-code-adapter/tests/test_skillsbench_toml_patch.py -q
```

检查脚本语法：

```bash
bash -n a3s-code-adapter/run_a3s_code_rl.sh
bash -n a3s-code-adapter/run_a3s_code_agent_traffic.sh
```
