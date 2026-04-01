# a3s-code-adapter for OpenClaw-RL

这是从 `a3s-code-rl` 迁移到 `OpenClaw-RL` 的 a3s-code Agent 数据采集适配层。

## 功能

通过真实的 a3s-code Agent 会话自动生成 RL 训练样本，替代手动交互式对话。

## 目录结构

```
a3s-code-adapter/
├── code_rl_api_server.py           # RL Proxy，拦截请求并记录样本
├── code_rl_rollout.py              # Rollout 接口，管理样本队列
├── a3s_code_agent_traffic_driver.py # 流量驱动器，持续生成会话
├── check_simulated_user_backends.py # 模拟用户后端健康检查
├── run_a3s_code_rl_4gpu.sh         # 主启动脚本（RL 训练）
├── run_a3s_code_agent_traffic.sh   # 流量驱动启动脚本
├── refresh_simulated_user_backends.sh # 刷新模拟用户后端列表
├── seed_data/                      # 种子任务数据
│   └── code_task_seeds.json
├── generated_workspaces/           # 运行时生成的会话工作区
├── generated_configs/              # 运行时生成的配置文件
└── results/                        # 运行日志和记录
```

## 快速启动

### 前置条件

1. 确保 `a3s-code` Python SDK 已安装：
   ```bash
   cd <A3S_CODE_REPO_ROOT>/sdk/python
   maturin develop --release
   ```

2. 配置模拟用户后端（可选，用于任务改写）

### 启动方式

**方式 1：一键启动（推荐）**
```bash
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/slime
bash ../a3s-code-adapter/run_a3s_code_rl_4gpu.sh
```

**方式 2：分离启动（调试用）**
```bash
# 终端 1：启动 RL 训练服务
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/slime
bash ../a3s-code-adapter/run_a3s_code_rl_4gpu.sh

# 终端 2：启动流量驱动（等 RL 服务启动后）
cd /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/a3s-code-adapter
bash run_a3s_code_agent_traffic.sh
```

## 关键环境变量

### RL 训练配置
- `NUM_GPUS=4` - GPU 总数
- `ACTOR_GPUS=2` - Actor 使用的 GPU 数
- `ROLLOUT_GPUS=2` - Rollout 使用的 GPU 数
- `TP_TRAIN=2` - Actor 张量并行度
- `TP_SGLANG=2` - Rollout 张量并行度
- `ENABLE_PRM=1` - 启用 PRM 打分
- `PRM_BACKEND=external_openai` - PRM 后端类型

### 流量驱动配置
- `A3S_CODE_TRAFFIC_CONCURRENCY=1` - 并发会话数
- `A3S_CODE_MAX_MAIN_TURNS=4` - 最大主轮次
- `A3S_CODE_MAX_TOOL_ROUNDS=16` - 最大工具调用轮次
- `CODE_RL_MATCHED_CONTEXT_TOKENS=8192` - 上下文长度

### 模拟用户配置
- `SIMULATED_USER_MODEL_URL` - 模拟用户服务 URL
- `SIMULATED_USER_MODEL_NAME` - 模拟用户模型名称
- `SIMULATED_USER_API_KEY` - API 密钥

## 监控训练进度

```bash
# 查看运行信息
cat runs/<run_id>/launch_info.json

# 查看训练样本记录
tail -f runs/<run_id>/code_rl_record.jsonl

# 查看 PRM 打分记录
tail -f runs/<run_id>/code_rl_prm_record.jsonl

# 查看 Ray Dashboard
# 浏览器访问 http://localhost:8265
```

## 与 openclaw-combine 的区别

| 特性 | openclaw-combine | a3s-code-adapter |
|------|------------------|------------------|
| 数据来源 | 手动交互式对话 | 自动化 Agent 任务执行 |
| 启动方式 | `start_chat.sh` + `interactive_chat.py` | `run_a3s_code_rl_4gpu.sh` |
| 种子数据 | 无 | `seed_data/code_task_seeds.json` |
| 工作区隔离 | 无 | 每个会话独立工作区 |
| 模拟用户 | 无 | 支持任务改写增加多样性 |
| Rollout 函数 | `openclaw_combine_rollout` | `code_rl_rollout` |
| API Server | `openclaw_combine_api_server` | `code_rl_api_server` |

## 故障排查

### 问题：a3s_code 模块导入失败
```bash
# 检查 a3s-code SDK 是否已安装
python3 -c "import a3s_code; print(a3s_code.__file__)"

# 如果未安装，编译安装
cd <A3S_CODE_REPO_ROOT>/sdk/python
maturin develop --release
```

### 问题：RL Proxy 端口 30000 被占用
```bash
# 检查端口占用
ss -ltnp | grep 30000

# 清理旧进程
pkill -f "code_rl_api_server"
pkill -f "sglang"
ray stop --force
```

### 问题：模拟用户后端不可用
```bash
# 手动刷新后端列表
bash refresh_simulated_user_backends.sh

# 查看可用后端
cat simulated_user_backends.json
```

## 迁移说明

本适配层从 `/mnt/shared-storage-user/puyuan/SafeCode/a3s-code-rl` 迁移而来，主要调整：

1. 路径适配：指向 OpenClaw-RL 的 `slime/` 和 `Megatron-LM/`
2. 模型配置：使用 `qwen3-4B.sh`（rotary_base=1M）
3. 启动方式：使用 `ray job submit` + `train_async.py`（与 OpenClaw-RL 一致）
4. 保持兼容：保留 openclaw-combine 的现有功能

## 相关文档

- [a3s-code-rl 原始 README](../../SafeCode/a3s-code-rl/README.md)
- [OpenClaw-RL 主文档](../README.md)
