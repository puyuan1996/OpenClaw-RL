# mode B aligned eval — 训练侧对齐的 Terminal-Bench 评测

## TL;DR

用 Harbor 评测 terminal-rl 训练出的 checkpoint 时，Harbor 自带的 `terminus-2` agent 与训练侧的 `CamelAgent` 在 prompt、tool schema、迭代上限、采样参数上都不一样，测出来的分数分不清是"模型能力差"还是"评测 harness 和训练不一致"。本目录提供一个 Harbor `BaseAgent` 适配器 `OpenClawCamelAgent`，让 Harbor 继续负责 docker-compose 生命周期、任务装配和 verifier，而 agent 驱动改用训练时那条代码路径（`CamelAgent` + `SGLangTurnClient` + `TerminalToolkit`）。两条评测路线分别称为 mode A（Harbor 原生 `terminus-2`）和 mode B（本适配器）。跑一次评测需要两步：`launchers/launch_sglang.sh` 起模型服务，`launchers/run_harbor_eval.sh` 起评测。逐项对齐表、历史评测结果和已知边界见 [`../../docs/HARBOR_CAMEL_MODE_B_zh.md`](../../docs/HARBOR_CAMEL_MODE_B_zh.md)；在受限 GPU worker 上从零搭 Docker / 代理 / 镜像预拉的完整运维流程见 [`../../docs/TBV21_HARBOR_FULL_EVAL_zh.md`](../../docs/TBV21_HARBOR_FULL_EVAL_zh.md)。

## 目录布局

| 路径 | 内容 |
|---|---|
| `adapter/openclaw_camel_adapter.py` | Harbor `BaseAgent` 适配器，唯一的运行时代码 |
| `launchers/launch_sglang.sh` | 起 SGLang 服务，评测相关 flag 全部按历史评测口径钉死 |
| `launchers/run_harbor_eval.sh` | 起 Harbor 评测，full 与 smoke 由环境变量区分 |
| `harbor_job_report.py` | 解析任意 Harbor job 目录：进度、聚合分数、解出任务、错误分布；`--watch` 轮询到结束 |
| `analysis/` | issue #21–#25 的一次性复现脚本，见下面「analysis/ 的定位」 |

## 快速开始

先起模型服务。`MODEL_DIR` 同时用于权重和 tokenizer，因为 chat template 必须来自被评测的那个 checkpoint。

```bash
MODEL_DIR=/path/to/qwen3-8b-hf \
SERVED_NAME=qwen3-8b-modeB \
TP_SIZE=4 \
bash launchers/launch_sglang.sh
```

等 `curl -s http://127.0.0.1:30000/v1/models` 能返回 `SERVED_NAME` 之后，先跑单任务 smoke，确认适配器、SGLang、Docker 三方都握手成功：

```bash
SERVED_NAME=qwen3-8b-modeB \
MODEL_DIR=/path/to/qwen3-8b-hf \
DATASET_DIR=/path/to/terminal-bench-dataset \
TASK_ID=git-multibranch K=1 N_CONCURRENT=1 JOBS_DIR=/tmp/smoke_jobs \
bash launchers/run_harbor_eval.sh
```

smoke 通过后跑全量。89 任务 × k=3 是历史评测使用的规模，单次约 4–12 小时，取决于超时任务的比例。

```bash
SERVED_NAME=qwen3-8b-modeB \
MODEL_DIR=/path/to/qwen3-8b-hf \
DATASET_DIR=/path/to/terminal-bench-dataset \
K=3 N_CONCURRENT=4 JOBS_DIR=/path/to/jobs \
bash launchers/run_harbor_eval.sh
```

## 读结果

`harbor_job_report.py` 解析 job 目录，不需要每次现写 JSON 遍历：

```bash
python harbor_job_report.py /path/to/jobs/<job-name>            # 人读
python harbor_job_report.py /path/to/jobs/<job-name> --json     # 入库
python harbor_job_report.py /path/to/jobs/<job-name> --watch    # 轮询到 finished_at 出现
```

它同时打印两个分母：Harbor 的报告口径 `reward_sum / n_total_trials`，以及只对跑到 verifier 的 trial 求的均值。后者更高，因为它把"没跑到 verifier 就报错"的 trial 移出了分母；两个数并排出现，是为了让这个差距无处可藏。

## 换 checkpoint 不需要改代码

`SERVED_NAME` 和 `MODEL_DIR` 是每次评测都必须显式给的两个参数，适配器没有为它们保留默认值：served name 填错会静默评测到另一个模型，tokenizer 目录填错会静默换掉 chat template，两种错误都不会报错、只会让分数变得无法解释，所以宁可启动即失败。其余对齐 knob（`max_iteration`、`temperature`、`max_new_tokens`、`non_think_mode`、`tool_call_parser` 等）都在适配器 `__init__` 里暴露成 kwargs，用 `--agent-kwarg key=value` 覆盖；覆盖任何一个都会让结果不再与 [`../../docs/HARBOR_CAMEL_MODE_B_zh.md`](../../docs/HARBOR_CAMEL_MODE_B_zh.md) 里记录的历史评测同口径。

## analysis/ 的定位

`analysis/` 下 12 个脚本是 issue #21–#25 出图和出表用的一次性复现脚本，不是通用工具，也没有命令行接口。它们把 job 目录和输出路径写死在模块级常量里，例如 `analysis/build_aligned_jsonl.py:32-34` 的 `BASE` / `OUT_JSONL` / `OUT_SUMMARY`；12 个脚本里只有 `build_base_comparison_figs.py` 用到 `argparse`。要针对新的 job 目录重跑，只能改脚本顶部的常量。这里原样保留它们，是为了让 issue #21–#25 的图表可以被重新生成、结论可以被复核，而不是把它们当作后续评测的分析工具链。

## 依赖

适配器从自身位置向上三级解析出 `terminal-rl/` 包根，所以在任意 checkout、任意 cwd 下都能跑；需要指向另一个 checkout 时设 `OPENCLAW_TERMINAL_RL_DIR`。解析失败会立刻抛 `RuntimeError` 并说明期望的目录结构，而不是等到 `import agent.camel_agent` 时报一个没有上下文的 `ModuleNotFoundError`。运行时还需要 `harbor`、`sglang`、`camel-ai`、`transformers`、`httpx`，以及可用的 `docker compose` 和当前用户在 `docker` 组内。`slime` 可有可无：装了就复用 `slime.utils.http_utils`，没装则适配器注入一个等价的 httpx 实现。
