# Remote worker (pool server)

This directory runs on the **CPU worker**: a pool server that manages Docker containers and executes terminal tasks on behalf of GPU training nodes.

For from-zero setup, hardening, watchdog, and recovery procedures, see the **operations runbook**: [`../docs/cpu_worker_docker_ops.md`](../docs/cpu_worker_docker_ops.md).

---

## Active scripts (current workflow)

### Setup & recovery (run on a fresh / broken CPU worker)

| Script | When to use |
|---|---|
| `setup_new_worker.sh` | Scenario A entry: first-time setup on a brand-new machine. Installs Docker/Compose, writes daemon config, hardens proxy/base images, installs watchdog, and verifies a build. |
| `fix_dockerd_and_proxy.sh` | Scenario B entry: one-shot recovery when Docker/proxy/build path is broken. Watchdog-aware; internally calls `prebuild_proxied_base_images.sh`. |
| `docker_worker_doctor.sh` | Log-aware diagnosis and repair wrapper. Use `diagnose` to analyze GPU train logs plus CPU worker Docker state; use `soft-repair` / `full-repair` for recovery. |
| `prebuild_proxied_base_images.sh` | Wraps the top base images with `apt.conf.d` proxy injection — mandatory in proxied environments because Ubuntu apt does not honor `HTTP_PROXY` env var |
| `restart_docker_force.sh` | Manual force-restart of dockerd (bypasses systemctl, used by watchdog and as escape hatch) |

### Steady-state (every training run)

| Script / file | Role |
|---|---|
| `run_pool_server_pu_v2.sh` | Hardened pool server launcher. Sources `/etc/seta_build_proxy.env`, sanity-checks dockerd, configures capacity, starts uvicorn |
| `pool_server.py` | FastAPI service exposed on port 18081 |
| `terminal_env.py` | Environment client used by pool server |
| `docker_compose_utils.py` | Helper to build / up / down compose stacks |
| `compose_override.yaml` | Optional override (set `COMPOSE_OVERRIDE_PATH=` to use) |

### Watchdog (recommended for >4h runs)

| File | Role |
|---|---|
| `docker_watchdog_v2.sh` | Main watchdog loop. Auto-restarts dockerd on hang; monitors pool_server `/healthz`; cleans address-pool exhaustion |
| `docker-watchdog.service` | systemd unit. Install with `systemctl enable --now docker-watchdog` |

### Manual ops

| Script | Role |
|---|---|
| `cleanup_docker_cache.sh` | Safe cleanup of build cache + stopped containers + dangling images (won't kill running) |
| `diag_docker_failures_lite.sh` | Lightweight diagnostic safe to run in parallel with training |

---

## Quick start (assuming machine already set up)

From the repo root:

```bash
# The launcher auto-sources /etc/seta_build_proxy.env when present.
bash terminal-rl/remote/run_pool_server_pu_v2.sh
```

Then on the GPU worker:

```bash
export WORKER_URLS="http://<this-cpu-worker-ip>:18081"
bash terminal-rl/terminal-rl_qwen3-8b_pu.sh
```

For first-time setup and recovery, follow [`../docs/cpu_worker_docker_ops.md`](../docs/cpu_worker_docker_ops.md).

For a failed training run, start with a log-aware diagnosis:

```bash
bash terminal-rl/remote/docker_worker_doctor.sh diagnose \
  --train-log /mnt/shared-storage-user/puyuan/code/OpenClaw-RL/runs/<run>/logs/train.log
```

---

## Optional environment variables (read by `pool_server.py`)

| Variable | Default | Description |
|---|---|---|
| `DATASET_DIR` | `terminal-rl/dataset` | Path to the task dataset directory |
| `TBENCH_OUTPUT_ROOT` | `terminal-rl/build_outputs` | Root for build/output artifacts |
| `ENV_SERVER_PORT` | `18081` | Port the pool server listens on |
| `WORKER_MAX_TASKS` | `16` | Max tasks allocated per worker |
| `WORKER_MAX_RUNS_PER_TASK` | `8` | Max concurrent runs per task |
| `TBENCH_DOCKER_IMAGE_SOURCE` | `build` | `build` or `pull` — build locally or pull from registry |
| `TBENCH_DOCKER_PULL_PREFIX` | — | Image prefix for `pull` mode |
| `COMPOSE_OVERRIDE_PATH` | — | Optional Docker Compose override file |

Capacity sizing rule (from issue #3 §1):
```
WORKER_MAX_TASKS × WORKER_MAX_RUNS_PER_TASK ≥ rollout_batch_size × n_samples_per_prompt
```

For 8×4 (current default), 16×8=128 has been observed to saturate dockerd. v2 launcher defaults to a more conservative 64×16=1024 to leave headroom.

---

## Archived scripts

Earlier versions and one-off fixes are preserved in `../archive/remote/` (see [`../archive/README.md`](../archive/README.md)).
