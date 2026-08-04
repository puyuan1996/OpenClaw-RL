#!/usr/bin/env bash
# terminal-rl_qwen3-8b_exploration_pu.sh
# Exploration-augmented training wrapper around terminal-rl_qwen3-8b_pu.sh.
#
# All options default OFF -> baseline training semantics when all disabled.
# Dataset and algorithm selection intentionally follow terminal-rl_qwen3-8b_pu.sh:
#   DATASET=seta|safety|agentharm|mixed|tau2
#   ALGO=grpo|dapo
#
# USAGE:
#   bash terminal-rl/terminal-rl_qwen3-8b_exploration_pu.sh              # pure baseline
#   DATASET=safety ALGO=dapo bash ...exploration_pu.sh                   # ASB + DAPO
#   EXPLORATION_PROFILE=robust_dapo_lite bash ...exploration_pu.sh       # conservative DAPO exploration preset
#   EXPLORATION_PROFILE=spear_lite bash ...exploration_pu.sh             # SPEAR-style intrinsic curriculum
#   EXPLORE_ENTROPY_COEF=0.01 bash ...exploration_pu.sh                  # +entropy bonus
#   EXPLORE_THINK_MODE=1 bash ...exploration_pu.sh                        # +think mode
#   EXPLORE_INTRINSIC=1 bash ...exploration_pu.sh                         # +intrinsic reward
#   EXPLORE_SAFETY_FILTER=1 bash ...exploration_pu.sh                     # +safety filter
#   ALGO=dapo EXPLORE_CDE_ACTOR=1 bash ...exploration_pu.sh               # +CDE actor/PPL bonus
#   EXPLORE_ENTROPY_COEF=0.01 EXPLORE_INTRINSIC=1 bash ...exploration_pu.sh # combined
#
# BASELINE OPTIONS (same names/defaults as terminal-rl_qwen3-8b_pu.sh):
#   DATASET                  : seta|safety|agentharm|mixed|tau2 (default seta)
#   ALGO                     : grpo|dapo (default grpo)
#   MIX_SETA_RATIO           : mixed seta ratio
#   MIX_SAFETY_RATIO         : mixed Agent-SafetyBench ratio
#   MIX_AGENTHARM_RATIO      : mixed AgentHarm ratio
#   SETA_SAFETY              : none|clawsentry (default clawsentry)
#   SAFETY_BENCH_REWARD      : rule|dense_rule|clawsentry (default rule)
#   AGENTHARM_REWARD         : rule|dense_rule|clawsentry (default rule)
#   TERMINAL_STRUCTURED_METRICS: Emit per-dataset JSON reward breakdowns (default 1)
#   TERMINAL_METRICS_JSONL   : Override JSONL path (default <run_dir>/logs/metrics.jsonl)
#
# EXPLORATION OPTIONS:
#   EXPLORATION_PROFILE      : off|robust_dapo_lite|spear_lite (default off)
#   EXPLORE_ENTROPY_COEF      : Entropy bonus coefficient (default 0.0 = OFF)
#                               Recommended: 0.005 ~ 0.02 (AEPO-style)
#   EXPLORE_THINK_MODE        : Enable Qwen3 think mode (0=OFF, 1=ON)
#   EXPLORE_TEMP_HIGH         : Rollout temperature override (empty = inherit baseline 1.0)
#   EXPLORE_INTRINSIC         : Count-based intrinsic reward (0=OFF, 1=ON)
#   EXPLORE_INTRINSIC_COEF    : Intrinsic reward weight (default 0.1)
#   EXPLORE_INTRINSIC_SCHEDULE: constant|cosine|linear (default constant)
#   EXPLORE_INTRINSIC_DECAY_STEPS: Schedule length in train steps (0=OFF)
#   EXPLORE_INTRINSIC_GRANULARITY: raw|signature (default raw)
#   EXPLORE_INTRINSIC_SCOPE   : process|episode (default process; robust_dapo_lite uses episode)
#   EXPLORE_SAFETY_FILTER     : Regex-based dangerous command penalty (0=OFF, 1=ON)
#   EXPLORE_SAFETY_FILTER_COEF: Safety penalty coefficient (default -0.5)
#   EXPLORE_MAX_TURN          : Override MAX_TURN (empty = inherit baseline 10)
#
# LaMer-inspired Options (from ICLR '26 "Meta-RL Induces Exploration in Language Agents"):
#   EXPLORE_LPRND             : LP-RND lifelong novelty bonus (0=OFF, 1=ON)
#                               Uses mean negative policy logprob as zero-cost novelty
#                               signal; no extra model parameters.
#   EXPLORE_LPRND_COEF        : LP-RND reward weight (default 0.05)
#   EXPLORE_LPRND_SCHEDULE    : constant|cosine|linear (default constant)
#   EXPLORE_LPRND_DECAY_STEPS : Schedule length in train steps (0=OFF)
#   EXPLORE_LPRND_CLIP        : LP-RND z-score clip (default 3.0)
#   EXPLORE_LPRND_WARMUP      : Number of trajectories before LP-RND stats start (default 32)
#   EXPLORE_ADVANTAGE_BONUS   : Add selected explore components after GRPO normalization (0=OFF, 1=ON)
#   EXPLORE_ADVANTAGE_BONUS_COMPONENTS: comma-separated reward keys (default explore_intrinsic_scaled)
#   EXPLORE_ADVANTAGE_BONUS_COEF: post-normalization bonus coefficient (default 1.0)
#   EXPLORE_ADVANTAGE_BONUS_CLIP: absolute bonus clip before coefficient (default 0.25)
#   EXPLORE_CDE_ACTOR         : CDE actor/PPL bonus (0=OFF, 1=ON)
#                               Adds omega * min(|r|/kappa, alpha * -mean_logprob)
#   EXPLORE_CDE_ACTOR_OMEGA   : CDE actor bonus weight (default 0.05)
#   EXPLORE_CDE_ACTOR_KAPPA   : Reward-magnitude clip divisor (default 2.0)
#   EXPLORE_CDE_ACTOR_ALPHA   : Log-PPL scale before clipping (default 0.1)
#   EXPLORE_CDE_ACTOR_REWARD_GATE: nonzero|positive|none (default nonzero)
#   EXPLORE_CDE_ACTOR_DECAY_STEPS: Linear omega decay steps (0=OFF)
#   EXPLORE_RETRY_ATTEMPTS    : Multi-attempt reflection (1=OFF/baseline, 2-3=ON)
#                               LaMer uses 3; each failed attempt generates a reflection turn
#   EXPLORE_RETRY_TRAJ_GAMMA  : Cross-attempt discount factor (default 1.0=OFF; LaMer uses 0.6)
#                               Rewards in earlier attempts are discounted, incentivising faster solve

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── Baseline-compatible Dataset / Algorithm Options ─────────────────
# Keep these names/defaults aligned with terminal-rl_qwen3-8b_pu.sh so switching
# between the main and exploration scripts does not require mental translation.
USER_ALGO_SET="${ALGO+x}"
DATASET="${DATASET:-seta}"
ALGO="${ALGO:-grpo}"
case "${DATASET}" in
  seta|safety|agentharm|mixed|tau2) ;;
  *)
    echo "[ERROR] Unknown DATASET=${DATASET}. Use: seta|safety|agentharm|mixed|tau2" >&2
    exit 1
    ;;
esac
case "${ALGO}" in
  grpo|dapo) ;;
  *)
    echo "[ERROR] Unknown ALGO=${ALGO}. Use: grpo|dapo" >&2
    exit 1
    ;;
esac
export DATASET ALGO
SETA_SAFETY="${SETA_SAFETY:-clawsentry}"
SAFETY_BENCH_REWARD="${SAFETY_BENCH_REWARD:-rule}"
AGENTHARM_REWARD="${AGENTHARM_REWARD:-rule}"
SAFETY_REWARD_COEF="${SAFETY_REWARD_COEF:-0.3}"
export SETA_SAFETY SAFETY_BENCH_REWARD AGENTHARM_REWARD SAFETY_REWARD_COEF

# ── Exploration Options (all default OFF for baseline compatibility) ──
EXPLORATION_PROFILE="${EXPLORATION_PROFILE:-${EXPLORE_PROFILE:-off}}"
EXPLORE_ENTROPY_COEF="${EXPLORE_ENTROPY_COEF:-0.0}"
EXPLORE_THINK_MODE="${EXPLORE_THINK_MODE:-0}"
EXPLORE_TEMP_HIGH="${EXPLORE_TEMP_HIGH:-}"
EXPLORE_INTRINSIC="${EXPLORE_INTRINSIC:-0}"
EXPLORE_INTRINSIC_COEF="${EXPLORE_INTRINSIC_COEF:-0.1}"
EXPLORE_INTRINSIC_SCHEDULE="${EXPLORE_INTRINSIC_SCHEDULE:-constant}"
EXPLORE_INTRINSIC_DECAY_STEPS="${EXPLORE_INTRINSIC_DECAY_STEPS:-0}"
EXPLORE_INTRINSIC_GRANULARITY="${EXPLORE_INTRINSIC_GRANULARITY:-raw}"
EXPLORE_INTRINSIC_SCOPE="${EXPLORE_INTRINSIC_SCOPE:-process}"
EXPLORE_SAFETY_FILTER="${EXPLORE_SAFETY_FILTER:-0}"
EXPLORE_SAFETY_FILTER_COEF="${EXPLORE_SAFETY_FILTER_COEF:--0.5}"
EXPLORE_MAX_TURN="${EXPLORE_MAX_TURN:-}"
# LaMer-inspired
EXPLORE_LPRND="${EXPLORE_LPRND:-0}"
EXPLORE_LPRND_COEF="${EXPLORE_LPRND_COEF:-0.05}"
EXPLORE_LPRND_SCHEDULE="${EXPLORE_LPRND_SCHEDULE:-constant}"
EXPLORE_LPRND_DECAY_STEPS="${EXPLORE_LPRND_DECAY_STEPS:-0}"
EXPLORE_LPRND_CLIP="${EXPLORE_LPRND_CLIP:-3.0}"
EXPLORE_LPRND_WARMUP="${EXPLORE_LPRND_WARMUP:-32}"
EXPLORE_ADVANTAGE_BONUS="${EXPLORE_ADVANTAGE_BONUS:-0}"
EXPLORE_ADVANTAGE_BONUS_COMPONENTS="${EXPLORE_ADVANTAGE_BONUS_COMPONENTS:-explore_intrinsic_scaled}"
EXPLORE_ADVANTAGE_BONUS_COEF="${EXPLORE_ADVANTAGE_BONUS_COEF:-1.0}"
EXPLORE_ADVANTAGE_BONUS_CLIP="${EXPLORE_ADVANTAGE_BONUS_CLIP:-0.25}"
EXPLORE_CDE_ACTOR="${EXPLORE_CDE_ACTOR:-0}"
EXPLORE_CDE_ACTOR_OMEGA="${EXPLORE_CDE_ACTOR_OMEGA:-0.05}"
EXPLORE_CDE_ACTOR_KAPPA="${EXPLORE_CDE_ACTOR_KAPPA:-2.0}"
EXPLORE_CDE_ACTOR_ALPHA="${EXPLORE_CDE_ACTOR_ALPHA:-0.1}"
EXPLORE_CDE_ACTOR_DECAY_STEPS="${EXPLORE_CDE_ACTOR_DECAY_STEPS:-0}"
EXPLORE_CDE_ACTOR_REWARD_GATE="${EXPLORE_CDE_ACTOR_REWARD_GATE:-nonzero}"
EXPLORE_RETRY_ATTEMPTS="${EXPLORE_RETRY_ATTEMPTS:-1}"
EXPLORE_RETRY_TRAJ_GAMMA="${EXPLORE_RETRY_TRAJ_GAMMA:-1.0}"
TERMINAL_STRUCTURED_METRICS="${TERMINAL_STRUCTURED_METRICS:-1}"
TERMINAL_METRICS_JSONL="${TERMINAL_METRICS_JSONL:-}"

case "${EXPLORATION_PROFILE}" in
  off|"") ;;
  robust_dapo_lite)
    # Preferred low-risk exploration algorithm:
    # DAPO + small entropy regularization + episode-local signature novelty +
    # dangerous-command penalty. We intentionally keep LP-RND opt-in because it
    # needs more calibration and historically had process-local running stats.
    if [[ -z "${USER_ALGO_SET}" ]]; then
      ALGO="dapo"
      export ALGO
    elif [[ "${ALGO}" != "dapo" ]]; then
      echo "[explore] robust_dapo_lite requires DAPO; overriding ALGO=${ALGO} -> dapo"
      ALGO="dapo"
      export ALGO
    fi
    [[ "${EXPLORE_ENTROPY_COEF}" == "0.0" || "${EXPLORE_ENTROPY_COEF}" == "0" ]] && EXPLORE_ENTROPY_COEF="0.01"
    [[ "${EXPLORE_INTRINSIC}" == "0" ]] && EXPLORE_INTRINSIC="1"
    [[ "${EXPLORE_INTRINSIC_COEF}" == "0.1" ]] && EXPLORE_INTRINSIC_COEF="0.03"
    [[ "${EXPLORE_INTRINSIC_GRANULARITY}" == "raw" ]] && EXPLORE_INTRINSIC_GRANULARITY="signature"
    [[ "${EXPLORE_INTRINSIC_SCOPE}" == "process" ]] && EXPLORE_INTRINSIC_SCOPE="episode"
    [[ "${EXPLORE_SAFETY_FILTER}" == "0" ]] && EXPLORE_SAFETY_FILTER="1"
    ;;
  spear_lite)
    # SPEAR-lite keeps only the low-risk part that fits terminal-rl today:
    # curriculum-scheduled intrinsic tool/command reward. Full SPEAR SIL replay
    # and clip-cov regularization require actor/trainer changes and stay out of
    # this compatibility wrapper.
    if [[ -z "${USER_ALGO_SET}" ]]; then
      ALGO="dapo"
      export ALGO
    elif [[ "${ALGO}" != "dapo" ]]; then
      echo "[explore] spear_lite is calibrated for DAPO; overriding ALGO=${ALGO} -> dapo"
      ALGO="dapo"
      export ALGO
    fi
    [[ "${EXPLORE_INTRINSIC}" == "0" ]] && EXPLORE_INTRINSIC="1"
    [[ "${EXPLORE_INTRINSIC_COEF}" == "0.1" ]] && EXPLORE_INTRINSIC_COEF="0.03"
    [[ "${EXPLORE_INTRINSIC_SCHEDULE}" == "constant" ]] && EXPLORE_INTRINSIC_SCHEDULE="cosine"
    [[ "${EXPLORE_INTRINSIC_DECAY_STEPS}" == "0" ]] && EXPLORE_INTRINSIC_DECAY_STEPS="200"
    [[ "${EXPLORE_INTRINSIC_GRANULARITY}" == "raw" ]] && EXPLORE_INTRINSIC_GRANULARITY="signature"
    [[ "${EXPLORE_INTRINSIC_SCOPE}" == "process" ]] && EXPLORE_INTRINSIC_SCOPE="episode"
    ;;
  *)
    echo "[ERROR] Unknown EXPLORATION_PROFILE=${EXPLORATION_PROFILE}. Use: off|robust_dapo_lite|spear_lite" >&2
    exit 1
    ;;
esac

echo "========================================"
echo "  Exploration Options"
echo "  DATASET         = ${DATASET}"
echo "  ALGO            = ${ALGO}"
if [[ "${DATASET}" == "mixed" ]]; then
echo "  MIX_RATIOS      = seta:${MIX_SETA_RATIO:-<default>} safety:${MIX_SAFETY_RATIO:-<default>} agentharm:${MIX_AGENTHARM_RATIO:-<unset>}"
fi
echo "  REWARD_MODES    = seta:${SETA_SAFETY} safety:${SAFETY_BENCH_REWARD} agentharm:${AGENTHARM_REWARD} coef:${SAFETY_REWARD_COEF}"
echo "  STRUCTURED_LOGS = ${TERMINAL_STRUCTURED_METRICS} (jsonl=${TERMINAL_METRICS_JSONL:-<run_dir>/logs/metrics.jsonl})"
echo "  PROFILE         = ${EXPLORATION_PROFILE:-off}"
echo "  ENTROPY_COEF    = ${EXPLORE_ENTROPY_COEF}"
echo "  THINK_MODE      = ${EXPLORE_THINK_MODE}"
echo "  TEMP_HIGH       = ${EXPLORE_TEMP_HIGH:-<inherit>}"
echo "  INTRINSIC       = ${EXPLORE_INTRINSIC} (coef=${EXPLORE_INTRINSIC_COEF}, schedule=${EXPLORE_INTRINSIC_SCHEDULE}/${EXPLORE_INTRINSIC_DECAY_STEPS}, granularity=${EXPLORE_INTRINSIC_GRANULARITY}, scope=${EXPLORE_INTRINSIC_SCOPE})"
echo "  SAFETY_FILTER   = ${EXPLORE_SAFETY_FILTER} (coef=${EXPLORE_SAFETY_FILTER_COEF})"
echo "  MAX_TURN        = ${EXPLORE_MAX_TURN:-<inherit>}"
echo "  LPRND           = ${EXPLORE_LPRND} (coef=${EXPLORE_LPRND_COEF}, schedule=${EXPLORE_LPRND_SCHEDULE}/${EXPLORE_LPRND_DECAY_STEPS}, clip=${EXPLORE_LPRND_CLIP}, warmup=${EXPLORE_LPRND_WARMUP}) [LaMer]"
echo "  POST_NORM_BONUS = ${EXPLORE_ADVANTAGE_BONUS} (components=${EXPLORE_ADVANTAGE_BONUS_COMPONENTS}, coef=${EXPLORE_ADVANTAGE_BONUS_COEF}, clip=${EXPLORE_ADVANTAGE_BONUS_CLIP})"
echo "  CDE_ACTOR       = ${EXPLORE_CDE_ACTOR} (omega=${EXPLORE_CDE_ACTOR_OMEGA}, alpha=${EXPLORE_CDE_ACTOR_ALPHA}, kappa=${EXPLORE_CDE_ACTOR_KAPPA}, gate=${EXPLORE_CDE_ACTOR_REWARD_GATE}, decay_steps=${EXPLORE_CDE_ACTOR_DECAY_STEPS}) [RLVR CDE]"
echo "  RETRY_ATTEMPTS  = ${EXPLORE_RETRY_ATTEMPTS} (traj_gamma=${EXPLORE_RETRY_TRAJ_GAMMA}) [LaMer]"
echo "========================================"

# ── 1. Entropy bonus -> pass to baseline via EXTRA_ALGO_ARGS ──
# Bug fix: the old wrapper used EXTRA_GRPO_ARGS only. terminal-rl_qwen3-8b_pu.sh
# intentionally ignores EXTRA_GRPO_ARGS when ALGO=dapo, so entropy silently
# disappeared exactly in the DAPO exploration setting this script is meant to run.
# EXTRA_ALGO_ARGS is consumed by both GRPO and DAPO branches.
EXTRA_ALGO_ARGS="${EXTRA_ALGO_ARGS:-}"
if [[ "${EXPLORE_ENTROPY_COEF}" != "0" && "${EXPLORE_ENTROPY_COEF}" != "0.0" ]]; then
  EXTRA_ALGO_ARGS="${EXTRA_ALGO_ARGS} --entropy-coef ${EXPLORE_ENTROPY_COEF}"
  echo "[explore] entropy bonus enabled: ${EXPLORE_ENTROPY_COEF}"
fi
export EXTRA_ALGO_ARGS

# ── 2. Think mode → switch to rollout_qwen3_think.yaml ──
if [[ "${EXPLORE_THINK_MODE}" == "1" ]]; then
  THINK_YAML="${SCRIPT_DIR}/configs/rollout_qwen3_think.yaml"
  if [[ -f "${THINK_YAML}" ]]; then
    export CUSTOM_CONFIG_PATH="${THINK_YAML}"
    echo "[explore] think mode ON: ${THINK_YAML}"
  else
    echo "[WARN] ${THINK_YAML} not found, think mode skipped" >&2
  fi
fi

# ── 3. Temperature override → env var, baseline reads it ──
if [[ -n "${EXPLORE_TEMP_HIGH}" ]]; then
  export ROLLOUT_TEMPERATURE="${EXPLORE_TEMP_HIGH}"
  echo "[explore] rollout temperature overridden to ${EXPLORE_TEMP_HIGH}"
fi

# ── 4. Intrinsic reward → env vars for generate.py ──
if [[ "${EXPLORE_INTRINSIC}" == "1" ]]; then
  export EXPLORE_INTRINSIC_ENABLED="1"
  export EXPLORE_INTRINSIC_COEF
  export EXPLORE_INTRINSIC_SCHEDULE
  export EXPLORE_INTRINSIC_DECAY_STEPS
  export EXPLORE_INTRINSIC_GRANULARITY
  export EXPLORE_INTRINSIC_SCOPE
  echo "[explore] intrinsic reward ON (coef=${EXPLORE_INTRINSIC_COEF}, schedule=${EXPLORE_INTRINSIC_SCHEDULE}/${EXPLORE_INTRINSIC_DECAY_STEPS})"
fi

# ── 5. Safety filter → env vars for generate.py ──
if [[ "${EXPLORE_SAFETY_FILTER}" == "1" ]]; then
  export EXPLORE_SAFETY_FILTER_ENABLED="1"
  export EXPLORE_SAFETY_FILTER_COEF
  echo "[explore] safety pre-filter ON (penalty_coef=${EXPLORE_SAFETY_FILTER_COEF})"
fi

# ── 6. MAX_TURN override ──
if [[ -n "${EXPLORE_MAX_TURN}" ]]; then
  export MAX_TURN="${EXPLORE_MAX_TURN}"
  echo "[explore] MAX_TURN overridden to ${EXPLORE_MAX_TURN}"
fi

# ── 7. LP-RND lifelong novelty (LaMer-inspired) ──
if [[ "${EXPLORE_LPRND}" == "1" ]]; then
  export EXPLORE_LPRND_ENABLED="1"
  export EXPLORE_LPRND_COEF
  export EXPLORE_LPRND_SCHEDULE
  export EXPLORE_LPRND_DECAY_STEPS
  export EXPLORE_LPRND_CLIP
  export EXPLORE_LPRND_WARMUP
  echo "[explore] LP-RND lifelong novelty ON (coef=${EXPLORE_LPRND_COEF}, schedule=${EXPLORE_LPRND_SCHEDULE}/${EXPLORE_LPRND_DECAY_STEPS})"
fi

# ── 8. Optional post-normalization exploration bonus ──
if [[ "${EXPLORE_ADVANTAGE_BONUS}" == "1" ]]; then
  export EXPLORE_ADVANTAGE_BONUS_ENABLED="1"
  export EXPLORE_ADVANTAGE_BONUS_COMPONENTS
  export EXPLORE_ADVANTAGE_BONUS_COEF
  export EXPLORE_ADVANTAGE_BONUS_CLIP
  echo "[explore] post-normalization bonus ON (components=${EXPLORE_ADVANTAGE_BONUS_COMPONENTS}, coef=${EXPLORE_ADVANTAGE_BONUS_COEF}, clip=${EXPLORE_ADVANTAGE_BONUS_CLIP})"
fi

# ── 9. CDE actor curiosity bonus (RLVR PPL bonus) ──
if [[ "${EXPLORE_CDE_ACTOR}" == "1" ]]; then
  export EXPLORE_CDE_ACTOR_ENABLED="1"
  export EXPLORE_CDE_ACTOR_OMEGA
  export EXPLORE_CDE_ACTOR_KAPPA
  export EXPLORE_CDE_ACTOR_ALPHA
  export EXPLORE_CDE_ACTOR_DECAY_STEPS
  export EXPLORE_CDE_ACTOR_REWARD_GATE
  echo "[explore] CDE actor/PPL bonus ON (omega=${EXPLORE_CDE_ACTOR_OMEGA}, alpha=${EXPLORE_CDE_ACTOR_ALPHA}, kappa=${EXPLORE_CDE_ACTOR_KAPPA}, gate=${EXPLORE_CDE_ACTOR_REWARD_GATE})"
fi

# ── 10. Multi-attempt reflection (LaMer-inspired) ──
if [[ "${EXPLORE_RETRY_ATTEMPTS}" != "1" ]]; then
  export EXPLORE_RETRY_ATTEMPTS
  export EXPLORE_RETRY_TRAJ_GAMMA
  echo "[explore] multi-attempt reflection ON (attempts=${EXPLORE_RETRY_ATTEMPTS}, traj_gamma=${EXPLORE_RETRY_TRAJ_GAMMA})"
  echo "[WARN] Multi-attempt requires agent_runner support (not yet implemented in terminal-rl)" >&2
fi

# ── 11. Build RUN_ID suffix for easy identification ──
SUF=""
[[ "${EXPLORATION_PROFILE:-off}" != "off" && -n "${EXPLORATION_PROFILE:-}" ]] && SUF="${SUF}_${EXPLORATION_PROFILE}"
[[ "${EXPLORE_ENTROPY_COEF}" != "0" && "${EXPLORE_ENTROPY_COEF}" != "0.0" ]] && SUF="${SUF}_ent${EXPLORE_ENTROPY_COEF}"
[[ "${EXPLORE_THINK_MODE}" == "1" ]] && SUF="${SUF}_think"
[[ -n "${EXPLORE_TEMP_HIGH}" ]] && SUF="${SUF}_T${EXPLORE_TEMP_HIGH}"
[[ "${EXPLORE_INTRINSIC}" == "1" ]] && SUF="${SUF}_int"
[[ "${EXPLORE_INTRINSIC}" == "1" && "${EXPLORE_INTRINSIC_SCHEDULE}" != "constant" ]] && SUF="${SUF}_${EXPLORE_INTRINSIC_SCHEDULE}${EXPLORE_INTRINSIC_DECAY_STEPS}"
[[ "${EXPLORE_SAFETY_FILTER}" == "1" ]] && SUF="${SUF}_safe"
[[ "${EXPLORE_LPRND}" == "1" ]] && SUF="${SUF}_lprnd"
[[ "${EXPLORE_LPRND}" == "1" && "${EXPLORE_LPRND_SCHEDULE}" != "constant" ]] && SUF="${SUF}_${EXPLORE_LPRND_SCHEDULE}${EXPLORE_LPRND_DECAY_STEPS}"
[[ "${EXPLORE_ADVANTAGE_BONUS}" == "1" ]] && SUF="${SUF}_postnorm"
[[ "${EXPLORE_CDE_ACTOR}" == "1" ]] && SUF="${SUF}_cdeact${EXPLORE_CDE_ACTOR_OMEGA}"
[[ "${EXPLORE_RETRY_ATTEMPTS}" != "1" ]] && SUF="${SUF}_retry${EXPLORE_RETRY_ATTEMPTS}"

if [[ -n "${SUF}" ]]; then
  TS="${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
  else
    GPUS=0
  fi
  if [[ "${GPUS}" -le 0 ]]; then
    GPUS="${NUM_GPUS:-4}"
  fi
  export RUN_TIMESTAMP="${TS}"
  export RUN_ID="${RUN_ID:-terminal-rl_qwen3-8b_${GPUS}gpu_${DATASET}_${ALGO}_explore${SUF}_${TS}}"
  echo "[explore] RUN_ID=${RUN_ID}"
fi

# Export all knobs so the main script can persist them in run_config and forward
# them to Ray runtime_env. Without this, generate.py workers do not reliably see
# exploration settings under ray job submit.
export EXPLORATION_PROFILE
export EXPLORE_ENTROPY_COEF EXPLORE_THINK_MODE EXPLORE_TEMP_HIGH
export EXPLORE_INTRINSIC EXPLORE_INTRINSIC_COEF EXPLORE_INTRINSIC_SCHEDULE EXPLORE_INTRINSIC_DECAY_STEPS EXPLORE_INTRINSIC_GRANULARITY EXPLORE_INTRINSIC_SCOPE
export EXPLORE_SAFETY_FILTER EXPLORE_SAFETY_FILTER_COEF
export EXPLORE_MAX_TURN EXPLORE_LPRND EXPLORE_LPRND_COEF EXPLORE_LPRND_SCHEDULE EXPLORE_LPRND_DECAY_STEPS EXPLORE_LPRND_CLIP EXPLORE_LPRND_WARMUP
export EXPLORE_ADVANTAGE_BONUS EXPLORE_ADVANTAGE_BONUS_COMPONENTS EXPLORE_ADVANTAGE_BONUS_COEF EXPLORE_ADVANTAGE_BONUS_CLIP
export EXPLORE_CDE_ACTOR EXPLORE_CDE_ACTOR_OMEGA EXPLORE_CDE_ACTOR_KAPPA EXPLORE_CDE_ACTOR_ALPHA EXPLORE_CDE_ACTOR_DECAY_STEPS EXPLORE_CDE_ACTOR_REWARD_GATE
export EXPLORE_RETRY_ATTEMPTS EXPLORE_RETRY_TRAJ_GAMMA
export TERMINAL_STRUCTURED_METRICS TERMINAL_METRICS_JSONL

# ── 12. Execute baseline script ──
exec bash "${SCRIPT_DIR}/terminal-rl_qwen3-8b_pu.sh" "$@"
