#!/usr/bin/env bash
# a3s-code-adapter training launcher.
# Default GPU allocation is conservative and can be overridden with NUM_GPUS,
# ACTOR_GPUS, ROLLOUT_GPUS, TP_TRAIN, PP_TRAIN, TP_SGLANG, and COLOCATE.
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${NVIDIA_VISIBLE_DEVICES:-}" && "${NVIDIA_VISIBLE_DEVICES}" != "all" && "${NVIDIA_VISIBLE_DEVICES}" != "void" ]]; then
  if [[ "${NVIDIA_VISIBLE_DEVICES}" == *GPU-* ]] && command -v nvidia-smi >/dev/null 2>&1; then
    mapped_cuda_devices="$(
      NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}" nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits |
        awk -F, '
          BEGIN {
            split(ENVIRON["NVIDIA_VISIBLE_DEVICES"], want, ",")
            for (i in want) wanted[want[i]] = 1
          }
          {
            gsub(/^[ \t]+|[ \t]+$/, "", $1)
            gsub(/^[ \t]+|[ \t]+$/, "", $2)
            if ($2 in wanted) {
              out = out (out == "" ? "" : ",") $1
            }
          }
          END { print out }
        '
    )"
    if [[ -n "${mapped_cuda_devices}" ]]; then
      export CUDA_VISIBLE_DEVICES="${mapped_cuda_devices}"
    fi
  else
    export CUDA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"
  fi
fi

# Ensure the intended Python environment is used even when conda activate is not available.
if [[ -z "${CONDA_ENV:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    CONDA_ENV="${CONDA_PREFIX}"
  else
    CONDA_ENV="$(python3 - <<'PY'
import sys
print(sys.prefix)
PY
)"
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python3}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
export CONDA_ENV
export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_SITE_PACKAGES="${CONDA_ENV}/lib/python${PYTHON_VERSION}/site-packages"
TORCH_LIB_DIR="${PYTHON_SITE_PACKAGES}/torch/lib"
CUDA_RUNTIME_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/lib"
CUDA_NVRTC_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib"
CUDNN_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib"
CURAND_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/curand/lib"
CUBLAS_INCLUDE_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cublas/include"
CUBLAS_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cublas/lib"
CUDNN_INCLUDE_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cudnn/include"
CUDA_NVRTC_INCLUDE_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cuda_nvrtc/include"
HOST_CUDA_DRIVER_LIB_DIR="/usr/local/nvidia/lib64"
CUDA_TARGET_DIR="${CONDA_ENV}/targets/x86_64-linux"
CUDA_NVCC_SITE_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cuda_nvcc"

if [[ ! -d "${CUDA_RUNTIME_LIB_DIR}" ]]; then
  echo "missing CUDA runtime dir: ${CUDA_RUNTIME_LIB_DIR}"
  exit 1
fi

LD_LIBRARY_PATH_PARTS=("${CONDA_ENV}/lib")
for lib_dir in "${TORCH_LIB_DIR}" "${CUDA_RUNTIME_LIB_DIR}" "${CUDA_NVRTC_LIB_DIR}" "${CUDNN_LIB_DIR}" "${CURAND_LIB_DIR}" "${CUBLAS_LIB_DIR}" "${HOST_CUDA_DRIVER_LIB_DIR}"; do
  if [[ -d "${lib_dir}" ]]; then
    LD_LIBRARY_PATH_PARTS+=("${lib_dir}")
  fi
done
LD_LIBRARY_PATH_PREFIX="$(IFS=:; echo "${LD_LIBRARY_PATH_PARTS[*]}")"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_PREFIX}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_ROOT="${PROJECT_ROOT}/slime"
MEGATRON_ROOT="${PROJECT_ROOT}/Megatron-LM"
CODE_RL_DIR="${A3S_CODE_ADAPTER_DIR:-${SCRIPT_DIR}}"
A3S_CODE_MODEL_VARIANT="${A3S_CODE_MODEL_VARIANT:-qwen3.5-4b}"
TRAIN_BACKEND="${TRAIN_BACKEND:-${A3S_CODE_TRAIN_BACKEND:-megatron}}"
case "${TRAIN_BACKEND}" in
  megatron|fsdp) ;;
  *)
    echo "unsupported TRAIN_BACKEND=${TRAIN_BACKEND}; expected megatron or fsdp" >&2
    exit 1
    ;;
esac
export TRAIN_BACKEND
DEFAULT_ARTIFACT_ROOT="${PROJECT_ROOT}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${DEFAULT_ARTIFACT_ROOT}}"

if [[ ! -d "${SLIME_ROOT}" ]]; then
  echo "missing SLIME_ROOT: ${SLIME_ROOT}"; exit 1
fi

case "${A3S_CODE_MODEL_VARIANT}" in
  qwen3.5-4b)
    MODEL_RUN_TAG="qwen35_4b"
    MODEL_FAMILY="qwen3.5-4b"
    MODEL_SCRIPT="qwen3.5-4B.sh"
    MODEL_DEFAULT_HF_CKPT="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    MODEL_DEFAULT_SAVE_CKPT="${ARTIFACT_ROOT:-${PROJECT_ROOT}}/ckpt/qwen35-4b-code-rl"
    MODEL_DEFAULT_SERVED_NAME="qwen3.5-4b"
    MODEL_DEFAULT_REASONING_PARSER="qwen3"
    MODEL_DEFAULT_TOOL_CALL_PARSER="qwen3_coder"
    MODEL_DEFAULT_SGLANG_LANGUAGE_ONLY="1"
    MODEL_DEFAULT_QWEN35_TEXT_ONLY_BRIDGE="1"
    MODEL_DEFAULT_CKPT_ROTARY_BASE="10000000"
    MODEL_DEFAULT_MEGATRON_TO_HF_MODE="bridge"
    MODEL_DEFAULT_WANDB_GROUP="qwen35-4b-code-rl"
    ;;
  qwen3-4b)
    MODEL_RUN_TAG="qwen3_4b"
    MODEL_FAMILY="qwen3-4b"
    MODEL_SCRIPT="qwen3-4B.sh"
    MODEL_DEFAULT_HF_CKPT="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/eb25fbe4f35f7147763bc24445679d1c00588d89"
    MODEL_DEFAULT_SAVE_CKPT="${ARTIFACT_ROOT:-${PROJECT_ROOT}}/ckpt/qwen3-4b-code-rl"
    MODEL_DEFAULT_SERVED_NAME="qwen3-4b-instruct-2507"
    MODEL_DEFAULT_REASONING_PARSER=""
    MODEL_DEFAULT_TOOL_CALL_PARSER="qwen"
    MODEL_DEFAULT_SGLANG_LANGUAGE_ONLY="0"
    MODEL_DEFAULT_QWEN35_TEXT_ONLY_BRIDGE="0"
    MODEL_DEFAULT_CKPT_ROTARY_BASE="5000000"
    MODEL_DEFAULT_MEGATRON_TO_HF_MODE="bridge"
    MODEL_DEFAULT_WANDB_GROUP="qwen3-4b-code-rl"
    ;;
  qwen3.6-35b-a3b)
    MODEL_RUN_TAG="qwen36_35b_a3b"
    MODEL_FAMILY="qwen3.6-35b-a3b"
    MODEL_SCRIPT="qwen3.6-35B-A3B.sh"
    MODEL_DEFAULT_HF_CKPT="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/7da1103448ba36029c34ce1a9a741dfe93ee0c50"
    MODEL_DEFAULT_SAVE_CKPT="${ARTIFACT_ROOT:-${PROJECT_ROOT}}/ckpt/qwen36-35b-a3b-code-rl"
    MODEL_DEFAULT_SERVED_NAME="qwen3.6-35b-a3b"
    MODEL_DEFAULT_REASONING_PARSER="qwen3"
    MODEL_DEFAULT_TOOL_CALL_PARSER="qwen3_coder"
    MODEL_DEFAULT_SGLANG_LANGUAGE_ONLY="1"
    MODEL_DEFAULT_QWEN35_TEXT_ONLY_BRIDGE="1"
    MODEL_DEFAULT_CKPT_ROTARY_BASE="10000000"
    MODEL_DEFAULT_MEGATRON_TO_HF_MODE="bridge"
    MODEL_DEFAULT_WANDB_GROUP="qwen36-35b-a3b-code-rl"
    ;;
  qwen3.5-122b-a10b-fp8)
    MODEL_RUN_TAG="qwen35_122b_a10b_fp8"
    MODEL_FAMILY="qwen3.5-122b-a10b-fp8"
    MODEL_SCRIPT="qwen3.5-122B-A10B.sh"
    MODEL_DEFAULT_HF_CKPT="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/fb53b9f3bdaab287c597d4e943783153ec527e06"
    MODEL_DEFAULT_SAVE_CKPT="${ARTIFACT_ROOT:-${PROJECT_ROOT}}/ckpt/qwen35-122b-a10b-fp8-code-rl"
    MODEL_DEFAULT_SERVED_NAME="qwen3.5-122b-a10b-fp8"
    MODEL_DEFAULT_REASONING_PARSER="qwen3"
    MODEL_DEFAULT_TOOL_CALL_PARSER="qwen3_coder"
    MODEL_DEFAULT_SGLANG_LANGUAGE_ONLY="1"
    MODEL_DEFAULT_QWEN35_TEXT_ONLY_BRIDGE="1"
    MODEL_DEFAULT_CKPT_ROTARY_BASE="10000000"
    MODEL_DEFAULT_MEGATRON_TO_HF_MODE="bridge"
    MODEL_DEFAULT_WANDB_GROUP="qwen35-122b-a10b-fp8-code-rl"
    ;;
  qwen3.5-122b-a10b)
    MODEL_RUN_TAG="qwen35_122b_a10b"
    MODEL_FAMILY="qwen3.5-122b-a10b"
    MODEL_SCRIPT="qwen3.5-122B-A10B.sh"
    MODEL_DEFAULT_HF_CKPT="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99"
    MODEL_DEFAULT_SAVE_CKPT="${ARTIFACT_ROOT:-${PROJECT_ROOT}}/ckpt/qwen35-122b-a10b-code-rl"
    MODEL_DEFAULT_SERVED_NAME="qwen3.5-122b-a10b"
    MODEL_DEFAULT_REASONING_PARSER="qwen3"
    MODEL_DEFAULT_TOOL_CALL_PARSER="qwen3_coder"
    MODEL_DEFAULT_SGLANG_LANGUAGE_ONLY="1"
    MODEL_DEFAULT_QWEN35_TEXT_ONLY_BRIDGE="1"
    MODEL_DEFAULT_CKPT_ROTARY_BASE="10000000"
    MODEL_DEFAULT_MEGATRON_TO_HF_MODE="bridge"
    MODEL_DEFAULT_WANDB_GROUP="qwen35-122b-a10b-code-rl"
    ;;
  qwen3-next-80b-a3b-instruct)
    MODEL_RUN_TAG="qwen3_next_80b_a3b_instruct"
    MODEL_FAMILY="qwen3-next-80b-a3b-instruct"
    MODEL_SCRIPT="qwen3-next-80B-A3B.sh"
    MODEL_DEFAULT_HF_CKPT="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct/snapshots/609718eef2e2279fd6654e39cec856ec70906535"
    MODEL_DEFAULT_SAVE_CKPT="${ARTIFACT_ROOT:-${PROJECT_ROOT}}/ckpt/qwen3-next-80b-a3b-instruct-code-rl"
    MODEL_DEFAULT_SERVED_NAME="qwen3-next-80b-a3b-instruct"
    MODEL_DEFAULT_REASONING_PARSER=""
    MODEL_DEFAULT_TOOL_CALL_PARSER="qwen"
    MODEL_DEFAULT_SGLANG_LANGUAGE_ONLY="0"
    MODEL_DEFAULT_QWEN35_TEXT_ONLY_BRIDGE="0"
    MODEL_DEFAULT_CKPT_ROTARY_BASE="10000000"
    MODEL_DEFAULT_MEGATRON_TO_HF_MODE="bridge"
    MODEL_DEFAULT_WANDB_GROUP="qwen3-next-80b-a3b-instruct-code-rl"
    ;;
  glm4.7-flash)
    MODEL_RUN_TAG="glm47_flash"
    MODEL_FAMILY="glm4.7-flash"
    MODEL_SCRIPT="glm4.7-30B-A3B.sh"
    MODEL_DEFAULT_HF_CKPT="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
    MODEL_DEFAULT_SAVE_CKPT="${ARTIFACT_ROOT:-${PROJECT_ROOT}}/ckpt/glm47-flash-code-rl"
    MODEL_DEFAULT_SERVED_NAME="glm4.7-flash-code-rl"
    MODEL_DEFAULT_REASONING_PARSER="glm45"
    MODEL_DEFAULT_TOOL_CALL_PARSER="glm45"
    MODEL_DEFAULT_SGLANG_LANGUAGE_ONLY="0"
    MODEL_DEFAULT_QWEN35_TEXT_ONLY_BRIDGE="0"
    MODEL_DEFAULT_CKPT_ROTARY_BASE="1000000"
    MODEL_DEFAULT_MEGATRON_TO_HF_MODE="raw"
    MODEL_DEFAULT_WANDB_GROUP="glm47-flash-code-rl"
    ;;
  *)
    echo "unsupported A3S_CODE_MODEL_VARIANT=${A3S_CODE_MODEL_VARIANT}" >&2
    exit 1
    ;;
esac

# ── Artifact root ────────────────────────────────────────────────
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}}"
RUN_ID="${RUN_ID:-a3s_code_rl_${MODEL_RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACT_ROOT}/runs/${RUN_ID}}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}"
export A3S_CODE_RUN_ID="${RUN_ID}"
export A3S_CODE_RUN_ROOT="${RUN_ROOT}"
export A3S_CODE_WORKSPACE_ROOT="${A3S_CODE_WORKSPACE_ROOT:-${RUN_ROOT}/a3s_workspaces}"
export A3S_CODE_CONFIG_ROOT="${A3S_CODE_CONFIG_ROOT:-${RUN_ROOT}/a3s_configs}"
export A3S_CODE_WORKSPACE_TEMPLATE_CACHE_ROOT="${A3S_CODE_WORKSPACE_TEMPLATE_CACHE_ROOT:-${RUN_ROOT}/a3s_workspace_template_cache}"
export A3S_CODE_RESULTS_DIR="${A3S_CODE_RESULTS_DIR:-${RUN_ROOT}/a3s_results}"
export A3S_CODE_TRAFFIC_RECORD_FILE="${A3S_CODE_TRAFFIC_RECORD_FILE:-${A3S_CODE_RESULTS_DIR}/a3s_code_agent_traffic.jsonl}"
mkdir -p "${A3S_CODE_RESULTS_DIR}"

LOCAL_CACHE_ROOT="${A3S_CODE_LOCAL_CACHE_ROOT:-/tmp/ocrl-${MODEL_RUN_TAG}-$$}"
mkdir -p \
  "${LOCAL_CACHE_ROOT}/xdg" \
  "${LOCAL_CACHE_ROOT}/flashinfer" \
  "${LOCAL_CACHE_ROOT}/flashinfer/cubins" \
  "${LOCAL_CACHE_ROOT}/triton" \
  "${LOCAL_CACHE_ROOT}/torchinductor" \
  "${LOCAL_CACHE_ROOT}/torch_extensions" \
  "${LOCAL_CACHE_ROOT}/tvm-ffi" \
  "${LOCAL_CACHE_ROOT}/hf" \
  "${LOCAL_CACHE_ROOT}/tmp"
RAY_TMPDIR="${A3S_CODE_RAY_TMPDIR:-/tmp/ray-${MODEL_RUN_TAG}-$$}"
mkdir -p "${RAY_TMPDIR}"
export A3S_CODE_LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT}"
export A3S_CODE_RAY_TMPDIR="${RAY_TMPDIR}"
export RAY_TMPDIR="${RAY_TMPDIR}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LOCAL_CACHE_ROOT}/xdg}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${LOCAL_CACHE_ROOT}/flashinfer}"
export FLASHINFER_CUBIN_DIR="${FLASHINFER_CUBIN_DIR:-${LOCAL_CACHE_ROOT}/flashinfer/cubins}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${LOCAL_CACHE_ROOT}/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_CACHE_ROOT}/torch_extensions}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-${LOCAL_CACHE_ROOT}/tvm-ffi}"
export HF_HOME="${HF_HOME:-${LOCAL_CACHE_ROOT}/hf}"
export TMPDIR="${A3S_CODE_TMPDIR:-${LOCAL_CACHE_ROOT}/tmp}"
mkdir -p "${TMPDIR}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF
fi

CUDA_SHIM_ROOT="${RUN_ROOT}/cuda-home"
mkdir -p "${CUDA_SHIM_ROOT}/targets/x86_64-linux"
mkdir -p "${CUDA_SHIM_ROOT}/bin"
if [[ -x "${CONDA_ENV}/bin/nvcc" ]]; then
  ln -sfn "${CONDA_ENV}/bin/nvcc" "${CUDA_SHIM_ROOT}/bin/nvcc"
fi
if [[ -d "${CUDA_TARGET_DIR}/include" ]]; then
  ln -sfn "${CONDA_ENV}/include" "${CUDA_SHIM_ROOT}/include"
  ln -sfn "${CONDA_ENV}/include" "${CUDA_SHIM_ROOT}/targets/x86_64-linux/include"
fi
if [[ -d "${CUDA_TARGET_DIR}/lib" ]]; then
  ln -sfn "${CUDA_TARGET_DIR}/lib" "${CUDA_SHIM_ROOT}/lib64"
  ln -sfn "${CUDA_TARGET_DIR}/lib" "${CUDA_SHIM_ROOT}/targets/x86_64-linux/lib"
else
  ln -sfn "${CUDA_RUNTIME_LIB_DIR}" "${CUDA_SHIM_ROOT}/lib64"
  ln -sfn "${CUDA_RUNTIME_LIB_DIR}" "${CUDA_SHIM_ROOT}/targets/x86_64-linux/lib"
fi
if [[ -d "${CONDA_ENV}/nvvm" ]]; then
  ln -sfn "${CONDA_ENV}/nvvm" "${CUDA_SHIM_ROOT}/nvvm"
elif [[ -d "${CUDA_NVCC_SITE_DIR}/nvvm" ]]; then
  ln -sfn "${CUDA_NVCC_SITE_DIR}/nvvm" "${CUDA_SHIM_ROOT}/nvvm"
fi
export CUDA_HOME="${CUDA_SHIM_ROOT}"
export CUDA_PATH="${CUDA_SHIM_ROOT}"
export CUDA_LIB_PATH="${CUDA_HOME}/targets/x86_64-linux/lib"
export CUDNN_HOME="${PYTHON_SITE_PACKAGES}/nvidia/cudnn"
export CUDNN_PATH="${CUDNN_HOME}"
export PATH="${CUDA_HOME}/bin:${CUDA_HOME}/nvvm/bin:${CONDA_ENV}/nvvm/bin:${PATH}"
if [[ -d "${CUDA_HOME}/nvvm/bin" ]]; then
  export CICC_PATH="${CUDA_HOME}/nvvm/bin"
elif [[ -d "${CONDA_ENV}/nvvm/bin" ]]; then
  export CICC_PATH="${CONDA_ENV}/nvvm/bin"
fi
CUDA_INCLUDE_PARTS=()
for include_dir in \
  "${CUDA_HOME}/include" \
  "${CUDA_HOME}/targets/x86_64-linux/include" \
  "${CONDA_ENV}/include" \
  "${CUDA_TARGET_DIR}/include" \
  "${CUDA_RUNTIME_LIB_DIR%/lib}/include" \
  "${CUDA_NVRTC_INCLUDE_DIR}" \
  "${CUBLAS_INCLUDE_DIR}" \
  "${CUDNN_INCLUDE_DIR}"; do
  if [[ -d "$include_dir" ]]; then
    CUDA_INCLUDE_PARTS+=("$include_dir")
  fi
done
if (( ${#CUDA_INCLUDE_PARTS[@]} > 0 )); then
  CUDA_INCLUDE_PREFIX="$(IFS=:; echo "${CUDA_INCLUDE_PARTS[*]}")"
  export CPATH="${CUDA_INCLUDE_PREFIX}${CPATH:+:${CPATH}}"
  export C_INCLUDE_PATH="${CUDA_INCLUDE_PREFIX}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
  export CPLUS_INCLUDE_PATH="${CUDA_INCLUDE_PREFIX}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
fi
CUDA_LIBRARY_PARTS=()
for lib_dir in "${CUDA_HOME}/lib64" "${CUDA_HOME}/targets/x86_64-linux/lib" "${CUDA_TARGET_DIR}/lib" "${CUDA_RUNTIME_LIB_DIR}" "${CONDA_ENV}/lib" "${CUBLAS_LIB_DIR}" "${CUDNN_LIB_DIR}" "${CUDA_NVRTC_LIB_DIR}" "${HOST_CUDA_DRIVER_LIB_DIR}"; do
  if [[ -d "$lib_dir" ]]; then
    CUDA_LIBRARY_PARTS+=("$lib_dir")
  fi
done
if (( ${#CUDA_LIBRARY_PARTS[@]} > 0 )); then
  CUDA_LIBRARY_PREFIX="$(IFS=:; echo "${CUDA_LIBRARY_PARTS[*]}")"
  export LIBRARY_PATH="${CUDA_LIBRARY_PREFIX}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi
if [[ ! -e "${CUDA_LIB_PATH}/libcudart.so.12" && -e "${CUDA_HOME}/lib64/libcudart.so.12" ]]; then
  export CUDA_LIB_PATH="${CUDA_HOME}/lib64"
fi

# ── Kill stale processes ─────────────────────────────────────────
if [[ "${A3S_CODE_EXTERNAL_RAY:-0}" == "1" ]]; then
  echo "Skipping ray stop because A3S_CODE_EXTERNAL_RAY=1"
else
  ray stop --force 2>/dev/null || true
fi
if [[ "${A3S_CODE_SKIP_STALE_SGLANG_CLEANUP:-0}" == "1" ]]; then
  echo "Skipping broad SGLang cleanup because A3S_CODE_SKIP_STALE_SGLANG_CLEANUP=1"
else
  pkill -f "sglang" 2>/dev/null || true
fi
pkill -f "code_rl_api_server" 2>/dev/null || true
sleep 2

# ── GPU allocation ───────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-4}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${NUM_GPUS}}"
ACTOR_GPUS="${ACTOR_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
PRM_GPUS="${PRM_GPUS:-1}"
ENABLE_PRM="${ENABLE_PRM:-1}"
PRM_BACKEND="${PRM_BACKEND:-external_openai}"  # external_openai | local_sglang | disabled
TP_TRAIN="${TP_TRAIN:-2}"   # tensor-parallel for Megatron actor
PP_TRAIN="${PP_TRAIN:-1}"   # pipeline-parallel for Megatron actor
CP_TRAIN="${CP_TRAIN:-1}"   # context-parallel for long-context Megatron actor
TP_SGLANG="${TP_SGLANG:-2}" # tensor-parallel for SGLang rollout / PRM
COLOCATE="${COLOCATE:-0}"

EFFECTIVE_PRM_GPUS=0
EFFECTIVE_PRM_BACKEND="disabled"
if [[ "${ENABLE_PRM}" == "1" && "${PRM_BACKEND}" == "local_sglang" ]]; then
  EFFECTIVE_PRM_GPUS="${PRM_GPUS}"
  EFFECTIVE_PRM_BACKEND="local_sglang"
elif [[ "${ENABLE_PRM}" == "1" && "${PRM_BACKEND}" == "external_openai" ]]; then
  EFFECTIVE_PRM_BACKEND="external_openai"
fi

ACTOR_TOTAL_GPUS=$(( ACTOR_NUM_NODES * ACTOR_GPUS ))
TOTAL_NUM_NODES="${TOTAL_NUM_NODES:-${NUM_NODES:-${ACTOR_NUM_NODES}}}"
TOTAL_GPUS="${TOTAL_GPUS:-$(( TOTAL_NUM_NODES * NUM_GPUS_PER_NODE ))}"

if [[ "${COLOCATE}" == "1" ]]; then
  if (( ACTOR_GPUS > NUM_GPUS_PER_NODE || ROLLOUT_GPUS > TOTAL_GPUS || EFFECTIVE_PRM_GPUS > TOTAL_GPUS )); then
    echo "Under COLOCATE=1, ACTOR_GPUS must be <= NUM_GPUS_PER_NODE, and total ROLLOUT_GPUS/PRM_GPUS must fit TOTAL_GPUS"
    echo "TOTAL_NUM_NODES=${TOTAL_NUM_NODES}, ACTOR_NUM_NODES=${ACTOR_NUM_NODES}, ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${EFFECTIVE_PRM_GPUS}, NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}, TOTAL_GPUS=${TOTAL_GPUS}"
    exit 1
  fi
else
  if (( ACTOR_TOTAL_GPUS + ROLLOUT_GPUS + EFFECTIVE_PRM_GPUS > TOTAL_GPUS )); then
    echo "ACTOR_TOTAL_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= TOTAL_GPUS"
    echo "TOTAL_NUM_NODES=${TOTAL_NUM_NODES}, ACTOR_NUM_NODES=${ACTOR_NUM_NODES}, ACTOR_GPUS=${ACTOR_GPUS}, ACTOR_TOTAL_GPUS=${ACTOR_TOTAL_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${EFFECTIVE_PRM_GPUS}, TOTAL_GPUS=${TOTAL_GPUS}"
    exit 1
  fi
fi

if (( TP_TRAIN > ACTOR_GPUS || ACTOR_TOTAL_GPUS % TP_TRAIN != 0 )); then
  echo "TP_TRAIN must fit one node and divide ACTOR_TOTAL_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, ACTOR_TOTAL_GPUS=${ACTOR_TOTAL_GPUS}, TP_TRAIN=${TP_TRAIN}"
  exit 1
fi

if (( CP_TRAIN < 1 || ACTOR_TOTAL_GPUS % (TP_TRAIN * PP_TRAIN * CP_TRAIN) != 0 )); then
  echo "TP_TRAIN * PP_TRAIN * CP_TRAIN must divide ACTOR_TOTAL_GPUS"
  echo "ACTOR_TOTAL_GPUS=${ACTOR_TOTAL_GPUS}, TP_TRAIN=${TP_TRAIN}, PP_TRAIN=${PP_TRAIN}, CP_TRAIN=${CP_TRAIN}"
  exit 1
fi
EXPERT_PARALLEL_PRODUCT=$(( ${ETP_TRAIN:-1} * ${EP_TRAIN:-1} * PP_TRAIN ))
if (( EXPERT_PARALLEL_PRODUCT < 1 || ACTOR_TOTAL_GPUS % EXPERT_PARALLEL_PRODUCT != 0 )); then
  echo "ETP_TRAIN * EP_TRAIN * PP_TRAIN must divide ACTOR_TOTAL_GPUS"
  echo "ACTOR_TOTAL_GPUS=${ACTOR_TOTAL_GPUS}, ETP_TRAIN=${ETP_TRAIN:-1}, EP_TRAIN=${EP_TRAIN:-1}, PP_TRAIN=${PP_TRAIN}"
  exit 1
fi

if (( TP_SGLANG > ROLLOUT_GPUS || ROLLOUT_GPUS % TP_SGLANG != 0 )); then
  echo "TP_SGLANG must divide ROLLOUT_GPUS"
  echo "ROLLOUT_GPUS=${ROLLOUT_GPUS}, TP_SGLANG=${TP_SGLANG}"
  exit 1
fi

# ── Model paths ──────────────────────────────────────────────────
HF_CKPT="${HF_CKPT:-${MODEL_DEFAULT_HF_CKPT}}"
REF_LOAD="${REF_LOAD:-}"
SAVE_CKPT="${SAVE_CKPT:-${MODEL_DEFAULT_SAVE_CKPT}}"
MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-${MODEL_DEFAULT_MEGATRON_TO_HF_MODE}}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-${HF_CKPT}}"
KIMI_BASE_URL="${KIMI_BASE_URL:-http://s-20260204175507-cqflp.ailab-pj.pjh-service.org.cn}"
mkdir -p "${SAVE_CKPT}"

# ── Model arch args ──────────────────────────────────────────────
if [[ "${TRAIN_BACKEND}" == "megatron" ]]; then
  source "${SLIME_ROOT}/scripts/models/${MODEL_SCRIPT}"
  MODEL_ARGS+=(--train-backend megatron)
else
  MODEL_ARGS=(--train-backend fsdp)
fi

# ── Serving / API config ─────────────────────────────────────────
export SGLANG_API_KEY="${SGLANG_API_KEY:-apiKey}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_DEFAULT_SERVED_NAME}}"
export HOST="0.0.0.0"
export PORT="${PORT:-30000}"

# ── Recording config ─────────────────────────────────────────────
export CODE_RL_RECORD_ENABLED="${CODE_RL_RECORD_ENABLED:-1}"
export CODE_RL_RECORD_FILE="${CODE_RL_RECORD_FILE:-${RUN_ROOT}/code_rl_record.jsonl}"
export CODE_RL_PRM_RECORD_FILE="${CODE_RL_PRM_RECORD_FILE:-${RUN_ROOT}/code_rl_prm_record.jsonl}"
export CODE_RL_FEEDBACK_RECORD_FILE="${CODE_RL_FEEDBACK_RECORD_FILE:-${RUN_ROOT}/code_rl_feedback_record.jsonl}"
export CODE_RL_TRACE_RECORD_FILE="${CODE_RL_TRACE_RECORD_FILE:-${RUN_ROOT}/code_rl_trace.jsonl}"
export CODE_RL_PURGE_RECORD_FILES_ON_PAUSE="${CODE_RL_PURGE_RECORD_FILES_ON_PAUSE:-0}"
export CODE_RL_SUBMIT_SIDE="${CODE_RL_SUBMIT_SIDE:-0}"
export CODE_RL_TRAIN_SIDE="${CODE_RL_TRAIN_SIDE:-0}"
export CODE_RL_REWARD_MODE="${CODE_RL_REWARD_MODE:-prm}"
export CODE_RL_REQUIRE_VERIFIER_FEEDBACK="${CODE_RL_REQUIRE_VERIFIER_FEEDBACK:-0}"
export CODE_RL_SESSION_IDLE_FLUSH_SEC="${CODE_RL_SESSION_IDLE_FLUSH_SEC:-30}"
export CODE_RL_MATCHED_CONTEXT_TOKENS="${CODE_RL_MATCHED_CONTEXT_TOKENS:-16384}"
export CODE_RL_MAX_TRAIN_TOKENS="${CODE_RL_MAX_TRAIN_TOKENS:-${CODE_RL_MATCHED_CONTEXT_TOKENS}}"
export CODE_RL_PAUSE_WAIT_TIMEOUT_SEC="${CODE_RL_PAUSE_WAIT_TIMEOUT_SEC:-1800}"
export CODE_RL_PAUSE_DRAIN_TIMEOUT_SEC="${CODE_RL_PAUSE_DRAIN_TIMEOUT_SEC:-1800}"
export CODE_RL_ROLLOUT_IDLE_TIMEOUT_SEC="${CODE_RL_ROLLOUT_IDLE_TIMEOUT_SEC:-3600}"
export CODE_RL_ROLLOUT_DRAIN_TIMEOUT_SEC="${CODE_RL_ROLLOUT_DRAIN_TIMEOUT_SEC:-0}"
export CODE_RL_PRM_TIMEOUT_SEC="${CODE_RL_PRM_TIMEOUT_SEC:-180}"
export CODE_RL_CONTEXT_SAFETY_MARGIN="${CODE_RL_CONTEXT_SAFETY_MARGIN:-512}"

export CONTEXT_LENGTH="${CONTEXT_LENGTH:-${CODE_RL_MATCHED_CONTEXT_TOKENS}}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
export REASONING_PARSER="${REASONING_PARSER:-${MODEL_DEFAULT_REASONING_PARSER}}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-${MODEL_DEFAULT_TOOL_CALL_PARSER}}"
export SGLANG_LANGUAGE_ONLY="${SGLANG_LANGUAGE_ONLY:-${MODEL_DEFAULT_SGLANG_LANGUAGE_ONLY}}"
export SLIME_QWEN35_TEXT_ONLY_BRIDGE="${SLIME_QWEN35_TEXT_ONLY_BRIDGE:-${MODEL_DEFAULT_QWEN35_TEXT_ONLY_BRIDGE}}"
export A3S_CODE_ROUTER_DISABLE_CIRCUIT_BREAKER="${A3S_CODE_ROUTER_DISABLE_CIRCUIT_BREAKER:-1}"
export A3S_CODE_ROUTER_DISABLE_HEALTH_CHECK="${A3S_CODE_ROUTER_DISABLE_HEALTH_CHECK:-1}"
export A3S_CODE_ROUTER_HEALTH_FAILURE_THRESHOLD="${A3S_CODE_ROUTER_HEALTH_FAILURE_THRESHOLD:-10}"
export A3S_CODE_ROUTER_HEALTH_SUCCESS_THRESHOLD="${A3S_CODE_ROUTER_HEALTH_SUCCESS_THRESHOLD:-1}"
export A3S_CODE_ROUTER_HEALTH_CHECK_TIMEOUT_SECS="${A3S_CODE_ROUTER_HEALTH_CHECK_TIMEOUT_SECS:-30}"
export A3S_CODE_ROUTER_HEALTH_CHECK_INTERVAL_SECS="${A3S_CODE_ROUTER_HEALTH_CHECK_INTERVAL_SECS:-120}"
export A3S_CODE_ROUTER_CB_FAILURE_THRESHOLD="${A3S_CODE_ROUTER_CB_FAILURE_THRESHOLD:-1000000}"
export A3S_CODE_ROUTER_CB_SUCCESS_THRESHOLD="${A3S_CODE_ROUTER_CB_SUCCESS_THRESHOLD:-1}"
export A3S_CODE_ROUTER_CB_TIMEOUT_DURATION_SECS="${A3S_CODE_ROUTER_CB_TIMEOUT_DURATION_SECS:-10}"
export A3S_CODE_ROUTER_BALANCE_ABS_THRESHOLD="${A3S_CODE_ROUTER_BALANCE_ABS_THRESHOLD:-0}"
export PRM_M="${PRM_M:-3}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-${CONTEXT_LENGTH}}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${CODE_RL_MAX_TRAIN_TOKENS}}"
export TRAIN_SEQ_LENGTH="${TRAIN_SEQ_LENGTH:-${CONTEXT_LENGTH}}"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-false}"
export SLIME_HOST_IP="${SLIME_HOST_IP:-}"
export CODE_RL_MAX_RESPONSE_TOKENS="${CODE_RL_MAX_RESPONSE_TOKENS:-4096}"
export CODE_RL_DROP_REPETITIVE_SAMPLES="${CODE_RL_DROP_REPETITIVE_SAMPLES:-0}"
export SLIME_PPO_RATIO_SAFE_BOUND="${SLIME_PPO_RATIO_SAFE_BOUND:-20.0}"
export CLIP_GRAD="${CLIP_GRAD:-0.5}"
export POLICY_LR="${POLICY_LR:-5e-6}"
export POLICY_WEIGHT_DECAY="${POLICY_WEIGHT_DECAY:-0.01}"
export POLICY_ADAM_EPS="${POLICY_ADAM_EPS:-1e-6}"
export POLICY_KL_LOSS_COEF="${POLICY_KL_LOSS_COEF:-0.01}"
export POLICY_USE_KL_LOSS="${POLICY_USE_KL_LOSS:-1}"
export ENABLE_REF_MODEL="${ENABLE_REF_MODEL:-${POLICY_USE_KL_LOSS}}"
export POLICY_EPS_CLIP_C="${POLICY_EPS_CLIP_C:-3.0}"
export POLICY_NORMALIZE_ADVANTAGES="${POLICY_NORMALIZE_ADVANTAGES:-1}"
export POLICY_USE_ROLLOUT_LOGPROBS="${POLICY_USE_ROLLOUT_LOGPROBS:-1}"
export DISABLE_BF16_REDUCED_PRECISION_MATMUL="${DISABLE_BF16_REDUCED_PRECISION_MATMUL:-1}"

if [[ "${ENABLE_PRM}" == "1" && "${PRM_BACKEND}" == "external_openai" ]]; then
  export CODE_RL_PRM_OPENAI_URL="${CODE_RL_PRM_OPENAI_URL:-${KIMI_BASE_URL}/v1/chat/completions}"
  export CODE_RL_PRM_OPENAI_MODEL_NAME="${CODE_RL_PRM_OPENAI_MODEL_NAME:-kimi-k2.5}"
  export CODE_RL_PRM_API_KEY="${CODE_RL_PRM_API_KEY:-}"
  export CODE_RL_PRM_HEALTH_URL="${CODE_RL_PRM_HEALTH_URL:-${KIMI_BASE_URL}/v1/models}"
else
  unset CODE_RL_PRM_OPENAI_URL || true
  unset CODE_RL_PRM_OPENAI_MODEL_NAME || true
  unset CODE_RL_PRM_API_KEY || true
  unset CODE_RL_PRM_HEALTH_URL || true
fi

# ── Ray ──────────────────────────────────────────────────────────
export RAY_health_check_failure_threshold="${RAY_health_check_failure_threshold:-20}"
export RAY_health_check_period_ms="${RAY_health_check_period_ms:-5000}"
export RAY_health_check_timeout_ms="${RAY_health_check_timeout_ms:-30000}"
export RAY_num_heartbeats_timeout="${RAY_num_heartbeats_timeout:-60}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.995}"
if [[ -n "${RAY_memory_monitor_refresh_ms:-}" ]]; then
  export RAY_memory_monitor_refresh_ms
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
NO_PROXY_BASE="127.0.0.1,localhost,0.0.0.0,::1,${MASTER_ADDR},10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,100.96.0.0/12,.pjlab.org.cn,.svc,.svc.cluster.local"
for local_ip in $(hostname -I 2>/dev/null || true); do
  if [[ -n "${local_ip}" ]]; then
    NO_PROXY_BASE="${NO_PROXY_BASE},${local_ip}"
  fi
done
if [[ -n "${MY_NO_PROXY:-}" ]]; then
  NO_PROXY_BASE="${NO_PROXY_BASE},${MY_NO_PROXY}"
fi
if [[ -n "${A3S_CODE_NO_PROXY_EXTRA:-}" ]]; then
  NO_PROXY_BASE="${NO_PROXY_BASE},${A3S_CODE_NO_PROXY_EXTRA}"
fi
export no_proxy="${NO_PROXY_BASE}"
export NO_PROXY="${NO_PROXY_BASE}"

if [[ "${A3S_CODE_EXTERNAL_RAY:-0}" == "1" ]]; then
  echo "Using existing Ray cluster: ${RAY_ADDRESS:-auto}"
else
  ray start \
    --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --temp-dir "${RAY_TMPDIR}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265
fi

# ── Checkpoint args ──────────────────────────────────────────────
CKPT_ARGS=(
  --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
  --hf-checkpoint "${HF_CKPT}"
  --save "${SAVE_CKPT}"
  --save-interval "${SAVE_INTERVAL:-100}"
)
if [[ "${TRAIN_BACKEND}" == "megatron" ]]; then
  CKPT_ARGS+=(--rotary-base "${MODEL_DEFAULT_CKPT_ROTARY_BASE}")
fi
if [[ "${SAVE_OPTIMIZER:-0}" != "1" ]]; then
  CKPT_ARGS+=(--no-save-optim)
fi
if [[ -n "${REF_LOAD}" ]]; then
  CKPT_ARGS+=(--ref-load "${REF_LOAD}")
fi
if [[ -n "${LOAD_CKPT:-}" ]]; then
  CKPT_ARGS+=(--load "${LOAD_CKPT}")
fi
if [[ -n "${START_ROLLOUT_ID:-}" ]]; then
  CKPT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi
if [[ "${NO_LOAD_OPTIM:-0}" == "1" ]]; then
  if [[ "${TRAIN_BACKEND}" == "megatron" ]]; then
    CKPT_ARGS+=(--no-load-optim)
  else
    echo "Ignoring NO_LOAD_OPTIM=1 for TRAIN_BACKEND=fsdp; FSDP auto-detects missing optimizer state."
  fi
fi
if [[ "${NO_LOAD_RNG:-0}" == "1" ]]; then
  if [[ "${TRAIN_BACKEND}" == "megatron" ]]; then
    CKPT_ARGS+=(--no-load-rng)
  else
    echo "Ignoring NO_LOAD_RNG=1 for TRAIN_BACKEND=fsdp; FSDP does not expose --no-load-rng."
  fi
fi

# ── Rollout ──────────────────────────────────────────────────────
ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --rollout-function-path code_rl_rollout.generate_rollout_code_rl
  --num-rollout "${NUM_ROLLOUT:-100000000}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-1}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.6}"
  --reward-key score
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-1}"
)

# ── Performance ─────────────────────────────────────────────────
if [[ "${TRAIN_BACKEND}" == "megatron" ]]; then
  PERF_ARGS=(
    --tensor-model-parallel-size "${TP_TRAIN}"
    --pipeline-model-parallel-size "${PP_TRAIN}"
    --context-parallel-size "${CP_TRAIN}"
    --expert-model-parallel-size "${EP_TRAIN:-1}"
    --expert-tensor-parallel-size "${ETP_TRAIN:-1}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
    --seq-length "${TRAIN_SEQ_LENGTH}"
    --qkv-format "${QKV_FORMAT:-bshd}"
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --log-probs-chunk-size 1024
  )
  if [[ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS:-}" ]]; then
    PERF_ARGS+=(--decoder-last-pipeline-num-layers "${DECODER_LAST_PIPELINE_NUM_LAYERS}")
  fi
  if [[ "${ENABLE_DYNAMIC_GLOBAL_BATCH_SIZE:-0}" == "1" ]]; then
    PERF_ARGS+=(--use-dynamic-global-batch-size)
  fi
  if [[ "${ENABLE_SEQUENCE_PARALLEL:-0}" == "1" ]]; then
    PERF_ARGS+=(--sequence-parallel)
  fi
  if [[ "${RL_USE_SEQUENCE_PACKING:-0}" == "1" ]]; then
    PERF_ARGS+=(--rl-use-sequence-packing)
    if [[ -n "${RL_SEQUENCE_PACKING_BIN_SIZE:-}" ]]; then
      PERF_ARGS+=(--rl-sequence-packing-bin-size "${RL_SEQUENCE_PACKING_BIN_SIZE}")
    fi
    if [[ -n "${RL_SEQUENCE_PACKING_ALGO:-}" ]]; then
      PERF_ARGS+=(--rl-sequence-packing-algo "${RL_SEQUENCE_PACKING_ALGO}")
    fi
  fi
  if [[ -n "${DISTRIBUTED_TIMEOUT_SECONDS_AFTER_INIT:-}" ]]; then
    PERF_ARGS+=(--distributed-timeout-seconds-after-init "${DISTRIBUTED_TIMEOUT_SECONDS_AFTER_INIT}")
  fi
else
  PERF_ARGS=(
    --attn-implementation "${FSDP_ATTN_IMPLEMENTATION:-flash_attention_2}"
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  )
  if [[ "${FSDP_GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
    PERF_ARGS+=(--gradient-checkpointing)
  fi
  if [[ "${FSDP_CPU_OFFLOAD:-0}" == "1" ]]; then
    PERF_ARGS+=(--fsdp-cpu-offload)
  fi
fi
if [[ "${ENABLE_DYNAMIC_BATCH_SIZE:-0}" == "1" ]]; then
  PERF_ARGS+=(--use-dynamic-batch-size)
fi
if [[ -n "${DISTRIBUTED_TIMEOUT_MINUTES:-}" ]]; then
  PERF_ARGS+=(--distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}")
fi

# ── GRPO ─────────────────────────────────────────────────────────
GRPO_ARGS=(
  --advantage-estimator grpo
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --eps-clip-c "${POLICY_EPS_CLIP_C}"
)
if [[ "${POLICY_USE_KL_LOSS}" == "1" && "${POLICY_KL_LOSS_COEF}" != "0" && "${POLICY_KL_LOSS_COEF}" != "0.0" ]]; then
  GRPO_ARGS+=(
    --use-kl-loss
    --kl-loss-coef "${POLICY_KL_LOSS_COEF}"
    --kl-loss-type low_var_kl
  )
fi

if [[ "${POLICY_NORMALIZE_ADVANTAGES}" == "1" ]]; then
  GRPO_ARGS+=(--normalize-advantages)
fi

if [[ "${POLICY_USE_ROLLOUT_LOGPROBS}" == "1" ]]; then
  GRPO_ARGS+=(--use-rollout-logprobs)
fi

# ── Optimizer ────────────────────────────────────────────────────
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${POLICY_LR}"
  --lr-decay-style constant
  --weight-decay "${POLICY_WEIGHT_DECAY}"
  --adam-beta1 0.9
  --adam-beta2 0.98
  --adam-eps "${POLICY_ADAM_EPS}"
  --clip-grad "${CLIP_GRAD}"
)
if [[ "${ENABLE_OPTIMIZER_CPU_OFFLOAD:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--optimizer-cpu-offload)
fi
if [[ "${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--overlap-cpu-optimizer-d2h-h2d)
fi
if [[ "${USE_PRECISION_AWARE_OPTIMIZER:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--use-precision-aware-optimizer)
fi
if [[ -n "${MAIN_GRADS_DTYPE:-}" ]]; then
  OPTIMIZER_ARGS+=(--main-grads-dtype "${MAIN_GRADS_DTYPE}")
fi
if [[ "${USE_DISTRIBUTED_OPTIMIZER:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(--use-distributed-optimizer)
fi

# ── SGLang ───────────────────────────────────────────────────────
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${TP_SGLANG}"
  --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
  --sglang-mem-fraction-static "${MEM_FRACTION_STATIC}"
  --sglang-context-length "${CONTEXT_LENGTH}"
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND:-triton}"
  --sglang-sampling-backend "${SGLANG_SAMPLING_BACKEND:-pytorch}"
  --router-cb-failure-threshold "${A3S_CODE_ROUTER_CB_FAILURE_THRESHOLD}"
  --router-cb-success-threshold "${A3S_CODE_ROUTER_CB_SUCCESS_THRESHOLD}"
  --router-cb-timeout-duration-secs "${A3S_CODE_ROUTER_CB_TIMEOUT_DURATION_SECS}"
  --router-health-failure-threshold "${A3S_CODE_ROUTER_HEALTH_FAILURE_THRESHOLD}"
  --router-health-success-threshold "${A3S_CODE_ROUTER_HEALTH_SUCCESS_THRESHOLD}"
  --router-health-check-timeout-secs "${A3S_CODE_ROUTER_HEALTH_CHECK_TIMEOUT_SECS}"
  --router-health-check-interval-secs "${A3S_CODE_ROUTER_HEALTH_CHECK_INTERVAL_SECS}"
  --router-balance-abs-threshold "${A3S_CODE_ROUTER_BALANCE_ABS_THRESHOLD}"
)

if [[ -n "${REASONING_PARSER}" ]]; then
  SGLANG_ARGS+=(--sglang-reasoning-parser "${REASONING_PARSER}")
fi

if [[ -n "${SGLANG_QUANTIZATION:-}" ]]; then
  SGLANG_ARGS+=(--sglang-quantization "${SGLANG_QUANTIZATION}")
fi

if [[ -n "${SGLANG_MOE_RUNNER_BACKEND:-}" ]]; then
  SGLANG_ARGS+=(--sglang-moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND}")
fi

if [[ "${SGLANG_DISABLE_FLASHINFER_AUTOTUNE:-0}" == "1" || "${SGLANG_DISABLE_FLASHINFER_AUTOTUNE:-0}" == "true" ]]; then
  SGLANG_ARGS+=(--sglang-disable-flashinfer-autotune)
fi

if [[ -n "${SGLANG_WATCHDOG_TIMEOUT:-}" ]]; then
  SGLANG_ARGS+=(--sglang-watchdog-timeout "${SGLANG_WATCHDOG_TIMEOUT}")
fi

if [[ -n "${SGLANG_SOFT_WATCHDOG_TIMEOUT:-}" ]]; then
  SGLANG_ARGS+=(--sglang-soft-watchdog-timeout "${SGLANG_SOFT_WATCHDOG_TIMEOUT}")
fi

if [[ -n "${SGLANG_LINEAR_ATTN_BACKEND:-}" ]]; then
  SGLANG_ARGS+=(--sglang-linear-attn-backend "${SGLANG_LINEAR_ATTN_BACKEND}")
fi

if [[ -n "${SGLANG_LINEAR_ATTN_PREFILL_BACKEND:-}" ]]; then
  SGLANG_ARGS+=(--sglang-linear-attn-prefill-backend "${SGLANG_LINEAR_ATTN_PREFILL_BACKEND}")
fi

if [[ -n "${SGLANG_LINEAR_ATTN_DECODE_BACKEND:-}" ]]; then
  SGLANG_ARGS+=(--sglang-linear-attn-decode-backend "${SGLANG_LINEAR_ATTN_DECODE_BACKEND}")
fi

if [[ -n "${SGLANG_MAMBA_BACKEND:-}" ]]; then
  SGLANG_ARGS+=(--sglang-mamba-backend "${SGLANG_MAMBA_BACKEND}")
fi

if [[ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]]; then
  SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi

if [[ -n "${SGLANG_MAX_TOTAL_TOKENS:-}" ]]; then
  SGLANG_ARGS+=(--sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}")
fi

if [[ -n "${SGLANG_MAX_PREFILL_TOKENS:-}" ]]; then
  SGLANG_ARGS+=(--sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}")
fi

if [[ -n "${SGLANG_CHUNKED_PREFILL_SIZE:-}" ]]; then
  SGLANG_ARGS+=(--sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}")
fi

if [[ -n "${SGLANG_MAX_MAMBA_CACHE_SIZE:-}" ]]; then
  SGLANG_ARGS+=(--sglang-max-mamba-cache-size "${SGLANG_MAX_MAMBA_CACHE_SIZE}")
fi

if [[ -n "${SGLANG_MAMBA_FULL_MEMORY_RATIO:-}" ]]; then
  SGLANG_ARGS+=(--sglang-mamba-full-memory-ratio "${SGLANG_MAMBA_FULL_MEMORY_RATIO}")
fi

if [[ -n "${SGLANG_MAMBA_SCHEDULER_STRATEGY:-}" ]]; then
  SGLANG_ARGS+=(--sglang-mamba-scheduler-strategy "${SGLANG_MAMBA_SCHEDULER_STRATEGY}")
fi

if [[ "${SGLANG_DISABLE_RADIX_CACHE:-0}" == "1" || "${SGLANG_DISABLE_RADIX_CACHE:-0}" == "true" ]]; then
  SGLANG_ARGS+=(--sglang-disable-radix-cache)
fi

if [[ "${SGLANG_DISABLE_CUDA_GRAPH:-1}" == "1" || "${SGLANG_DISABLE_CUDA_GRAPH:-1}" == "true" ]]; then
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi
if [[ "${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-1}" == "1" || "${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-1}" == "true" ]]; then
  SGLANG_ARGS+=(--sglang-disable-piecewise-cuda-graph)
fi
if [[ "${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-1}" == "1" || "${SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-1}" == "true" ]]; then
  SGLANG_ARGS+=(--sglang-disable-custom-all-reduce)
fi
if [[ "${SGLANG_DISABLE_OVERLAP_SCHEDULE:-1}" == "1" || "${SGLANG_DISABLE_OVERLAP_SCHEDULE:-1}" == "true" ]]; then
  SGLANG_ARGS+=(--sglang-disable-overlap-schedule)
fi

if [[ "${SGLANG_LANGUAGE_ONLY}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-language-only)
fi

if [[ "${A3S_CODE_ROUTER_DISABLE_CIRCUIT_BREAKER}" == "1" ]]; then
  SGLANG_ARGS+=(--router-disable-circuit-breaker)
fi

if [[ "${A3S_CODE_ROUTER_DISABLE_HEALTH_CHECK}" == "1" ]]; then
  SGLANG_ARGS+=(--router-disable-health-check)
fi

# ── PRM ──────────────────────────────────────────────────────────
if [[ "${ENABLE_PRM}" == "1" && "${PRM_BACKEND}" == "local_sglang" ]]; then
  PRM_ARGS=(
    --prm-enable
    --prm-num-gpus "${PRM_GPUS}"
    --prm-num-gpus-per-engine "${TP_SGLANG}"
    --prm-model-path "${PRM_MODEL_PATH}"
    --prm-m "${PRM_M}"
    --prm-temperature "${PRM_TEMPERATURE:-0.6}"
    --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-8192}"
  )
else
  PRM_ARGS=()
fi

# ── LoRA ─────────────────────────────────────────────────────────
LORA_ARGS=()
if [[ "${USE_LORA:-0}" == "1" ]]; then
  if [[ "${TRAIN_BACKEND}" != "fsdp" ]]; then
    echo "USE_LORA=1 requires TRAIN_BACKEND=fsdp; current TRAIN_BACKEND=${TRAIN_BACKEND}" >&2
    exit 1
  fi
  LORA_ARGS+=(
    --use-lora
    --lora-rank "${LORA_RANK:-16}"
    --lora-alpha "${LORA_ALPHA:-32}"
    --lora-dropout "${LORA_DROPOUT:-0.0}"
  )
  if [[ -n "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}" ]]; then
    LORA_ARGS+=(--lora-target-modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}")
  fi
  if [[ -n "${LORA_MODULES_TO_SAVE:-}" ]]; then
    LORA_ARGS+=(--lora-modules-to-save "${LORA_MODULES_TO_SAVE}")
  fi
fi

# ── Custom generate / reward ─────────────────────────────────────
CUSTOM_ARGS=(
  --custom-generate-function-path code_rl_api_server.generate
  --custom-rm-path code_rl_api_server.reward_func
)

COLOCATE_ARGS=()
if [[ "${COLOCATE}" == "1" ]]; then
  COLOCATE_ARGS+=(--colocate)
fi

# ── Benchmark eval ───────────────────────────────────────────────
EVAL_ARGS=()
A3S_CODE_ENABLE_BENCHMARK_EVAL="${A3S_CODE_ENABLE_BENCHMARK_EVAL:-0}"
if [[ "${A3S_CODE_ENABLE_BENCHMARK_EVAL}" == "1" ]]; then
  BENCHMARK_EVAL_DIR="${RUN_ROOT}/benchmark_eval"
  A3S_CODE_SKILLSBENCH_ROOT="${A3S_CODE_SKILLSBENCH_ROOT:-${HOME}/workspace/skillsbench}"
  A3S_CODE_CLAWMARK_ROOT="${A3S_CODE_CLAWMARK_ROOT:-${HOME}/workspace/ClawMark}"
  if [[ -d "${A3S_CODE_SKILLSBENCH_ROOT}" && -d "${A3S_CODE_CLAWMARK_ROOT}" ]]; then
    "${PYTHON_BIN}" "${CODE_RL_DIR}/a3s_code_benchmarks/benchmark_eval_builder.py" \
      --skillsbench-root "${A3S_CODE_SKILLSBENCH_ROOT}" \
      --clawmark-root "${A3S_CODE_CLAWMARK_ROOT}" \
      --output-dir "${BENCHMARK_EVAL_DIR}" \
      --skillsbench-max-tasks "${A3S_CODE_SKILLSBENCH_EVAL_MAX_TASKS:-24}" \
      --clawmark-max-tasks "${A3S_CODE_CLAWMARK_EVAL_MAX_TASKS:-24}" \
      --eval-max-response-len "${A3S_CODE_EVAL_MAX_RESPONSE_LEN:-2048}"
    EVAL_ARGS+=(
      --eval-interval "${A3S_CODE_EVAL_INTERVAL:-20}"
      --eval-config "${BENCHMARK_EVAL_DIR}/benchmark_eval_config.json"
    )
    if [[ "${A3S_CODE_SKIP_EVAL_BEFORE_TRAIN:-0}" == "1" ]]; then
      EVAL_ARGS+=(--skip-eval-before-train)
    fi
  else
    echo "benchmark eval roots missing: skillsbench=${A3S_CODE_SKILLSBENCH_ROOT} clawmark=${A3S_CODE_CLAWMARK_ROOT}" >&2
  fi
fi

# ── Official benchmark eval trigger ──────────────────────────────
export A3S_CODE_ENABLE_OFFICIAL_BENCHMARK_EVAL="${A3S_CODE_ENABLE_OFFICIAL_BENCHMARK_EVAL:-0}"
export A3S_CODE_OFFICIAL_BENCHMARK_SCRIPT="${A3S_CODE_OFFICIAL_BENCHMARK_SCRIPT:-${CODE_RL_DIR}/a3s_code_benchmarks/official/official_benchmark_eval.py}"
export A3S_CODE_OFFICIAL_BENCHMARK_OUTPUT_DIR="${A3S_CODE_OFFICIAL_BENCHMARK_OUTPUT_DIR:-${RUN_ROOT}/official_benchmark_eval}"
export A3S_CODE_OFFICIAL_BENCHMARK_SUITES="${A3S_CODE_OFFICIAL_BENCHMARK_SUITES:-skillsbench,clawmark}"
export A3S_CODE_OFFICIAL_BENCHMARK_WAIT_AT_EXIT="${A3S_CODE_OFFICIAL_BENCHMARK_WAIT_AT_EXIT:-1}"
export A3S_CODE_OFFICIAL_BENCHMARK_EVAL_INTERVAL="${A3S_CODE_OFFICIAL_BENCHMARK_EVAL_INTERVAL:-${A3S_CODE_EVAL_INTERVAL:-20}}"
export A3S_CODE_OFFICIAL_SKILLSBENCH_MAX_TASKS="${A3S_CODE_OFFICIAL_SKILLSBENCH_MAX_TASKS:-0}"
export A3S_CODE_OFFICIAL_CLAWMARK_MAX_TASKS="${A3S_CODE_OFFICIAL_CLAWMARK_MAX_TASKS:-0}"
export A3S_CODE_OFFICIAL_SKILLSBENCH_REPEATS="${A3S_CODE_OFFICIAL_SKILLSBENCH_REPEATS:-1}"
export A3S_CODE_OFFICIAL_CLAWMARK_REPEATS="${A3S_CODE_OFFICIAL_CLAWMARK_REPEATS:-1}"
export A3S_CODE_OFFICIAL_SKILLSBENCH_TIMEOUT_SEC="${A3S_CODE_OFFICIAL_SKILLSBENCH_TIMEOUT_SEC:-0}"

# ── Misc ─────────────────────────────────────────────────────────
if [[ "${TRAIN_BACKEND}" == "megatron" ]]; then
  MISC_ARGS=(
    --transformer-impl local
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend unfused
    --no-rope-fusion
    --no-masked-softmax-fusion
    --no-bias-dropout-fusion
    --no-gradient-accumulation-fusion
  )
  if [[ "${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-0}" == "1" ]]; then
    MISC_ARGS+=(--accumulate-allreduce-grads-in-fp32)
  fi
  if [[ "${GRAD_REDUCE_IN_BF16:-0}" == "1" ]]; then
    MISC_ARGS+=(--grad-reduce-in-bf16)
  fi
  if [[ "${RL_OFFLOAD_OPTIMIZER_DURING_INFERENCE:-0}" == "1" ]]; then
    MISC_ARGS+=(--rl-offload-optimizer-during-inference)
  fi
else
  MISC_ARGS=()
fi

TRAIN_ENV_VARS_JSON="$("${PYTHON_BIN}" - <<'PY'
import json
import os

names = [
    "A3S_CODE_LOCAL_CACHE_ROOT",
    "A3S_CODE_RAY_TMPDIR",
    "RAY_TMPDIR",
    "XDG_CACHE_HOME",
    "FLASHINFER_WORKSPACE_BASE",
    "FLASHINFER_CUBIN_DIR",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TVM_FFI_CACHE_DIR",
    "HF_HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_LIB_PATH",
    "CUDNN_HOME",
    "CUDNN_PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "CICC_PATH",
    "CC",
    "CXX",
    "CUDAHOSTCXX",
    "TORCH_CUDA_ARCH_LIST",
    "PYTORCH_CUDA_ALLOC_CONF",
    "MLP_SOCKET_IFNAME",
    "A3S_CODE_NETWORK_IFNAME",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "NCCL_IB_DISABLE",
    "NCCL_CUMEM_ENABLE",
    "NCCL_P2P_LEVEL",
    "NCCL_NVLS_ENABLE",
    "NCCL_MIN_CTAS",
    "NCCL_PXN_DISABLE",
    "NCCL_RUNTIME_CONNECT",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "TORCH_NCCL_AVOID_RECORD_STREAMS",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_BLOCKING_WAIT",
    "TORCH_NCCL_DUMP_ON_TIMEOUT",
    "TORCH_NCCL_TRACE_BUFFER_SIZE",
    "TORCH_NCCL_COORD_CHECK_MILSEC",
    "TORCH_NCCL_ENABLE_MONITORING",
    "TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN",
    "OMPI_MCA_pml",
    "OMPI_MCA_btl",
    "OMPI_MCA_routed",
    "OMPI_MCA_routed_radix",
    "OMPI_MCA_plm_rsh_no_tree_spawn",
    "OMPI_MCA_oob_tcp_if_include",
    "OMPI_MCA_btl_tcp_if_include",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION",
    "SGLANG_FLUSH_CACHE_TIMEOUT_SEC",
    "SGLANG_API_KEY",
    "CODE_RL_SUBMIT_SIDE",
    "CODE_RL_TRAIN_SIDE",
    "CODE_RL_REWARD_MODE",
    "CODE_RL_REQUIRE_VERIFIER_FEEDBACK",
    "CODE_RL_PRM_OPENAI_URL",
    "CODE_RL_PRM_OPENAI_MODEL_NAME",
    "CODE_RL_PRM_API_KEY",
    "CODE_RL_PRM_HEALTH_URL",
    "CODE_RL_PAUSE_WAIT_TIMEOUT_SEC",
    "CODE_RL_PAUSE_DRAIN_TIMEOUT_SEC",
    "CODE_RL_ROLLOUT_IDLE_TIMEOUT_SEC",
    "CODE_RL_ROLLOUT_DRAIN_TIMEOUT_SEC",
    "CODE_RL_GENERATION_TIMEOUT_SEC",
    "CODE_RL_PRM_TIMEOUT_SEC",
    "CODE_RL_CONTEXT_SAFETY_MARGIN",
    "CODE_RL_MAX_RESPONSE_TOKENS",
    "CODE_RL_DROP_REPETITIVE_SAMPLES",
]
print(json.dumps({name: os.environ[name] for name in names if os.environ.get(name)}, separators=(",", ":")))
PY
)"
TRAIN_ENV_ARGS=(
  --train-env-vars "${TRAIN_ENV_VARS_JSON}"
)

if [[ "${TRAIN_BACKEND}" == "megatron" && "${DISABLE_BF16_REDUCED_PRECISION_MATMUL}" == "1" ]]; then
  MISC_ARGS+=(--disable-bf16-reduced-precision-matmul)
fi

# ── Wandb ────────────────────────────────────────────────────────
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-a3s_code_rl}"
WANDB_GROUP="${WANDB_GROUP:-${MODEL_DEFAULT_WANDB_GROUP}}"
WANDB_KEY_VALUE="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [[ "${USE_WANDB}" == "1" && -n "${WANDB_KEY_VALUE}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi

# ── Runtime env ──────────────────────────────────────────────────
RUNTIME_ENV_JSON="$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_ROOT}:${CODE_RL_DIR}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "LD_LIBRARY_PATH": "${LD_LIBRARY_PATH}",
    "CUDA_HOME": "${CUDA_HOME}",
    "CUDA_PATH": "${CUDA_PATH}",
    "CUDA_LIB_PATH": "${CUDA_LIB_PATH}",
    "CUDNN_HOME": "${CUDNN_HOME}",
    "CUDNN_PATH": "${CUDNN_PATH}",
    "LIBRARY_PATH": "${LIBRARY_PATH:-}",
    "CPATH": "${CPATH:-}",
    "C_INCLUDE_PATH": "${C_INCLUDE_PATH:-}",
    "CPLUS_INCLUDE_PATH": "${CPLUS_INCLUDE_PATH:-}",
    "CICC_PATH": "${CICC_PATH:-}",
    "CC": "${CC}",
    "CXX": "${CXX}",
    "CUDAHOSTCXX": "${CUDAHOSTCXX}",
    "TORCH_CUDA_ARCH_LIST": "${TORCH_CUDA_ARCH_LIST}",
    "MLP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME:-}",
    "A3S_CODE_NETWORK_IFNAME": "${A3S_CODE_NETWORK_IFNAME:-}",
    "NCCL_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME:-}",
    "GLOO_SOCKET_IFNAME": "${GLOO_SOCKET_IFNAME:-}",
    "TP_SOCKET_IFNAME": "${TP_SOCKET_IFNAME:-}",
    "NCCL_IB_DISABLE": "${NCCL_IB_DISABLE:-}",
    "NCCL_CUMEM_ENABLE": "${NCCL_CUMEM_ENABLE:-}",
    "NCCL_P2P_LEVEL": "${NCCL_P2P_LEVEL:-}",
    "NCCL_NVLS_ENABLE": "${NCCL_NVLS_ENABLE:-}",
    "NCCL_MIN_CTAS": "${NCCL_MIN_CTAS:-}",
    "NCCL_PXN_DISABLE": "${NCCL_PXN_DISABLE:-}",
    "NCCL_RUNTIME_CONNECT": "${NCCL_RUNTIME_CONNECT:-}",
    "NCCL_DEBUG": "${NCCL_DEBUG:-}",
    "NCCL_DEBUG_SUBSYS": "${NCCL_DEBUG_SUBSYS:-}",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "${TORCH_NCCL_AVOID_RECORD_STREAMS:-}",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "${TORCH_NCCL_ASYNC_ERROR_HANDLING:-}",
    "TORCH_NCCL_BLOCKING_WAIT": "${TORCH_NCCL_BLOCKING_WAIT:-}",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "${TORCH_NCCL_DUMP_ON_TIMEOUT:-}",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "${TORCH_NCCL_TRACE_BUFFER_SIZE:-}",
    "TORCH_NCCL_COORD_CHECK_MILSEC": "${TORCH_NCCL_COORD_CHECK_MILSEC:-}",
    "TORCH_NCCL_ENABLE_MONITORING": "${TORCH_NCCL_ENABLE_MONITORING:-}",
    "TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN": "${TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN:-}",
    "OMPI_MCA_pml": "${OMPI_MCA_pml:-}",
    "OMPI_MCA_btl": "${OMPI_MCA_btl:-}",
    "OMPI_MCA_routed": "${OMPI_MCA_routed:-}",
    "OMPI_MCA_routed_radix": "${OMPI_MCA_routed_radix:-}",
    "OMPI_MCA_plm_rsh_no_tree_spawn": "${OMPI_MCA_plm_rsh_no_tree_spawn:-}",
    "OMPI_MCA_oob_tcp_if_include": "${OMPI_MCA_oob_tcp_if_include:-}",
    "OMPI_MCA_btl_tcp_if_include": "${OMPI_MCA_btl_tcp_if_include:-}",
    "A3S_CODE_LOCAL_CACHE_ROOT": "${A3S_CODE_LOCAL_CACHE_ROOT}",
    "A3S_CODE_RAY_TMPDIR": "${A3S_CODE_RAY_TMPDIR}",
    "RAY_TMPDIR": "${RAY_TMPDIR}",
    "XDG_CACHE_HOME": "${XDG_CACHE_HOME}",
    "FLASHINFER_WORKSPACE_BASE": "${FLASHINFER_WORKSPACE_BASE}",
    "FLASHINFER_CUBIN_DIR": "${FLASHINFER_CUBIN_DIR}",
    "TRITON_CACHE_DIR": "${TRITON_CACHE_DIR}",
    "TORCHINDUCTOR_CACHE_DIR": "${TORCHINDUCTOR_CACHE_DIR}",
    "TORCH_EXTENSIONS_DIR": "${TORCH_EXTENSIONS_DIR}",
    "TVM_FFI_CACHE_DIR": "${TVM_FFI_CACHE_DIR}",
    "HF_HOME": "${HF_HOME}",
    "TMPDIR": "${TMPDIR}",
    "TEMP": "${TEMP}",
    "TMP": "${TMP}",
    "no_proxy": "${NO_PROXY_BASE}",
    "NO_PROXY": "${NO_PROXY_BASE}",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION}",
    "SLIME_HOST_IP": "${SLIME_HOST_IP}",
    "CODE_RL_PRM_OPENAI_URL": "${CODE_RL_PRM_OPENAI_URL:-}",
    "CODE_RL_PRM_OPENAI_MODEL_NAME": "${CODE_RL_PRM_OPENAI_MODEL_NAME:-}",
    "CODE_RL_PRM_API_KEY": "${CODE_RL_PRM_API_KEY:-}",
    "CODE_RL_PRM_HEALTH_URL": "${CODE_RL_PRM_HEALTH_URL:-}",
    "CODE_RL_BENCHMARK_EVAL_JUDGE_URL": "${CODE_RL_BENCHMARK_EVAL_JUDGE_URL:-}",
    "CODE_RL_BENCHMARK_EVAL_JUDGE_MODEL_NAME": "${CODE_RL_BENCHMARK_EVAL_JUDGE_MODEL_NAME:-}",
    "CODE_RL_BENCHMARK_EVAL_JUDGE_API_KEY": "${CODE_RL_BENCHMARK_EVAL_JUDGE_API_KEY:-}",
    "CODE_RL_BENCHMARK_EVAL_JUDGE_TIMEOUT_SEC": "${CODE_RL_BENCHMARK_EVAL_JUDGE_TIMEOUT_SEC:-90}",
    "CODE_RL_PAUSE_WAIT_TIMEOUT_SEC": "${CODE_RL_PAUSE_WAIT_TIMEOUT_SEC:-}",
    "CODE_RL_PAUSE_DRAIN_TIMEOUT_SEC": "${CODE_RL_PAUSE_DRAIN_TIMEOUT_SEC:-}",
    "CODE_RL_ROLLOUT_IDLE_TIMEOUT_SEC": "${CODE_RL_ROLLOUT_IDLE_TIMEOUT_SEC:-}",
    "CODE_RL_ROLLOUT_DRAIN_TIMEOUT_SEC": "${CODE_RL_ROLLOUT_DRAIN_TIMEOUT_SEC:-}",
    "CODE_RL_GENERATION_TIMEOUT_SEC": "${CODE_RL_GENERATION_TIMEOUT_SEC:-}",
    "CODE_RL_PRM_TIMEOUT_SEC": "${CODE_RL_PRM_TIMEOUT_SEC}",
    "CODE_RL_CONTEXT_SAFETY_MARGIN": "${CODE_RL_CONTEXT_SAFETY_MARGIN}",
    "CODE_RL_MAX_RESPONSE_TOKENS": "${CODE_RL_MAX_RESPONSE_TOKENS}",
    "CODE_RL_DROP_REPETITIVE_SAMPLES": "${CODE_RL_DROP_REPETITIVE_SAMPLES}",
    "CODE_RL_SUBMIT_SIDE": "${CODE_RL_SUBMIT_SIDE}",
    "CODE_RL_TRAIN_SIDE": "${CODE_RL_TRAIN_SIDE}",
    "CODE_RL_REWARD_MODE": "${CODE_RL_REWARD_MODE}",
    "CODE_RL_REQUIRE_VERIFIER_FEEDBACK": "${CODE_RL_REQUIRE_VERIFIER_FEEDBACK}",
    "A3S_CODE_SEED_DATA_FILE": "${A3S_CODE_SEED_DATA_FILE:-}",
    "A3S_CODE_TASK_TEMPLATE_ROOT": "${A3S_CODE_TASK_TEMPLATE_ROOT:-}",
    "A3S_CODE_ENABLE_TASK_VERIFIER_REWARD": "${A3S_CODE_ENABLE_TASK_VERIFIER_REWARD:-1}",
    "A3S_CODE_VERIFIER_FALLBACK_TO_TEST_COMMAND": "${A3S_CODE_VERIFIER_FALLBACK_TO_TEST_COMMAND:-1}",
    "A3S_CODE_TASK_VERIFIER_TIMEOUT_SEC": "${A3S_CODE_TASK_VERIFIER_TIMEOUT_SEC:-180}",
    "A3S_CODE_BUILTIN_SKILLS": "${A3S_CODE_BUILTIN_SKILLS:-1}",
    "A3S_CODE_PLANNING": "${A3S_CODE_PLANNING:-1}",
    "A3S_CODE_PLANNING_MODE": "${A3S_CODE_PLANNING_MODE:-}",
    "SLIME_PPO_RATIO_SAFE_BOUND": "${SLIME_PPO_RATIO_SAFE_BOUND}",
    "SLIME_QWEN35_TEXT_ONLY_BRIDGE": "${SLIME_QWEN35_TEXT_ONLY_BRIDGE}"
  }
}
JSON
)"
RUNTIME_ENV_JSON="$("${PYTHON_BIN}" -c 'import json,sys; payload=json.load(sys.stdin); payload["env_vars"]={k:v for k,v in payload.get("env_vars",{}).items() if v != ""}; print(json.dumps(payload,separators=(",",":")))' <<<"${RUNTIME_ENV_JSON}")"

# ── Save launch info ─────────────────────────────────────────────
cat > "${RUN_ROOT}/launch_info.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "run_root": "${RUN_ROOT}",
  "model_family": "${MODEL_FAMILY}",
  "train_backend": "${TRAIN_BACKEND}",
  "use_lora": $([[ "${USE_LORA:-0}" == "1" ]] && echo true || echo false),
  "lora_rank": ${LORA_RANK:-16},
  "lora_alpha": ${LORA_ALPHA:-32},
  "lora_target_modules": "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}",
  "hf_ckpt": "${HF_CKPT}",
  "save_ckpt": "${SAVE_CKPT}",
  "num_gpus": ${NUM_GPUS},
  "num_gpus_per_node": ${NUM_GPUS_PER_NODE},
  "actor_num_nodes": ${ACTOR_NUM_NODES},
  "actor_total_gpus": ${ACTOR_TOTAL_GPUS},
  "total_gpus": ${TOTAL_GPUS},
  "actor_gpus": ${ACTOR_GPUS},
  "rollout_gpus": ${ROLLOUT_GPUS},
  "prm_gpus": ${EFFECTIVE_PRM_GPUS},
  "tp_train": ${TP_TRAIN},
  "pp_train": ${PP_TRAIN},
  "cp_train": ${CP_TRAIN},
  "ep_train": ${EP_TRAIN:-1},
  "etp_train": ${ETP_TRAIN:-1},
  "tp_sglang": ${TP_SGLANG},
  "network_ifname": "${MLP_SOCKET_IFNAME:-}",
  "nccl_socket_ifname": "${NCCL_SOCKET_IFNAME:-}",
  "nccl_ib_disable": "${NCCL_IB_DISABLE:-}",
  "nccl_runtime_connect": "${NCCL_RUNTIME_CONNECT:-}",
  "torch_nccl_async_error_handling": "${TORCH_NCCL_ASYNC_ERROR_HANDLING:-}",
  "torch_nccl_blocking_wait": "${TORCH_NCCL_BLOCKING_WAIT:-}",
  "torch_nccl_dump_on_timeout": "${TORCH_NCCL_DUMP_ON_TIMEOUT:-}",
  "torch_nccl_trace_buffer_size": "${TORCH_NCCL_TRACE_BUFFER_SIZE:-}",
  "torch_nccl_coord_check_milsec": "${TORCH_NCCL_COORD_CHECK_MILSEC:-}",
  "torch_nccl_enable_monitoring": "${TORCH_NCCL_ENABLE_MONITORING:-}",
  "torch_nccl_log_cpp_stack_on_unclean_shutdown": "${TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN:-}",
  "local_cache_root": "${A3S_CODE_LOCAL_CACHE_ROOT}",
  "ray_tmpdir": "${RAY_TMPDIR}",
  "tvm_ffi_cache_dir": "${TVM_FFI_CACHE_DIR}",
  "ray_memory_usage_threshold": "${RAY_memory_usage_threshold}",
  "ray_memory_monitor_refresh_ms": "${RAY_memory_monitor_refresh_ms:-}",
  "colocate": $([[ "${COLOCATE}" == "1" ]] && echo true || echo false),
  "dynamic_batch_size": $([[ "${ENABLE_DYNAMIC_BATCH_SIZE:-0}" == "1" ]] && echo true || echo false),
  "optimizer_cpu_offload": $([[ "${ENABLE_OPTIMIZER_CPU_OFFLOAD:-0}" == "1" ]] && echo true || echo false),
  "precision_aware_optimizer": $([[ "${USE_PRECISION_AWARE_OPTIMIZER:-0}" == "1" ]] && echo true || echo false),
  "main_grads_dtype": "${MAIN_GRADS_DTYPE:-}",
  "grad_reduce_in_bf16": $([[ "${GRAD_REDUCE_IN_BF16:-0}" == "1" ]] && echo true || echo false),
  "accumulate_allreduce_grads_in_fp32": $([[ "${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-0}" == "1" ]] && echo true || echo false),
  "rl_offload_optimizer_during_inference": $([[ "${RL_OFFLOAD_OPTIMIZER_DURING_INFERENCE:-0}" == "1" ]] && echo true || echo false),
  "use_distributed_optimizer": $([[ "${USE_DISTRIBUTED_OPTIMIZER:-0}" == "1" ]] && echo true || echo false),
  "enable_prm": $([[ "${ENABLE_PRM}" == "1" ]] && echo true || echo false),
  "prm_backend": "${EFFECTIVE_PRM_BACKEND}",
  "prm_model": "$([[ "${EFFECTIVE_PRM_BACKEND}" == "external_openai" ]] && printf '%s' "${CODE_RL_PRM_OPENAI_MODEL_NAME}" || printf '%s' "${PRM_MODEL_PATH}")",
  "code_rl_pause_wait_timeout_sec": "${CODE_RL_PAUSE_WAIT_TIMEOUT_SEC:-}",
  "code_rl_pause_drain_timeout_sec": "${CODE_RL_PAUSE_DRAIN_TIMEOUT_SEC:-}",
  "code_rl_rollout_idle_timeout_sec": "${CODE_RL_ROLLOUT_IDLE_TIMEOUT_SEC:-}",
  "code_rl_rollout_drain_timeout_sec": "${CODE_RL_ROLLOUT_DRAIN_TIMEOUT_SEC:-}",
  "router_balance_abs_threshold": "${A3S_CODE_ROUTER_BALANCE_ABS_THRESHOLD:-}",
  "seed_data_file": "${A3S_CODE_SEED_DATA_FILE:-${CODE_RL_DIR}/seed_data/code_task_seeds.json}",
  "task_template_root": "${A3S_CODE_TASK_TEMPLATE_ROOT:-${CODE_RL_DIR}/task_templates}",
  "task_verifier_reward_enabled": $([[ "${A3S_CODE_ENABLE_TASK_VERIFIER_REWARD:-1}" == "1" ]] && echo true || echo false),
  "a3s_code_builtin_skills": $([[ "${A3S_CODE_BUILTIN_SKILLS:-1}" == "1" ]] && echo true || echo false),
  "a3s_code_planning": "${A3S_CODE_PLANNING:-1}",
  "a3s_code_planning_mode": "${A3S_CODE_PLANNING_MODE:-}",
  "benchmark_eval_enabled": $([[ "${A3S_CODE_ENABLE_BENCHMARK_EVAL}" == "1" ]] && echo true || echo false),
  "timestamp": "$(date -Iseconds)"
}
EOF

echo "=== RUN_ROOT: ${RUN_ROOT} ==="

# ── Launch (direct driver, avoids ray job submit env issues) ──────
export PYTHONPATH="${MEGATRON_ROOT}:${CODE_RL_DIR}:${SLIME_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OPENCLAW_RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}"

cd "${SLIME_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/openclaw-rl/train_async_direct_driver.py" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${TRAIN_ENV_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${COLOCATE_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${LORA_ARGS[@]}" \
  "${PRM_ARGS[@]}"
