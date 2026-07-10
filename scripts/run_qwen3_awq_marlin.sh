#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AIRLLM_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
PYTHON_BIN_DIR="$(dirname -- "${PYTHON}")"
GPU_LAYER_GROUP_SIZE="${AIRLLM_GPU_LAYER_GROUP_SIZE:-9}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "AirLLM Python environment not found: ${PYTHON}" >&2
    echo "Create it with: uv venv --python 3.12 .venv" >&2
    exit 2
fi

PYTHON_CUDA_HOME="$("${PYTHON}" - <<'PY'
from pathlib import Path
import site

for site_packages in site.getsitepackages():
    candidate = Path(site_packages) / "nvidia" / "cu13"
    if (candidate / "bin" / "nvcc").is_file():
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY
)" || {
    echo "The CUDA 13.0 Marlin build toolchain is missing from ${PYTHON}." >&2
    echo "Install it with:" >&2
    echo "uv pip install --python .venv/bin/python ninja nvidia-cuda-nvcc==13.0.88 nvidia-cuda-cccl==13.0.85 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 nvidia-cuda-runtime==13.0.88 nvidia-cublas==13.0.2.14" >&2
    exit 2
}

CUDART_PATH="$(find "${PYTHON_CUDA_HOME}/lib" -maxdepth 1 -type f -name 'libcudart.so.*' -print -quit)"
if [[ -z "${CUDART_PATH}" ]]; then
    echo "CUDA runtime library not found under ${PYTHON_CUDA_HOME}/lib" >&2
    exit 2
fi

CUDA_LINK_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}/airllm/cuda-cu130-link"
CUDART_LINK="${CUDA_LINK_DIR}/libcudart.so"
mkdir -p "${CUDA_LINK_DIR}"
if [[ -L "${CUDART_LINK}" ]]; then
    CURRENT_CUDART="$(readlink -f "${CUDART_LINK}" 2>/dev/null || true)"
    EXPECTED_CUDART="$(readlink -f "${CUDART_PATH}")"
    if [[ "${CURRENT_CUDART}" != "${EXPECTED_CUDART}" ]]; then
        ln -sfn "${CUDART_PATH}" "${CUDART_LINK}"
    fi
elif [[ -e "${CUDART_LINK}" ]]; then
    echo "Refusing to replace non-symlink CUDA linker path: ${CUDART_LINK}" >&2
    exit 2
else
    ln -s "${CUDART_PATH}" "${CUDART_LINK}"
fi

export CUDA_HOME="${PYTHON_CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PYTHON_BIN_DIR}:${PATH}"
export LIBRARY_PATH="${CUDA_LINK_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GPTQMODEL_TORCH_EXTENSIONS_DIR="${GPTQMODEL_TORCH_EXTENSIONS_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/airllm/gptqmodel-torch-extensions}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_FORCE_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_WEIGHTS_ONLY_LOAD:-1}"

if [[ "${1:-}" == "--context-limit-benchmark" ]]; then
    shift
    exec "${PYTHON}" "${ROOT_DIR}/benchmarks/benchmark_context_limit.py" \
        --layers-per-gpu-group "${GPU_LAYER_GROUP_SIZE}" \
        --max-gpu-layer-fraction 0.5 \
        --prefetch-groups 8 \
        --cpu-prefetch-workers 2 \
        --cpu-layer-cache-gib 16 \
        --awq-backend gemm_triton \
        "$@"
fi

if [[ "${1:-}" == "--benchmark" ]]; then
    shift
    exec "${PYTHON}" "${ROOT_DIR}/benchmarks/benchmark_group_streaming.py" \
        --model-path /mnt/s/ai-cache/huggingface/hub/models--Qwen--Qwen3-4B-AWQ \
        --device cuda:0 \
        --layers-per-gpu-group "${GPU_LAYER_GROUP_SIZE}" \
        --max-gpu-layer-fraction 0.5 \
        --prefetch-groups 8 \
        --cpu-prefetch-workers 2 \
        --cpu-layer-cache-gib 16 \
        --cpu-layer-cache-policy static \
        --no-persistent-gpu-residency \
        --awq-backend gemm_triton \
        --cache-implementation dynamic \
        "$@"
fi

for argument in "$@"; do
    case "${argument}" in
        --persistent-gpu-residency)
            echo "The prepared streaming launcher enforces half-or-less decoder-weight residency." >&2
            echo "Full-model GPU residency is outside this fork's large-model benchmark target." >&2
            exit 2
            ;;
    esac
done

exec "${PYTHON}" "${ROOT_DIR}/scripts/run_qwen3_awq.py" \
    --layers-per-gpu-group "${GPU_LAYER_GROUP_SIZE}" \
    --max-gpu-layer-fraction 0.5 \
    --prefetch-groups 8 \
    --cpu-prefetch-workers 2 \
    --cpu-layer-cache-gib 16 \
    --cpu-layer-cache-policy static \
    --no-persistent-gpu-residency \
    --awq-backend gemm_triton \
    --cache-implementation dynamic \
    --max-input-tokens 40448 \
    --max-new-tokens 512 \
    "$@"
