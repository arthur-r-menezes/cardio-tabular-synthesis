#!/usr/bin/env bash

# SAMPLE.sh - sample synthetic data from trained models.

# Usage:

#   ./SAMPLE.sh --gpus 0,1 --models ctgan tabddpm tabsyn great stasy --dataname cardio [--parallel] [--steps 1000] [--ddim]

# Defaults:

#   gpus=0, dataname=cardio, models=(tabddpm ctgan dpctgan tabsyn great stasy), steps=1000, ddim=false

set -u -o pipefail

DATANAME="cardio"
declare -a GPUS=("0")
declare -a MODELS=("tabddpm" "ctgan" "dpctgan" "tabsyn" "great" "stasy")
PARALLEL=false
DDIM_FLAG=""
STEPS="1000"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_DIR}/sample_${DATANAME}_${TS}.txt"

# Parse args

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpus) shift; GPUS=(); IFS=',' read -ra TOKS <<< "${1}"; for t in "${TOKS[@]}"; do GPUS+=("$t"); done; shift ;;
    -m|--models) shift; MODELS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do MODELS+=("$1"); shift; done ;;
    -d|--dataname) DATANAME="$2"; shift 2 ;;
    -p|--parallel) PARALLEL=true; shift ;;
    --ddim) DDIM_FLAG="--ddim"; shift ;;
    --steps) STEPS="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 --gpus 0,1 --models ... --dataname cardio [--parallel] [--ddim] [--steps N]"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "Start: $(date)" | tee -a "${SUMMARY}"
echo "Dataname: ${DATANAME}" | tee -a "${SUMMARY}"
echo "Models: ${MODELS[*]}" | tee -a "${SUMMARY}"
echo "GPUs: ${GPUS[*]}" | tee -a "${SUMMARY}"
echo "Parallel: ${PARALLEL}" | tee -a "${SUMMARY}"
echo "DDIM: ${DDIM_FLAG:+true}" | tee -a "${SUMMARY}"
echo "Steps: ${STEPS}" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

gpu_idx=0
next_gpu() { local g="${GPUS[$gpu_idx]}"; gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} )); echo "${g}"; }

run_single() {
  local method="$1"
  local gpu="$2"
  local ts log; ts="$(date +%Y%m%d_%H%M%S)"; log="${LOG_DIR}/sample_${DATANAME}_${method}_${ts}.log"

  # Optional --load fix for CTGAN/DP-CTGAN path mismatch
  local extra=""
  if [[ "${method}" == "ctgan" ]]; then
    extra="--load ${REPO_ROOT}/baselines/ctgan/ckpt/${DATANAME}.pkl"
  elif [[ "${method}" == "dpctgan" ]]; then
    extra="--load ${REPO_ROOT}/baselines/ctgan/ckpt/${DATANAME}.pkl"
  fi

  echo "[run] ${method} on GPU ${gpu} -> ${log}" | tee -a "${SUMMARY}"
  (
    cd "${REPO_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    if [[ "${method}" == "tabddpm" ]]; then
      python main.py --dataname "${DATANAME}" --method tabddpm --mode sample --gpu "${gpu}" ${DDIM_FLAG} --steps "${STEPS}" ${extra}
    else
      python main.py --dataname "${DATANAME}" --method "${method}" --mode sample --gpu "${gpu}" ${extra}
    fi
  ) > >(tee "${log}") 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${method} (rc=${rc}) See ${log}" | tee -a "${SUMMARY}"
  else
    echo "[OK]   ${method} -> synthetic/${DATANAME}/${method}.csv" | tee -a "${SUMMARY}"
  fi
  return $rc
}

sequential_exec() {
  for method in "${MODELS[@]}"; do
    gpu="$(next_gpu)"
    run_single "${method}" "${gpu}"
  done
}

parallel_exec() {
  declare -a PIDS=()
  for method in "${MODELS[@]}"; do
    gpu="$(next_gpu)"
    ( run_single "${method}" "${gpu}" ) &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do wait "$pid"; done
}

if [[ "${PARALLEL}" == true ]]; then
  parallel_exec
else
  sequential_exec
fi

echo ""
echo "All done. Summary at ${SUMMARY}"