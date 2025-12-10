#!/usr/bin/env bash

# Train multiple tabular synthesis models with logs and GPU selection.

# Usage:

#   ./TRAIN.sh --gpus 0,1 --models ctgan tabddpm tabsyn great stasy --dataname cardio [--parallel]

# Defaults:

#   gpus=0, dataname=cardio, models=(tabddpm ctgan dpctgan tabsyn great stasy)

# Notes:

#   - VAE is auto-run before TabSyn and is not a standalone training option.

#   - In --parallel mode, one background process is started per requested model.

#     If more models than GPUs are provided, GPUs may be shared concurrently.

set -u -o pipefail

# Defaults

DATANAME="cardio"
declare -a GPUS=("0")
declare -a MODELS=("tabddpm" "ctgan" "dpctgan" "tabsyn" "great" "stasy")
PARALLEL=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_DIR}/summary_${DATANAME}_${TIMESTAMP}.txt"

# Parse args

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpus)
      shift
      GPUS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        IFS=',' read -ra TOKS <<< "$1"
        for t in "${TOKS[@]}"; do GPUS+=("$t"); done
        shift
      done
      ;;
    -m|--models)
      shift
      MODELS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    -d|--dataname)
      DATANAME="$2"
      shift 2
      ;;
    -p|--parallel)
      PARALLEL=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --gpus 0,1 --models ctgan tabddpm tabsyn great stasy --dataname cardio [--parallel]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

echo "Start: $(date)" | tee -a "${SUMMARY}"
echo "Dataname: ${DATANAME}" | tee -a "${SUMMARY}"
echo "Models: ${MODELS[*]}" | tee -a "${SUMMARY}"
echo "GPUs: ${GPUS[*]}" | tee -a "${SUMMARY}"
echo "Parallel: ${PARALLEL}" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

gpu_idx=0
next_gpu() {
  local gpu="${GPUS[$gpu_idx]}"
  gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
  echo "${gpu}"
}

run_single() {
  local method="$1"
  local gpu_id="$2"
  local run_ts
  run_ts="$(date +%Y%m%d_%H%M%S)"
  local log="${LOG_DIR}/train_${DATANAME}_${method}_${run_ts}.log"

  echo "[run] ${method} on GPU ${gpu_id} -> ${log}" | tee -a "${SUMMARY}"
  (
    cd "${REPO_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    python main.py --dataname "${DATANAME}" --method "${method}" --mode train --gpu "${gpu_id}"
  ) > >(tee "${log}") 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${method} (rc=${rc}) See ${log}" | tee -a "${SUMMARY}"
  else
    echo "[OK]   ${method}" | tee -a "${SUMMARY}"
  fi
  return $rc
}

run_tabsyn_pipeline() {
  local gpu_id="$1"
  # VAE first (internal), then TabSyn
  run_single "vae" "${gpu_id}"
  run_single "tabsyn" "${gpu_id}"
}

sequential_exec() {
  for model in "${MODELS[@]}"; do
    if [[ "${model}" == "vae" ]]; then
      echo "[warn] 'vae' is internal to TabSyn; ignoring." | tee -a "${SUMMARY}"
      continue
    fi
    gpu="$(next_gpu)"
    case "${model}" in
      tabsyn) run_tabsyn_pipeline "${gpu}" ;;
      tabddpm|ctgan|dpctgan|great|stasy) run_single "${model}" "${gpu}" ;;
      *)
        echo "[warn] Unknown model '${model}'. Skipping." | tee -a "${SUMMARY}"
        ;;
    esac
  done
}

parallel_exec() {
  declare -a PIDS=()
  for model in "${MODELS[@]}"; do
    if [[ "${model}" == "vae" ]]; then
      echo "[warn] 'vae' is internal to TabSyn; ignoring." | tee -a "${SUMMARY}"
      continue
    fi
    gpu="$(next_gpu)"
    case "${model}" in
      tabsyn)
        ( run_tabsyn_pipeline "${gpu}" ) &
        PIDS+=("$!")
        ;;
      tabddpm|ctgan|dpctgan|great|stasy)
        ( run_single "${model}" "${gpu}" ) &
        PIDS+=("$!")
        ;;
      *)
        echo "[warn] Unknown model '${model}'. Skipping." | tee -a "${SUMMARY}"
        ;;
    esac
  done

  # Wait for all background jobs
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
}

if [[ "${PARALLEL}" == true ]]; then
  parallel_exec
else
  sequential_exec
fi

echo ""
echo "All done. Summary at ${SUMMARY}"