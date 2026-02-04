#!/usr/bin/env bash

# Run evaluation (density, detection, quality, MLE, pMSE, t-SNE) on all synthetic datasets

# and generate comparative plots.

#

# Usage:

#   ./UTILITY.sh --dataname cardio --models tabddpm ctgan dpctgan tabsyn great stasy

#

# Defaults:

#   dataname=cardio, models=(tabddpm ctgan dpctgan tabsyn great stasy)

set -u -o pipefail

DATANAME="cardio"
declare -a MODELS=("tabddpm" "ctgan" "dpctgan" "tabsyn" "great" "stasy")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_DIR}/utility_${DATANAME}_${TS}.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dataname)
      DATANAME="$2"
      shift 2
      ;;
    -m|--models)
      shift
      MODELS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    -h|--help)
      echo "Usage: $0 --dataname cardio --models tabddpm ctgan dpctgan tabsyn great stasy"
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
echo "" | tee -a "${SUMMARY}"

ensure_eval_deps() {
  python - <<'PY' >/dev/null 2>&1 || {
try:
    import sdmetrics  # noqa
    import synthcity  # noqa
    import xgboost  # noqa
    import matplotlib  # noqa
    import seaborn  # noqa
    import sklearn  # noqa
    print("OK")
except Exception:
    raise SystemExit(1)
PY
  if [[ $? -ne 0 ]]; then
    echo "[deps] Installing eval dependencies (sdmetrics, synthcity, xgboost, seaborn, scikit-learn)..." | tee -a "${SUMMARY}"
    pip install --quiet sdmetrics synthcity xgboost seaborn scikit-learn || {
      echo "[deps] pip install failed. Please install manually." | tee -a "${SUMMARY}"
      exit 1
    }
  fi
}

ensure_eval_deps

REAL_PATH="${REPO_ROOT}/synthetic/${DATANAME}/real.csv"
TEST_PATH="${REPO_ROOT}/synthetic/${DATANAME}/test.csv"
if [[ ! -f "${REAL_PATH}" ]]; then
  echo "[warn] Missing ${REAL_PATH}. Ensure TRAIN.sh ran (it saves synthetic/{dataname}/real.csv)." | tee -a "${SUMMARY}"
fi
if [[ ! -f "${TEST_PATH}" ]]; then
  echo "[warn] Missing ${TEST_PATH}. Ensure SAMPLE.sh ran (it saves synthetic/{dataname}/test.csv)." | tee -a "${SUMMARY}"
fi

eval_detection_one() {
  local method="$1"
  local out_dir="${REPO_ROOT}/eval/detection/${DATANAME}"
  mkdir -p "${out_dir}"
  local score
  score="$(python "${REPO_ROOT}/eval/eval_detection.py" --dataname "${DATANAME}" --model "${method}" | tail -n 1 | awk -F ': ' '{print $2}')"
  if [[ -z "${score}" ]]; then
    echo "[detect] ${method}: failed to parse detection score" | tee -a "${SUMMARY}"
  else
    echo "${score}" > "${out_dir}/${method}.txt"
    echo "[detect] ${method}: ${score}" | tee -a "${SUMMARY}"
  fi
}

for method in "${MODELS[@]}"; do
  SYN_PATH="${REPO_ROOT}/synthetic/${DATANAME}/${method}.csv"
  if [[ ! -f "${SYN_PATH}" ]]; then
    echo "[skip] ${method}: synthetic file not found at ${SYN_PATH}" | tee -a "${SUMMARY}"
    continue
  fi

  echo "[eval] ${DATANAME} / ${method}" | tee -a "${SUMMARY}"

  # Density (Shape/Trend + CSV details)
  python "${REPO_ROOT}/eval/eval_density.py" --dataname "${DATANAME}" --model "${method}" >> "${SUMMARY}" 2>&1

  # Detection (capture score to file)
  eval_detection_one "${method}"

  # Quality (Alpha/Beta to txt)
  python "${REPO_ROOT}/eval/eval_quality.py" --dataname "${DATANAME}" --model "${method}" >> "${SUMMARY}" 2>&1

  # MLE (writes JSON with metrics)
  python "${REPO_ROOT}/eval/eval_mle.py" --dataname "${DATANAME}" --model "${method}" >> "${SUMMARY}" 2>&1

  # pMSE-Ratio (writes eval/pmse/{dataname}/{method}.txt)
  python "${REPO_ROOT}/eval/eval_pmse.py" --dataname "${DATANAME}" --model "${method}" >> "${SUMMARY}" 2>&1

  # t-SNE latent visualization (writes eval/tsne/{dataname}/{method}.png)
  python "${REPO_ROOT}/eval/eval_tsne.py" --dataname "${DATANAME}" --model "${method}" >> "${SUMMARY}" 2>&1
done

# Comparative plots (aggregates density, detection, quality, MLE, pMSE)

python "${REPO_ROOT}/eval/plot_compare.py" --dataname "${DATANAME}" --models "${MODELS[*]}" >> "${SUMMARY}" 2>&1

echo "" | tee -a "${SUMMARY}"
echo "Done. Summary at ${SUMMARY}" | tee -a "${SUMMARY}"