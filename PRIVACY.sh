#!/usr/bin/env bash

# PRIVACY.sh - run privacy evaluations (DCR + DOMIAS MIA) on synthetic datasets.

#

# Usage:

#   ./PRIVACY.sh --dataname cardio --models tabddpm ctgan dpctgan tabsyn great stasy

#

# Defaults:

#   dataname=cardio

#   models=(tabddpm ctgan dpctgan tabsyn great stasy)

set -u -o pipefail

DATANAME="cardio"
declare -a MODELS=("tabddpm" "ctgan" "dpctgan" "tabsyn" "great" "stasy")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_DIR}/privacy_${DATANAME}_${TS}.txt"

# DOMIAS args

DOMIAS_MEM=500           # how many real training points are considered “members” in the attack.
DOMIAS_REF=5000          # size of the reference dataset for estimating p_R.
DOMIAS_SYN=10000         # how many synthetic points DOMIAS uses internally.
DOMIAS_DENSITY="prior"   # Density Estimator ("prior / "kde" / "bnaf"). (BNAF is strongest but slowest).
DOMIAS_EPOCHS=2000       # number of epochs for the internal CTGAN used in baseline LOGAN_0.

# Parse args

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dataname)
      DATANAME="$2"; shift 2 ;;
    -m|--models)
      shift; MODELS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        MODELS+=("$1"); shift; done ;;
    --domias-mem)
      DOMIAS_MEM="$2"; shift 2 ;;
    --domias-ref)
      DOMIAS_REF="$2"; shift 2 ;;
    --domias-syn)
      DOMIAS_SYN="$2"; shift 2 ;;
    --domias-density)
      DOMIAS_DENSITY="$2"; shift 2 ;;
    --domias-epochs)
      DOMIAS_EPOCHS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --dataname cardio --models ... [--domias-mem N] [--domias-ref N] [--domias-syn N] [--domias-density prior|kde|bnaf] [--domias-epochs N]"
      exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "Start: $(date)" | tee -a "${SUMMARY}"
echo "Dataname: ${DATANAME}" | tee -a "${SUMMARY}"
echo "Models: ${MODELS[*]}" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

# Ensure DOMIAS deps are installed

python - <<'PY' >/dev/null 2>&1 || {
try:
    import domias  # noqa
    import geomloss  # noqa
except Exception:
    raise SystemExit(1)
PY
if [[ $? -ne 0 ]]; then
  echo "[deps] Installing DOMIAS dependencies (geomloss, tensorboardX)..." | tee -a "${SUMMARY}"
  pip install --quiet geomloss tensorboardX || {
    echo "[deps] pip install failed. Please install geomloss and tensorboardX manually." | tee -a "${SUMMARY}"
    exit 1
  }
fi

for method in "${MODELS[@]}"; do
  SYN_PATH="${REPO_ROOT}/synthetic/${DATANAME}/${method}.csv"
  if [[ ! -f "${SYN_PATH}" ]]; then
    echo "[skip] ${method}: synthetic file not found at ${SYN_PATH}" | tee -a "${SUMMARY}"
    continue
  fi

  echo "[privacy] ${DATANAME} / ${method}" | tee -a "${SUMMARY}"

  # Make sure Python can see local packages (src, baselines, domias, etc.)
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

  # 1) Distance to Closest Record (DCR)
  dcr_line="$(python "${REPO_ROOT}/eval/eval_dcr.py" --dataname "${DATANAME}" --model "${method}" | tail -n 1)"
  # Expected format: "<dataname>-<model>, DCR Score = <value>"
  dcr_score="$(echo "${dcr_line}" | awk -F '=' '{print $2}' | tr -d ' ')"
  if [[ -z "${dcr_score}" ]]; then
    echo "  [dcr] failed to parse DCR score from: ${dcr_line}" | tee -a "${SUMMARY}"
  else
    out_dir="${REPO_ROOT}/eval/privacy/${DATANAME}"
    mkdir -p "${out_dir}"
    echo "${dcr_score}" > "${out_dir}/${method}_dcr.txt"
    echo "  [dcr] score=${dcr_score}" | tee -a "${SUMMARY}"
  fi

  # 2) DOMIAS membership inference attack
  domias_line="$(python "${REPO_ROOT}/eval/eval_domias.py" \
    --dataname "${DATANAME}" \
    --model "${method}" \
    --mem_set_size "${DOMIAS_MEM}" \
    --reference_set_size "${DOMIAS_REF}" \
    --synthetic_size "${DOMIAS_SYN}" \
    --density_estimator "${DOMIAS_DENSITY}" \
    --training_epochs "${DOMIAS_EPOCHS}" \
    | tail -n 1)"
  echo "  [domias] ${domias_line}" | tee -a "${SUMMARY}"
done

echo "" | tee -a "${SUMMARY}"
echo "Done. Summary at ${SUMMARY}" | tee -a "${SUMMARY}"