#!/usr/bin/env bash
# Run every baseline end-to-end on a small smoke dataset.
#
# Usage:
#   bash tests/smoke_all.sh [DATASET]
# Env:
#   BASELINES=...                override the baseline list
#   EPOCHS, BS, DM                train hyperparams
#   PE_PARTS, REL_DAY_BIAS        forwarded to *_md / mtan / strats baselines
#   DEVICE=cpu|mps|cuda           default: cpu (laptop MPS lacks a transformer kernel)
#   KG_TAG=...                    also smoke htr_former_kg with this KG (needs
#                                 processed_data/<DATASET>/patient_graphs.<KG_TAG>.pkl)

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-smoke_200}"
EPOCHS="${EPOCHS:-2}"
BS="${BS:-8}"
DM="${DM:-64}"
PE_PARTS="${PE_PARTS:-abs,gap,domain,interaction}"
REL_DAY_BIAS="${REL_DAY_BIAS:-on}"
DEVICE="${DEVICE:-cpu}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

if [[ ! -d "processed_data/${DATASET}" ]]; then
  echo "[build] processed_data/${DATASET} not found — building..."
  if [[ "${DATASET}" =~ ^smoke_([0-9]+)$ ]]; then
    N="${BASH_REMATCH[1]}"
    python build_data.py --n-patients "${N}" --tag "${DATASET}" --skip-text
  else
    echo "Dataset '${DATASET}' is not a smoke build tag; please build it manually."
    exit 1
  fi
fi

DEFAULT_BASELINES=(
  # original 7 (legacy variants)
  medbert tlstm time2vec transformehr behrt cehrbert multimodal_behrt
  # multidomain-adapted 7 (composite PE + rel-day bias)
  medbert_md tlstm_md time2vec_md transformehr_md behrt_md cehrbert_md multimodal_behrt_md
  # new irregular-time baselines
  mtan strats
)

# allow override via BASELINES env var (space-separated)
if [[ -n "${BASELINES:-}" ]]; then
  read -ra BASELINES_ARR <<<"${BASELINES}"
else
  BASELINES_ARR=("${DEFAULT_BASELINES[@]}")
  if [[ -n "${KG_TAG:-}" ]]; then
    BASELINES_ARR+=(htr_former_kg)
  fi
fi

mkdir -p reports
SUMMARY="reports/${DATASET}_smoke_summary.tsv"
printf "baseline\tvariant\ttrain_sec\tauc\tauprc\tbrier\tacc\tf1\tpe_parts\trel_day_bias\n" > "${SUMMARY}"

for B in "${BASELINES_ARR[@]}"; do
  echo "==============================================================="
  echo "=== Smoke: ${B} on ${DATASET}"
  echo "==============================================================="
  START=$(date +%s)
  EXTRA_ARGS=()
  if [[ "${B}" == *_md || "${B}" == mtan || "${B}" == strats ]]; then
    EXTRA_ARGS+=(--pe-parts "${PE_PARTS}")
    if [[ "${B}" == *_md ]]; then
      EXTRA_ARGS+=(--rel-day-bias "${REL_DAY_BIAS}")
    fi
  fi
  if [[ "${B}" == htr_former_kg ]]; then
    EXTRA_ARGS+=(--kg-tag "${KG_TAG:?htr_former_kg needs KG_TAG}")
  fi
  python run.py --baseline "${B}" --dataset "${DATASET}" \
      --epochs "${EPOCHS}" --batch-size "${BS}" --d-model "${DM}" \
      --device "${DEVICE}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "reports/${DATASET}_${B}.log"
  END=$(date +%s)
  DUR=$((END - START))

  METRICS_FILE="predictions/${DATASET}/${B}/metrics.json"
  if [[ -f "${METRICS_FILE}" ]]; then
    python - <<PY >> "${SUMMARY}"
import json
m = json.load(open("${METRICS_FILE}"))
e = m.get("extra", {})
v = e.get("variant", "-")
pp = e.get("pe_parts", "-")
rb = e.get("rel_day_bias", "-")
print(f"${B}\t{v}\t${DUR}\t{m['auc']:.4f}\t{m['auprc']:.4f}\t{m['brier']:.4f}\t{m['accuracy']:.4f}\t{m['f1']:.4f}\t{pp}\t{rb}")
PY
  fi
done

echo
echo "=============================================================="
echo "Summary:"
column -t -s $'\t' "${SUMMARY}"
