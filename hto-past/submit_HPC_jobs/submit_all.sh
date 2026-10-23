#!/bin/bash
# Submit every (baseline × dataset) combination as a separate SLURM job.
#
# Default baseline set = original 7 + 7 multidomain + 2 irregular-ts = 16 jobs per dataset.
#
# Usage:
#   bash submit_HPC_jobs/submit_all.sh [DATASETS ...]
# Env overrides:
#   BASELINES="medbert tlstm ..."    # override baseline list
#   EPOCHS=15 BATCH_SIZE=64 LR=2e-4 D_MODEL=256
#   PE_PARTS="abs,gap,domain,interaction"
#   REL_DAY_BIAS=on|off
#   RUN_SUFFIX=...                   # adds suffix to output dir
#   PE_ABLATION=1                    # if set, also submit reduced-PE variants
#                                    # (full, no_interaction, no_domain, no_gap, abs_only)

set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(full)
fi

DEFAULT_BASELINES="medbert tlstm time2vec transformehr behrt cehrbert multimodal_behrt \
                   medbert_md tlstm_md time2vec_md transformehr_md behrt_md cehrbert_md multimodal_behrt_md \
                   mtan strats"

BASELINES="${BASELINES:-${DEFAULT_BASELINES}}"

mkdir -p logs

SUBMITTED=()
submit() {
  local BL="$1"; local DS="$2"; local PE="$3"; local RB="$4"; local SFX="$5"
  local JOBNAME="hto_${BL}_${DS}${SFX:+_$SFX}"
  local EXPORTS="ALL,BASELINE=${BL},DATASET=${DS},PE_PARTS=${PE},REL_DAY_BIAS=${RB}"
  if [[ -n "${SFX}" ]]; then
    EXPORTS="${EXPORTS},RUN_SUFFIX=${SFX}"
  fi
  for var in EPOCHS BATCH_SIZE LR D_MODEL HTO_ROOT VENV; do
    if [[ -n "${!var:-}" ]]; then
      EXPORTS="${EXPORTS},${var}=${!var}"
    fi
  done
  echo "sbatch --job-name=${JOBNAME} --export=${EXPORTS}"
  local JID=$(sbatch --parsable \
                     --job-name="${JOBNAME}" \
                     --export="${EXPORTS}" \
                     submit_HPC_jobs/run_one.sh)
  echo "  -> jobid ${JID}"
  SUBMITTED+=("${JID} ${JOBNAME}")
}

for DS in "${DATASETS[@]}"; do
  if [[ ! -d "processed_data/${DS}" ]]; then
    echo "!! Skipping ${DS}: processed_data/${DS} does not exist on submit host"
    continue
  fi

  for BL in ${BASELINES}; do
    # headline run with full composite PE + rel-day bias on
    submit "${BL}" "${DS}" "${PE_PARTS:-abs,gap,domain,interaction}" "${REL_DAY_BIAS:-on}" "${RUN_SUFFIX:-}"
  done

  # optional PE ablation sweep (only meaningful for *_md baselines)
  if [[ "${PE_ABLATION:-0}" == "1" ]]; then
    for BL in ${BASELINES}; do
      if [[ "${BL}" != *_md ]]; then continue; fi
      submit "${BL}" "${DS}" "abs,gap,domain"       "on"  "no_interaction"
      submit "${BL}" "${DS}" "abs,gap"              "on"  "no_domain"
      submit "${BL}" "${DS}" "abs"                  "on"  "abs_only"
      submit "${BL}" "${DS}" "abs,gap,domain,interaction" "off" "no_rel_bias"
    done
  fi
done

echo
echo "Submitted ${#SUBMITTED[@]} jobs:"
printf '  %s\n' "${SUBMITTED[@]}"
