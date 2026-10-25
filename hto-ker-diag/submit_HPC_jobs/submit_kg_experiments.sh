#!/usr/bin/env bash
# Fan out the KG diagnosis-tower experiment matrix: every config x every seed, one SLURM job each.
#
# Usage:
#   bash submit_HPC_jobs/submit_kg_experiments.sh <DATASET> <KG_TAG>
#   DRY_RUN=1 bash submit_HPC_jobs/submit_kg_experiments.sh full full     # print the jobs only
# Env:
#   SEEDS="42 43 44 45 46"   EPOCHS, BATCH_SIZE, LR, D_MODEL, HTO_ROOT, VENV (forwarded to run_one.sh)
#
# Needs, under processed_data/: <DATASET>/ (build_data.py) and, for each KG tag used below,
# kg/<tag>/transr.pt (build_kg.py) and <DATASET>/patient_graphs.<tag>.pkl (build_patient_graphs.py):
#   <KG_TAG>           the real graph
#   <KG_TAG>_isa       build_kg.py --isa-only
#   <KG_TAG>_rewired   build_kg.py --rewire-seed 1
# Results: predictions/<DATASET>/<baseline>_<config>_s<seed>/metrics.json
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:?usage: submit_kg_experiments.sh <DATASET> <KG_TAG>}"
KG="${2:?usage: submit_kg_experiments.sh <DATASET> <KG_TAG>}"
SEEDS="${SEEDS:-42 43 44 45 46}"
mkdir -p logs

# config name | baseline | run.py KG flags
CONFIGS=(
  "baseline|htr_former_ca|"
  "kg|htr_former_kg|--kg-tag ${KG}"
  "kg_add|htr_former_kg|--kg-tag ${KG} --kg-fusion add"
  "kg_no_time|htr_former_kg|--kg-tag ${KG} --kg-time off"
  "kg_tau_shuffled|htr_former_kg|--kg-tag ${KG} --kg-tau shuffled"
  "kg_no_transr|htr_former_kg|--kg-tag ${KG} --kg-no-transr"
  "kg_isa|htr_former_kg|--kg-tag ${KG}_isa"
  "kg_rewired|htr_former_kg|--kg-tag ${KG}_rewired"
)

n=0
for cfg in "${CONFIGS[@]}"; do
  IFS="|" read -r name baseline kg_args <<<"${cfg}"
  for seed in ${SEEDS}; do
    suffix="${name}_s${seed}"
    # KG_ARGS contains spaces, so it travels in the environment (--export=ALL), not in the --export list
    exports="ALL,BASELINE=${baseline},DATASET=${DATASET},SEED=${seed},RUN_SUFFIX=${suffix}"
    if [[ -n "${DRY_RUN:-}" ]]; then
      echo "KG_ARGS=\"${kg_args}\" sbatch --job-name=hto_${suffix} --export=${exports} submit_HPC_jobs/run_one.sh"
    else
      KG_ARGS="${kg_args}" sbatch --job-name="hto_${suffix}" --export="${exports}" submit_HPC_jobs/run_one.sh
    fi
    n=$((n + 1))
  done
done
echo "${n} jobs"
