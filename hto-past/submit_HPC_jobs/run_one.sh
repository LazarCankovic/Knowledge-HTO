#!/bin/bash
# Template for a single SLURM job: one (baseline × dataset × PE config).
#
# Required env vars (passed via sbatch --export):
#   BASELINE   e.g. medbert_md
#   DATASET    e.g. full
#
# Optional env vars:
#   EPOCHS=10                  training epochs
#   BATCH_SIZE=32              batch size
#   LR=3e-4                    learning rate
#   D_MODEL=128                model dim
#   PE_PARTS="abs,gap,domain,interaction"   CompositePE ablation subset
#   REL_DAY_BIAS="on"|"off"    relative-day attention bias
#   RUN_SUFFIX=...             appended to predictions/<dataset>/<baseline><suffix>/
#   HTO_ROOT=/fs/atipa/data/nsf-jg/hto/models
#   VENV=myenv
#
#SBATCH --job-name=hto
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

: "${BASELINE:?set BASELINE via --export}"
: "${DATASET:?set DATASET via --export}"
: "${HTO_ROOT:=/fs/atipa/data/nsf-jg/hto/hto2}"
: "${VENV:=}"
: "${EPOCHS:=10}"
: "${BATCH_SIZE:=32}"
: "${LR:=3e-4}"
: "${D_MODEL:=128}"
: "${PE_PARTS:=abs,gap,domain,interaction}"
: "${REL_DAY_BIAS:=on}"
: "${RUN_SUFFIX:=}"

cd "${HTO_ROOT}"
if [[ -n "${VENV}" ]]; then
  source "${VENV}/bin/activate"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs predictions reports

echo "=== HTO run: baseline=${BASELINE} dataset=${DATASET} pe_parts=${PE_PARTS} rel_day_bias=${REL_DAY_BIAS} suffix=${RUN_SUFFIX}"
echo "=== host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-default} time=$(date -Is)"
nvidia-smi -L || true

EXTRA_ARGS=()
case "${BASELINE}" in
  *_md | mtan | strats)
    EXTRA_ARGS+=(--pe-parts "${PE_PARTS}")
    ;;
esac
case "${BASELINE}" in
  *_md)
    EXTRA_ARGS+=(--rel-day-bias "${REL_DAY_BIAS}")
    ;;
esac
if [[ -n "${RUN_SUFFIX}" ]]; then
  EXTRA_ARGS+=(--run-suffix "${RUN_SUFFIX}")
fi

python run.py \
    --baseline "${BASELINE}" \
    --dataset "${DATASET}" \
    --epochs  "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --d-model "${D_MODEL}" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "=== done $(date -Is)"
