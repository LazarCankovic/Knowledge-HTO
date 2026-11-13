#!/bin/bash
# One graph-enrichment stage per SLURM job (see README "Graph enrichment").
#
#   STAGE   tests | ctxkg | stage1 | graphs | ctxgraphs | stage2 | stage3 | eval
#           graphs = build_patient_graphs.py (adds edge_hops, needed by stage2)
#           ctxkg / ctxgraphs = labs, vitals and symptoms (build_context_kg.py, build_context_patient_graphs.py);
#           then run stage1, stage2 and eval with KG_TAG=ctx (stage3 picks the context up by itself)
# Required env vars (sbatch --export): STAGE, HTO_ROOT (this folder), VENV, MIMIC_DIR, CODES
#   ctxkg also: BASE_KG (build_lab_kg.py output), HPO (hp.json), LABITEMS (d_labitems.csv), PRIMEKG, MONDO,
#               ICD_SNOMED (default processed_data/kg/full/icd_snomed_map.parquet)
#   stage1 also: MONDO, PRIMEKG, DISEASE_FEATURES, SNOMED, GEMS (and an HF cache with the model)
#   stage3 also: RXNORM (rrf/ folder)
# Optional: DATASET=full_storetime  KG_TAG=full  HF_HOME=...  EXTRA_ARGS="..."
#
# Example (stage 1 on one GPU, then stage 3 on CPU after it):
#   sbatch -p gpuqs --gres=gpu:a100:1 -c 8 --mem=48G -t 04:00:00 --export=ALL,STAGE=stage1,... run_graph_enrichment.sh
#   sbatch -p cpuqs -c 4 --mem=48G -t 04:00:00 --dependency=afterok:<id> --export=ALL,STAGE=stage3,... run_graph_enrichment.sh
#
#SBATCH --job-name=hto-enrich
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

: "${STAGE:?set STAGE}"
: "${HTO_ROOT:?set HTO_ROOT}"
: "${VENV:?set VENV}"
: "${MIMIC_DIR:?set MIMIC_DIR}"
: "${CODES:?set CODES}"
: "${DATASET:=full_storetime}"
: "${KG_TAG:=full}"
: "${EXTRA_ARGS:=}"

cd "${HTO_ROOT}"
source "${VENV}/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p logs
read -ra EXTRA <<<"${EXTRA_ARGS}"

echo "=== stage=${STAGE} dataset=${DATASET} kg=${KG_TAG} host=$(hostname) time=$(date -Is)"
nvidia-smi -L 2>/dev/null || true

case "${STAGE}" in
  tests)
    python -m pytest tests/test_graph_enrichment.py tests/test_context_graph.py tests/test_kg.py -q -p no:cacheprovider ;;
  ctxkg)
    python build_context_kg.py --base-kg-dir "${BASE_KG:?}" \
      --icd-snomed-map "${ICD_SNOMED:-processed_data/kg/full/icd_snomed_map.parquet}" --codes "${CODES}" \
      --labitems "${LABITEMS:?}" --hpo "${HPO:?}" --primekg "${PRIMEKG:?}" --mondo "${MONDO:?}" "${EXTRA[@]}" ;;
  ctxgraphs)
    python build_context_patient_graphs.py --dataset "${DATASET}" --kg-tag "${KG_TAG}" --mimic-dir "${MIMIC_DIR}" \
      "${EXTRA[@]}" ;;
  stage1)
    python 1_extract_text_embeddings.py --codes "${CODES}" --mondo "${MONDO:?}" --primekg "${PRIMEKG:?}" \
      --disease-features "${DISEASE_FEATURES:?}" --kg-dir "processed_data/kg/${KG_TAG}" \
      --snomed-dir "${SNOMED:?}" --gems "${GEMS:?}" "${EXTRA[@]}" ;;
  graphs)
    python build_patient_graphs.py --dataset "${DATASET}" --kg-tag "${KG_TAG}" --mimic-dir "${MIMIC_DIR}" "${EXTRA[@]}" ;;
  stage2)
    python 2_enrich_patient_graphs.py --dataset "${DATASET}" --kg-tag "${KG_TAG}" --mimic-dir "${MIMIC_DIR}" "${EXTRA[@]}" ;;
  stage3)
    python 3_build_treatment_block.py --dataset "${DATASET}" --mimic-dir "${MIMIC_DIR}" \
      --rxnorm-dir "${RXNORM:?}" "${EXTRA[@]}" ;;
  eval)
    python eval_graph_enrichment.py --dataset "${DATASET}" --kg-tag "${KG_TAG}" --mimic-dir "${MIMIC_DIR}" \
      --codes "${CODES}" ;;
  *)
    echo "unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac

echo "=== done $(date -Is)"
