# Poster evidence: separate diagnosis KG and lab KG models

**Status:** Four full `full_storetime` runs finished. Initial CSV values for jobs 96828 and 96829 were transcribed from rounded SLURM logs; run `collect.py` **on the HPC** before submitting the package to replace them with exact `metrics.json` values and generate aggregate diagnostics. This directory contains no MIMIC patient records.

## People and contributions

| Person | Contribution | What still needs confirming |
| --- | --- | --- |
| Lazar Cankovic | BigQuery cohort export, lab-to-LOINC/SNOMED graph and separate lab KG tower, HPC runs and poster results | Confirm desired author order and exact contribution wording |
| Diagnosis-tower teammate | Diagnosis ICD-to-SNOMED graph and separate diagnosis KG model in `diagnosis-tower-mods` | Add teammate name and confirm contribution wording |

The models are **separate**: `htr_former_kg` changes the diagnosis tower; `htr_former_lab_kg` changes the lab tower. The plain `htr_former_ca` is the common comparator. Other towers remain in the four-tower architecture.

## What the evidence supports so far

The shared `full_storetime` dataset has 38,364 patients, with train/validation/test counts 26,855/5,755/5,754. The primary metric is test AUROC, with validation AUROC used to choose the best epoch. Jobs 96731, 96828 and 96829 report approximately 0.7080 (plain), 0.7174 (additive lab KG), and 0.7062 (diagnosis KG) test AUROC, respectively. The gated lab KG job 96730 reports 0.7056. Exact figures, counts and source locations go in `evidence_table.csv` after `collect.py` runs.

**Interpretation:** additive lab KG is about 0.0094 higher than plain on this one test run, while the diagnosis KG is about 0.0018 lower. Additive lab validation AUROC (~0.7070) was below plain (~0.7098). The additive variant was inspected after the gated lab model's test results were known, so this is exploratory and should not be framed as a confirmed improvement. Accuracy and Brier also favor other runs in some comparisons. The older `hto-past` AUROC 0.7924 came from a different cohort/vocabulary and must not appear on this comparison plot.

## Setup, inputs and split

This repository folder contains `requirements.txt` and Python code under `src/hto/`. On the HPC the existing environment was `/data/nsf-jg/hto/models/myenv/bin/python`, with `PYTHONPATH=/data/nsf-jg/hto/hto2/python_extra_mpl` and `LD_LIBRARY_PATH` unset. For a new environment, see `requirements.txt` and the project `README.md`; run `python -m unittest discover -s tests` or `python -m pytest tests/test_kg.py tests/test_lab_kg.py tests/test_lab_snomed_graph.py -q` once pytest is installed.

MIMIC-IV requires credentialed access. The BigQuery export in `extraction/export_from_bigquery.py` creates patient-level parquet under `data/processed/mimic-iv-3.1/mimic_data/`, `tables/splits.csv`, `icd_inventory.csv`, and `d_labitems.csv`. The fixed split CSV was supplied to `build_data.py` and written as `processed_data/full_storetime/split.json`; vocabulary was fitted on training patients. The lab source uses `storetime` (result availability). External terminology inputs: SNOMED CT US RF2 September 2026, LOINC Ontology RF2 2.83, MIMIC item-to-LOINC mapping, and CMS 2018 ICD-9 GEM. Their original downloads and the MIMIC data are outside this evidence package and should remain in approved storage.

## Commands used (from `hto-ker-diag/`)

```bash
python build_data.py --tag full_storetime --only-allcodes \
  --mimic-dir data/processed/mimic-iv-3.1/mimic_data \
  --split-csv data/processed/mimic-iv-3.1/tables/splits.csv

# SNOMED diagnosis KG: map, graph, then TransR; see build_kg.py --help for input paths.
python build_kg.py map --tag full --codes data/processed/mimic-iv-3.1/tables/icd_inventory.csv \
  --snomed-dir /path/to/SNOMED_RF2 --gems /path/to/2018_I9gem.txt
python build_kg.py graph --tag full --codes data/processed/mimic-iv-3.1/tables/icd_inventory.csv \
  --snomed-dir /path/to/SNOMED_RF2
python build_kg.py pretrain --tag full --device cuda
python build_patient_graphs.py --dataset full_storetime --kg-tag full \
  --mimic-dir data/processed/mimic-iv-3.1/mimic_data

# Lab KG extends the diagnosis graph through MIMIC itemid -> LOINC -> SNOMED.
python build_lab_kg.py graph --tag lab_snomed --diagnosis-kg-tag full \
  --labitems data/processed/mimic-iv-3.1/tables/d_labitems.csv \
  --loinc-map /path/to/d_labitems_to_loinc.csv \
  --snomed-dir /path/to/SNOMED_RF2 --loinc-extension-dir /path/to/LOINC_RF2
python build_lab_kg.py pretrain --tag lab_snomed --device cuda
python build_lab_patient_graphs.py --dataset full_storetime --kg-tag lab_snomed \
  --mimic-dir data/processed/mimic-iv-3.1/mimic_data

python run.py --baseline htr_former_ca --dataset full_storetime --epochs 20 --batch-size 32 --seed 42 --device cuda
python run.py --baseline htr_former_lab_kg --dataset full_storetime --kg-tag lab_snomed \
  --epochs 20 --batch-size 32 --seed 42 --device cuda
python run.py --baseline htr_former_lab_kg --dataset full_storetime --kg-tag lab_snomed \
  --kg-fusion add --epochs 20 --batch-size 32 --seed 42 --device cuda --run-suffix add_s42
python run.py --baseline htr_former_kg --dataset full_storetime --kg-tag full \
  --epochs 20 --batch-size 32 --seed 42 --device cuda
```

Each full run produces `predictions/full_storetime/<name>/metrics.json` (aggregate outcomes and configuration), `predictions.npz` (patient-level predictions: **do not share**) and plots. `runs.csv` includes all known SLURM jobs and failures. `experiment_plan.csv` records completed exploratory comparisons and missing controls.

## Regenerate and package on the HPC

```bash
cd /data/nsf-jg/hto/hto2/Knowledge-HTO-lab/hto-ker-diag
env -u LD_LIBRARY_PATH PYTHONPATH=/data/nsf-jg/hto/hto2/python_extra_mpl \
  /data/nsf-jg/hto/models/myenv/bin/python poster_evidence/collect.py

# Optional: make an actual source snapshot, including local changes not yet committed.
tar -czf poster_evidence/code_snapshot.tar.gz \
  --exclude='__pycache__' run.py build_data.py build_kg.py build_lab_kg.py \
  build_patient_graphs.py build_lab_patient_graphs.py requirements.txt src tests \
  extraction/export_from_bigquery.py submit_HPC_jobs

# Contains aggregate CSV/Markdown and code metadata, with no predictions or raw MIMIC files.
tar -czf poster_evidence_upload.tar.gz poster_evidence
```

`code_manifest.txt` records the exact HPC Git HEAD, dirty/untracked status and SHA-256 per Python file. This matters because the HPC worktree includes applied patches. Keep the source snapshot as well if the assignment requires a runnable copy of the code. Review the files before distributing them.

## Diagnostic status and limits

`diagnostics/README.md` documents generated error counts, per-medication subsets, probability-profile examples, threshold sensitivity and descriptive paired bootstrap intervals. A true clinical **case example** with features and outcome is still pending team review within the approved MIMIC environment; aggregate prototypes do not replace it. No shuffled/rewired graph control or multiple training seeds were run on this cohort. Any bootstrap interval describes uncertainty among test patients for these trained models and does not correct for repeated inspection of the test set.

## Review update

The reviewed archive filters `diagnostics/subgroups_by_target.csv` to target groups with at least 30 test patients (23 groups per model remain). The included collector applies the same rule on future runs. `code_manifest.txt` records the original HPC collection-time source; see `review_notes.md` for this later collector change.
