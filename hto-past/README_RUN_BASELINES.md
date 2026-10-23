# HTO Baseline Runner

This folder contains the code needed to build processed HTO-style datasets and
run the registered baseline models.

## Expected Data Layout

Put your already preprocessed parquet files under:

```text
mimic_data/
  final_patients_table.parquet
  final_diagnoses.parquet
  final_medications.parquet
  final_labs.parquet
  final_vitals.parquet
  final_sequences.parquet
```

Required columns:

```text
final_patients_table.parquet: patient_id, label, latest_htn_med_name, bps, bpd
final_diagnoses.parquet: patient_id, visit_id, time, diag
final_medications.parquet: patient_id, visit_id, time, medication
final_labs.parquet: patient_id, visit_id, time, lab_key, lab_value
final_vitals.parquet: patient_id, visit_id, time, vital_key, vital_value
final_sequences.parquet: patient_id, sequence
```

`final_sequences.parquet` is optional if you build with `--skip-text`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build Processed Data

```bash
python build_data.py --tag mydata --skip-text
```

This creates:

```text
processed_data/mydata/
  canonical.pkl
  split.json
  codes_flat.train.pkl
  codes_flat.test.pkl
  codes_flat.vocab.json
  codes_visit.train.pkl
  codes_visit.test.pkl
  codes_visit.vocab.json
  multimodal.train.pkl
  multimodal.test.pkl
  multimodal.vocab.json
  multidomain_visit.train.pkl
  multidomain_visit.test.pkl
  multidomain_visit.vocab.json
```

## Run One Baseline

```bash
python run.py \
  --baseline medbert \
  --dataset mydata \
  --epochs 10 \
  --batch-size 32 \
  --lr 3e-4 \
  --d-model 128 \
  --device cuda
```

Use `--device cpu` if CUDA is unavailable.

## Run All Registered Baselines

```bash
BASELINES="medbert tlstm time2vec transformehr behrt cehrbert multimodal_behrt medbert_md tlstm_md time2vec_md transformehr_md behrt_md cehrbert_md multimodal_behrt_md mtan strats"

for B in $BASELINES; do
  EXTRA=""
  case "$B" in *_md|mtan|strats) EXTRA="$EXTRA --pe-parts abs,gap,domain,interaction" ;; esac
  case "$B" in *_md) EXTRA="$EXTRA --rel-day-bias on" ;; esac

  python run.py \
    --baseline "$B" \
    --dataset mydata \
    --epochs 10 \
    --batch-size 32 \
    --lr 3e-4 \
    --d-model 128 \
    --device cuda \
    $EXTRA
done
```

Outputs are written to:

```text
predictions/mydata/<baseline>/
  metrics.json
  predictions.npz
  confusion.png
  roc.png
  pr.png
  prob_hist.png
```

## Registered Baselines

```text
medbert
tlstm
time2vec
transformehr
behrt
cehrbert
multimodal_behrt
medbert_md
tlstm_md
time2vec_md
transformehr_md
behrt_md
cehrbert_md
multimodal_behrt_md
mtan
strats
```
