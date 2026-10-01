# Pipeline Diagram
MIMIC-IV BigQuery
        ↓
Broad hypertension cohort
ICD-10: I10/I11/I12/I13/I15/I16
ICD-9: 401–405, 4372
        ↓
Identify hypertension-treatment medications
        ↓
Select LAST observed HTO event per patient
        ↓
Require diagnosis history before/at cutoff
        ↓
Require valid BP during days 1–90 after cutoff
        ↓
Extract all diagnoses before cutoff
        ↓
Extract all medications before cutoff
        ↓
Normalize medication names
        ↓
Order visits chronologically
        ↓
DX events followed by MED events within visit
        ↓
Flatten into patient sequence
        ↓
Guarantee index HTO token is present
        ↓
Create outcome:
median SBP <120 AND median DBP <80
        ↓
Final dataset:
38,364 patients






# Command for the running the model/eval pipeline
python train.py --config configs/logistic_real.yaml

# Data Explanation
The preprocessing pipeline is executed in the MIMIC-IV Colab notebook using Google BigQuery. Cells are executed sequentially from cohort construction through diagnosis/medication extraction, sequence generation, BP outcome construction, and final CSV export.

