"""Deeper inspection: unique values, label structure, sequence field structure."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "mimic_data"

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 260)

patients = pd.read_parquet(ROOT / "final_patients_table.parquet")
print("final_patients_table columns:", list(patients.columns))
print(patients.dtypes.to_string())
print(patients.head(5).to_string())
print("n_patients:", patients["patient_id"].nunique())
print("bps stats:", patients["bps"].describe().to_string())
print("bpd stats:", patients["bpd"].describe().to_string())
print("latest_htn_med_name values:", patients["latest_htn_med_name"].value_counts(dropna=False).head(20).to_string())
print()

seq = pd.read_parquet(ROOT / "final_sequences.parquet")
print("final_sequences columns:", list(seq.columns))
print(seq.dtypes.to_string())
print("n_patients:", seq["patient_id"].nunique())
r = seq.iloc[0]
print("First row patient_id:", r["patient_id"])
print("  latest_bp_time:", r["latest_bp_time"])
print("  bps / bpd / med:", r["bps"], r["bpd"], r["latest_htn_med_name"])
print("  history type:", type(r["history"]))
if r["history"] is not None:
    try:
        h = r["history"]
        print("  history len:", len(h))
        print("  history[0]:", h[0] if len(h) else "(empty)")
    except Exception as e:
        print("  history repr:", repr(r["history"])[:500])
print("  sequence type:", type(r["sequence"]))
if r["sequence"] is not None:
    try:
        s = r["sequence"]
        print("  sequence len:", len(s))
        if len(s):
            print("  sequence[0]:", s[0])
            print("  sequence[:3]:", s[:3])
    except Exception as e:
        print("  sequence repr:", repr(r["sequence"])[:500])

print()
print("sequence lengths distribution:")
seq_lens = seq["sequence"].apply(lambda x: 0 if x is None else len(x))
print(seq_lens.describe().to_string())
print("hist_lens:")
hist_lens = seq["history"].apply(lambda x: 0 if x is None else len(x))
print(hist_lens.describe().to_string())

# check the "extra" columns (compared to patients)
extra = set(seq.columns) - set(patients.columns)
print("extra in sequences:", extra)

# final_vitals
vit = pd.read_parquet(ROOT / "final_vitals.parquet")
print("vital_key counts:")
print(vit["vital_key"].value_counts().to_string())
print("vital_value dtype sample (unique first 10):", vit["vital_value"].head(10).tolist())
