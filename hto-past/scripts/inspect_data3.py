"""Inspect the encoded sequence strings + label distribution."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "mimic_data"

pd.set_option("display.max_columns", 60)

seq = pd.read_parquet(ROOT / "final_sequences.parquet")
print("label counts:")
print(seq["label"].value_counts(dropna=False).to_string())
print()
r = seq.iloc[0]
print("patient_id:", r["patient_id"])
print("sequence_length:", r["sequence_length"])
print()
print("--- history (first 1200 chars) ---")
print(r["history"][:1200])
print()
print("--- sequence (first 1200 chars) ---")
print(r["sequence"][:1200])
print()
print("--- sequence (last 600 chars) ---")
print(r["sequence"][-600:])
