"""Dump schema + head for every parquet in mimic_data/."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "mimic_data"

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 260)

for pq in sorted(ROOT.glob("*.parquet")):
    print("=" * 100)
    print(pq.name)
    df = pd.read_parquet(pq)
    print(f"shape: {df.shape}")
    print("dtypes:")
    print(df.dtypes.to_string())
    print("head:")
    print(df.head(3).to_string())
    if "subject_id" in df.columns:
        print(f"n_subjects: {df['subject_id'].nunique()}")
    for tcol in ("time", "time_days", "days", "day", "t"):
        if tcol in df.columns:
            print(f"{tcol}: min={df[tcol].min()} max={df[tcol].max()}")
    print()
