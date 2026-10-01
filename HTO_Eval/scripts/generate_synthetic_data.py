#!/usr/bin/env python3
"""Generate a small synthetic dataset matching the real schema, so the
skeleton pipeline is runnable end-to-end before real data exists.

Delete sample_data/synthetic_patients.csv (and point configs at your real
file) once you have real data -- nothing else needs to change.
"""
import json
import random

import pandas as pd

random.seed(42)

DX_CODES = ["DX_9_4019", "DX_9_25000", "DX_10_I10", "DX_10_E119", "DX_10_I509", "DX_9_2720"]
MED_CODES = ["MED_METFORMIN", "MED_AMLODIPINE", "MED_LISINOPRIL", "MED_HYDROCHLOROTHIAZIDE", "MED_ATORVASTATIN"]

N_PATIENTS = 500
POS_RATE = 0.28  # matches the ~72/28 real-world distribution described


def make_sequence(is_positive: bool) -> list:
    n_tokens = random.randint(4, 14)
    tokens = []
    for _ in range(n_tokens):
        if random.random() < 0.55:
            tokens.append(random.choice(DX_CODES))
        else:
            tokens.append(random.choice(MED_CODES))
    # Give a mild, noisy signal correlated with the label so a baseline
    # model has something non-trivial to learn on synthetic data.
    if is_positive:
        tokens.append(random.choice(["MED_LISINOPRIL", "MED_AMLODIPINE"]))
    else:
        tokens.append(random.choice(["DX_9_4019", "DX_10_I10"]))
    return tokens


def main():
    rows = []
    for i in range(N_PATIENTS):
        is_positive = random.random() < POS_RATE
        rows.append(
            {
                "patient_id": f"SYNTHETIC_{i:04d}",
                "sequence": json.dumps(make_sequence(is_positive)),
                "label": int(is_positive),
            }
        )
    df = pd.DataFrame(rows)
    out_path = "sample_data/synthetic_patients.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic patients to {out_path}")
    print(df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
