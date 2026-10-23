"""Build canonical per-patient records from mimic_data/*.parquet.

Canonical record:
    {
      "patient_id": int,
      "label": int,
      "target_med": str,
      "bps": float, "bpd": float, "latest_bp_time": pd.Timestamp,
      "visits": [
          {
            "day_offset": float,   # days since patient's first observed event
            "diag_codes": [str, ...],
            "med_codes":  [str, ...],
            "lab_num":    [(lab_key, lab_value), ...],
            "vital_num":  [(vital_key, vital_value), ...],
          }, ...
      ],
    }

Visits are grouped by (patient_id, visit_id). Rows with NULL visit_id get
grouped by day bucket so they still form coherent events.

We support a `n_patients` sample for smoke tests — if set, we keep a
stratified (by label) sample of that many patients and filter every event
parquet to just those patient ids before aggregating.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MIMIC_DIR = Path(__file__).resolve().parents[3] / "mimic_data"
PROCESSED_DIR = Path(__file__).resolve().parents[3] / "processed_data"


def _load_patients() -> pd.DataFrame:
    return pd.read_parquet(MIMIC_DIR / "final_patients_table.parquet")


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    g1 = df[df["label"] == 1]
    g0 = df[df["label"] == 0]
    frac = min(1.0, n / len(df))
    s1 = g1.sample(n=min(len(g1), max(1, int(round(len(g1) * frac)))), random_state=seed)
    s0 = g0.sample(n=min(len(g0), max(1, int(round(len(g0) * frac)))), random_state=seed)
    out = pd.concat([s1, s0]).sort_values("patient_id").reset_index(drop=True)
    # trim or pad to exactly n if possible
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).sort_values("patient_id").reset_index(drop=True)
    return out


def _filter_to_patients(df: pd.DataFrame, keep_ids: set) -> pd.DataFrame:
    return df[df["patient_id"].isin(keep_ids)].reset_index(drop=True)


def _visit_key(df: pd.DataFrame) -> pd.Series:
    """Visit key: visit_id if present, else patient_id + day bucket."""
    vid = df["visit_id"].astype("Int64")
    day = df["time"].dt.floor("D").astype("int64")  # ns
    fallback = "NA_" + df["patient_id"].astype(str) + "_" + day.astype(str)
    return np.where(vid.notna(), "V_" + vid.astype(str), fallback)


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def build_canonical(
    n_patients: Optional[int] = None,
    out_path: Optional[Path] = None,
    seed: int = 42,
) -> list[dict]:
    patients = _load_patients()
    if n_patients is not None and n_patients < len(patients):
        patients = _stratified_sample(patients, n=n_patients, seed=seed)
    keep_ids = set(patients["patient_id"].astype("int64").tolist())

    diag = _filter_to_patients(pd.read_parquet(MIMIC_DIR / "final_diagnoses.parquet"), keep_ids)
    meds = _filter_to_patients(pd.read_parquet(MIMIC_DIR / "final_medications.parquet"), keep_ids)
    labs = _filter_to_patients(pd.read_parquet(MIMIC_DIR / "final_labs.parquet"), keep_ids)
    vit  = _filter_to_patients(pd.read_parquet(MIMIC_DIR / "final_vitals.parquet"),   keep_ids)

    # first-event time per patient (for day_offset)
    firsts = pd.concat([
        diag[["patient_id", "time"]],
        meds[["patient_id", "time"]],
        labs[["patient_id", "time"]],
        vit[["patient_id", "time"]],
    ], ignore_index=True).groupby("patient_id")["time"].min().rename("t0")
    first_time = firsts.to_dict()

    # cutoff time per patient — filter out events after latest_bp_time
    cutoff_time = patients.set_index("patient_id")["latest_bp_time"].to_dict()

    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cutoff"] = df["patient_id"].map(cutoff_time)
        df = df[df["time"] <= df["cutoff"]].reset_index(drop=True)
        df["visit_key"] = _visit_key(df)
        df["t0"] = df["patient_id"].map(first_time)
        df["day_offset"] = (df["time"] - df["t0"]).dt.total_seconds() / 86400.0
        return df

    diag = _enrich(diag)
    meds = _enrich(meds)
    labs = _enrich(labs)
    vit  = _enrich(vit)

    # numeric vital values (some are strings in source)
    vit["vital_value_num"] = pd.to_numeric(vit["vital_value"], errors="coerce")

    # Per-patient dict of visits
    records: list[dict] = []
    # index each event parquet by patient for fast row slicing
    diag_g = diag.groupby("patient_id")
    meds_g = meds.groupby("patient_id")
    labs_g = labs.groupby("patient_id")
    vit_g  = vit.groupby("patient_id")

    for _, prow in patients.iterrows():
        pid = int(prow["patient_id"])
        visits: dict[str, dict] = {}

        def _get_group(g, pid):
            try:
                return g.get_group(pid)
            except KeyError:
                return None

        gd = _get_group(diag_g, pid)
        if gd is not None:
            for _, r in gd.iterrows():
                vk = r["visit_key"]
                v = visits.setdefault(vk, {"day_offset": float(r["day_offset"]),
                                          "diag_codes": [], "med_codes": [],
                                          "lab_num": [], "vital_num": []})
                if pd.notna(r["diag"]):
                    v["diag_codes"].append(str(r["diag"]))
                v["day_offset"] = min(v["day_offset"], float(r["day_offset"]))

        gm = _get_group(meds_g, pid)
        if gm is not None:
            for _, r in gm.iterrows():
                vk = r["visit_key"]
                v = visits.setdefault(vk, {"day_offset": float(r["day_offset"]),
                                          "diag_codes": [], "med_codes": [],
                                          "lab_num": [], "vital_num": []})
                if pd.notna(r["medication"]):
                    v["med_codes"].append(str(r["medication"]))
                v["day_offset"] = min(v["day_offset"], float(r["day_offset"]))

        gl = _get_group(labs_g, pid)
        if gl is not None:
            for _, r in gl.iterrows():
                vk = r["visit_key"]
                v = visits.setdefault(vk, {"day_offset": float(r["day_offset"]),
                                          "diag_codes": [], "med_codes": [],
                                          "lab_num": [], "vital_num": []})
                val = _safe_float(r["lab_value"])
                if val is not None and pd.notna(r["lab_key"]):
                    v["lab_num"].append((str(r["lab_key"]), val))
                v["day_offset"] = min(v["day_offset"], float(r["day_offset"]))

        gv = _get_group(vit_g, pid)
        if gv is not None:
            for _, r in gv.iterrows():
                vk = r["visit_key"]
                v = visits.setdefault(vk, {"day_offset": float(r["day_offset"]),
                                          "diag_codes": [], "med_codes": [],
                                          "lab_num": [], "vital_num": []})
                val = _safe_float(r["vital_value_num"])
                if val is not None and pd.notna(r["vital_key"]):
                    v["vital_num"].append((str(r["vital_key"]), val))
                v["day_offset"] = min(v["day_offset"], float(r["day_offset"]))

        # sort visits chronologically, drop totally empty visits
        visit_list = [v for v in visits.values() if (v["diag_codes"] or v["med_codes"] or v["lab_num"] or v["vital_num"])]
        visit_list.sort(key=lambda v: v["day_offset"])

        rec = {
            "patient_id": pid,
            "label": int(prow["label"]),
            "target_med": str(prow["latest_htn_med_name"]),
            "bps": _safe_float(prow.get("bps")),
            "bpd": _safe_float(prow.get("bpd")),
            "visits": visit_list,
        }
        records.append(rec)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(records, f)

    return records


def load_canonical(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=None, help="Sample size for smoke tests")
    ap.add_argument("--out", type=str, default=None, help="Output pickle path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out) if args.out else PROCESSED_DIR / (
        f"canonical_n{args.n_patients}.pkl" if args.n_patients else "canonical_full.pkl"
    )
    print(f"Building canonical records (n_patients={args.n_patients}) -> {out}")
    recs = build_canonical(n_patients=args.n_patients, out_path=out, seed=args.seed)
    print(f"Built {len(recs)} records; mean visits/patient={np.mean([len(r['visits']) for r in recs]):.1f}")
    print(f"Label distribution: {pd.Series([r['label'] for r in recs]).value_counts().to_dict()}")
