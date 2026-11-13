"""Labs and vitals for the context graph and the treatment block.

A fixed panel (resources/measurement_panel.tsv) selects clinically relevant measurements: MIMIC lab_key
(+ fluid) for labs, vital_key for vitals (OMR and ICU blood pressure, BMI, weight, height). Each has
optional low / high thresholds; a value below low is state "low", above high is "high". A state names the
HPO phenotype and the PrimeKG disease it indicates (resolved by name at build time, never guessed).

Graph codes (observed in the patient graph like diagnosis codes):
  meas:<measure>            every measurement
  meas:<measure>:<state>    every abnormal measurement
Value features for a measurement node, from rows strictly before t0:
  value_attr = [has_value, z(last value), z(last - previous value), abnormal (-1 low / 0 / +1 high)]
  z uses a robust centre / scale (median, IQR / 1.349) fitted on training patients' pre-t0 values.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RESOURCES = Path(__file__).resolve().parent / "resources"
VALUE_DIM = 4
VALUE_FEATURES = ["has_value", "last_value_z", "delta_z", "abnormal"]
Z_CLIP = 5.0


@dataclass
class MeasurementConfig:
    condition_window_days: float = 365.0   # Stage 3: the last value in this window before t0 sets the state


def _split(v) -> list[str]:
    return [s.strip() for s in str(v).split("|") if s.strip()] if pd.notna(v) and str(v).strip() else []


def load_panel(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or RESOURCES / "measurement_panel.tsv", sep="\t", dtype=str, keep_default_na=False)
    for c in ("low", "high"):
        df[c] = pd.to_numeric(df[c].replace("", np.nan))
    for c in ("keys", "low_phenotype", "high_phenotype", "low_disease", "high_disease"):
        df[c] = df[c].map(_split)
    if df["measure"].duplicated().any():
        raise ValueError("measurement panel has duplicate measures")
    return df.reset_index(drop=True)


def measure_key(measure: str, state: str | None = None) -> str:
    return f"meas:{measure}" if not state else f"meas:{measure}:{state}"


def load_measurements(mimic_dir: Path, panel: pd.DataFrame) -> pd.DataFrame:
    """patient_id, visit_id, time, measure, value (numeric rows of the panel only)."""
    import pyarrow.parquet as pq
    mimic_dir = Path(mimic_dir)
    out = []
    labs = panel[panel["source"] == "lab"]
    if len(labs):
        lab_of = {(k, fl): m for m, keys, fl in zip(labs["measure"], labs["keys"], labs["fluid"]) for k in keys}
        t = pq.read_table(mimic_dir / "final_labs.parquet",
                          columns=["patient_id", "visit_id", "time", "lab_key", "fluid", "lab_value"],
                          filters=[("lab_key", "in", sorted({k for k, _ in lab_of}))])
        df = t.replace_schema_metadata(None).to_pandas()
        df["measure"] = [lab_of.get((k, fl)) for k, fl in zip(df["lab_key"], df["fluid"])]
        df["value"] = pd.to_numeric(df["lab_value"], errors="coerce")
        out.append(df[["patient_id", "visit_id", "time", "measure", "value"]])
    vit = panel[panel["source"] == "vital"]
    if len(vit):
        vital_of = {k: m for m, keys in zip(vit["measure"], vit["keys"]) for k in keys}
        t = pq.read_table(mimic_dir / "final_vitals.parquet",
                          columns=["patient_id", "visit_id", "time", "vital_key", "vital_value"],
                          filters=[("vital_key", "in", sorted(vital_of))])
        df = t.replace_schema_metadata(None).to_pandas()
        df["measure"] = df["vital_key"].map(vital_of)
        df["value"] = pd.to_numeric(df["vital_value"], errors="coerce")
        out.append(df[["patient_id", "visit_id", "time", "measure", "value"]])
    if not out:
        return pd.DataFrame(columns=["patient_id", "visit_id", "time", "measure", "value"])
    m = pd.concat(out, ignore_index=True).dropna(subset=["measure", "value", "time"])
    m["time"] = pd.to_datetime(m["time"])
    return m.reset_index(drop=True)


def add_states(m: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Adds state: 'low', 'high' or ''."""
    p = panel.set_index("measure")
    low, high = m["measure"].map(p["low"]), m["measure"].map(p["high"])
    state = np.where(m["value"] < low, "low", np.where(m["value"] > high, "high", ""))
    return m.assign(state=state)


def before_t0(m: pd.DataFrame, index_time: pd.Series) -> pd.DataFrame:
    t0 = pd.to_datetime(m["patient_id"].map(index_time))
    return m[pd.to_datetime(m["time"]) < t0]


def measurement_codes(m: pd.DataFrame) -> pd.DataFrame:
    """patient_id, visit_id, time, code: one meas:<measure> row per measurement, plus one
    meas:<measure>:<state> row per abnormal measurement. visit_id falls back to the calendar day."""
    m = m.copy()
    day = pd.to_datetime(m["time"]).dt.strftime("d%Y%m%d")
    v = m["visit_id"]
    if pd.api.types.is_float_dtype(v):
        v = v.astype("Int64")
    m["visit_id"] = v.astype("string").fillna(day)
    base = m.assign(code="meas:" + m["measure"])
    ab = m[m["state"] != ""]
    ab = ab.assign(code="meas:" + ab["measure"] + ":" + ab["state"])
    return pd.concat([base, ab], ignore_index=True)[["patient_id", "visit_id", "time", "code"]]


def training_patient_ids(data_dir: Path, mimic_dir: Path) -> tuple[set[int] | None, str]:
    """Training patients for fitting the value scale: <data_dir>/split.json, else the extraction's
    tables/splits.csv next to mimic_data. Returns (ids or None, source)."""
    path = Path(data_dir) / "split.json"
    if path.exists():
        from hto.data.splits import load_split
        return set(load_split(path)["train_ids"]), str(path)
    csv = Path(mimic_dir).parent / "tables" / "splits.csv"
    if csv.exists():
        s = pd.read_csv(csv, usecols=["patient_id", "split"])
        return set(s.loc[s["split"].astype(str) == "train", "patient_id"].astype(int)), str(csv)
    return None, "none (all patients)"


def fit_value_scale(m_pre: pd.DataFrame, patient_ids=None) -> dict[str, tuple[float, float]]:
    """measure -> (median, IQR / 1.349), from pre-t0 rows of the given (training) patients."""
    sub = m_pre if patient_ids is None else m_pre[m_pre["patient_id"].isin(set(patient_ids))]
    out = {}
    for meas, v in sub.groupby("measure")["value"]:
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        scale = (q3 - q1) / 1.349
        if not scale > 0:
            scale = float(v.std()) if v.std() > 0 else 1.0
        out[meas] = (float(med), float(scale))
    return out


def last_values(m_pre: pd.DataFrame) -> pd.DataFrame:
    """Per (patient_id, measure): last and previous value, last time, count and last state (pre-t0 rows)."""
    s = m_pre.sort_values(["patient_id", "measure", "time"], kind="stable")
    g = s.groupby(["patient_id", "measure"], sort=False)
    s = s.assign(prev_value=g["value"].shift(1), n=g["value"].transform("size"))
    last = s.groupby(["patient_id", "measure"], sort=False).tail(1)
    out = pd.DataFrame({"patient_id": last["patient_id"], "measure": last["measure"],
                        "last_value": last["value"], "last_time": last["time"],
                        "last_state": last["state"] if "state" in last else "", "n": last["n"],
                        "prev_value": last["prev_value"]})
    return out.reset_index(drop=True)


def value_features(last: pd.DataFrame, scale: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """last_values output -> patient_id, measure, has_value, last_value_z, delta_z, abnormal."""
    c = last["measure"].map(lambda k: scale.get(k, (0.0, 1.0))[0])
    s = last["measure"].map(lambda k: scale.get(k, (0.0, 1.0))[1])
    z = ((last["last_value"] - c) / s).clip(-Z_CLIP, Z_CLIP)
    dz = ((last["last_value"] - last["prev_value"]) / s).clip(-Z_CLIP, Z_CLIP).fillna(0.0)
    ab = last["last_state"].map({"low": -1.0, "high": 1.0}).fillna(0.0)
    return pd.DataFrame({"patient_id": last["patient_id"], "measure": last["measure"], "has_value": 1.0,
                         "last_value_z": z.astype(float), "delta_z": dz.astype(float), "abnormal": ab})


def current_states(m_pre: pd.DataFrame, index_time: pd.Series, window_days: float) -> pd.DataFrame:
    """patient_id, measure, state, time: measurements whose last value within window_days before t0 is
    abnormal (the patient's current abnormal state)."""
    last = last_values(m_pre)
    t0 = pd.to_datetime(last["patient_id"].map(index_time))
    recent = (t0 - pd.to_datetime(last["last_time"])) <= pd.Timedelta(days=window_days)
    out = last[recent & (last["last_state"] != "")]
    return out.rename(columns={"last_state": "state", "last_time": "time"})[["patient_id", "measure", "state", "time"]]
