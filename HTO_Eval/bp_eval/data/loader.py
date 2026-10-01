"""Load and validate the patient-level sequence dataset.

Expected schema (one row per patient):
    patient_id : str/int, unique
    sequence   : list[str] of DX_* / MED_* tokens, ordered by visit,
                 ending at the last observed HTO cutoff
    label      : int in {0, 1}
                 1 = median SBP < 120 AND median DBP < 80 during the
                     1-90 day post-HTO window, else 0

On disk the sequence column is stored as a JSON-encoded list of strings
(so the file can round-trip through CSV). Parquet/pickle inputs may store
it as a native Python list already; both are handled.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

REQUIRED_COLUMNS = ["patient_id", "sequence", "label"]
VALID_LABELS = {0, 1}


@dataclass
class LoadResult:
    """Container returned by load_dataset: the clean frame plus a record
    of anything dropped, so exclusions are never silent.
    """
    df: pd.DataFrame
    excluded: pd.DataFrame  # columns: patient_id, reason
    warnings: List[str] = field(default_factory=list)


def _parse_sequence(value):
    """Normalize a sequence cell to a python list[str]."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        # Fall back: allow a plain whitespace or comma separated token string
        if "," in stripped:
            return [t.strip() for t in stripped.split(",") if t.strip()]
        return stripped.split()
    return []


def load_dataset(data_path: str) -> LoadResult:
    """Load the patient dataframe from CSV/parquet/pickle and validate it.

    Rows are excluded (not silently dropped) when:
      - patient_id is missing or duplicated (first occurrence kept)
      - sequence is missing/empty after parsing
      - label is missing or not in {0, 1}

    Every exclusion is recorded with a reason in `excluded` so it can be
    written to excluded_patients.csv as part of the run record.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. This is a skeleton run — "
            "point config['data_path'] at your real dataset once it's ready."
        )

    if path.suffix == ".csv":
        raw = pd.read_csv(path)
    elif path.suffix in (".parquet", ".pq"):
        raw = pd.read_parquet(path)
    elif path.suffix in (".pkl", ".pickle"):
        raw = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported data file extension: {path.suffix}")

    raw = raw.reset_index(drop=True)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(
            f"Data file is missing required columns: {missing_cols}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    warnings = []
    excluded_rows = []
    keep_mask = []

    seen_ids = set()
    parsed_sequences = raw["sequence"].apply(_parse_sequence)

    for idx, row in raw.iterrows():
        pid = row["patient_id"]
        reason = None

        if pd.isna(pid) or str(pid).strip() == "":
            reason = "missing_patient_id"
        elif pid in seen_ids:
            reason = "duplicate_patient_id"
        elif len(parsed_sequences.iloc[idx]) == 0:
            reason = "empty_sequence"
        else:
            label = pd.to_numeric(
            row["label"],
            errors="coerce"
            )
            if pd.isna(label):
                reason = "missing_or_invalid_label"
            elif label not in VALID_LABELS:
                reason = f"invalid_label_value:{row['label']}"

        if reason is not None:
            excluded_rows.append({"patient_id": pid, "reason": reason})
            keep_mask.append(False)
        else:
            seen_ids.add(pid)
            keep_mask.append(True)

    clean = raw.loc[keep_mask].copy()
    clean["sequence"] = parsed_sequences.loc[keep_mask]
    clean["label"] = clean["label"].astype(int)
    clean = clean.reset_index(drop=True)

    excluded_df = pd.DataFrame(excluded_rows, columns=["patient_id", "reason"])

    if len(excluded_df) > 0:
        warnings.append(
            f"{len(excluded_df)} patient(s) excluded during load; see excluded_patients.csv"
        )

    if len(clean) == 0:
        raise ValueError("No valid rows remained after data validation/exclusion.")

    label_counts = clean["label"].value_counts(normalize=True).to_dict()
    warnings.append(f"Label distribution after cleaning: {label_counts}")

    return LoadResult(df=clean, excluded=excluded_df, warnings=warnings)
