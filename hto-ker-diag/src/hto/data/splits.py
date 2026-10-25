"""Patient-level 70/10/20 stratified split stable across dataset variants."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split


def make_split(
    patient_ids: Iterable[int],
    labels: Iterable[int],
    out_path: Path,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
) -> dict:
    pids = np.asarray(list(patient_ids), dtype=np.int64)
    y = np.asarray(list(labels), dtype=np.int64)

    # first carve out test (20%)
    tr_val, te, y_tr_val, _ = train_test_split(
        pids, y, test_size=test_size, random_state=seed, stratify=y
    )
    # then carve out val from remaining (10% of total = 12.5% of 80%)
    val_frac = val_size / (1.0 - test_size)
    tr, va, _, _ = train_test_split(
        tr_val, y_tr_val, test_size=val_frac, random_state=seed, stratify=y_tr_val
    )
    split = {
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "train_ids": sorted(int(x) for x in tr),
        "val_ids":   sorted(int(x) for x in va),
        "test_ids":  sorted(int(x) for x in te),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    return split


def split_from_csv(csv_path: Path, patient_ids: Iterable[int], out_path: Path) -> dict:
    """Write split.json from an existing patient_id,split CSV (e.g. the extraction's tables/splits.csv),
    so every consumer of the cohort uses the same split. Every patient must be assigned."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    assign = dict(zip(df["patient_id"].astype(int), df["split"].astype(str)))
    pids = [int(p) for p in patient_ids]
    missing = [p for p in pids if p not in assign]
    if missing:
        raise ValueError(f"{len(missing)} patients missing from {csv_path}, e.g. {missing[:5]}")
    bad = sorted({assign[p] for p in pids} - {"train", "val", "test"})
    if bad:
        raise ValueError(f"unknown split names in {csv_path}: {bad}")
    split = {
        "source": str(csv_path),
        "train_ids": sorted(p for p in pids if assign[p] == "train"),
        "val_ids":   sorted(p for p in pids if assign[p] == "val"),
        "test_ids":  sorted(p for p in pids if assign[p] == "test"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    return split


def load_split(path: Path) -> dict:
    with open(path) as f:
        s = json.load(f)
    s["train_ids"] = set(s["train_ids"])
    s["val_ids"]   = set(s.get("val_ids", []))
    s["test_ids"]  = set(s["test_ids"])
    return s
