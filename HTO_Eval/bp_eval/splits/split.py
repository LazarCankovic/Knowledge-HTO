"""Patient-level stratified 80/10/10 split, created once and reused forever.

The whole point of this module is that `train.py` for model #1 and
`train.py` for model #37 read the exact same three ID lists, so every
model in the registry.csv is evaluated on the identical held-out test set.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_VAL_SIZE = 0.10
DEFAULT_TEST_SIZE = 0.10
DEFAULT_RANDOM_STATE = 42


@dataclass
class SplitIds:
    split_id: str
    train_ids: List
    val_ids: List
    test_ids: List


def _compute_split_id(train_ids, val_ids, test_ids, random_state) -> str:
    """A short deterministic fingerprint of the split, used as the
    'split identifier' in the run record. If the underlying patient set
    or random_state ever changes, this hash changes too, so a stale split
    can never be silently reused across incompatible datasets.
    """
    payload = json.dumps(
        {
            "train": sorted(map(str, train_ids)),
            "val": sorted(map(str, val_ids)),
            "test": sorted(map(str, test_ids)),
            "random_state": random_state,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _write_ids(split_dir: Path, name: str, ids: List) -> None:
    with open(split_dir / f"{name}_ids.json", "w") as f:
        json.dump([str(i) for i in ids], f, indent=2)


def _read_ids(split_dir: Path, name: str) -> List[str]:
    with open(split_dir / f"{name}_ids.json") as f:
        return json.load(f)


def get_or_create_split(
    df: pd.DataFrame,
    split_dir: str,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> SplitIds:
    """Return the persisted train/val/test patient-ID split, creating it
    on first call and reusing it on every call after that.

    Stratifies on `label` so the ~72/28 class balance is preserved in all
    three splits. Operates at the patient level (df already has exactly
    one row per patient_id).
    """
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)

    train_file = split_path / "train_ids.json"
    val_file = split_path / "val_ids.json"
    test_file = split_path / "test_ids.json"

    if train_file.exists() and val_file.exists() and test_file.exists():
        train_ids = _read_ids(split_path, "train")
        val_ids = _read_ids(split_path, "val")
        test_ids = _read_ids(split_path, "test")

        current_ids = set(df["patient_id"].astype(str))
        split_ids_all = set(train_ids) | set(val_ids) | set(test_ids)

        new_patients = current_ids - split_ids_all
        missing_patients = split_ids_all - current_ids
        if new_patients:
            print(
                f"[split] WARNING: {len(new_patients)} patient(s) in the current "
                "dataset are not in the saved split (added to the dataset after "
                "the split was created). They will be excluded from this run "
                "so the held-out test set stays untouched. Regenerate the split "
                "deliberately (delete the split_dir files) if you want to include them."
            )
        if missing_patients:
            print(
                f"[split] WARNING: {len(missing_patients)} patient(s) from the saved "
                "split are no longer present in the current dataset."
            )

        split_id = _compute_split_id(train_ids, val_ids, test_ids, random_state)
        return SplitIds(split_id, train_ids, val_ids, test_ids)

    # No split saved yet -> create it once, deterministically, and persist it.
    labels = df["label"].values
    ids = df["patient_id"].astype(str).values

    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        ids, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    # val_size is a fraction of the *original* dataset; rescale relative to
    # the remaining train_val portion.
    relative_val_size = val_size / (1.0 - test_size)
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids,
        train_val_labels,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_labels,
    )

    _write_ids(split_path, "train", train_ids)
    _write_ids(split_path, "val", val_ids)
    _write_ids(split_path, "test", test_ids)

    split_id = _compute_split_id(train_ids, val_ids, test_ids, random_state)
    with open(split_path / "split_meta.json", "w") as f:
        json.dump(
            {
                "split_id": split_id,
                "random_state": random_state,
                "val_size": val_size,
                "test_size": test_size,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "n_test": len(test_ids),
            },
            f,
            indent=2,
        )

    return SplitIds(split_id, list(train_ids), list(val_ids), list(test_ids))


def apply_split(df: pd.DataFrame, split: SplitIds):
    """Slice a dataframe into (train_df, val_df, test_df) using saved IDs."""
    pid = df["patient_id"].astype(str)
    train_df = df.loc[pid.isin(split.train_ids)].reset_index(drop=True)
    val_df = df.loc[pid.isin(split.val_ids)].reset_index(drop=True)
    test_df = df.loc[pid.isin(split.test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df
