"""Class weight computation — strictly from the training split.

Never call this on val_df or test_df or the full dataframe: that would
leak information about the held-out class balance into training.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def compute_train_class_weights(train_df: pd.DataFrame) -> Dict[int, float]:
    """Balanced class weights (inverse frequency) from train_df['label'] only."""
    classes = np.array([0, 1])
    y_train = train_df["label"].values
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def maybe_compute_class_weights(train_df: pd.DataFrame, enabled: bool) -> Optional[Dict[int, float]]:
    if not enabled:
        return None
    return compute_train_class_weights(train_df)
