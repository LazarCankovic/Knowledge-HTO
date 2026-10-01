"""Model-agnostic interface.

Every model — Logistic Regression today, an LSTM or BERT tomorrow — plugs
into the exact same training/eval harness by implementing this contract.
train.py never imports a specific model class directly except through the
MODEL_REGISTRY in bp_eval/models/__init__.py, so adding a new model means:
  1. subclass BaseBPModel
  2. register it in MODEL_REGISTRY
  3. point config['model_type'] at it
No changes to train.py, evaluation, or tracking are needed.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd


class BaseBPModel(ABC):
    """Common contract for every model in this project.

    Sequences are passed through as the raw list[str] of DX_*/MED_* tokens
    (df['sequence']) — each model wrapper owns its own featurization
    (e.g. TF-IDF bag-of-tokens for Logistic Regression, a token->id vocab
    + padding for an LSTM, a tokenizer for BERT).
    """

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> "BaseBPModel":
        """Fit on train_df only.

        val_df, when provided, must be used ONLY for early stopping,
        hyperparameter selection already baked into this call, or
        diagnostics logged during training — never for anything that
        touches test_df. Models with no notion of early stopping (e.g.
        plain Logistic Regression) may ignore val_df for fitting; the
        harness still uses val_df afterwards for threshold tuning.

        class_weight, when provided, must have been computed from the
        training split only (see bp_eval/utils/class_weights.py).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(label=1) for each row in df, shape (n_rows,)."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the fitted model (+ any fitted preprocessing, e.g. a
        vectorizer or tokenizer) to `path` so it can be reloaded for
        inference without retraining.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseBPModel":
        raise NotImplementedError

    def get_params(self) -> dict:
        """Hyperparameters to log in the run record's config.json."""
        return dict(self.hyperparams)
