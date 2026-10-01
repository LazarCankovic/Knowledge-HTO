"""Placeholder for a sequence-model (LSTM) implementation of BaseBPModel.

This file exists so the harness (train.py, MODEL_REGISTRY) already has a
real entry point for 'lstm' — swap the body of fit/predict_proba/save/load
for a real implementation without touching anything else in the project.

Sketch of what a real implementation needs to do:
  - fit(): build a token -> integer vocabulary from train_df['sequence']
    only (never from val/test, to avoid leakage), pad/truncate sequences
    to a fixed max length, wrap in a DataLoader, train an nn.LSTM +
    linear head with BCEWithLogitsLoss (weighted by class_weight if
    given), and use val_df each epoch for early stopping (e.g. stop when
    val loss/AUC stops improving for `patience` epochs) and to pick the
    best-performing checkpoint.
  - predict_proba(): tokenize+pad df['sequence'] with the *same* vocab
    fit on train, run the saved best checkpoint, sigmoid the logits.
  - save()/load(): persist both the vocab and the model state_dict.

Requires the optional `torch` dependency (not installed by default — see
requirements.txt) since it isn't needed until this model is actually used.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bp_eval.models.base import BaseBPModel


class LSTMModel(BaseBPModel):
    def __init__(self, hidden_size: int = 64, num_layers: int = 1, max_len: int = 256,
                 epochs: int = 20, patience: int = 3, **hyperparams):
        super().__init__(
            hidden_size=hidden_size, num_layers=num_layers, max_len=max_len,
            epochs=epochs, patience=patience, **hyperparams,
        )
        self._not_implemented_message = (
            "LSTMModel is a registered placeholder, not yet implemented. "
            "See the module docstring in bp_eval/models/lstm_stub.py for "
            "the implementation sketch (vocab from train only, pad/truncate, "
            "early stopping on val_df)."
        )

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None,
            class_weight: Optional[Dict[int, float]] = None) -> "LSTMModel":
        raise NotImplementedError(self._not_implemented_message)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(self._not_implemented_message)

    def save(self, path: str) -> None:
        raise NotImplementedError(self._not_implemented_message)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        raise NotImplementedError(
            "LSTMModel is a registered placeholder, not yet implemented."
        )
