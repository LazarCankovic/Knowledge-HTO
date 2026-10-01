"""Placeholder for a transformer (BERT-style) implementation of BaseBPModel.

Same purpose as lstm_stub.py: registers 'bert' as a real MODEL_REGISTRY
entry so config-driven swapping works today; fill in the body when ready.

Sketch of what a real implementation needs to do:
  - Treat each DX_*/MED_* token as a vocabulary item for a small
    from-scratch transformer, OR map tokens to a pretrained clinical
    tokenizer's vocabulary (e.g. add DX_/MED_ tokens as special tokens)
    if fine-tuning a pretrained checkpoint (e.g. a ClinicalBERT variant).
  - fit(): tokenize+pad train_df['sequence'] (vocab/tokenizer state built
    from train only), fine-tune with a classification head, use val_df for
    early stopping / best-checkpoint selection exactly like the LSTM.
  - predict_proba(): run the same tokenizer + saved checkpoint on df,
    softmax/sigmoid the output logits.
  - save()/load(): persist tokenizer/vocab state + model weights.

Requires optional `torch` + `transformers` dependencies (not installed by
default — see requirements.txt).
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bp_eval.models.base import BaseBPModel


class BERTModel(BaseBPModel):
    def __init__(self, pretrained_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 max_len: int = 256, epochs: int = 5, patience: int = 2,
                 learning_rate: float = 2e-5, **hyperparams):
        super().__init__(
            pretrained_name=pretrained_name, max_len=max_len, epochs=epochs,
            patience=patience, learning_rate=learning_rate, **hyperparams,
        )
        self._not_implemented_message = (
            "BERTModel is a registered placeholder, not yet implemented. "
            "See the module docstring in bp_eval/models/bert_stub.py."
        )

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None,
            class_weight: Optional[Dict[int, float]] = None) -> "BERTModel":
        raise NotImplementedError(self._not_implemented_message)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(self._not_implemented_message)

    def save(self, path: str) -> None:
        raise NotImplementedError(self._not_implemented_message)

    @classmethod
    def load(cls, path: str) -> "BERTModel":
        raise NotImplementedError(
            "BERTModel is a registered placeholder, not yet implemented."
        )
