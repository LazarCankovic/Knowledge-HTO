"""Baseline model: TF-IDF over the token sequence + Logistic Regression.

This is the first model to run through the harness end-to-end, proving the
pipeline (split -> fit -> threshold-tune -> eval -> log) works before any
sequence model (LSTM/BERT) is plugged in.
"""
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from bp_eval.models.base import BaseBPModel


def _join_tokens(sequence) -> str:
    # TF-IDF operates on whitespace-joined token strings; DX_*/MED_* tokens
    # are already atomic so a simple space-join + whitespace tokenizer is
    # enough (no free-text tokenization needed).
    return " ".join(sequence)


def _identity_preprocessor(x):
    # Module-level (not a lambda) so the fitted vectorizer can be pickled
    # by joblib for save()/load().
    return x


class LogisticRegressionBaseline(BaseBPModel):
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        ngram_range=(1, 1),
        min_df: int = 1,
        **hyperparams,
    ):
        super().__init__(
            C=C, max_iter=max_iter, ngram_range=tuple(ngram_range), min_df=min_df, **hyperparams
        )
        self.C = C
        self.max_iter = max_iter
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[LogisticRegression] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> "LogisticRegressionBaseline":
        texts = train_df["sequence"].apply(_join_tokens)

        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=_identity_preprocessor,
            token_pattern=None,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        X_train = self.vectorizer.fit_transform(texts)
        y_train = train_df["label"].values

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=class_weight,  # None or {0: w0, 1: w1} from train only
        )
        self.model.fit(X_train, y_train)
        # No early stopping for plain LR; val_df is intentionally unused
        # here and left to the harness for threshold tuning.
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model has not been fit yet.")
        texts = df["sequence"].apply(_join_tokens)
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model, "hyperparams": self.hyperparams}, path)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionBaseline":
        payload = joblib.load(path)
        instance = cls(**payload["hyperparams"])
        instance.vectorizer = payload["vectorizer"]
        instance.model = payload["model"]
        return instance
