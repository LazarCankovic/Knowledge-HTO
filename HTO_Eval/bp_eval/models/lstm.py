"""PyTorch LSTM baseline for longitudinal DX_/MED_ token sequences."""

from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from bp_eval.models.base import BaseBPModel


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

PAD_ID = 0
UNK_ID = 1


class SequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Dict[str, int],
        max_len: int
    ):
        self.sequences = df["sequence"].tolist()
        self.labels = df["label"].astype(float).tolist()

        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Keep most recent clinical history
        seq = seq[-self.max_len:]

        token_ids = [
            self.vocab.get(token, UNK_ID)
            for token in seq
        ]

        if len(token_ids) == 0:
            token_ids = [UNK_ID]

        length = len(token_ids)

        # Pad to max_len
        if length < self.max_len:
            token_ids = token_ids + (
                [PAD_ID] * (self.max_len - length)
            )

        x = torch.tensor(
            token_ids,
            dtype=torch.long
        )

        length = torch.tensor(
            length,
            dtype=torch.long
        )

        y = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )

        return x, length, y


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_layers,
        dropout
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_ID
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, input_ids, lengths):

        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # Final hidden state = patient representation
        patient_representation = hidden[-1]

        patient_representation = self.dropout(
            patient_representation
        )

        logits = self.classifier(
            patient_representation
        ).squeeze(1)

        return logits


class LSTMModel(BaseBPModel):

    def __init__(
        self,
        embedding_dim=128,
        hidden_size=128,
        num_layers=1,
        max_len=512,
        batch_size=64,
        learning_rate=0.001,
        epochs=10,
        patience=3,
        dropout=0.2,
        weight_decay=0.0,
        **hyperparams
    ):

        super().__init__(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            max_len=max_len,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            dropout=dropout,
            weight_decay=weight_decay,
            **hyperparams
        )

        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.max_len = int(max_len)
        self.batch_size = int(batch_size)

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)

        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)

        self.vocab = None
        self.model = None

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.training_history = []

    def build_vocab(self, train_df):

        # TRAIN SET ONLY
        all_tokens = sorted({
            token
            for sequence in train_df["sequence"]
            for token in sequence
        })

        vocab = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID
        }

        for token in all_tokens:
            vocab[token] = len(vocab)

        return vocab

    def build_network(self):

        self.model = LSTMClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

        self.model = self.model.to(
            self.device
        )

    def make_loader(
        self,
        df,
        shuffle=False
    ):

        dataset = SequenceDataset(
            df,
            self.vocab,
            self.max_len
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )

        return loader

    def evaluate_loss(
        self,
        df,
        criterion
    ):

        loader = self.make_loader(
            df,
            shuffle=False
        )

        self.model.eval()

        total_loss = 0
        total_samples = 0

        with torch.no_grad():

            for x, lengths, labels in loader:

                x = x.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(
                    x,
                    lengths
                )

                loss = criterion(
                    logits,
                    labels
                )

                batch_size = labels.size(0)

                total_loss += (
                    loss.item() * batch_size
                )

                total_samples += batch_size

        return (
            total_loss /
            max(total_samples, 1)
        )

    def fit(
        self,
        train_df,
        val_df=None,
        class_weight=None
    ):

        # ---------------------------------
        # Vocabulary: TRAIN ONLY
        # ---------------------------------

        self.vocab = self.build_vocab(
            train_df
        )

        self.build_network()

        train_loader = self.make_loader(
            train_df,
            shuffle=True
        )

        # ---------------------------------
        # Class imbalance handling
        # ---------------------------------

        if class_weight is not None:

            pos_weight_value = (
                class_weight[1]
                /
                class_weight[0]
            )

            pos_weight = torch.tensor(
                [pos_weight_value],
                dtype=torch.float32,
                device=self.device
            )

            print(
                f"[lstm] pos_weight="
                f"{pos_weight_value:.4f}"
            )

        else:
            pos_weight = None

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        best_state = deepcopy(
            self.model.state_dict()
        )

        best_val_loss = float("inf")

        epochs_without_improvement = 0

        print(
            f"[lstm] device={self.device}"
        )

        print(
            f"[lstm] vocab size="
            f"{len(self.vocab)}"
        )

        # ---------------------------------
        # Training loop
        # ---------------------------------

        for epoch in range(
            1,
            self.epochs + 1
        ):

            self.model.train()

            total_loss = 0
            total_samples = 0

            for (
                x,
                lengths,
                labels
            ) in train_loader:

                x = x.to(self.device)

                lengths = lengths.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                optimizer.zero_grad()

                logits = self.model(
                    x,
                    lengths
                )

                loss = criterion(
                    logits,
                    labels
                )

                loss.backward()

                # Helps stabilize LSTM training
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                batch_size = labels.size(0)

                total_loss += (
                    loss.item()
                    * batch_size
                )

                total_samples += (
                    batch_size
                )

            train_loss = (
                total_loss
                /
                total_samples
            )

            # ---------------------------------
            # Validation / early stopping
            # ---------------------------------

            if (
                val_df is not None
                and len(val_df) > 0
            ):

                val_loss = self.evaluate_loss(
                    val_df,
                    criterion
                )

                print(
                    f"[lstm] epoch={epoch:02d} "
                    f"train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f}"
                )

                self.training_history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })

                if (
                    val_loss
                    <
                    best_val_loss - 1e-5
                ):

                    best_val_loss = (
                        val_loss
                    )

                    best_state = deepcopy(
                        self.model.state_dict()
                    )

                    epochs_without_improvement = 0

                else:

                    epochs_without_improvement += 1

                    if (
                        epochs_without_improvement
                        >= self.patience
                    ):

                        print(
                            "[lstm] early stopping "
                            f"at epoch {epoch}"
                        )

                        break

            else:

                print(
                    f"[lstm] epoch={epoch:02d} "
                    f"train_loss={train_loss:.4f}"
                )

                best_state = deepcopy(
                    self.model.state_dict()
                )

        # Restore best validation checkpoint
        self.model.load_state_dict(
            best_state
        )

        return self

    def predict_proba(
        self,
        df
    ):

        if (
            self.model is None
            or self.vocab is None
        ):
            raise RuntimeError(
                "Model has not been fit."
            )

        loader = self.make_loader(
            df,
            shuffle=False
        )

        self.model.eval()

        probabilities = []

        with torch.no_grad():

            for (
                x,
                lengths,
                _
            ) in loader:

                x = x.to(self.device)

                lengths = lengths.to(
                    self.device
                )

                logits = self.model(
                    x,
                    lengths
                )

                probs = torch.sigmoid(
                    logits
                )

                probabilities.append(
                    probs.cpu().numpy()
                )

        return np.concatenate(
            probabilities
        )

    def save(
        self,
        path
    ):

        payload = {
            "hyperparams":
                self.hyperparams,

            "vocab":
                self.vocab,

            "state_dict":
                self.model.state_dict(),

            "training_history":
                self.training_history
        }

        torch.save(
            payload,
            path
        )

    @classmethod
    def load(
        cls,
        path
    ):

        payload = torch.load(
            path,
            map_location="cpu"
        )

        instance = cls(
            **payload["hyperparams"]
        )

        instance.vocab = (
            payload["vocab"]
        )

        instance.build_network()

        instance.model.load_state_dict(
            payload["state_dict"]
        )

        instance.model.eval()

        instance.training_history = (
            payload.get(
                "training_history",
                []
            )
        )

        return instance