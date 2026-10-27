"""Med-BERT (Rasmy et al., 2020, arXiv:2005.12833).

Original: BERT pretrained on diagnosis-code sequences with MLM + "prolonged
length of stay" auxiliary task, then fine-tuned on a downstream task. We
implement a single-task, supervised-only variant that keeps the core
architectural idea: a BERT encoder over a flat code sequence with learned
token + segment (visit) embeddings, then CLS head.

Variant: codes_flat (we additionally pass a per-token visit-index segment
embedding reconstructed from day_offsets).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .common import SinusoidalPositionalEmbedding, pad_2d, pad_floats


class MedBERT(nn.Module):
    name = "medbert"
    needs_variant = "codes_flat"

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 max_len: int = 512, dropout: float = 0.1, n_segments: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_segments = n_segments
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_len=max_len + 2)
        self.seg_emb = nn.Embedding(n_segments + 1, d_model)
        self.dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        # Prepend CLS to each token stream so we can read it out.
        ids_list = [[2] + it["token_ids"] for it in items]  # 2 = CLS id
        day_list = [[0.0] + list(it["day_offsets"]) for it in items]
        ids, mask = pad_2d(ids_list, pad_value=0, max_len=self.max_len)

        # Build visit-index "segments" from unique day_offsets (older -> 0).
        seg_ids = torch.zeros_like(ids)
        for i, (ids_i, days_i) in enumerate(zip(ids_list, day_list)):
            # map each day to a bucketed segment id in [1, n_segments]
            uniq = sorted(set(days_i))
            mapping = {d: min(k + 1, self.n_segments) for k, d in enumerate(uniq)}
            segs = [mapping[d] for d in days_i]
            segs = segs[-self.max_len:]
            seg_ids[i, : len(segs)] = torch.tensor(segs, dtype=torch.long)

        labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        patient_ids = torch.tensor([int(it["patient_id"]) for it in items], dtype=torch.long)
        return {"ids": ids, "mask": mask, "seg_ids": seg_ids, "labels": labels, "patient_ids": patient_ids}

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ids, mask, seg_ids = batch["ids"], batch["mask"], batch["seg_ids"]
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(ids) + self.pos_emb(pos) + self.seg_emb(seg_ids)
        x = self.dropout(x)
        attn_mask = mask < 0.5  # True = padding
        h = self.encoder(x, src_key_padding_mask=attn_mask)
        cls = h[:, 0, :]
        return self.cls(cls)
