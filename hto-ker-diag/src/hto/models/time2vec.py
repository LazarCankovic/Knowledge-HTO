"""Time2Vec (Kazemi et al. 2019, arXiv:1907.05321).

Core idea: learn time representation t2v(t) = [omega_0 * t + phi_0,
sin(omega_i t + phi_i) for i>0]. Frequently paired with a seq model.

Here we fuse Time2Vec with a small Transformer over the flat code stream.
The time feature is the per-token `day_offset`.

Variant: codes_flat.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .common import SinusoidalPositionalEmbedding, pad_2d, pad_floats


class Time2Vec(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(k - 1) * 0.01)
        self.b = nn.Parameter(torch.randn(k - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [..., 1] or [...]
        if t.dim() and t.shape[-1] != 1:
            t = t.unsqueeze(-1)
        v0 = self.w0 * t + self.b0
        vi = torch.sin(self.w * t + self.b)
        return torch.cat([v0, vi], dim=-1)


class Time2VecModel(nn.Module):
    name = "time2vec"
    needs_variant = "codes_flat"

    def __init__(self, vocab_size: int, d_model: int = 128, t2v_k: int = 32, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_len=max_len + 2)
        self.t2v = Time2Vec(t2v_k)
        self.time_proj = nn.Linear(t2v_k, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        ids_list = [[2] + it["token_ids"] for it in items]
        day_list = [[0.0] + list(it["day_offsets"]) for it in items]
        ids, mask = pad_2d(ids_list, max_len=self.max_len)
        days, _ = pad_floats(day_list, max_len=self.max_len)
        labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        patient_ids = torch.tensor([int(it["patient_id"]) for it in items], dtype=torch.long)
        return {"ids": ids, "mask": mask, "days": days, "labels": labels, "patient_ids": patient_ids}

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ids, mask, days = batch["ids"], batch["mask"], batch["days"]
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, -1)
        # scale days by a factor before t2v to keep gradient well-conditioned
        t = days / 365.0
        tfeat = self.time_proj(self.t2v(t.unsqueeze(-1))).squeeze(-2) if t.dim() == 2 else self.time_proj(self.t2v(t))
        # time_proj output: [B, L, d_model]
        if tfeat.dim() == 4:
            tfeat = tfeat.squeeze(-2)
        x = self.tok_emb(ids) + self.pos_emb(pos) + tfeat
        x = self.dropout(x)
        h = self.encoder(x, src_key_padding_mask=mask < 0.5)
        return self.cls(h[:, 0, :])
