"""STraTS — Self-supervised Transformer for Time-Series (Tipirneni & Reddy, KDD 2022).

Core idea: represent every observation as a (feature, value, time) triplet
and encode the triplet additively; then a set-transformer-style self-
attention with a learned [CLS] produces the patient embedding. STraTS does
not care about order — it cares about what was observed and when.

We adapt this to multidomain_visit by treating every event (diag code, med,
lab bin with numeric value, vital bin with numeric value) as a triplet:
  - feature  = token id
  - value    = numeric (0 for symbolic events)
  - time     = day offset

Continuous value embedding (for lab/vital) uses Time2Vec-style Fourier on
z-scored values. Time uses Time2VecDays. Symbolic events (value_mask = 0)
skip the value term.
"""
from __future__ import annotations

from typing import List

import math
import torch
import torch.nn as nn

from .composite_pe import Time2VecDays
from .multidomain_batcher import build_flat_batch


class _FourierFeature(nn.Module):
    def __init__(self, k: int, init_scale: float = 1.0):
        super().__init__()
        omegas = torch.linspace(0.1, 4.0, steps=k - 1) * init_scale
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(omegas)
        self.b = nn.Parameter(torch.zeros(k - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        v0 = self.w0 * x + self.b0
        vi = torch.sin(self.w * x + self.b)
        return torch.cat([v0, vi], dim=-1)


class STraTS(nn.Module):
    name = "strats"
    needs_variant = "multidomain_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 1024, dropout: float = 0.1,
                 t2v_k: int = 32, val_k: int = 32):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.domain_emb = nn.Embedding(5, d_model)
        self.time_f = Time2VecDays(k=t2v_k)
        self.time_proj = nn.Linear(t2v_k, d_model)
        self.val_f = _FourierFeature(k=val_k)
        self.val_proj = nn.Linear(val_k, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 2),
        )
        self.register_buffer("day_scale", torch.tensor(365.0))
        self.register_buffer("num_mean", torch.zeros(1))
        self.register_buffer("num_std", torch.ones(1))
        self._stats_fit = False

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        return build_flat_batch(items, max_len=self.max_len)

    def _fit_stats(self, batch):
        if self._stats_fit or not self.training:
            return
        d = batch["day"][batch["mask"] > 0.5]
        if d.numel() > 32:
            self.day_scale.copy_(d.max().detach().clamp_min(1.0))
        v = batch["num"][batch["num_mask"] > 0.5]
        if v.numel() > 32:
            self.num_mean.copy_(v.mean().detach())
            self.num_std.copy_(v.std().clamp_min(1e-3).detach())
            self._stats_fit = True

    def forward(self, batch) -> torch.Tensor:
        self._fit_stats(batch)
        B = batch["ids"].size(0)
        feat_e = self.tok_emb(batch["ids"]) + self.domain_emb(batch["domain_id"])

        t_norm = batch["day"] / self.day_scale.clamp_min(1.0)
        time_e = self.time_proj(self.time_f(t_norm))

        val_z = (batch["num"] - self.num_mean) / self.num_std
        val_e = self.val_proj(self.val_f(val_z)) * batch["num_mask"].unsqueeze(-1)

        x = feat_e + time_e + val_e
        # prepend CLS
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        key_pad = torch.cat([torch.zeros(B, 1, device=x.device), 1 - batch["mask"]], dim=1) < 0.5
        # key_pad: True means PAD. But we built it as `0 means pad`. Fix:
        cls_col = torch.ones(B, 1, device=x.device)
        mask2 = torch.cat([cls_col, batch["mask"]], dim=1)
        key_pad = mask2 < 0.5
        h = self.enc(x, src_key_padding_mask=key_pad)
        return self.cls(self.drop(h[:, 0, :]))
