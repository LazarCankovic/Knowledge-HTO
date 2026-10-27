"""mTAN — Multi-Time Attention Networks (Shukla & Marlin, ICLR 2021).

Key idea: a set of R learned "reference time points" attend via multi-head
attention over the observed events, each of which carries a learnable
continuous time embedding. The R reference embeddings are then pooled and
fed to a classifier.

Input (from multidomain_visit.flat_*): for each event we have a token id, a
domain id, a scalar numeric value (0 for symbolic events), a numeric mask,
and a day offset. We embed the event as `tok_emb + domain_emb + num_proj`
and place it at day `t`. Reference times are uniformly spaced on
[0, max_day_seen] at init with their positions learnable.
"""
from __future__ import annotations

from typing import List

import math
import torch
import torch.nn as nn

from .composite_pe import Time2VecDays
from .multidomain_batcher import build_flat_batch


class _MHA_with_time_KV(nn.Module):
    """Multi-head cross attention where K carries a time embedding for each
    observed event and Q carries a time embedding for each reference point."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d = d_model
        self.dh = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_pad: torch.Tensor) -> torch.Tensor:
        B, R, _ = q.shape
        _, L, _ = k.shape
        Q = self.Wq(q).reshape(B, R, self.h, self.dh).transpose(1, 2)
        K = self.Wk(k).reshape(B, L, self.h, self.dh).transpose(1, 2)
        V = self.Wv(v).reshape(B, L, self.h, self.dh).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dh)
        scores = scores.masked_fill(key_pad.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        o = torch.matmul(attn, V)                                       # [B, h, R, dh]
        return self.out(o.transpose(1, 2).reshape(B, R, self.d))


class MTAN(nn.Module):
    name = "mtan"
    needs_variant = "multidomain_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_ref: int = 32, t2v_k: int = 32, max_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.domain_emb = nn.Embedding(5, d_model)
        self.num_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.time_embed = Time2VecDays(k=t2v_k)
        self.time_proj = nn.Linear(t2v_k, d_model)
        self.ref_times = nn.Parameter(torch.linspace(0.0, 1.0, n_ref))
        self.ref_query = nn.Parameter(torch.randn(1, n_ref, d_model) * 0.02)
        self.attn = _MHA_with_time_KV(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 2),
        )
        # running max day (to rescale ref_times into the observed range)
        self.register_buffer("day_scale", torch.tensor(365.0))
        self._stats_fit = False

        # stats for numeric
        self.register_buffer("num_mean", torch.zeros(1))
        self.register_buffer("num_std", torch.ones(1))

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
        num_z = (batch["num"] - self.num_mean) / self.num_std
        num_z = (num_z * batch["num_mask"]).unsqueeze(-1)

        # observed event representation
        ev = self.tok_emb(batch["ids"]) + self.domain_emb(batch["domain_id"]) + self.num_proj(num_z)
        t_obs = batch["day"] / self.day_scale.clamp_min(1.0)
        ev = ev + self.time_proj(self.time_embed(t_obs))

        # reference queries
        B = ev.size(0)
        ref_q = self.ref_query.expand(B, -1, -1) + self.time_proj(self.time_embed(self.ref_times)).unsqueeze(0)

        key_pad = batch["mask"] < 0.5
        attended = self.attn(ref_q, ev, ev, key_pad)
        pooled = self.norm(attended).mean(dim=1)
        return self.cls(self.dropout(pooled))
