"""Composite positional encoding + relative-day attention bias.

Motivated by TransformEHR's time handling (a single `Linear(day/365)`
added to visit embeddings) but decomposed into interpretable components
that can be independently ablated:

    PE(event) = P_pos(visit_idx)                   # within-sequence position
              + T_abs(day_offset)                  # Time2Vec over days
              + T_gap(delta_prev_days)             # Time2Vec over visit gaps
              + D(domain_id)                       # domain embedding
              + DT(domain_id, day_offset)          # per-domain time modulation

Each component is gated by `parts` -- pass any subset of
{"abs", "gap", "domain", "interaction"}. The sinusoidal within-sequence
position (P_pos) is always present.

RelativeDayBias produces an additive attention bias phi(|Δd_i − Δd_j|)
shaped [B, H, L, L] that a custom encoder threads into every layer.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Time2VecDays(nn.Module):
    """Time2Vec with frequencies initialised to span 1/day ... 1/(10 years).

    Input t: tensor of days (any shape). Output: tensor of shape [..., k].
    """
    def __init__(self, k: int = 32, min_days: float = 1.0, max_days: float = 10 * 365.0):
        super().__init__()
        # Log-uniform initial omegas across [1/max_days, 1/min_days].
        omegas = torch.logspace(
            math.log10(1.0 / max_days),
            math.log10(1.0 / min_days),
            steps=k - 1,
            base=10.0,
        ) * (2 * math.pi)
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w  = nn.Parameter(omegas)
        self.b  = nn.Parameter(torch.zeros(k - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [...]
        t = t.unsqueeze(-1)  # [..., 1]
        v0 = self.w0 * t + self.b0
        vi = torch.sin(self.w * t + self.b)
        return torch.cat([v0, vi], dim=-1)


class CompositePE(nn.Module):
    """Composite positional encoding.

    Always-on: sinusoidal within-sequence position P_pos(visit_idx_or_pos).

    Optional (controlled by `parts`):
      - "abs"         : T_abs(day_offset)    — Time2Vec
      - "gap"         : T_gap(delta_prev)    — Time2Vec
      - "domain"      : D(domain_id)         — Embedding
      - "interaction" : small MLP on concat([Time2Vec(day), onehot(domain)])
    """
    ALL_PARTS = ("abs", "gap", "domain", "interaction")

    def __init__(
        self,
        d_model: int,
        parts: Iterable[str] = ALL_PARTS,
        n_domains: int = 5,
        t2v_k: int = 32,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        parts = tuple(p for p in parts if p in self.ALL_PARTS)
        self.parts = parts
        self.d_model = d_model
        self.n_domains = n_domains

        self.register_buffer("pos_pe", _sinusoidal_pe(max_seq_len, d_model), persistent=False)

        if "abs" in parts:
            self.t2v_abs = Time2VecDays(k=t2v_k)
            self.abs_proj = nn.Linear(t2v_k, d_model)
        if "gap" in parts:
            self.t2v_gap = Time2VecDays(k=t2v_k, min_days=0.1, max_days=365.0 * 2)
            self.gap_proj = nn.Linear(t2v_k, d_model)
        if "domain" in parts:
            self.domain_emb = nn.Embedding(n_domains, d_model)
        if "interaction" in parts:
            self.t2v_inter = Time2VecDays(k=t2v_k)
            self.inter_mlp = nn.Sequential(
                nn.Linear(t2v_k + n_domains, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

    def forward(
        self,
        position_idx: torch.Tensor,   # long [B, L]
        day_offset: torch.Tensor,     # float [B, L]
        delta_prev: torch.Tensor,     # float [B, L]
        domain_id: torch.Tensor,      # long  [B, L]
    ) -> torch.Tensor:
        pe = self.pos_pe[position_idx.clamp_min(0).long()]
        if "abs" in self.parts:
            pe = pe + self.abs_proj(self.t2v_abs(day_offset))
        if "gap" in self.parts:
            pe = pe + self.gap_proj(self.t2v_gap(delta_prev))
        if "domain" in self.parts:
            pe = pe + self.domain_emb(domain_id.clamp_min(0).long())
        if "interaction" in self.parts:
            dom_oh = F.one_hot(domain_id.clamp_min(0).long(), num_classes=self.n_domains).float()
            tfeat = self.t2v_inter(day_offset)
            pe = pe + self.inter_mlp(torch.cat([tfeat, dom_oh], dim=-1))
        return pe


class RelativeDayBias(nn.Module):
    """Per-head additive attention bias phi(|Δday_i - Δday_j|).

    Output shape [B, H, L, L]. Meant to be passed through the stack via our
    custom encoder which adds it to attention logits each layer.
    """
    def __init__(self, n_heads: int, hidden: int = 16):
        super().__init__()
        self.n_heads = n_heads
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, n_heads),
        )

    def forward(self, days: torch.Tensor, key_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # days: [B, L]
        B, L = days.shape
        dt = (days.unsqueeze(-1) - days.unsqueeze(-2)).abs()   # [B, L, L]
        feats = torch.stack([dt, torch.log1p(dt), 1.0 / (1.0 + dt)], dim=-1)
        bias = self.mlp(feats)                                  # [B, L, L, H]
        bias = bias.permute(0, 3, 1, 2).contiguous()            # [B, H, L, L]
        if key_pad_mask is not None:
            # key_pad_mask: [B, L] with True => pad
            m = key_pad_mask.unsqueeze(1).unsqueeze(2)          # [B, 1, 1, L]
            bias = bias.masked_fill(m, float("-inf"))
        return bias


# ---------------- Custom encoder that threads additive attention bias ----------------

class _MHAWithBias(nn.Module):
    """Standard multi-head attention with an additive [B, H, L, L] bias term."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_pad_mask: torch.Tensor | None, attn_bias: torch.Tensor | None) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # each [B, H, L, d_head]
        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L]
        if attn_bias is not None:
            scores = scores + attn_bias
        if key_pad_mask is not None:
            scores = scores.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                            # [B, H, L, d_head]
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)
        return self.out(out)


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _MHAWithBias(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_pad_mask, attn_bias):
        x = x + self.dropout(self.attn(self.ln1(x), key_pad_mask, attn_bias))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class BiasableEncoder(nn.Module):
    """A transformer encoder stack that accepts an additive attention bias
    shape [B, H, L, L]. Used when `rel_day_bias=on`."""
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            _EncoderBlock(d_model, n_heads, d_model * ffn_mult, dropout) for _ in range(n_layers)
        ])
        self.n_heads = n_heads
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_pad_mask: torch.Tensor | None = None,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_pad_mask, attn_bias)
        return self.ln_out(x)
