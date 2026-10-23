"""Small shared backbone for *_md (multidomain) baselines.

Every *_md baseline needs the same plumbing:
  - look up token embeddings
  - (optionally) add CompositePE components
  - (optionally) run encoder with a RelativeDayBias
  - pool at [CLS] and classify

`MDBackboneFlat` handles the flat-stream variants (medbert_md, time2vec_md,
behrt_md, cehrbert_md). Hierarchical baselines (tlstm_md, transformehr_md,
multimodal_behrt_md) use CompositePE and BiasableEncoder directly.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .composite_pe import (
    BiasableEncoder,
    CompositePE,
    RelativeDayBias,
    _sinusoidal_pe,
)


def _parse_parts(parts) -> list[str]:
    if parts is None:
        return []
    if isinstance(parts, str):
        return [p.strip() for p in parts.split(",") if p.strip()]
    return list(parts)


class MDBackboneFlat(nn.Module):
    """Token embedding + composite PE + transformer encoder + CLS readout.

    The `extra_emb_fn(batch, device) -> Tensor[B, L, D]` callback lets a
    subclass add its own signal (e.g. an "[ATT]" bucket embedding for
    cehrbert_md, or a visit-index alt-segment embedding for behrt_md).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 1024,
        pe_parts: Iterable[str] | str | None = ("abs", "gap", "domain", "interaction"),
        rel_day_bias: bool = True,
        n_domains: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_heads = n_heads
        self.rel_day_bias_on = bool(rel_day_bias)

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe_parts = _parse_parts(pe_parts)
        self.pe = CompositePE(
            d_model=d_model, parts=self.pe_parts,
            n_domains=n_domains, max_seq_len=max_len + 2,
        )
        self.dropout = nn.Dropout(dropout)

        if self.rel_day_bias_on:
            self.encoder = BiasableEncoder(d_model, n_heads, n_layers, dropout)
            self.rel_bias = RelativeDayBias(n_heads=n_heads)
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                dropout=dropout, activation="gelu", batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.rel_bias = None

        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def encode(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        pos: torch.Tensor,
        day: torch.Tensor,
        delta: torch.Tensor,
        domain: torch.Tensor,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.tok_emb(ids) + self.pe(pos, day, delta, domain)
        if extra is not None:
            x = x + extra
        x = self.dropout(x)
        key_pad = mask < 0.5
        if self.rel_bias is not None:
            attn_bias = self.rel_bias(day, key_pad_mask=None)  # masking handled via key_pad
            h = self.encoder(x, key_pad_mask=key_pad, attn_bias=attn_bias)
        else:
            h = self.encoder(x, src_key_padding_mask=key_pad)
        return h

    def classify(self, h: torch.Tensor) -> torch.Tensor:
        return self.cls(h[:, 0, :])


# ---- a helper reused by hierarchical md models ----

class _SinPos(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.register_buffer("pe", _sinusoidal_pe(max_len, d_model), persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.pe[idx.clamp_min(0).long()]
