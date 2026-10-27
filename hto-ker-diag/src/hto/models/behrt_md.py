"""BEHRT on multidomain_visit.

BEHRT's original time handling is: token + age-at-visit + visit-position +
alternating-segment embeddings summed into the input. We don't have age but
we keep the visit-position + alternating-segment idea as an additional
signal on top of CompositePE.

Visit index is taken from `batch['visit_idx']` (per-token). The alternating
segment is 0/1 by visit_idx % 2.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .md_backbone import MDBackboneFlat
from .multidomain_batcher import build_flat_batch


class BEHRT_MD(nn.Module):
    name = "behrt_md"
    needs_variant = "multidomain_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, max_len: int = 1024, dropout: float = 0.1,
                 n_visits: int = 96,
                 pe_parts="abs,gap,domain,interaction", rel_day_bias: bool = True):
        super().__init__()
        self.max_len = max_len
        self.visit_emb = nn.Embedding(n_visits + 2, d_model)
        self.seg_alt_emb = nn.Embedding(2, d_model)
        self.n_visits = n_visits
        self.backbone = MDBackboneFlat(
            vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, max_len=max_len, pe_parts=pe_parts, rel_day_bias=rel_day_bias,
        )

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        return build_flat_batch(items, max_len=self.max_len)

    def forward(self, batch) -> torch.Tensor:
        vi = batch["visit_idx"].clamp_max(self.n_visits)
        alt = (vi % 2).long()
        extra = self.visit_emb(vi) + self.seg_alt_emb(alt)
        h = self.backbone.encode(
            ids=batch["ids"], mask=batch["mask"], pos=batch["pos"],
            day=batch["day"], delta=batch["delta"], domain=batch["domain_id"],
            extra=extra,
        )
        return self.backbone.classify(h)
