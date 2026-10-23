"""CEHR-BEHRT on multidomain_visit.

Keeps the W/M/Y ATT-token bucketing from CEHR-BEHRT applied to the
delta-prev-visit days (already in the batch). ATT tokens get a separate
embedding table so they don't contaminate the code vocab. CompositePE's
`gap` component captures the continuous gap signal while these discrete
tokens mark the symbolic W/M/Y transitions.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .md_backbone import MDBackboneFlat
from .multidomain_batcher import build_flat_batch


def _att_bucket(dt: torch.Tensor) -> torch.Tensor:
    # 0 = <1w, 1 = <1mo, 2 = >=1mo
    out = torch.zeros_like(dt, dtype=torch.long)
    out = torch.where(dt >= 7.0,  torch.ones_like(out),    out)
    out = torch.where(dt >= 30.0, torch.full_like(out, 2), out)
    return out


class CEHRBEHRT_MD(nn.Module):
    name = "cehrbert_md"
    needs_variant = "multidomain_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, max_len: int = 1024, dropout: float = 0.1,
                 n_visits: int = 96,
                 pe_parts="abs,gap,domain,interaction", rel_day_bias: bool = True):
        super().__init__()
        self.max_len = max_len
        self.visit_emb = nn.Embedding(n_visits + 2, d_model)
        self.att_emb = nn.Embedding(3, d_model)
        self.n_visits = n_visits
        self.backbone = MDBackboneFlat(
            vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, max_len=max_len, pe_parts=pe_parts, rel_day_bias=rel_day_bias,
        )

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        return build_flat_batch(items, max_len=self.max_len)

    def forward(self, batch) -> torch.Tensor:
        vi = batch["visit_idx"].clamp_max(self.n_visits)
        bucket = _att_bucket(batch["delta"])                  # [B, L]
        # ATT signal: apply bucket embedding only at the *first* token of each
        # new visit — elsewhere it's zeroed.
        is_boundary = (vi[:, 1:] != vi[:, :-1])
        pad = torch.ones_like(vi[:, :1], dtype=torch.bool)
        first_of_visit = torch.cat([pad, is_boundary], dim=1)
        att_vec = self.att_emb(bucket) * first_of_visit.unsqueeze(-1).float()
        extra = self.visit_emb(vi) + att_vec
        h = self.backbone.encode(
            ids=batch["ids"], mask=batch["mask"], pos=batch["pos"],
            day=batch["day"], delta=batch["delta"], domain=batch["domain_id"],
            extra=extra,
        )
        return self.backbone.classify(h)
