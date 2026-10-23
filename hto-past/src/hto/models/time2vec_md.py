"""Time2Vec on multidomain_visit.

The original Time2Vec model adds a learnable Fourier time feature to each
token. In the multidomain setting we keep the same *idea* but explicitly
restrict CompositePE to its "abs" component — i.e. a Time2Vec over days
added to token embeddings — so it cleanly isolates the Time2Vec mechanism
from the other PE components. Rel-day bias and gap/domain parts can still
be enabled via flags for ablation.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .md_backbone import MDBackboneFlat
from .multidomain_batcher import build_flat_batch


class Time2Vec_MD(nn.Module):
    name = "time2vec_md"
    needs_variant = "multidomain_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 1024, dropout: float = 0.1,
                 pe_parts="abs", rel_day_bias: bool = False):
        super().__init__()
        self.max_len = max_len
        self.backbone = MDBackboneFlat(
            vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, max_len=max_len, pe_parts=pe_parts, rel_day_bias=rel_day_bias,
        )

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        return build_flat_batch(items, max_len=self.max_len)

    def forward(self, batch) -> torch.Tensor:
        h = self.backbone.encode(
            ids=batch["ids"], mask=batch["mask"], pos=batch["pos"],
            day=batch["day"], delta=batch["delta"], domain=batch["domain_id"],
        )
        return self.backbone.classify(h)
