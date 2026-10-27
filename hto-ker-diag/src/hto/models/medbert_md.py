"""Med-BERT on multidomain_visit.

Differences from the original Med-BERT:
  - input is the full multidomain flat stream (diag + med + lab + vital,
    delimited by [EOD]/[EOM]/[EOL]/[EOVI]/[EOV])
  - the discrete visit "segment" embedding is replaced by the CompositePE's
    (optional) domain + time components
  - optional relative-day attention bias

Configurable via pe_parts ∈ {"", "abs", "abs,gap", ..., "abs,gap,domain,interaction"}.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .md_backbone import MDBackboneFlat
from .multidomain_batcher import build_flat_batch


class MedBERT_MD(nn.Module):
    name = "medbert_md"
    needs_variant = "multidomain_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 1024, dropout: float = 0.1,
                 pe_parts="abs,gap,domain,interaction", rel_day_bias: bool = True):
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
