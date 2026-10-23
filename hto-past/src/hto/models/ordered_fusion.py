from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn


class OrderedFusionModel(nn.Module):
    """
    Approach 3:
    Build one single time-ordered sequence across all domains.

    Batch keys:
    event_ids   [B, T]
    domain_ids  [B, T]   0=diag, 1=med, 2=lab, 3=vit
    times       [B, T]
    positions   [B, T]
    values      [B, T]
    has_value   [B, T]
    mask        [B, T]
    """
    name = "ordered_fusion"

    def __init__(
        self,
        vocab_size: int,
        n_domains: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_events: int = 512,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.domain_emb = nn.Embedding(n_domains, d_model)
        self.pos_emb = nn.Embedding(max_events + 2, d_model)
        self.time_proj = nn.Linear(1, d_model)

        # Numeric side-channel for lab/vital events
        self.value_proj = nn.Sequential(
            nn.Linear(2, d_model),   # [value, has_value]
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learned fusion at event level
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        event_ids = batch["event_ids"]          # [B, T]
        domain_ids = batch["domain_ids"]        # [B, T]
        times = batch["times"]                  # [B, T]
        positions = batch["positions"]          # [B, T]
        values = batch["values"].unsqueeze(-1)  # [B, T, 1]
        has_value = batch["has_value"].unsqueeze(-1)  # [B, T, 1]
        mask = batch["mask"]                    # [B, T]

        tok = self.token_emb(event_ids)
        dom = self.domain_emb(domain_ids)
        pos = self.pos_emb(positions)
        tim = self.time_proj(times.unsqueeze(-1))

        base = tok + dom + pos + tim

        val = self.value_proj(torch.cat([values, has_value], dim=-1))

        # Learned fusion of semantic event info + numeric value info
        h = self.fusion(torch.cat([base, val], dim=-1))

        cls = self.cls_token.expand(event_ids.size(0), 1, -1)
        seq = torch.cat([cls, h], dim=1)

        key_pad = torch.cat(
            [torch.ones(event_ids.size(0), 1, device=event_ids.device), mask],
            dim=1
        ) < 0.5

        out = self.encoder(seq, src_key_padding_mask=key_pad)
        logits = self.classifier(self.dropout(out[:, 0, :]))
        return logits