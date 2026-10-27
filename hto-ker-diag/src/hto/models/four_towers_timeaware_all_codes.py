from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class VisitCodeTower(nn.Module):
    """
    One token-sequence tower per visit.
    Works for diagnoses, medications, labs-tokenized, vitals-tokenized.
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, vmask: torch.Tensor) -> torch.Tensor:
        """
        ids  : [B, V, L]
        mask : [B, V, L]
        vmask: [B, V]
        """
        B, V, L = ids.shape

        x = self.emb(ids).reshape(B * V, L, -1)
        cls = self.cls.expand(B * V, 1, -1)
        seq = torch.cat([cls, x], dim=1)

        key_pad = torch.cat(
            [torch.zeros(B * V, 1, device=ids.device), 1 - mask.reshape(B * V, L)],
            dim=1,
        ).bool()

        h = self.enc(seq, src_key_padding_mask=key_pad)[:, 0, :]
        h = h.reshape(B, V, -1)
        return h * vmask.unsqueeze(-1)


class FourTowersTimeAwareAllCodes(nn.Module):
    """
    All 4 domains use VisitCodeTower.
    Time is added AFTER fusion, before patient-level transformer.
    """
    name = "four_towers_timeaware_allcodes"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_visit_layers: int = 2,
        n_patient_layers: int = 2,
        max_visits: int = 64,
        dropout: float = 0.1,
        n_classes: int = 2,
        time_fn: str = "log",   # "log" | "exp_30" | "exp_90" | "exp_180"
    ):
        super().__init__()
        self.max_visits = max_visits
        self.time_fn = time_fn

        self.diag_tower = VisitCodeTower(vocab_size, d_model, n_heads, n_visit_layers, dropout)
        self.med_tower  = VisitCodeTower(vocab_size, d_model, n_heads, n_visit_layers, dropout)
        self.lab_tower  = VisitCodeTower(vocab_size, d_model, n_heads, n_visit_layers, dropout)
        self.vit_tower  = VisitCodeTower(vocab_size, d_model, n_heads, n_visit_layers, dropout)

        self.visit_fusion = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pat_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pat_cls, std=0.02)

        self.time_proj = nn.Linear(1, d_model)
        # learned positional embedding (vs original fixed sinusoid)
        self.pat_pos = nn.Embedding(max_visits + 2, d_model)

        pat_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.pat_enc = nn.TransformerEncoder(pat_layer, num_layers=n_patient_layers)

        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def encode_time(self, days: torch.Tensor) -> torch.Tensor:
        """Scale days-since-HTM_FRx according to the selected time_fn."""
        if self.time_fn == "log":
            return torch.log1p(days) / math.log1p(365)
        elif self.time_fn == "exp_30":
            return torch.exp(-days / 30)
        elif self.time_fn == "exp_90":
            return torch.exp(-days / 90)
        elif self.time_fn == "exp_180":
            return torch.exp(-days / 180)
        else:
            raise ValueError(f"Unknown time_fn: {self.time_fn}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Expected batch keys:
          diag_ids, diag_mask
          med_ids,  med_mask
          lab_ids,  lab_mask
          vit_ids,  vit_mask
          days   : [B, V]  days relative to HTM_FRx (prescription date)
          vmask  : [B, V]
        """
        vmask = batch["vmask"]
        days  = batch["days"]   # anchored to HTM_FRx

        h_diag = self.diag_tower(batch["diag_ids"], batch["diag_mask"], vmask)
        h_med  = self.med_tower( batch["med_ids"],  batch["med_mask"],  vmask)
        h_lab  = self.lab_tower( batch["lab_ids"],  batch["lab_mask"],  vmask)
        h_vit  = self.vit_tower( batch["vit_ids"],  batch["vit_mask"],  vmask)

        h_cat   = torch.cat([h_diag, h_med, h_lab, h_vit], dim=-1)
        h_visit = self.visit_fusion(h_cat)

        # time embedding: scaled scalar → d_model vector
        days_enc = self.encode_time(days)
        tpe = self.time_proj(days_enc.unsqueeze(-1))

        B, V, _ = h_visit.shape
        pos = torch.arange(V, device=h_visit.device).unsqueeze(0).expand(B, -1)

        # learned positional embedding lookup (not sinusoidal)
        pat_in = h_visit + tpe + self.pat_pos(pos)

        pat_cls = self.pat_cls.expand(B, 1, -1)
        pat_seq = torch.cat([pat_cls, pat_in], dim=1)

        pat_mask = torch.cat([torch.ones(B, 1, device=h_visit.device), vmask], dim=1)
        h_pat = self.pat_enc(pat_seq, src_key_padding_mask=pat_mask < 0.5)

        return self.cls_head(self.dropout(h_pat[:, 0, :]))
