from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :positions.size(1), :].expand(positions.size(0), -1, -1)


class VisitCodeTower(nn.Module):
	# good for now
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
        B, V, L = ids.shape

        x = self.emb(ids).reshape(B * V, L, -1)
        cls = self.cls.expand(B * V, 1, -1)
        seq = torch.cat([cls, x], dim=1)

        key_pad = torch.cat(
            [torch.zeros(B * V, 1, device=ids.device), 1 - mask.reshape(B * V, L)],
            dim=1
        ).bool()

        h = self.enc(seq, src_key_padding_mask=key_pad)[:, 0, :]
        h = h.reshape(B, V, -1)
        return h * vmask.unsqueeze(-1)


class NumericTower(nn.Module):
	# MODIFY
	# Approach 1:
	# Consider input for labs: take all the labs, organize them in a way: [LCODE] I123 [LVAL] 0.23 [LCODE] I456 [LVAL] 0.245 [LCODE] I789 [LVAL] 0.54
	# CHECK WITH CHAT OR CLAUDE: check for the sequence

    def __init__(self, n_feat: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * n_feat, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.register_buffer("mean", torch.zeros(n_feat))
        self.register_buffer("std", torch.ones(n_feat))
        self._fit = False

    def maybe_fit_stats(self, values: torch.Tensor, nmask: torch.Tensor) -> None:
        if self._fit or not self.training:
            return

        flat = values.reshape(-1, values.shape[-1])
        flatm = nmask.reshape(-1, nmask.shape[-1])
	# keep for now -  v0
	# should be modified for labs, vitals
        cnt = flatm.sum(0).clamp_min(1.0)
        mean = (flat * flatm).sum(0) / cnt
        var = (((flat - mean) * flatm) ** 2).sum(0) / cnt
        std = var.sqrt().clamp_min(1e-3)

        self.mean.copy_(mean.detach())
        self.std.copy_(std.detach())
        self._fit = True

    def forward(self, values: torch.Tensor, nmask: torch.Tensor) -> torch.Tensor:
        self.maybe_fit_stats(values, nmask)

        z = (values - self.mean) / self.std
        z = z * nmask

        return self.proj(torch.cat([z, nmask], dim=-1))


class FourTowersTimeAwareModel(nn.Module):
    name = "four_towers_timeaware"

    def __init__(
        self,
        diag_vocab_size: int,
        med_vocab_size: int,
        n_lab: int,
        n_vit: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_visit_layers: int = 2,
        n_patient_layers: int = 2,
        max_visits: int = 64,
        dropout: float = 0.1,
        n_classes: int = 2,
        time_mode: str = "linear",   # "linear" | "log"
    ):
        super().__init__()
        self.max_visits = max_visits
        self.time_mode = time_mode

        # 4 towers
        self.diag_tower = VisitCodeTower(diag_vocab_size, d_model, n_heads, n_visit_layers, dropout)
        self.med_tower = VisitCodeTower(med_vocab_size, d_model, n_heads, n_visit_layers, dropout)
        self.lab_tower = NumericTower(n_lab, d_model)
        self.vit_tower = NumericTower(n_vit, d_model)

        # learned fusion after concatenation
        self.visit_fusion = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # patient-level CLS token
        self.pat_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pat_cls, std=0.02)

        # time and position
        self.time_proj = nn.Linear(1, d_model)
        self.pat_pos = SinusoidalPositionalEmbedding(d_model, max_len=max_visits + 2)

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
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def encode_time(self, days: torch.Tensor) -> torch.Tensor:
        if self.time_mode == "linear":
            return days
        elif self.time_mode == "log":
            return torch.log1p(days)
        else:
            raise ValueError(f"Unknown time_mode: {self.time_mode}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        vmask = batch["vmask"]
        days = batch["days"]

        # 4 domain encodings per visit
        h_diag = self.diag_tower(batch["diag_ids"], batch["diag_mask"], vmask)                # [B,V,D]
        h_med = self.med_tower(batch["med_ids"], batch["med_mask"], vmask)                    # [B,V,D]
        h_lab = self.lab_tower(batch["lab_vals"], batch["lab_mask"]) * vmask.unsqueeze(-1)    # [B,V,D]
        h_vit = self.vit_tower(batch["vit_vals"], batch["vit_mask"]) * vmask.unsqueeze(-1)    # [B,V,D]

        # multimodal fusion
        h_cat = torch.cat([h_diag, h_med, h_lab, h_vit], dim=-1)                              # [B,V,4D]
        h_visit = self.visit_fusion(h_cat)                                                     # [B,V,D]

        # time-aware patient modeling
        days_enc = self.encode_time(days)
        tpe = self.time_proj(days_enc.unsqueeze(-1))

        B, V, _ = h_visit.shape
        pos = torch.arange(V, device=h_visit.device).unsqueeze(0).expand(B, -1)

        pat_in = h_visit + tpe + self.pat_pos(pos)

        pat_cls = self.pat_cls.expand(B, 1, -1)
        pat_seq = torch.cat([pat_cls, pat_in], dim=1)

        pat_mask = torch.cat([torch.ones(B, 1, device=h_visit.device), vmask], dim=1)
        h_pat = self.pat_enc(pat_seq, src_key_padding_mask=pat_mask < 0.5)

        return self.cls(self.dropout(h_pat[:, 0, :]))

