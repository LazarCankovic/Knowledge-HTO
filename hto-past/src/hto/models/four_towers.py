from __future__ import annotations

from typing import Dict
import math
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
    """
    Tower for one discrete domain per visit, e.g. diagnoses or medications.
    ids:  [B, V, L]
    mask: [B, V, L]
    vmask:[B, V]
    output: [B, V, D]
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
    """
    Tower for one numeric domain per visit, e.g. labs or vitals.
    values: [B, V, F]
    nmask:  [B, V, F]
    output: [B, V, D]
    """
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


class PatientSequenceEncoder(nn.Module):
    """
    Models the visit sequence over time.
    input: [B, V, D]
    output: [B, D]
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int, max_visits: int, dropout: float):
        super().__init__()
        self.pat_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pat_cls, std=0.02)

        self.visit_emb = nn.Embedding(max_visits + 2, d_model)
        self.time_proj = nn.Linear(1, d_model)
        self.pos = SinusoidalPositionalEmbedding(d_model, max_len=max_visits + 2)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visit_repr: torch.Tensor, days: torch.Tensor, vmask: torch.Tensor) -> torch.Tensor:
        B, V, _ = visit_repr.shape
        visit_idx = (torch.arange(V, device=visit_repr.device) + 1).clamp_max(V).unsqueeze(0).expand(B, -1)
        pos_idx = torch.arange(V, device=visit_repr.device).unsqueeze(0).expand(B, -1)

        x = (
            visit_repr
            + self.visit_emb(visit_idx)
            + self.time_proj(days.unsqueeze(-1))
            + self.pos(pos_idx)
        )

        pat_cls = self.pat_cls.expand(B, 1, -1)
        seq = torch.cat([pat_cls, x], dim=1)
        mask = torch.cat([torch.ones(B, 1, device=visit_repr.device), vmask], dim=1)

        h = self.enc(seq, src_key_padding_mask=mask < 0.5)
        return self.dropout(h[:, 0, :])


class FourTowerConcatModel(nn.Module):
    """
    Approach 1:
    - diagnosis tower
    - medication tower
    - lab tower
    - vital tower
    - concatenate 4 embeddings
    - 2 FC fusion layers
    - patient-level transformer
    - classifier
    """
    name = "four_tower_concat"

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
    ):
        super().__init__()

        self.diag_tower = VisitCodeTower(diag_vocab_size, d_model, n_heads, n_visit_layers, dropout)
        self.med_tower = VisitCodeTower(med_vocab_size, d_model, n_heads, n_visit_layers, dropout)
        self.lab_tower = NumericTower(n_lab, d_model)
        self.vit_tower = NumericTower(n_vit, d_model)

        self.visit_fusion = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.patient_encoder = PatientSequenceEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_patient_layers,
            max_visits=max_visits,
            dropout=dropout,
        )

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        vmask = batch["vmask"]
        days = batch["days"]

        h_diag = self.diag_tower(batch["diag_ids"], batch["diag_mask"], vmask)
        h_med = self.med_tower(batch["med_ids"], batch["med_mask"], vmask)
        h_lab = self.lab_tower(batch["lab_vals"], batch["lab_mask"]) * vmask.unsqueeze(-1)
        h_vit = self.vit_tower(batch["vit_vals"], batch["vit_mask"]) * vmask.unsqueeze(-1)

        h = torch.cat([h_diag, h_med, h_lab, h_vit], dim=-1)
        h_visit = self.visit_fusion(h)
        h_patient = self.patient_encoder(h_visit, days, vmask)
        logits = self.classifier(h_patient)
        return logits