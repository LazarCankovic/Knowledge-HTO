from __future__ import annotations
import math
from typing import Dict
import torch
import torch.nn as nn


class VisitCodeTower(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=4*d_model, dropout=dropout, activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, ids, mask, vmask):
        B, V, L = ids.shape
        x = self.emb(ids).reshape(B*V, L, -1)
        cls = self.cls.expand(B*V, 1, -1)
        seq = torch.cat([cls, x], dim=1)
        key_pad = torch.cat([torch.zeros(B*V, 1, device=ids.device),
                             1 - mask.reshape(B*V, L)], dim=1).bool()
        h = self.enc(seq, src_key_padding_mask=key_pad)[:, 0, :]
        return h.reshape(B, V, -1) * vmask.unsqueeze(-1)


class SingleTowerTimeAware(nn.Module):
    """A1+Time: one shared tower over merged allcodes sequence + temporal encoding."""
    name = "single_tower_timeaware"

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_visit_layers=2,
                 n_patient_layers=2, max_visits=64, dropout=0.1, n_classes=2,
                 time_fn="log", **kwargs):
        super().__init__()
        self.max_visits = max_visits
        self.time_fn = time_fn

        # single shared tower for all domains merged
        self.tower = VisitCodeTower(vocab_size, d_model, n_heads, n_visit_layers, dropout)

        self.pat_cls  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pat_cls, std=0.02)
        self.time_proj = nn.Linear(1, d_model)
        self.pat_pos   = nn.Embedding(max_visits + 2, d_model)

        pat_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=4*d_model, dropout=dropout, activation="gelu", batch_first=True)
        self.pat_enc = nn.TransformerEncoder(pat_layer, num_layers=n_patient_layers)
        self.dropout  = nn.Dropout(dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_classes))

    def encode_time(self, days):
        if self.time_fn == "log":
            return torch.log1p(days) / math.log1p(365)
        elif self.time_fn == "exp_30":
            return torch.exp(-days / 30)
        elif self.time_fn == "exp_90":
            return torch.exp(-days / 90)
        elif self.time_fn == "exp_180":
            return torch.exp(-days / 180)
        raise ValueError(f"Unknown time_fn: {self.time_fn}")

    def forward(self, batch: Dict[str, torch.Tensor]):
        vmask = batch["vmask"]
        days  = batch["days"]

        # all_ids: merged sequence of all domain codes per visit [B, V, L]
        all_ids  = batch["all_ids"]
        all_mask = batch["all_mask"]

        h_visit = self.tower(all_ids, all_mask, vmask)

        days_enc = self.encode_time(days)
        tpe = self.time_proj(days_enc.unsqueeze(-1))

        B, V, _ = h_visit.shape
        pos = torch.arange(V, device=h_visit.device).unsqueeze(0).expand(B, -1)
        pat_in = h_visit + tpe + self.pat_pos(pos)

        pat_cls = self.pat_cls.expand(B, 1, -1)
        pat_seq = torch.cat([pat_cls, pat_in], dim=1)
        pat_mask = torch.cat([torch.ones(B, 1, device=h_visit.device), vmask], dim=1)
        h_pat = self.pat_enc(pat_seq, src_key_padding_mask=pat_mask < 0.5)
        return self.cls_head(self.dropout(h_pat[:, 0, :]))
