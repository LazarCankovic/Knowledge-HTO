"""Time-aware LSTM (Baytas et al., KDD 2017).

Original: a modified LSTM cell where the short-term memory C_s is decayed
by g(dt) before being added back with long-term memory. g(dt) is a
monotonically decreasing function of the elapsed time between visits.

We use the exact cell from the paper with g(dt) = 1 / log(e + dt) and
operate over per-visit aggregated embeddings (mean of code embeddings in
the visit). The input to the LSTM is the per-visit embedding; the time
gap is measured in days between consecutive visits.

Variant: codes_visit.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .common import pad_3d


class TLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # gates: input, forget, output, candidate
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        # short-term decomposition network
        self.W_d = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def _g(dt: torch.Tensor) -> torch.Tensor:
        # 1 / log(e + dt)
        return 1.0 / torch.log(torch.exp(torch.tensor(1.0, device=dt.device)) + dt.clamp_min(0.0))

    def forward(self, x: torch.Tensor, dt: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x [B, D_in], dt [B], h [B, H], c [B, H]
        # Decompose C into short- and long-term.
        C_s = torch.tanh(self.W_d(c))
        C_s_hat = C_s * self._g(dt).unsqueeze(-1)
        C_t = c - C_s
        C_star = C_t + C_s_hat

        i = torch.sigmoid(self.W_i(x) + self.U_i(h))
        f = torch.sigmoid(self.W_f(x) + self.U_f(h))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h))
        g = torch.tanh(self.W_c(x) + self.U_c(h))

        c_new = f * C_star + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class TLSTM(nn.Module):
    name = "tlstm"
    needs_variant = "codes_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cell = TLSTMCell(d_model, hidden)
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        visits = [it["visits"] for it in items]
        ids, vmask, cmask = pad_3d(visits)

        # day offsets -> per-visit dt (diff), leading 0 for the first visit
        B, V = vmask.shape
        dts = torch.zeros((B, V), dtype=torch.float32)
        for i, it in enumerate(items):
            dos = it["day_offsets"][-V:]
            if len(dos) > 1:
                prev = dos[0]
                for j, d in enumerate(dos):
                    dts[i, j] = max(0.0, d - prev)
                    prev = d

        labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        patient_ids = torch.tensor([int(it["patient_id"]) for it in items], dtype=torch.long)
        return {"ids": ids, "vmask": vmask, "cmask": cmask, "dts": dts,
                "labels": labels, "patient_ids": patient_ids}

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ids, vmask, cmask, dts = batch["ids"], batch["vmask"], batch["cmask"], batch["dts"]
        B, V, L = ids.shape

        tok = self.tok_emb(ids)                       # [B, V, L, D]
        denom = cmask.sum(-1, keepdim=True).clamp_min(1.0)
        vis_emb = (tok * cmask.unsqueeze(-1)).sum(2) / denom  # [B, V, D]

        h = torch.zeros(B, self.hidden, device=ids.device)
        c = torch.zeros(B, self.hidden, device=ids.device)
        last_h = h
        for t in range(V):
            x_t = vis_emb[:, t, :]
            dt_t = dts[:, t]
            h_new, c_new = self.cell(x_t, dt_t, h, c)
            # if this visit is padded, keep h,c unchanged
            alive = vmask[:, t].unsqueeze(-1)
            h = alive * h_new + (1 - alive) * h
            c = alive * c_new + (1 - alive) * c
            last_h = h
        return self.cls(self.dropout(last_h))
