"""TransformEHR (Yang et al., Nature Communications 2023).

Hierarchical transformer: an encoder on the set of codes within a visit
(mean-pooling a [VCLS] token) followed by a patient-level transformer
over the resulting per-visit embeddings. The visit-level embedding also
adds a day-offset time embedding.

Variant: codes_visit.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from .common import SinusoidalPositionalEmbedding, pad_3d


class TransformEHR(nn.Module):
    name = "transformehr"
    needs_variant = "codes_visit"

    # time_fn options: "log" | "exp_30" | "exp_90" | "exp_180"
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_visit_layers: int = 2, n_patient_layers: int = 2,
                 max_visits: int = 64, max_codes: int = 64, dropout: float = 0.1,
                 time_fn: str = "log"):
        super().__init__()
        self.max_visits = max_visits
        self.time_fn = time_fn
        self.max_codes = max_codes
        # padding_idx=0 so pad tokens contribute zero gradient and don't shift embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # learned [VCLS] token that accumulates the full visit summary after attention
        self.visit_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        # small normal init instead of zeros so VCLS breaks symmetry on the first forward pass
        nn.init.normal_(self.visit_cls, std=0.02)
        vis_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.visit_enc = nn.TransformerEncoder(vis_layer, num_layers=n_visit_layers)

        # learned [PCLS] token whose final hidden state is the patient-level representation used for classification
        self.pat_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pat_cls, std=0.02)
        # projects scalar day offset (normalized by 365) into d_model so time adds directly to visit embeddings
        self.time_proj = nn.Linear(1, d_model)
        # +2 for [PCLS] and one extra buffer so sinusoid covers the full padded sequence length
        self.pat_pos = SinusoidalPositionalEmbedding(d_model, max_len=max_visits + 2)
        pat_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.pat_enc = nn.TransformerEncoder(pat_layer, num_layers=n_patient_layers)
        self.dropout = nn.Dropout(dropout)
        # two-layer MLP head: hidden layer with GELU avoids dying-ReLU on the final binary logits
        self.cls = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        visits = [it["visits"] for it in items]
        # pad_3d truncates to max_visits/max_codes and returns both visit-level and code-level masks
        ids, vmask, cmask = pad_3d(visits, max_visits=self.max_visits, max_codes=self.max_codes)
        B, V = vmask.shape
        days = torch.zeros((B, V), dtype=torch.float32)
        for i, it in enumerate(items):
            # take the last V offsets so longer sequences are truncated from the front, keeping recency
            dos = it["day_offsets"][-V:]
            for j, d in enumerate(dos):
                if self.time_fn == "log":
                    days[i, j] = math.log1p(d) / math.log1p(365)
                elif self.time_fn == "exp_30":
                    days[i, j] = math.exp(-d / 30)
                elif self.time_fn == "exp_90":
                    days[i, j] = math.exp(-d / 90)
                elif self.time_fn == "exp_180":
                    days[i, j] = math.exp(-d / 180)
                else:
                    raise ValueError(f"Unknown time_fn: {self.time_fn!r}. Choose log | exp_30 | exp_90 | exp_180")
        labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        patient_ids = torch.tensor([int(it["patient_id"]) for it in items], dtype=torch.long)
        return {"ids": ids, "vmask": vmask, "cmask": cmask, "days": days,
                "labels": labels, "patient_ids": patient_ids}

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ids, vmask, cmask, days = batch["ids"], batch["vmask"], batch["cmask"], batch["days"]
        B, V, L = ids.shape

        tok = self.tok_emb(ids)                            # [B, V, L, D]
        # expand without copying memory — all visits in the batch share the same VCLS initial value
        vcls = self.visit_cls.expand(B * V, 1, -1)         # [B*V, 1, D]
        flat = tok.reshape(B * V, L, -1)
        seq = torch.cat([vcls, flat], dim=1)               # [B*V, L+1, D]
        # VCLS position is never masked (zeros prepended) so attention can always write into it
        key_pad = torch.cat([torch.zeros(B * V, 1, device=ids.device), 1 - cmask.reshape(B * V, L)], dim=1).bool()
        h_visit = self.visit_enc(seq, src_key_padding_mask=key_pad)
        # index 0 is the VCLS output — the single vector summarizing all codes in this visit
        visit_emb = h_visit[:, 0, :].reshape(B, V, -1)      # [B, V, D]
        # zero out padded visits so they don't inject signal into the patient-level transformer
        visit_emb = visit_emb * vmask.unsqueeze(-1)

        # three time signals added together: continuous elapsed time, visit order, sinusoidal position
        tpe = self.time_proj(days.unsqueeze(-1))
        pos = torch.arange(V, device=ids.device).unsqueeze(0).expand(B, -1)
        pat_in = visit_emb + tpe + self.pat_pos(pos)
        pat_cls = self.pat_cls.expand(B, 1, -1)
        pat_seq = torch.cat([pat_cls, pat_in], dim=1)       # [B, V+1, D]
        # PCLS is always unmasked (ones prepended) so it can attend to every real visit
        pat_mask = torch.cat([torch.ones(B, 1, device=ids.device), vmask], dim=1)
        h_pat = self.pat_enc(pat_seq, src_key_padding_mask=pat_mask < 0.5)
        # dropout before the classifier head reduces co-adaptation on the patient summary vector
        return self.cls(self.dropout(h_pat[:, 0, :]))
