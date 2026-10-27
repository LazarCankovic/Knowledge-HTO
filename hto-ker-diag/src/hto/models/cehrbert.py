"""CEHR-BEHRT (Pang et al., arXiv:2111.08585).

Extends BEHRT with artificial time tokens between visits ("ATT" tokens)
that encode the days-between-visits in coarse buckets (W/M/Y), plus an
age-at-visit embedding. We implement the ATT bucketing and continuous
time-embedding additive feature.

Variant: codes_visit.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .common import SinusoidalPositionalEmbedding

SEP_ID = 3  # PAD=0, UNK=1, CLS=2, SEP=3, MASK=4


def _att_bucket_token(dt_days: float) -> int:
    """Quantize days-between-visits into a small id space reserved at the top of the
    vocab (we use vocab_size-3..vocab_size-1 as ATT buckets). The model reserves
    those slots via vocab_size+3 extra embeddings (see __init__)."""
    if dt_days < 7:
        return 0  # W
    elif dt_days < 30:
        return 1  # M
    else:
        return 2  # Y


class CEHRBEHRT(nn.Module):
    name = "cehrbert"
    needs_variant = "codes_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 max_len: int = 512, n_visits: int = 64, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_visits = n_visits
        # Separate ATT embedding table (3 buckets) — keep codebook clean
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.att_emb = nn.Embedding(3, d_model)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_len=max_len + 2)
        self.visit_emb = nn.Embedding(n_visits + 2, d_model)
        self.time_emb = nn.Linear(1, d_model)  # continuous day_offset / 365
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        B = len(items)
        L = self.max_len
        ids = torch.zeros((B, L), dtype=torch.long)
        att_ids = torch.full((B, L), -1, dtype=torch.long)  # -1 => not an ATT token
        vidx = torch.zeros((B, L), dtype=torch.long)
        days = torch.zeros((B, L), dtype=torch.float32)
        mask = torch.zeros((B, L), dtype=torch.float32)

        for i, it in enumerate(items):
            visits = it["visits"]
            dos = it["day_offsets"]
            flat_tok, flat_att, flat_v, flat_d = [2], [-1], [0], [0.0]  # CLS start
            for j, visit in enumerate(visits):
                if j > 0:
                    dt = max(0.0, dos[j] - dos[j - 1])
                    flat_tok.append(0)  # placeholder id; we'll use att_ids instead
                    flat_att.append(_att_bucket_token(dt))
                    flat_v.append(min(j + 1, self.n_visits))
                    flat_d.append(dos[j])
                vi = min(j + 1, self.n_visits)
                for code in visit:
                    flat_tok.append(int(code))
                    flat_att.append(-1)
                    flat_v.append(vi)
                    flat_d.append(dos[j])
                flat_tok.append(SEP_ID); flat_att.append(-1); flat_v.append(vi); flat_d.append(dos[j])
            flat_tok, flat_att = flat_tok[:L], flat_att[:L]
            flat_v, flat_d = flat_v[:L], flat_d[:L]
            n = len(flat_tok)
            ids[i, :n] = torch.tensor(flat_tok, dtype=torch.long)
            att_ids[i, :n] = torch.tensor(flat_att, dtype=torch.long)
            vidx[i, :n] = torch.tensor(flat_v, dtype=torch.long)
            days[i, :n] = torch.tensor(flat_d, dtype=torch.float32) / 365.0
            mask[i, :n] = 1.0

        labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        patient_ids = torch.tensor([int(it["patient_id"]) for it in items], dtype=torch.long)
        return {"ids": ids, "att_ids": att_ids, "vidx": vidx, "days": days, "mask": mask,
                "labels": labels, "patient_ids": patient_ids}

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ids, att_ids, vidx, days, mask = batch["ids"], batch["att_ids"], batch["vidx"], batch["days"], batch["mask"]
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, -1)
        base = self.tok_emb(ids)
        # overlay ATT embedding where att_ids >= 0
        has_att = (att_ids >= 0)
        if has_att.any():
            safe_idx = att_ids.clamp_min(0)
            att_vec = self.att_emb(safe_idx)
            base = torch.where(has_att.unsqueeze(-1), att_vec, base)
        x = base + self.pos_emb(pos) + self.visit_emb(vidx) + self.time_emb(days.unsqueeze(-1))
        x = self.dropout(x)
        h = self.encoder(x, src_key_padding_mask=mask < 0.5)
        return self.cls(h[:, 0, :])
