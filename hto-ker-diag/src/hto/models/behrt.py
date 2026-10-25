"""BEHRT (Li et al., Scientific Reports 2020).

Original: BERT over diagnosis codes with three embeddings summed - code,
age-at-visit, and "visit position" (within the flat stream a [SEP] plus a
segment-alternating flag). We adapt: we don't have age, so we use a
"visit index" embedding and keep the [SEP]-separated flat form.

Variant: codes_visit (we flatten visits with [SEP] between them).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .common import SinusoidalPositionalEmbedding


SEP_ID = 3  # Vocab specials: PAD=0, UNK=1, CLS=2, SEP=3, MASK=4


class BEHRT(nn.Module):
    name = "behrt"
    needs_variant = "codes_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 max_len: int = 512, n_visits: int = 64, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.n_visits = n_visits
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_len=max_len + 2)
        self.visit_emb = nn.Embedding(n_visits + 2, d_model)
        # "age" is unknown; we include a two-bucket segment (alternating A/B) to match the original
        self.seg_emb = nn.Embedding(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def _flatten(self, items: List[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(items)
        ids = torch.zeros((B, self.max_len), dtype=torch.long)
        vidx = torch.zeros((B, self.max_len), dtype=torch.long)
        seg = torch.zeros((B, self.max_len), dtype=torch.long)
        mask = torch.zeros((B, self.max_len), dtype=torch.float32)
        for i, it in enumerate(items):
            flat, vix, segflags = [2], [0], [0]  # CLS start
            for j, visit in enumerate(it["visits"]):
                vi = min(j + 1, self.n_visits)
                flag = j % 2
                for code in visit:
                    flat.append(int(code))
                    vix.append(vi)
                    segflags.append(flag)
                flat.append(SEP_ID)
                vix.append(vi)
                segflags.append(flag)
            flat = flat[: self.max_len]
            vix = vix[: self.max_len]
            segflags = segflags[: self.max_len]
            ids[i, : len(flat)] = torch.tensor(flat, dtype=torch.long)
            vidx[i, : len(vix)] = torch.tensor(vix, dtype=torch.long)
            seg[i, : len(segflags)] = torch.tensor(segflags, dtype=torch.long)
            mask[i, : len(flat)] = 1.0
        return ids, vidx, seg, mask

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        ids, vidx, seg, mask = self._flatten(items)
        labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        patient_ids = torch.tensor([int(it["patient_id"]) for it in items], dtype=torch.long)
        return {"ids": ids, "vidx": vidx, "seg": seg, "mask": mask,
                "labels": labels, "patient_ids": patient_ids}

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ids, vidx, seg, mask = batch["ids"], batch["vidx"], batch["seg"], batch["mask"]
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(ids) + self.pos_emb(pos) + self.visit_emb(vidx) + self.seg_emb(seg)
        x = self.dropout(x)
        h = self.encoder(x, src_key_padding_mask=mask < 0.5)
        return self.cls(h[:, 0, :])
