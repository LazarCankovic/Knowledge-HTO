"""Multimodal-BEHRT (Mbaye et al., Front. Oncol. 2025).

Extends BEHRT by adding a continuous-valued numeric modality branch: per
visit, a fixed-size numeric feature vector (lab values + vital values with
missingness masks) is projected to the model dim and added to the visit-
level code embedding. We implement the hierarchical variant: visit-level
code attention, then patient-level transformer that fuses codes + numerics.

Variant: multimodal.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .common import SinusoidalPositionalEmbedding, pad_3d


###### - MY COMMENTS

# This model works in 2 stages:
# 1. For each visit, summarize the codes inside that visit.
# 2. Across visits, summarize the whole patient history.

class MultimodalBEHRT(nn.Module):
    name = "multimodal_behrt"
    needs_variant = "multimodal"

    # First we initialize all important model components.
    def __init__(self, vocab_size: int, n_lab: int = 12, n_vit: int = 3, d_model: int = 128,
                 n_heads: int = 4, n_visit_layers: int = 2, n_patient_layers: int = 2,
                 max_visits: int = 64, max_codes: int = 64, dropout: float = 0.1):
        super().__init__()
        self.max_visits = max_visits
        self.max_codes = max_codes
        self.n_lab = n_lab
        self.n_vit = n_vit

        # Embedding table for all discrete medical codes.
        # Each code id becomes a dense vector of size d_model.
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Learnable CLS token used to summarize one visit.
        # After visit-level attention, this token will represent the whole visit.
        self.visit_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.visit_cls, std=0.02)

        # Numeric projection branch.
        # Takes labs + vitals + missingness masks and maps them into model dimension.
        # This is the numeric modality branch of the model.
        self.num_proj = nn.Sequential(
            nn.Linear(2 * (n_lab + n_vit), d_model),  # values concatenated with mask
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learned embedding for visit number / visit index.
        self.visit_emb = nn.Embedding(max_visits + 2, d_model)

        # Linear projection for time (day offset).
        # This lets the model use temporal information.
        self.time_proj = nn.Linear(1, d_model)

        # Transformer encoder for codes inside a visit.
        # This summarizes all codes in one visit into one visit representation.
        vis_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.visit_enc = nn.TransformerEncoder(vis_layer, num_layers=n_visit_layers)

        # Transformer encoder across visits.
        # This models the sequence of visits over time for one patient.
        pat_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.pat_enc = nn.TransformerEncoder(pat_layer, num_layers=n_patient_layers)

        # Learnable CLS token for the whole patient history.
        # After patient-level attention, this token will represent the whole patient.
        self.pat_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pat_cls, std=0.02)

        # Positional embedding across visits.
        self.pat_pos = SinusoidalPositionalEmbedding(d_model, max_len=max_visits + 2)

        self.dropout = nn.Dropout(dropout)

        # Final classifier for binary prediction.
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)
        )

        # Running mean and std for numeric normalization.
        # IMPORTANT this is only a baseline method because it fits on the first training batch.
        self.register_buffer("num_mean", torch.zeros(n_lab + n_vit))
        self.register_buffer("num_std", torch.ones(n_lab + n_vit))
        self._stats_fit = False

    # Build one batch from a list of patient dictionaries.
    # This converts raw patient data into tensors the model can use.
    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        visits = [it["visits"] for it in items]

        # pad_3d returns:
        # ids   = padded code ids
        # vmask = visit mask (which visits are real)
        # cmask = code mask (which codes inside visits are real)
        ids, vmask, cmask = pad_3d(visits, max_visits=self.max_visits, max_codes=self.max_codes)
        B, V = vmask.shape
        n_feat = self.n_lab + self.n_vit

        # num   = numeric values (labs + vitals)
        # nmask = missingness mask for numeric values
        # days  = time offsets
        num = torch.zeros((B, V, n_feat), dtype=torch.float32)
        nmask = torch.zeros((B, V, n_feat), dtype=torch.float32)
        days = torch.zeros((B, V), dtype=torch.float32)

        # Convert raw lists into tensors
        for i, it in enumerate(items):
            dos = it["day_offsets"][-V:]
            labs = it["lab_vecs"][-V:]
            labsm = it["lab_masks"][-V:]
            vits = it["vital_vecs"][-V:]
            vitsm = it["vital_masks"][-V:]

            for j in range(len(dos)):
                # Convert day offsets into years
                days[i, j] = dos[j] / 365.0

                # Fill in lab values and masks
                if j < len(labs):
                    num[i, j, : self.n_lab] = torch.tensor(labs[j], dtype=torch.float32)
                    nmask[i, j, : self.n_lab] = torch.tensor(labsm[j], dtype=torch.float32)

                # Fill in vital values and masks
                if j < len(vits):
                    num[i, j, self.n_lab:] = torch.tensor(vits[j], dtype=torch.float32)
                    nmask[i, j, self.n_lab:] = torch.tensor(vitsm[j], dtype=torch.float32)

        labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        patient_ids = torch.tensor([int(it["patient_id"]) for it in items], dtype=torch.long)

        return {
            "ids": ids,
            "vmask": vmask,
            "cmask": cmask,
            "num": num,
            "nmask": nmask,
            "days": days,
            "labels": labels,
            "patient_ids": patient_ids
        }

    # Fit mean and std once, then use them to z-score labs and vitals.
    # This is used to normalize numeric features before fusion.
    def _maybe_fit_stats(self, num: torch.Tensor, nmask: torch.Tensor) -> None:
        if self._stats_fit or not self.training:
            return

        flat = num.reshape(-1, num.shape[-1])
        flatm = nmask.reshape(-1, nmask.shape[-1])

        # Count how many observed values each feature has
        cnt = flatm.sum(0).clamp_min(1.0)

        # Compute mean only over observed values
        mean = (flat * flatm).sum(0) / cnt

        # Compute variance only over observed values
        var = (((flat - mean) * flatm) ** 2).sum(0) / cnt
        std = var.sqrt().clamp_min(1e-3)

        self.num_mean.copy_(mean.detach())
        self.num_std.copy_(std.detach())
        self._stats_fit = True

    # Forward pass:
    # 1. Encode codes inside each visit
    # 2. Encode labs/vitals as numeric features
    # 3. Fuse them with time + position
    # 4. Encode the sequence of visits for the patient
    # 5. Predict label from patient CLS token
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ids, vmask, cmask = batch["ids"], batch["vmask"], batch["cmask"]
        num, nmask, days = batch["num"], batch["nmask"], batch["days"]
        B, V, L = ids.shape

        # Normalize numeric values
        self._maybe_fit_stats(num, nmask)
        num_z = (num - self.num_mean) / self.num_std
        num_z = num_z * nmask  # zero out missing numeric values

        # Convert code ids into embeddings
        tok = self.tok_emb(ids)

        # Add visit-level CLS token to every visit
        vcls = self.visit_cls.expand(B * V, 1, -1)

        # Flatten visits so each visit can be processed independently
        flat = tok.reshape(B * V, L, -1)

        # Mask padded codes
        key_pad = torch.cat(
            [torch.zeros(B * V, 1, device=ids.device), 1 - cmask.reshape(B * V, L)],
            dim=1
        ).bool()

        # Visit sequence = [visit_cls] + code embeddings
        seq = torch.cat([vcls, flat], dim=1)

        # Visit-level transformer:
        # take the CLS output as the summary of all codes in one visit
        h_visit = self.visit_enc(seq, src_key_padding_mask=key_pad)[:, 0, :].reshape(B, V, -1)

        # Zero out padded visits
        h_visit = h_visit * vmask.unsqueeze(-1)

        # Numeric branch:
        # project normalized labs + vitals + masks into model dimension
        num_feat = self.num_proj(torch.cat([num_z, nmask], dim=-1))  # [B, V, D]

        # Create visit positions
        pos = torch.arange(V, device=ids.device).unsqueeze(0).expand(B, -1)
        vi = (torch.arange(V, device=ids.device) + 1).clamp_max(self.max_visits).unsqueeze(0).expand(B, -1)

        # MULTIMODAL FUSION STEP:
        # Here the model combines:
        # - h_visit = summary of discrete codes
        # - num_feat = summary of numeric labs/vitals
        # - time projection = visit timing information
        # - visit embedding = visit number
        # - positional embedding = sequence position
        #
        # IMPORTANT:
        # right now fusion is just simple addition.
        # That works as a baseline, but it is also the main weakness.
        # A better model would keep diagnoses, medications, labs, and vitals
        # separate longer and use learned fusion (MLP / gating / attention).
        pat_in = h_visit + num_feat + self.time_proj(days.unsqueeze(-1)) + self.visit_emb(vi) + self.pat_pos(pos)

        # Add patient-level CLS token
        pat_cls = self.pat_cls.expand(B, 1, -1)
        pat_seq = torch.cat([pat_cls, pat_in], dim=1)

        # Mask padded visits
        pat_mask = torch.cat([torch.ones(B, 1, device=ids.device), vmask], dim=1)

        # Patient-level transformer:
        # this learns patterns across visits over time
        h_pat = self.pat_enc(pat_seq, src_key_padding_mask=pat_mask < 0.5)

        # Final binary prediction from the patient CLS token
        return self.cls(self.dropout(h_pat[:, 0, :]))