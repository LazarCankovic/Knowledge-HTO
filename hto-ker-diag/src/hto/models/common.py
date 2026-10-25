"""Shared helpers for the 7 baselines.

A baseline exposes:
    class Baseline:
        name: str
        needs_variant: str          # "codes_visit" | "codes_flat" | "multimodal" | "text_sequence"
        def __init__(self, vocab_size, **hp): ...
        def build_batch(self, items) -> dict[str, torch.Tensor]
        def forward(self, batch) -> torch.Tensor  # logits [B, 2]
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn


PAD_ID = 0  # Vocab specials are [PAD, UNK, CLS, SEP, MASK] → PAD=0, CLS=2


def pad_2d(seq_list: List[List[int]], pad_value: int = 0, max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """List[List[int]] -> (ids [B, L], mask [B, L])."""
    L = max_len if max_len is not None else max((len(s) for s in seq_list), default=1)
    L = max(L, 1)
    B = len(seq_list)
    ids = torch.full((B, L), pad_value, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.float32)
    for i, s in enumerate(seq_list):
        s = s[-L:]  # keep last L (most recent)
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1.0
    return ids, mask


def pad_3d(seq_of_seq: List[List[List[int]]], pad_value: int = 0, max_visits: int | None = None, max_codes: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """List[List[List[int]]] -> (ids [B, V, L], visit_mask [B, V], code_mask [B, V, L])."""
    V = max_visits if max_visits is not None else max((len(p) for p in seq_of_seq), default=1)
    L = max_codes if max_codes is not None else max((len(v) for p in seq_of_seq for v in p), default=1)
    V, L = max(V, 1), max(L, 1)
    B = len(seq_of_seq)
    ids = torch.full((B, V, L), pad_value, dtype=torch.long)
    vmask = torch.zeros((B, V), dtype=torch.float32)
    cmask = torch.zeros((B, V, L), dtype=torch.float32)
    for i, pat in enumerate(seq_of_seq):
        pat = pat[-V:]  # most recent V
        for j, v in enumerate(pat):
            v = v[:L]
            ids[i, j, : len(v)] = torch.tensor(v, dtype=torch.long)
            cmask[i, j, : len(v)] = 1.0
            if len(v):
                vmask[i, j] = 1.0
    return ids, vmask, cmask


def pad_floats(seq_list: List[List[float]], pad_value: float = 0.0, max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    L = max_len if max_len is not None else max((len(s) for s in seq_list), default=1)
    L = max(L, 1)
    B = len(seq_list)
    x = torch.full((B, L), float(pad_value), dtype=torch.float32)
    m = torch.zeros((B, L), dtype=torch.float32)
    for i, s in enumerate(seq_list):
        s = s[-L:]
        x[i, : len(s)] = torch.tensor(s, dtype=torch.float32)
        m[i, : len(s)] = 1.0
    return x, m


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: long tensor of any shape
        flat = positions.clamp_min(0).long()
        out = self.pe[flat]
        return out


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits [B, 2], targets [B] long in {0, 1}
        ls = torch.nn.functional.log_softmax(logits, dim=-1)
        t1 = targets.float()
        t0 = 1 - t1
        s = self.smoothing
        probs_target_0 = t0 * (1 - s) + (1 - t0) * s
        probs_target_1 = t1 * (1 - s) + (1 - t1) * s
        loss = -(probs_target_0 * ls[:, 0] + probs_target_1 * ls[:, 1])
        return loss.mean()
