"""Hierarchical *_md baselines: per-visit token encoder → per-patient encoder.

Used by:
  - transformehr_md:   Transformer(visit tokens) → Transformer(visits)
  - multimodal_behrt_md: same as above plus a numeric branch per visit
  - tlstm_md:          Transformer(visit tokens) → T-LSTM cell over visits

CompositePE is applied at the PATIENT level (per-visit), which is the
operating level of TransformEHR's time additions.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .composite_pe import BiasableEncoder, CompositePE, RelativeDayBias
from .md_backbone import _SinPos, _parse_parts
from .multidomain_batcher import build_visit_batch


class _VisitEncoder(nn.Module):
    """A small transformer that reads a visit's token list and returns its
    [VCLS] embedding."""
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.vcls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.vcls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, tok_emb: torch.Tensor, cmask: torch.Tensor) -> torch.Tensor:
        # tok_emb [BV, L, D], cmask [BV, L]
        BV, L, D = tok_emb.shape
        seq = torch.cat([self.vcls.expand(BV, -1, -1), tok_emb], dim=1)
        kp = torch.cat([torch.zeros(BV, 1, device=tok_emb.device), 1 - cmask], dim=1).bool()
        h = self.enc(seq, src_key_padding_mask=kp)
        return h[:, 0, :]


class _PatientEncoder(nn.Module):
    """Transformer over per-visit embeddings with CompositePE & optional
    relative-day bias. Prepends [PCLS] and returns its hidden state."""
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float,
                 max_visits: int, pe_parts, rel_day_bias: bool):
        super().__init__()
        self.pcls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pcls, std=0.02)
        self.pe = CompositePE(
            d_model=d_model, parts=_parse_parts(pe_parts),
            n_domains=5, max_seq_len=max_visits + 4,
        )
        self.rel_day_bias_on = bool(rel_day_bias)
        if self.rel_day_bias_on:
            self.enc = BiasableEncoder(d_model, n_heads, n_layers, dropout)
            self.rel_bias = RelativeDayBias(n_heads=n_heads)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                dropout=dropout, activation="gelu", batch_first=True,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.rel_bias = None

    def forward(self, visit_emb: torch.Tensor, vmask: torch.Tensor,
                v_day: torch.Tensor, v_delta: torch.Tensor) -> torch.Tensor:
        B, V, D = visit_emb.shape
        visit_idx = torch.arange(V, device=visit_emb.device).unsqueeze(0).expand(B, -1)
        # for visit-level PE we use a STRUCT domain id (4) by default -- the
        # composite PE's `domain` term then reduces to a single shared vector.
        dom = torch.full_like(visit_idx, 4)
        # inject CompositePE additions first, then add PCLS row
        visit_emb = visit_emb + self.pe(visit_idx, v_day, v_delta, dom)
        seq = torch.cat([self.pcls.expand(B, -1, -1), visit_emb], dim=1)
        mask = torch.cat([torch.ones(B, 1, device=visit_emb.device), vmask], dim=1)
        kp = mask < 0.5
        if self.rel_bias is not None:
            days_full = torch.cat([torch.zeros(B, 1, device=visit_emb.device), v_day], dim=1)
            attn_bias = self.rel_bias(days_full, key_pad_mask=None)
            h = self.enc(seq, key_pad_mask=kp, attn_bias=attn_bias)
        else:
            h = self.enc(seq, src_key_padding_mask=kp)
        return h[:, 0, :]


def _embed_visits(tok_emb_module: nn.Embedding, domain_emb_module: nn.Embedding | None,
                  num_proj_module: nn.Module | None,
                  batch: dict, include_numeric: bool) -> torch.Tensor:
    """Produce [B, V, L, D] token-level embedding summed with domain + numeric."""
    ids = batch["ids"]                      # [B, V, L]
    domain = batch["domain"]                # [B, V, L]
    x = tok_emb_module(ids)
    if domain_emb_module is not None:
        x = x + domain_emb_module(domain.clamp_min(0))
    if include_numeric and num_proj_module is not None:
        num = batch["num"] * batch["num_mask"]
        x = x + num_proj_module(num.unsqueeze(-1))
    return x


class TransformEHR_MD(nn.Module):
    name = "transformehr_md"
    needs_variant = "multidomain_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_visit_layers: int = 2, n_patient_layers: int = 2,
                 max_visits: int = 64, max_codes: int = 96, dropout: float = 0.1,
                 pe_parts="abs,gap,domain,interaction", rel_day_bias: bool = True):
        super().__init__()
        self.max_visits = max_visits
        self.max_codes = max_codes
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.domain_emb = nn.Embedding(5, d_model)
        self.visit_enc = _VisitEncoder(d_model, n_heads, n_visit_layers, dropout)
        self.pat = _PatientEncoder(d_model, n_heads, n_patient_layers, dropout,
                                   max_visits, pe_parts, rel_day_bias)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        return build_visit_batch(items, max_visits=self.max_visits, max_codes=self.max_codes)

    def forward(self, batch) -> torch.Tensor:
        tok = _embed_visits(self.tok_emb, self.domain_emb, None, batch, include_numeric=False)
        B, V, L, D = tok.shape
        visit_flat_emb = tok.reshape(B * V, L, D)
        cmask_flat = batch["cmask"].reshape(B * V, L)
        visit_vec = self.visit_enc(visit_flat_emb, cmask_flat).reshape(B, V, D)
        visit_vec = visit_vec * batch["vmask"].unsqueeze(-1)
        pat = self.pat(visit_vec, batch["vmask"], batch["v_day"], batch["v_delta"])
        return self.cls(self.dropout(pat))


class MultimodalBEHRT_MD(nn.Module):
    name = "multimodal_behrt_md"
    needs_variant = "multidomain_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_visit_layers: int = 2, n_patient_layers: int = 2,
                 max_visits: int = 64, max_codes: int = 96, dropout: float = 0.1,
                 pe_parts="abs,gap,domain,interaction", rel_day_bias: bool = True):
        super().__init__()
        self.max_visits = max_visits
        self.max_codes = max_codes
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.domain_emb = nn.Embedding(5, d_model)
        # numeric projector: takes per-token scalar value -> d_model contribution
        self.num_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.visit_enc = _VisitEncoder(d_model, n_heads, n_visit_layers, dropout)
        self.pat = _PatientEncoder(d_model, n_heads, n_patient_layers, dropout,
                                   max_visits, pe_parts, rel_day_bias)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

        # running stats for per-token numeric values (computed once on first training batch)
        self.register_buffer("num_mean", torch.zeros(1))
        self.register_buffer("num_std",  torch.ones(1))
        self._stats_fit = False

    def _maybe_fit_stats(self, batch):
        if self._stats_fit or not self.training:
            return
        v = batch["num"][batch["num_mask"] > 0.5]
        if v.numel() < 32:
            return
        self.num_mean.copy_(v.mean().detach())
        self.num_std.copy_(v.std().clamp_min(1e-3).detach())
        self._stats_fit = True

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        return build_visit_batch(items, max_visits=self.max_visits, max_codes=self.max_codes)

    def forward(self, batch) -> torch.Tensor:
        self._maybe_fit_stats(batch)
        num_z = (batch["num"] - self.num_mean) / self.num_std
        num_z = num_z * batch["num_mask"]
        batch = dict(batch); batch["num"] = num_z
        tok = _embed_visits(self.tok_emb, self.domain_emb, self.num_proj, batch, include_numeric=True)
        B, V, L, D = tok.shape
        cmask_flat = batch["cmask"].reshape(B * V, L)
        visit_vec = self.visit_enc(tok.reshape(B * V, L, D), cmask_flat).reshape(B, V, D)
        visit_vec = visit_vec * batch["vmask"].unsqueeze(-1)
        pat = self.pat(visit_vec, batch["vmask"], batch["v_day"], batch["v_delta"])
        return self.cls(self.dropout(pat))


class TLSTMCell_MD(nn.Module):
    """T-LSTM cell from the original paper: short-term memory decayed by
    g(dt) = 1/log(e + dt) between visits."""
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.h = d_hidden
        self.Wi = nn.Linear(d_in, d_hidden); self.Ui = nn.Linear(d_hidden, d_hidden, bias=False)
        self.Wf = nn.Linear(d_in, d_hidden); self.Uf = nn.Linear(d_hidden, d_hidden, bias=False)
        self.Wo = nn.Linear(d_in, d_hidden); self.Uo = nn.Linear(d_hidden, d_hidden, bias=False)
        self.Wc = nn.Linear(d_in, d_hidden); self.Uc = nn.Linear(d_hidden, d_hidden, bias=False)
        self.Wd = nn.Linear(d_hidden, d_hidden)

    def forward(self, x, dt, h, c):
        C_s = torch.tanh(self.Wd(c))
        g = 1.0 / torch.log(torch.exp(torch.tensor(1.0, device=dt.device)) + dt.clamp_min(0.0))
        C_s_hat = C_s * g.unsqueeze(-1)
        C_star = (c - C_s) + C_s_hat
        i = torch.sigmoid(self.Wi(x) + self.Ui(h))
        f = torch.sigmoid(self.Wf(x) + self.Uf(h))
        o = torch.sigmoid(self.Wo(x) + self.Uo(h))
        gc = torch.tanh(self.Wc(x) + self.Uc(h))
        c_new = f * C_star + i * gc
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class TLSTM_MD(nn.Module):
    name = "tlstm_md"
    needs_variant = "multidomain_visit"

    def __init__(self, vocab_size: int, d_model: int = 128, hidden: int = 128,
                 n_visit_layers: int = 1, max_visits: int = 64, max_codes: int = 96,
                 dropout: float = 0.2, n_heads: int = 4,
                 pe_parts="abs,gap,domain,interaction", rel_day_bias: bool = False):
        super().__init__()
        # rel_day_bias is defined here for uniformity but TLSTM is sequential
        # and doesn't consume attention bias; we ignore it silently.
        self.max_visits = max_visits
        self.max_codes = max_codes
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.domain_emb = nn.Embedding(5, d_model)
        self.visit_enc = _VisitEncoder(d_model, n_heads, n_visit_layers, dropout)
        self.pe = CompositePE(
            d_model=d_model, parts=_parse_parts(pe_parts),
            n_domains=5, max_seq_len=max_visits + 4,
        )
        self.cell = TLSTMCell_MD(d_model, hidden)
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def build_batch(self, items: List[dict]) -> dict[str, torch.Tensor]:
        return build_visit_batch(items, max_visits=self.max_visits, max_codes=self.max_codes)

    def forward(self, batch) -> torch.Tensor:
        tok = _embed_visits(self.tok_emb, self.domain_emb, None, batch, include_numeric=False)
        B, V, L, D = tok.shape
        visit_vec = self.visit_enc(
            tok.reshape(B * V, L, D),
            batch["cmask"].reshape(B * V, L),
        ).reshape(B, V, D)
        # add CompositePE at visit level
        visit_idx = torch.arange(V, device=tok.device).unsqueeze(0).expand(B, -1)
        dom = torch.full_like(visit_idx, 4)
        visit_vec = visit_vec + self.pe(visit_idx, batch["v_day"], batch["v_delta"], dom)
        visit_vec = visit_vec * batch["vmask"].unsqueeze(-1)

        h = torch.zeros(B, self.hidden, device=tok.device)
        c = torch.zeros(B, self.hidden, device=tok.device)
        for t in range(V):
            dt = batch["v_delta"][:, t]
            alive = batch["vmask"][:, t].unsqueeze(-1)
            h_new, c_new = self.cell(visit_vec[:, t, :], dt, h, c)
            h = alive * h_new + (1 - alive) * h
            c = alive * c_new + (1 - alive) * c
        return self.cls(self.dropout(h))
