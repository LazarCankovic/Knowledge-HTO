"""Time-aware knowledge-graph code embeddings (KerPrint local-KG propagation, pure PyTorch).

Each patient has its own graph (nodes are copies, so nothing is shared between patients in a
batch). With e_h the embedding of a receiving node h and e_u a neighbour reached through relation r
after tau days:

  f_tau        = W_t2 (1 - tanh^2(W_t1 tau / 180 + b_t1)) + b_t2               (KerPrint Eq. 1)
  pi(h,r,u,tau)= FFN(LN(M_r e_h) || LN(M_r e_u) || LN(f_tau))                  (Eq. 2, + LayerNorm)
  alpha        = softmax of pi over h's incoming edges                         (Eq. 3)
  e_N(h)       = sum_u alpha e_u                                               (Eq. 4)
  e_h^(l)      = LeakyReLU(W_l (e_h^(l-1) + e_N(h)^(l-1)))                     (Eq. 5)
  out          = Linear([e^(0) || ... || e^(L)])  (each L2-normalised; Linear zero-initialised)

Attention is computed once per forward pass from the layer-0 embeddings, as in the KerPrint code,
but **with gradients** (KerPrint computed it under torch.no_grad(), so its time MLP never trained).

The LayerNorms on the FFN inputs are not in KerPrint. Without them the attention is inert: TransR
embeddings have tiny raw norms (TransR scores normalised vectors), so the projections reach the FFN
at ~0.007 per dimension, ~10x smaller than f_tau. The logits then vary by ~0.006 across edges, the
softmax is ~uniform, and tau moves alpha by ~0.06% (measured on the demo graphs).
With zero_init_out (used by "add" fusion) the output layer starts at zero, so a model adding these
vectors starts out identical to the model without them. "gate" fusion (KGFusion) gets the same start
from its zero-initialised scale instead.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from hto.models.four_towers_timeaware_all_codes_cross_attn import VisitCodeTower


def edge_softmax(logits: torch.Tensor, dst: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Softmax of edge logits over the incoming edges of each destination node."""
    if logits.numel() == 0:
        return logits
    shift = torch.full((n_nodes,), float("-inf"), device=logits.device, dtype=logits.dtype)
    shift = shift.scatter_reduce(0, dst, logits.detach(), reduce="amax", include_self=True)
    ex = torch.exp(logits - shift[dst])
    den = torch.zeros(n_nodes, device=logits.device, dtype=logits.dtype).index_add(0, dst, ex)
    return ex / den[dst].clamp_min(1e-12)


class TimeAwareKGEmbedding(nn.Module):
    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        d_kg: int = 64,
        n_layers: int = 2,
        d_out: int = 128,
        time_dim: int = 16,
        tau_scale: float = 180.0,
        use_time: bool = True,
        dropout: float = 0.1,
        transr_path: str | Path | None = None,
        zero_init_out: bool = True,
    ):
        super().__init__()
        self.use_time = use_time
        self.tau_scale = tau_scale
        self.time_dim = time_dim

        self.entity = nn.Embedding(n_entities + 1, d_kg, padding_idx=0)
        self.W_R = nn.Parameter(torch.empty(n_relations + 1, d_kg, d_kg))
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.W_R)

        self.time_in = nn.Linear(1, time_dim)
        self.time_out = nn.Linear(time_dim, time_dim)
        self.ln_proj = nn.LayerNorm(d_kg)
        self.ln_time = nn.LayerNorm(time_dim)
        self.att = nn.Sequential(
            nn.Linear(2 * d_kg + time_dim, d_kg),
            nn.LeakyReLU(0.2),
            nn.Linear(d_kg, 1),
        )
        self.gcn = nn.ModuleList([nn.Linear(d_kg, d_kg) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear((n_layers + 1) * d_kg, d_out)
        if zero_init_out:     # "add" fusion: the KG term itself starts at 0
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        if transr_path is not None:
            self.load_transr(transr_path)

    def load_transr(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu")
        for name, src in (("entity", state["entity"]), ("W_R", state["W_R"])):
            dst = self.entity.weight if name == "entity" else self.W_R
            if tuple(src.shape) != tuple(dst.shape):
                raise ValueError(f"TransR {name} shape {tuple(src.shape)} != model {tuple(dst.shape)} "
                                 f"(check --kg-tag and the TransR --dim)")
            with torch.no_grad():
                dst.copy_(src)

    def time_feature(self, tau: torch.Tensor) -> torch.Tensor:
        if not self.use_time:
            return tau.new_zeros(tau.shape[0], self.time_dim)
        t = self.time_in((tau / self.tau_scale).unsqueeze(-1))
        return self.time_out(1.0 - torch.tanh(t) ** 2)

    def project(self, e: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        """M_r e for each edge, one matmul per relation (avoids an [E, d, d] gather)."""
        if rel.numel() == 0:
            return e
        order = torch.argsort(rel)
        rels, counts = torch.unique_consecutive(rel[order], return_counts=True)
        parts, start = [], 0
        for r, c in zip(rels.tolist(), counts.tolist()):
            idx = order[start:start + c]
            parts.append(e[idx] @ self.W_R[r])
            start += c
        out = torch.cat(parts)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=order.device)
        return out[inverse]

    def attention(self, e0, edge_index, edge_rel, edge_tau) -> torch.Tensor:
        src, dst = edge_index
        f_tau = self.time_feature(edge_tau)
        logits = self.att(torch.cat([
            self.ln_proj(self.project(e0[dst], edge_rel)),
            self.ln_proj(self.project(e0[src], edge_rel)),
            self.ln_time(f_tau) if self.use_time else f_tau,
        ], dim=-1)).squeeze(-1)
        return edge_softmax(logits, dst, e0.shape[0])

    def forward(self, node_ids, edge_index, edge_rel, edge_tau) -> torch.Tensor:
        """node_ids [N] global entity ids; edge_index [2, E] local (src, dst); returns [N, d_out]."""
        e0 = self.entity(node_ids)
        src, dst = edge_index
        alpha = self.attention(e0, edge_index, edge_rel, edge_tau)

        h = e0
        outs = [F.normalize(e0, dim=-1)]
        for lin in self.gcn:
            msg = self.dropout(alpha.unsqueeze(-1) * h[src])
            agg = torch.zeros_like(h).index_add(0, dst, msg)
            h = F.leaky_relu(lin(h + agg), 0.2)
            outs.append(F.normalize(h, dim=-1))
        return self.out(self.dropout(torch.cat(outs, dim=-1)))


class KGFusion(nn.Module):
    """Combines a code embedding with its KG vector.

    add   x = emb + kg                                (kg comes from a zero-initialised output layer)
    gate  kg_n = LayerNorm(kg) * has_node             (scale-matched to emb; 0 where there is no node)
          g    = sigmoid(W_g [emb || kg_n] + b_g)     (per code and dimension: how much to trust the KG)
          x    = emb + s * g * kg_n                   (s: learnable scalar, starts at 0)
    Both start exactly at x = emb. In gate mode s gets a gradient at the first step; the gate can learn
    to lean on the KG where the embedding is uninformative ([UNK] / rare codes) and less where it is not.
    """

    def __init__(self, d_model: int, mode: str = "gate"):
        super().__init__()
        if mode not in ("add", "gate"):
            raise ValueError(f"kg fusion must be 'add' or 'gate', got {mode!r}")
        self.mode = mode
        if mode == "gate":
            self.ln = nn.LayerNorm(d_model)
            self.gate = nn.Linear(2 * d_model, d_model)
            nn.init.zeros_(self.gate.weight)       # g = 0.5 everywhere at the start; learned from there
            nn.init.zeros_(self.gate.bias)
            self.scale = nn.Parameter(torch.zeros(1))
        self.last_gate: torch.Tensor | None = None  # [B, V, L] mean gate per position (diagnostics)

    def forward(self, emb: torch.Tensor, kg: torch.Tensor, has_node: torch.Tensor) -> torch.Tensor:
        if self.mode == "add":
            return emb + kg
        kg_n = self.ln(kg) * has_node.unsqueeze(-1).to(kg.dtype)
        g = torch.sigmoid(self.gate(torch.cat([emb, kg_n], dim=-1)))
        self.last_gate = g.detach().mean(-1)
        return emb + self.scale * g * kg_n


class KGVisitCodeTower(VisitCodeTower):
    """VisitCodeTower whose code embeddings are fused with a per-code KG vector (set extra_emb and
    extra_mask before forward). With extra_emb = None it computes exactly what VisitCodeTower computes."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, dropout, fusion: str = "gate"):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout)
        self.fusion = KGFusion(d_model, fusion)
        self.extra_emb: torch.Tensor | None = None
        self.extra_mask: torch.Tensor | None = None

    def forward(self, ids, mask, vmask):
        B, V, L = ids.shape

        x = self.emb(ids)
        if self.extra_emb is not None:
            x = self.fusion(x, self.extra_emb, self.extra_mask)
        x = x.reshape(B * V, L, -1)
        cls = self.cls.expand(B * V, 1, -1)
        seq = torch.cat([cls, x], dim=1)

        key_pad = torch.cat(
            [torch.zeros(B * V, 1, device=ids.device), 1 - mask.reshape(B * V, L)],
            dim=1,
        ).bool()

        h = self.enc(seq, src_key_padding_mask=key_pad)[:, 0, :]
        h = h.reshape(B, V, -1)

        return h * vmask.unsqueeze(-1)
