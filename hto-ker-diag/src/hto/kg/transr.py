"""TransR pretraining on the global graph (KerPrint / KGAT style).

score(h, r, t) = || W_r e_h + e_r - W_r e_t ||^2   (projected vectors L2-normalised, as in KGAT)
loss           = -log sigmoid(score(h, r, t') - score(h, r, t)) + l2 * ||embeddings||^2
with t' a uniformly sampled corrupt tail. Ids are 1-based; row 0 is padding.

A random 5% of triples is held out and reported as hits@10 against 100 random candidate tails
(chance = 0.10), so a broken run is visible before the embeddings are used.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransRConfig:
    dim: int = 64
    epochs: int = 50
    batch_size: int = 2048
    lr: float = 1e-3
    l2: float = 1e-5
    holdout_frac: float = 0.05
    device: str = "cpu"
    seed: int = 42


class TransR(nn.Module):
    def __init__(self, n_entities: int, n_relations: int, dim: int):
        super().__init__()
        self.entity = nn.Embedding(n_entities + 1, dim, padding_idx=0)
        self.relation = nn.Embedding(n_relations + 1, dim, padding_idx=0)
        self.W_R = nn.Parameter(torch.empty(n_relations + 1, dim, dim))
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        nn.init.xavier_uniform_(self.W_R)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        W = self.W_R[r]
        rh = F.normalize(torch.bmm(self.entity(h).unsqueeze(1), W).squeeze(1), dim=-1)
        rt = F.normalize(torch.bmm(self.entity(t).unsqueeze(1), W).squeeze(1), dim=-1)
        er = F.normalize(self.relation(r), dim=-1)
        return ((rh + er - rt) ** 2).sum(-1)

    def loss(self, h, r, t, t_neg, l2: float) -> torch.Tensor:
        pos, neg = self.score(h, r, t), self.score(h, r, t_neg)
        reg = (self.entity(h).pow(2).sum(-1) + self.entity(t).pow(2).sum(-1)
               + self.entity(t_neg).pow(2).sum(-1) + self.relation(r).pow(2).sum(-1)).mean()
        return -F.logsigmoid(neg - pos).mean() + l2 * reg


@torch.no_grad()
def hits_at_10(model: TransR, triples: torch.Tensor, n_entities: int, n_candidates: int = 100) -> float:
    """Share of held-out triples whose true tail ranks in the top 10 of itself + random candidates."""
    if len(triples) == 0:
        return float("nan")
    gen = torch.Generator().manual_seed(0)
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    cand = torch.randint(1, n_entities + 1, (len(triples), n_candidates - 1), generator=gen).to(h.device)
    true_score = model.score(h, r, t)
    k = cand.shape[1]
    cand_score = model.score(h.repeat_interleave(k), r.repeat_interleave(k), cand.reshape(-1)).view(-1, k)
    rank = 1 + (cand_score < true_score.unsqueeze(1)).sum(1)
    return float((rank <= 10).float().mean())


def pretrain_transr(graph: dict, cfg: TransRConfig, out_path: Path) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n_ent, n_rel = len(graph["entities"]), len(graph["relations"])
    triples = torch.as_tensor(graph["triples"][["h", "r", "t"]].to_numpy(), dtype=torch.long)

    perm = torch.as_tensor(rng.permutation(len(triples)))
    n_hold = int(len(triples) * cfg.holdout_frac)
    held, train = triples[perm[:n_hold]], triples[perm[n_hold:]]

    device = torch.device(cfg.device)
    model = TransR(n_ent, n_rel, cfg.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    held = held.to(device)
    print(f"[transr] entities={n_ent:,} relations={n_rel} train_triples={len(train):,} held_out={n_hold:,}")
    print(f"[transr] epoch 0 hits@10={hits_at_10(model, held, n_ent):.3f} (chance 0.10)")

    history = []
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        order = torch.randperm(len(train))
        total = 0.0
        for i in range(0, len(train), cfg.batch_size):
            batch = train[order[i:i + cfg.batch_size]].to(device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            t_neg = torch.randint(1, n_ent + 1, t.shape, device=device)
            loss = model.loss(h, r, t, t_neg, cfg.l2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        history.append(total / len(train))
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"[transr] epoch {epoch} loss={history[-1]:.4f} hits@10={hits_at_10(model, held, n_ent):.3f} "
                  f"({time.time() - start:.0f}s)")

    final_hits = hits_at_10(model, held, n_ent)
    out = {
        "entity": model.entity.weight.detach().cpu(),
        "relation": model.relation.weight.detach().cpu(),
        "W_R": model.W_R.detach().cpu(),
        "config": asdict(cfg),
        "n_entities": n_ent,
        "n_relations": n_rel,
        "loss_history": history,
        "heldout_hits_at_10": final_hits,
    }
    torch.save(out, out_path)
    print(f"[transr] saved {out_path}  held-out hits@10={final_hits:.3f}")
    return out
