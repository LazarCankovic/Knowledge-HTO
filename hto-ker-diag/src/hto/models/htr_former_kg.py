from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from hto.data import batch_builder_all_codes_local as bb
from hto.data.batch_builder_all_codes_local import build_allcodes_batch
from hto.models.htr_former_cross_attn import HTRFormerCrossAttn
from hto.models.kg_code_embedding import KGVisitCodeTower, TimeAwareKGEmbedding

MAX_VISITS, MAX_DIAG, MAX_MED, MAX_LAB, MAX_VIT = 64, 64, 64, 128, 128


class HTRFormerKG(HTRFormerCrossAttn):
    """HTR-Former (cross-domain attention) with a KG-enriched diagnosis tower.

    Each diagnosis code embedding is fused with KG(code | patient's time-aware graph) (see KGFusion):
      gate (default)  emb + s * sigmoid(W [emb || LN(kg)]) * LN(kg), s starting at 0
      add             emb + kg, kg from a zero-initialised output layer
    Either way training starts from plain HTR-Former.
    Codes without a graph node, and codes the vocab maps to [UNK], still get their KG term when the code
    is in the patient's graph: the lookup uses the raw code strings (visit["blocks_raw"]["diag"]).

    kg_patient_graphs  processed_data/<dataset>/patient_graphs.<kg-tag>.pkl (build_patient_graphs.py)
    kg_transr          processed_data/kg/<kg-tag>/transr.pt to initialise entity / relation matrices
    kg_time            False drops tau from the attention (ablation)
    kg_tau_mode        "real" | "shuffled" (tau permuted within each patient's graph: a time control)
    kg_fusion          "gate" | "add"
    kg_scale_lr_mult   learning-rate multiplier for the gate's scale s (its own param group, no weight
                       decay). s is one number starting at 0; at the base lr it moves ~lr per step, so the
                       KG entered at ~1% of the embedding's size after 150 steps (demo probe).
    """

    name = "htr_former_kg"

    def __init__(self, vocab_size, d_model=128, n_heads=4,
                 n_visit_layers=2, n_patient_layers=2,
                 max_visits=64, dropout=0.1, n_classes=2,
                 time_fn="log",
                 kg_patient_graphs: str | Path | None = None,
                 kg_transr: str | Path | None = None,
                 kg_layers: int = 2,
                 kg_dim: int = 64,
                 kg_time: bool = True,
                 kg_tau_mode: str = "real",
                 kg_fusion: str = "gate",
                 kg_scale_lr_mult: float = 30.0,
                 **kwargs):
        super().__init__(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_visit_layers=n_visit_layers, n_patient_layers=n_patient_layers,
            max_visits=max_visits, dropout=dropout,
            n_classes=n_classes, time_fn=time_fn,
        )
        if kg_patient_graphs is None:
            raise ValueError("htr_former_kg needs kg_patient_graphs (run build_patient_graphs.py, pass --kg-tag)")
        if kg_tau_mode not in ("real", "shuffled"):
            raise ValueError(f"kg_tau_mode must be 'real' or 'shuffled', got {kg_tau_mode!r}")
        with open(kg_patient_graphs, "rb") as f:
            data = pickle.load(f)
        self.patient_graphs: dict = data["graphs"]
        self.kg_tau_mode = kg_tau_mode
        self.kg_scale_lr_mult = kg_scale_lr_mult

        if kg_transr is not None and Path(kg_transr).exists():
            kg_dim = int(torch.load(kg_transr, map_location="cpu")["entity"].shape[1])
        else:
            kg_transr = None
        self.kg = TimeAwareKGEmbedding(
            n_entities=data["n_entities"], n_relations=data["n_relations"],
            d_kg=kg_dim, n_layers=kg_layers, d_out=d_model, use_time=kg_time,
            dropout=dropout, transr_path=kg_transr, zero_init_out=(kg_fusion == "add"),
        )
        # same architecture as the inherited tower, plus the KG fusion
        self.diag_tower = KGVisitCodeTower(vocab_size, d_model, n_heads, n_visit_layers, dropout, fusion=kg_fusion)

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """Optimizer groups (used by train.py): the gate scale gets lr * kg_scale_lr_mult and no weight decay."""
        fusion = self.diag_tower.fusion
        if fusion.mode != "gate":
            return [{"params": list(self.parameters())}]
        rest = [p for n, p in self.named_parameters() if n != "diag_tower.fusion.scale"]
        return [{"params": rest},
                {"params": [fusion.scale], "lr": lr * self.kg_scale_lr_mult, "weight_decay": 0.0}]

    def build_batch(self, items):
        batch = build_allcodes_batch(
            items,
            max_visits=MAX_VISITS,
            max_diag_codes=MAX_DIAG,
            max_med_codes=MAX_MED,
            max_lab_codes=MAX_LAB,
            max_vit_codes=MAX_VIT,
        )
        node_ids, edge_index, edge_rel, edge_tau = [], [], [], []
        diag_kg_idx = torch.full((len(items), MAX_VISITS, MAX_DIAG), -1, dtype=torch.long)
        offset = 0
        for i, it in enumerate(items):
            pid = int(it["patient_id"])
            g = self.patient_graphs.get(pid)
            if g is None:
                continue
            tau = g["edge_tau"]
            if self.kg_tau_mode == "shuffled":
                tau = np.random.default_rng(pid).permutation(tau)
            node_ids.append(g["node_ids"])
            edge_index.append(g["edge_index"] + offset)
            edge_rel.append(g["edge_rel"])
            edge_tau.append(tau)
            # same visit / code positions as build_allcodes_batch: last MAX_VISITS visits, first MAX_DIAG codes
            for j, visit in enumerate(it["visits"][-MAX_VISITS:]):
                for k, code in enumerate(visit.get("blocks_raw", {}).get("diag", [])[:MAX_DIAG]):
                    li = g["code_index"].get(code)
                    if li is not None:
                        diag_kg_idx[i, j, k] = offset + li
            offset += len(g["node_ids"])

        cat = lambda xs, dtype, empty: torch.as_tensor(np.concatenate(xs) if xs else np.zeros(empty), dtype=dtype)
        batch["kg_node_ids"] = cat(node_ids, torch.long, 0)
        batch["kg_edge_index"] = (torch.as_tensor(np.concatenate(edge_index, axis=1), dtype=torch.long)
                                  if edge_index else torch.zeros((2, 0), dtype=torch.long))
        batch["kg_edge_rel"] = cat(edge_rel, torch.long, 0)
        batch["kg_edge_tau"] = cat(edge_tau, torch.float32, 0)
        batch["diag_kg_idx"] = diag_kg_idx
        return batch

    def kg_code_vectors(self, batch) -> torch.Tensor:
        """[B, V, L, d_model] KG term per diagnosis position (zeros where the code has no graph node)."""
        h = self.kg(batch["kg_node_ids"], batch["kg_edge_index"], batch["kg_edge_rel"], batch["kg_edge_tau"])
        h = torch.cat([h.new_zeros(1, h.shape[1]), h], dim=0)
        return h[batch["diag_kg_idx"] + 1]

    @torch.no_grad()
    def kg_diagnostics(self, items, batch_size: int = 32, device: str = "cpu") -> dict:
        """How much the trained KG branch matters, on `items` (e.g. the test set):
          kg_to_emb_ratio     mean |fused embedding - code embedding| / |code embedding| over diagnosis
                              positions with a graph node (what the KG actually adds after fusion)
          max_dp_kg_removed   largest change in P(controlled) when the KG term is removed
          max_dp_time_off     largest change in P(controlled) when tau is dropped from the attention
          gate_scale, gate_mean_unk, gate_mean_known   (gate fusion) the learned scale s and the mean gate
                              at codes the vocab maps to [UNK] vs codes it knows
        A ratio near 0 means the branch never engaged (try more steps or a higher lr)."""
        was_training = self.training
        self.eval()
        fusion = self.diag_tower.fusion
        unk_id = bb.TOKEN_TO_ID.get("[UNK]", 1)
        ratios, dp_kg, dp_time = [], 0.0, 0.0
        gate_unk, gate_known = [], []
        for i in range(0, len(items), batch_size):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in self.build_batch(items[i:i + batch_size]).items()}
            pos = batch["diag_kg_idx"] >= 0
            if pos.any():
                emb = self.diag_tower.emb(batch["diag_ids"])
                delta = fusion(emb, self.kg_code_vectors(batch), pos) - emb
                ratios.append((delta[pos].norm(dim=-1) / emb[pos].norm(dim=-1)).mean().item())
                if fusion.mode == "gate":
                    unk = pos & (batch["diag_ids"] == unk_id)
                    gate_unk += fusion.last_gate[unk].tolist()
                    gate_known += fusion.last_gate[pos & ~unk].tolist()
            p = torch.softmax(self(batch), -1)[:, 1]
            no_kg = dict(batch, diag_kg_idx=torch.full_like(batch["diag_kg_idx"], -1))
            dp_kg = max(dp_kg, (p - torch.softmax(self(no_kg), -1)[:, 1]).abs().max().item())
            use_time = self.kg.use_time
            self.kg.use_time = False
            dp_time = max(dp_time, (p - torch.softmax(self(batch), -1)[:, 1]).abs().max().item())
            self.kg.use_time = use_time
        self.train(was_training)
        out = {"kg_to_emb_ratio": float(np.mean(ratios)) if ratios else 0.0,
               "max_dp_kg_removed": dp_kg, "max_dp_time_off": dp_time}
        if fusion.mode == "gate":
            out.update(gate_scale=float(fusion.scale.item()),
                       gate_mean_unk=float(np.mean(gate_unk)) if gate_unk else None,
                       gate_mean_known=float(np.mean(gate_known)) if gate_known else None)
        return out

    def forward(self, batch):
        self.diag_tower.extra_emb = self.kg_code_vectors(batch)
        self.diag_tower.extra_mask = batch["diag_kg_idx"] >= 0
        try:
            return super().forward(batch)
        finally:
            self.diag_tower.extra_emb = None
            self.diag_tower.extra_mask = None
