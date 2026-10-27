"""HTR-Former with patient-graph knowledge fused into lab code tokens only."""
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
TOKENS_PER_LAB = 4  # [LCODE], code, [LVAL], value bin


class HTRFormerLabKG(HTRFormerCrossAttn):
    """Keep the four-domain predictor; replace only its lab tower with KG fusion.

    Patient graphs use lab item IDs, independent of the model vocabulary. The
    KG vector is attached to the *code* token at position 4*k+1 for lab k;
    value-bin, marker and padding tokens retain their original embeddings.
    """

    name = "htr_former_lab_kg"

    def __init__(self, vocab_size, d_model=128, n_heads=4,
                 n_visit_layers=2, n_patient_layers=2, max_visits=64,
                 dropout=0.1, n_classes=2, time_fn="log",
                 kg_patient_graphs: str | Path | None = None,
                 kg_transr: str | Path | None = None, kg_layers: int = 2,
                 kg_dim: int = 64, kg_time: bool = True,
                 kg_tau_mode: str = "real", kg_fusion: str = "gate",
                 kg_scale_lr_mult: float = 30.0, **kwargs):
        super().__init__(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                         n_visit_layers=n_visit_layers, n_patient_layers=n_patient_layers,
                         max_visits=max_visits, dropout=dropout,
                         n_classes=n_classes, time_fn=time_fn)
        if kg_patient_graphs is None:
            raise ValueError("htr_former_lab_kg needs kg_patient_graphs; build the lab patient graphs first")
        if kg_tau_mode not in ("real", "shuffled"):
            raise ValueError(f"Unknown kg_tau_mode: {kg_tau_mode!r}")
        with open(kg_patient_graphs, "rb") as f:
            data = pickle.load(f)
        if data.get("domain") != "lab":
            raise ValueError("Expected lab patient graphs; diagnosis graphs cannot be used for labs")
        self.patient_graphs = data["graphs"]
        self.kg_tau_mode = kg_tau_mode
        self.kg_scale_lr_mult = kg_scale_lr_mult

        if kg_transr is not None and Path(kg_transr).exists():
            kg_dim = int(torch.load(kg_transr, map_location="cpu")["entity"].shape[1])
        else:
            kg_transr = None
        self.kg = TimeAwareKGEmbedding(
            n_entities=data["n_entities"], n_relations=data["n_relations"],
            d_kg=kg_dim, n_layers=kg_layers, d_out=d_model,
            use_time=kg_time, dropout=dropout, transr_path=kg_transr,
            zero_init_out=(kg_fusion == "add"),
        )
        self.lab_tower = KGVisitCodeTower(
            vocab_size, d_model, n_heads, n_visit_layers, dropout, fusion=kg_fusion
        )

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        fusion = self.lab_tower.fusion
        if fusion.mode != "gate":
            return [{"params": list(self.parameters())}]
        rest = [p for n, p in self.named_parameters() if n != "lab_tower.fusion.scale"]
        return [{"params": rest},
                {"params": [fusion.scale], "lr": lr * self.kg_scale_lr_mult,
                 "weight_decay": 0.0}]

    def build_batch(self, items):
        batch = build_allcodes_batch(
            items, max_visits=MAX_VISITS, max_diag_codes=MAX_DIAG,
            max_med_codes=MAX_MED, max_lab_codes=MAX_LAB, max_vit_codes=MAX_VIT,
        )
        node_ids, edge_index, edge_rel, edge_tau = [], [], [], []
        kg_idx = torch.full((len(items), MAX_VISITS, MAX_LAB), -1, dtype=torch.long)
        offset = 0
        for i, it in enumerate(items):
            g = self.patient_graphs.get(int(it["patient_id"]))
            # Validate the data even for patients without a matching graph.
            for j, visit in enumerate(it["visits"][-MAX_VISITS:]):
                blocks = visit["blocks"].get("lab", [])
                raw = visit.get("blocks_raw", {}).get("lab", [])
                n_measurements = max(0, len(blocks) - 1)  # omit [EOL]
                if len(raw) != n_measurements:
                    raise ValueError("Lab item IDs are missing/misaligned in processed data; "
                                     "rebuild canonical and allcodes_timeaware with this branch")
                if g is None:
                    continue
                for k, code in enumerate(raw[: MAX_LAB // TOKENS_PER_LAB]):
                    position = TOKENS_PER_LAB * k + 1
                    if code is not None and batch["lab_mask"][i, j, position] > 0:
                        li = g["code_index"].get(code)
                        if li is not None:
                            kg_idx[i, j, position] = offset + li
            if g is None:
                continue
            tau = g["edge_tau"]
            if self.kg_tau_mode == "shuffled":
                tau = np.random.default_rng(int(it["patient_id"])).permutation(tau)
            node_ids.append(g["node_ids"])
            edge_index.append(g["edge_index"] + offset)
            edge_rel.append(g["edge_rel"])
            edge_tau.append(tau)
            offset += len(g["node_ids"])

        def cat(xs, dtype):
            return torch.as_tensor(np.concatenate(xs) if xs else np.zeros(0), dtype=dtype)

        batch["kg_node_ids"] = cat(node_ids, torch.long)
        batch["kg_edge_index"] = (torch.as_tensor(np.concatenate(edge_index, axis=1), dtype=torch.long)
                                  if edge_index else torch.zeros((2, 0), dtype=torch.long))
        batch["kg_edge_rel"] = cat(edge_rel, torch.long)
        batch["kg_edge_tau"] = cat(edge_tau, torch.float32)
        batch["lab_kg_idx"] = kg_idx
        return batch

    def kg_code_vectors(self, batch) -> torch.Tensor:
        h = self.kg(batch["kg_node_ids"], batch["kg_edge_index"],
                    batch["kg_edge_rel"], batch["kg_edge_tau"])
        h = torch.cat([h.new_zeros(1, h.shape[1]), h], dim=0)
        return h[batch["lab_kg_idx"] + 1]

    @torch.no_grad()
    def kg_diagnostics(self, items, batch_size: int = 32, device: str = "cpu") -> dict:
        was_training = self.training
        self.eval()
        ratios, linked, possible = [], 0, 0
        max_dp_removed, max_dp_time_off = 0.0, 0.0
        fusion = self.lab_tower.fusion
        unk_id = bb.TOKEN_TO_ID.get("[UNK]", 1)
        gate_unk, gate_known = [], []
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            possible += sum(min(len(v.get("blocks_raw", {}).get("lab", [])), MAX_LAB // 4)
                            for it in chunk for v in it["visits"][-MAX_VISITS:])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in self.build_batch(chunk).items()}
            pos = batch["lab_kg_idx"] >= 0
            linked += int(pos.sum().item())
            if pos.any():
                emb = self.lab_tower.emb(batch["lab_ids"])
                delta = fusion(emb, self.kg_code_vectors(batch), pos) - emb
                ratios.append((delta[pos].norm(dim=-1) / emb[pos].norm(dim=-1).clamp_min(1e-12)).mean().item())
                if fusion.mode == "gate":
                    unk = pos & (batch["lab_ids"] == unk_id)
                    gate_unk += fusion.last_gate[unk].tolist()
                    gate_known += fusion.last_gate[pos & ~unk].tolist()
            p = torch.softmax(self(batch), dim=-1)[:, 1]
            no_kg = dict(batch, lab_kg_idx=torch.full_like(batch["lab_kg_idx"], -1))
            max_dp_removed = max(max_dp_removed,
                                 (p - torch.softmax(self(no_kg), dim=-1)[:, 1]).abs().max().item())
            old = self.kg.use_time
            try:
                self.kg.use_time = False
                max_dp_time_off = max(max_dp_time_off,
                    (p - torch.softmax(self(batch), dim=-1)[:, 1]).abs().max().item())
            finally:
                self.kg.use_time = old
        self.train(was_training)
        out = {"linked_lab_codes": linked, "total_lab_codes": possible,
               "lab_kg_coverage": linked / possible if possible else 0.0,
               "kg_to_emb_ratio": float(np.mean(ratios)) if ratios else 0.0,
               "max_dp_kg_removed": max_dp_removed,
               "max_dp_time_off": max_dp_time_off}
        if fusion.mode == "gate":
            out.update(gate_scale=float(fusion.scale.item()),
                       gate_mean_unk=float(np.mean(gate_unk)) if gate_unk else None,
                       gate_mean_known=float(np.mean(gate_known)) if gate_known else None)
        return out

    def forward(self, batch):
        self.lab_tower.extra_emb = self.kg_code_vectors(batch)
        self.lab_tower.extra_mask = batch["lab_kg_idx"] >= 0
        try:
            return super().forward(batch)
        finally:
            self.lab_tower.extra_emb = None
            self.lab_tower.extra_mask = None
