"""Tests for the KG-enriched diagnosis tower (htr_former_kg).

    python -m pytest tests/test_kg.py -q

Synthetic tests need no data. The demo integration test runs only when processed_data/demo and its
patient graphs exist (build_data.py + build_kg.py + build_patient_graphs.py).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hto.data import batch_builder_all_codes_local as bb  # noqa: E402
from hto.models.htr_former_cross_attn import HTRFormerCrossAttn  # noqa: E402
from hto.models.htr_former_kg import HTRFormerKG  # noqa: E402
from hto.models.kg_code_embedding import TimeAwareKGEmbedding, edge_softmax  # noqa: E402

N_ENT, N_REL, D = 30, 4, 16


def tiny_graph(seed: int, n_nodes: int = 6, n_edges: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "node_ids": rng.choice(np.arange(1, N_ENT + 1), n_nodes, replace=False),
        "edge_index": rng.integers(0, n_nodes, (2, n_edges)),
        "edge_rel": rng.integers(1, N_REL + 1, n_edges),
        "edge_tau": rng.uniform(0, 700, n_edges).astype(np.float32),
    }


def as_batch(graphs: list[dict]):
    offsets = np.cumsum([0] + [len(g["node_ids"]) for g in graphs[:-1]])
    return (
        torch.as_tensor(np.concatenate([g["node_ids"] for g in graphs])),
        torch.as_tensor(np.concatenate([g["edge_index"] + o for g, o in zip(graphs, offsets)], axis=1)),
        torch.as_tensor(np.concatenate([g["edge_rel"] for g in graphs])),
        torch.as_tensor(np.concatenate([g["edge_tau"] for g in graphs])),
    )


def kg_module(use_time: bool = True) -> TimeAwareKGEmbedding:
    torch.manual_seed(0)
    m = TimeAwareKGEmbedding(N_ENT, N_REL, d_kg=D, n_layers=2, d_out=8, use_time=use_time, dropout=0.0)
    torch.nn.init.normal_(m.out.weight)   # the zero-initialised output would hide everything below it
    return m.eval()


def test_edge_softmax_sums_to_one_per_destination():
    dst = torch.tensor([0, 0, 1, 1, 1, 3])
    alpha = edge_softmax(torch.randn(6), dst, n_nodes=4)
    sums = torch.zeros(4).index_add(0, dst, alpha)
    assert torch.allclose(sums[[0, 1, 3]], torch.ones(3))
    assert sums[2] == 0      # node without incoming edges
    assert edge_softmax(torch.zeros(0), torch.zeros(0, dtype=torch.long), 3).numel() == 0


def test_patients_do_not_leak_into_each_other():
    m = kg_module()
    a, b, b2 = tiny_graph(1), tiny_graph(2), tiny_graph(3)
    out_ab = m(*as_batch([a, b]))[: len(a["node_ids"])]
    out_ab2 = m(*as_batch([a, b2]))[: len(a["node_ids"])]
    out_a = m(*as_batch([a]))
    assert torch.allclose(out_ab, out_a, atol=1e-6)
    assert torch.allclose(out_ab2, out_a, atol=1e-6)


def test_time_attention_and_projections_get_gradients():
    m = kg_module().train()
    m(*as_batch([tiny_graph(1), tiny_graph(2)])).pow(2).sum().backward()
    for name in ["time_in.weight", "time_out.weight", "att.0.weight", "W_R", "entity.weight", "gcn.0.weight"]:
        grad = dict(m.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, f"no gradient reaches {name}"


def test_tau_changes_output_only_when_time_is_on():
    g = tiny_graph(1)
    g2 = dict(g, edge_tau=g["edge_tau"][::-1].copy())
    on = kg_module(use_time=True)
    assert not torch.allclose(on(*as_batch([g])), on(*as_batch([g2])))
    off = kg_module(use_time=False)
    assert torch.allclose(off(*as_batch([g])), off(*as_batch([g2])))


def test_output_starts_at_zero():
    torch.manual_seed(0)
    m = TimeAwareKGEmbedding(N_ENT, N_REL, d_kg=D, d_out=8).eval()
    assert m(*as_batch([tiny_graph(1)])).abs().max() == 0


# ---------- full model on synthetic items ----------

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOD]", "[EOM]", "[EOL]", "[EOVI]",
         "D_9_4019", "D_10_I10", "M_lisinopril", "L_Glucose_p4", "[LCODE]", "[LVAL]", "LVAL_BIN_10000"]


def synthetic_items():
    def visit(day, codes):
        ids = [VOCAB.index(f"D_{c}") if f"D_{c}" in VOCAB else 1 for c in codes]
        return {"day_offset": day, "delta_prev_days": 0.0,
                "blocks": {"diag": ids + [5], "med": [11, 6], "lab": [12, 7], "vital": []},
                "block_numeric": {"lab": [100.0, 0.0], "vital": []},
                "blocks_raw": {"diag": list(codes)}}
    return [
        {"patient_id": 1, "label": 1, "visits": [visit(0.0, ["9_4019", "9_25000"]), visit(200.0, ["10_I10"])]},
        {"patient_id": 2, "label": 0, "visits": [visit(0.0, ["10_I10"])]},
    ]


def synthetic_graphs(tmp_path: Path) -> Path:
    graphs = {
        1: {**tiny_graph(1), "code_index": {"9_4019": 0, "9_25000": 1, "10_I10": 2}},
        2: {**tiny_graph(2), "code_index": {"10_I10": 3}},
    }
    path = tmp_path / "patient_graphs.test.pkl"
    with open(path, "wb") as f:
        pickle.dump({"n_entities": N_ENT, "n_relations": N_REL, "graphs": graphs}, f)
    return path


FUSIONS = pytest.mark.parametrize("fusion", ["gate", "add"])


def make_models(tmp_path, fusion="gate"):
    bb.set_vocab(VOCAB)
    torch.manual_seed(0)
    kg = HTRFormerKG(len(VOCAB), d_model=32, n_heads=4, dropout=0.0,
                     kg_patient_graphs=synthetic_graphs(tmp_path), kg_dim=D, kg_fusion=fusion).eval()
    base = HTRFormerCrossAttn(len(VOCAB), d_model=32, n_heads=4, dropout=0.0).eval()
    missing, unexpected = base.load_state_dict(kg.state_dict(), strict=False)
    # everything the KG model has on top of htr_former_ca: the graph module and the fusion layer
    assert not missing and all(k.startswith(("kg.", "diag_tower.fusion.")) for k in unexpected)
    return kg, base


@FUSIONS
def test_kg_model_starts_identical_to_htr_former_ca(tmp_path, fusion):
    kg, base = make_models(tmp_path, fusion)
    items = synthetic_items()
    with torch.no_grad():
        assert torch.allclose(kg(kg.build_batch(items)), base(base.build_batch(items)), atol=1e-6)


def test_diag_positions_point_at_the_right_graph_nodes(tmp_path):
    kg, _ = make_models(tmp_path)
    batch = kg.build_batch(synthetic_items())
    idx, nodes = batch["diag_kg_idx"], batch["kg_node_ids"]
    g1, g2 = kg.patient_graphs[1], kg.patient_graphs[2]
    assert nodes[idx[0, 0, 0]] == g1["node_ids"][0]          # 9_4019 in patient 1, visit 0
    assert nodes[idx[0, 1, 0]] == g1["node_ids"][2]          # 10_I10 in patient 1, visit 1
    assert nodes[idx[1, 0, 0]] == g2["node_ids"][3]          # 10_I10 in patient 2 -> patient 2's own node
    assert idx[0, 0, 2] == -1                                 # the [EOD] terminator has no node
    assert (idx[1, 1:] == -1).all()                           # padding visits
    # 9_25000 is [UNK] in the vocab but still reaches its KG node through the raw code
    assert batch["diag_ids"][0, 0, 1] == 1 and nodes[idx[0, 0, 1]] == g1["node_ids"][1]


@FUSIONS
def test_kg_model_trains_the_kg_branch(tmp_path, fusion):
    kg, _ = make_models(tmp_path, fusion)
    kg.train()
    opt = torch.optim.Adam(kg.parameters(), lr=1e-2)
    batch = kg.build_batch(synthetic_items())
    labels = torch.tensor([1, 0])
    for _ in range(2):   # step 1 moves the zero-initialised scale / output layer, step 2 reaches the graph layers
        opt.zero_grad()
        torch.nn.functional.cross_entropy(kg(batch), labels).backward()
        opt.step()
    if fusion == "gate":
        assert kg.diag_tower.fusion.scale.abs().item() > 0
        assert kg.diag_tower.fusion.gate.weight.grad.abs().sum() > 0
    assert kg.kg.out.weight.abs().sum() > 0
    assert kg.kg.time_in.weight.grad.abs().sum() > 0


@FUSIONS
def test_param_groups_cover_every_parameter_once(tmp_path, fusion):
    kg, _ = make_models(tmp_path, fusion)
    groups = kg.param_groups(lr=1e-3, weight_decay=0.01)
    ids = [id(p) for g in groups for p in g["params"]]
    assert sorted(ids) == sorted(id(p) for p in kg.parameters())
    if fusion == "gate":
        scale_group = next(g for g in groups if any(p is kg.diag_tower.fusion.scale for p in g["params"]))
        assert len(scale_group["params"]) == 1
        assert scale_group["lr"] == pytest.approx(1e-3 * kg.kg_scale_lr_mult)
        assert scale_group["weight_decay"] == 0.0


def test_train_py_uses_the_model_param_groups(tmp_path):
    from hto.train import TrainConfig, train_and_eval
    kg, _ = make_models(tmp_path, "gate")
    seen = {}
    orig = torch.optim.AdamW

    def spy(params, **kw):
        seen["groups"] = params if isinstance(params, list) else None
        return orig(params, **kw)

    torch.optim.AdamW = spy
    try:
        items = synthetic_items()
        train_and_eval(kg, items, items, TrainConfig(epochs=1, batch_size=2, lr=1e-3), val_items=items)
    finally:
        torch.optim.AdamW = orig
    assert seen["groups"] is not None and any(g.get("lr") == pytest.approx(1e-3 * kg.kg_scale_lr_mult)
                                              for g in seen["groups"])


def test_gate_leaves_positions_without_a_graph_node_alone(tmp_path):
    kg, _ = make_models(tmp_path, "gate")
    fusion = kg.diag_tower.fusion
    with torch.no_grad():                      # move every fusion parameter away from its initial value
        fusion.scale.fill_(1.0)
        torch.nn.init.normal_(fusion.ln.bias)
        torch.nn.init.normal_(fusion.gate.weight)
    batch = kg.build_batch(synthetic_items())
    pos = batch["diag_kg_idx"] >= 0
    with torch.no_grad():
        emb = kg.diag_tower.emb(batch["diag_ids"])
        fused = fusion(emb, kg.kg_code_vectors(batch), pos)
    assert torch.equal(fused[~pos], emb[~pos])           # padding, [EOD], unmapped codes: untouched
    assert not torch.allclose(fused[pos], emb[pos])       # codes with a node: changed


# ---------- demo integration ----------

DEMO = ROOT / "processed_data" / "demo"


@pytest.mark.skipif(not (DEMO / "patient_graphs.demo.pkl").exists(), reason="demo KG data not built")
def test_demo_forward_and_coverage():
    from hto.data.variants import Vocab, load_variant
    vocab = Vocab.load(DEMO / "allcodes_timeaware.vocab.json")
    bb.set_vocab(vocab.id_to_token)
    items = load_variant(DEMO / "allcodes_timeaware.train.pkl")
    model = HTRFormerKG(len(vocab), kg_patient_graphs=DEMO / "patient_graphs.demo.pkl",
                        kg_transr=ROOT / "processed_data" / "kg" / "demo" / "transr.pt").eval()
    batch = model.build_batch(items[:8])
    with torch.no_grad():
        assert model(batch).shape == (8, 2)
    raw_positions = sum(min(len(v["blocks_raw"]["diag"]), 64) for it in items[:8] for v in it["visits"][-64:])
    linked = int((batch["diag_kg_idx"] >= 0).sum())
    assert raw_positions > 0 and linked / raw_positions > 0.8, f"only {linked}/{raw_positions} codes linked"
