"""Synthetic checks for the lab KG and token-position alignment (no MIMIC data)."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hto.data import batch_builder_all_codes_local as bb  # noqa: E402
from hto.data import build as build_module  # noqa: E402
from hto.data.variants import make_multidomain_visit  # noqa: E402
from hto.kg.lab_graph import build_lab_graph  # noqa: E402
from hto.kg.patient_graphs import build_all  # noqa: E402
from hto.models.htr_former_cross_attn import HTRFormerCrossAttn  # noqa: E402
from hto.models.htr_former_lab_kg import HTRFormerLabKG  # noqa: E402


VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOD]", "[EOM]", "[EOL]", "[EOVI]",
         "D_10_I10", "M_lisinopril", "L_Glucose_p4", "L_Sodium_p4",
         "[LCODE]", "[LVAL]", "LVAL_BIN_10000", "LVAL_BIN_10100"]


def items():
    def visit(day, lab_codes, lab_itemids, values):
        return {"day_offset": day, "blocks": {
                    "diag": [9, 5], "med": [10, 6],
                    "lab": lab_codes + ([7] if lab_codes else []), "vital": []},
                "block_numeric": {"lab": values + ([0.0] if values else []), "vital": []},
                "blocks_raw": {"diag": ["10_I10"], "lab": lab_itemids}}
    return [
        {"patient_id": 1, "label": 1, "visits": [
            visit(0, [11, 1], ["lab:11", "lab:12"], [100.0, 101.0]),
            visit(30, [], [], [])]},
        {"patient_id": 2, "label": 0, "visits": [
            visit(0, [12], ["lab:13"], [100.0])]},
    ]


def graphs_file(tmp_path):
    graphs = {
        1: {"node_ids": np.array([1, 2, 3]),
            # Both lab code nodes need competing incoming edges for time attention.
            "edge_index": np.array([[2, 1, 2, 0], [0, 0, 1, 1]]),
            "edge_rel": np.array([1, 1, 1, 1]),
            "edge_tau": np.array([0., 30., 20., 0.], np.float32),
            "code_index": {"lab:11": 0, "lab:12": 1}},
        2: {"node_ids": np.array([4, 5]),
            "edge_index": np.array([[0], [1]]),
            "edge_rel": np.array([1]), "edge_tau": np.array([0.], np.float32),
            "code_index": {"lab:13": 0}},
    }
    path = tmp_path / "patient_graphs.lab_test.pkl"
    with open(path, "wb") as f:
        pickle.dump({"domain": "lab", "n_entities": 5, "n_relations": 2,
                     "graphs": graphs}, f)
    return path


def model(tmp_path, fusion="gate"):
    bb.set_vocab(VOCAB)
    torch.manual_seed(0)
    return HTRFormerLabKG(len(VOCAB), d_model=32, n_heads=4,
                          dropout=0.0, kg_dim=16, kg_fusion=fusion,
                          kg_patient_graphs=graphs_file(tmp_path))


def test_itemid_distinguishes_same_label_and_patient_graph_has_time():
    metadata = pd.DataFrame([
        {"itemid": 11, "label": "Glucose", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 12, "label": "Glucose", "fluid": "Urine", "category": "Chemistry"},
    ])
    graph = build_lab_graph(metadata)
    assert {"lab:11", "lab:12"} <= set(graph["entities"]["key"])
    assert graph["stats"]["lab_codes"] == 2
    events = pd.DataFrame([
        {"patient_id": 1, "time": "2020-01-01", "code": "lab:11"},
        {"patient_id": 1, "time": "2020-01-04", "code": "lab:12"},
    ])
    patients, _ = build_all(events, graph, code_col="code", code_kind="lab",
                            path_time_priority=True)
    assert set(patients[1]["code_index"]) == {"lab:11", "lab:12"}
    assert np.any(patients[1]["edge_tau"] == 3)


def test_variant_keeps_itemids_aligned_with_lab_measurements():
    records = [{"patient_id": 1, "label": 1, "target_med": "lisinopril", "visits": [{
        "day_offset": 0.0, "diag_codes": [], "med_codes": [],
        "lab_num": [("Glucose", 100.0), ("Glucose", 101.0)],
        "lab_itemids": [11, 12], "vital_num": []}]}]
    variant, _ = make_multidomain_visit(records)
    assert variant[0]["visits"][0]["blocks_raw"]["lab"] == ["lab:11", "lab:12"]
    assert len(variant[0]["visits"][0]["blocks"]["lab"]) == 3  # includes [EOL]


def test_canonical_preserves_itemid_for_duplicate_lab_labels(monkeypatch):
    patients = pd.DataFrame([{"patient_id": 1, "label": 1,
                              "latest_bp_time": pd.Timestamp("2020-02-01"),
                              "latest_htn_med_name": "lisinopril", "bps": 120., "bpd": 80.}])

    def empty(columns):
        df = pd.DataFrame(columns=["patient_id", "time", "visit_id", *columns])
        df["time"] = pd.to_datetime(df["time"])
        return df

    frames = {"final_diagnoses.parquet": empty(["diag"]),
              "final_medications.parquet": empty(["medication"]),
              "final_vitals.parquet": empty(["vital_key", "vital_value"]),
              "final_labs.parquet": pd.DataFrame([
                  {"patient_id": 1, "visit_id": 101, "time": pd.Timestamp("2020-01-01"),
                   "lab_key": "Glucose", "lab_value": 100., "itemid": 11},
                  {"patient_id": 1, "visit_id": 101, "time": pd.Timestamp("2020-01-02"),
                   "lab_key": "Glucose", "lab_value": 101., "itemid": 12}])}
    monkeypatch.setattr(build_module, "_load_patients", lambda: patients)
    monkeypatch.setattr(pd, "read_parquet", lambda path: frames[path.name])
    visit = build_module.build_canonical()[0]["visits"][0]
    assert visit["lab_num"] == [("Glucose", 100.), ("Glucose", 101.)]
    assert visit["lab_itemids"] == [11, 12]


def test_kg_attaches_only_to_lab_code_token_positions(tmp_path):
    kg = model(tmp_path)
    batch = kg.build_batch(items())
    idx, nodes = batch["lab_kg_idx"], batch["kg_node_ids"]
    assert nodes[idx[0, 0, 1]] == 1  # first measurement code
    assert nodes[idx[0, 0, 5]] == 2  # second measurement code
    assert batch["lab_ids"][0, 0, 5] == 1  # [UNK] still has its own KG node
    assert nodes[idx[1, 0, 1]] == 4  # separate patient graph
    assert (idx[:, :, 0::4] == -1).all()  # [LCODE] markers
    assert (idx[:, :, 2::4] == -1).all()  # [LVAL] markers
    assert (idx[:, :, 3::4] == -1).all()  # value bins
    assert (idx[0, 1] == -1).all()        # visit without labs


@pytest.mark.parametrize("fusion", ["gate", "add"])
def test_zero_initialized_kg_starts_at_base_model(tmp_path, fusion):
    kg = model(tmp_path, fusion).eval()
    base = HTRFormerCrossAttn(len(VOCAB), d_model=32, n_heads=4, dropout=0.0).eval()
    missing, unexpected = base.load_state_dict(kg.state_dict(), strict=False)
    assert not missing and all(k.startswith(("kg.", "lab_tower.fusion.")) for k in unexpected)
    with torch.no_grad():
        assert torch.allclose(kg(kg.build_batch(items())), base(base.build_batch(items())), atol=1e-6)


def test_lab_graph_branch_receives_gradients_and_no_node_positions_stay_plain(tmp_path):
    kg = model(tmp_path).train()
    batch = kg.build_batch(items())
    opt = torch.optim.Adam(kg.parameters(), lr=1e-2)
    for _ in range(2):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(kg(batch), batch["labels"]).backward()
        opt.step()
    assert kg.lab_tower.fusion.scale.abs().item() > 0
    assert kg.kg.time_in.weight.grad.abs().sum() > 0
    pos = batch["lab_kg_idx"] >= 0
    emb = kg.lab_tower.emb(batch["lab_ids"])
    fused = kg.lab_tower.fusion(emb, kg.kg_code_vectors(batch), pos)
    assert torch.equal(fused[~pos], emb[~pos])


def test_old_variant_without_lab_itemids_is_rejected(tmp_path):
    kg = model(tmp_path)
    old = items()
    del old[0]["visits"][0]["blocks_raw"]["lab"]
    with pytest.raises(ValueError, match="Lab item IDs"):
        kg.build_batch(old)
