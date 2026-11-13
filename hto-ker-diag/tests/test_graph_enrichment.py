"""Tests for the graph-enrichment stages (1_extract_text_embeddings, 2_enrich_patient_graphs,
3_build_treatment_block). Synthetic data only.

    python -m pytest tests/test_graph_enrichment.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hto.kg.disease_map import (DiseaseMapConfig, attach_primekg, build_icd_mondo_map, coverage_report,  # noqa: E402
                                is_supplementary, load_mondo, snomed_associated_findings)
from hto.kg.enrich import code_history, enrich_graph, materialize, time_encoding  # noqa: E402
from hto.kg.patient_graphs import GlobalGraphIndex, PatientGraphConfig, build_patient_graph  # noqa: E402
from hto.kg.primekg import mondo_to_primekg  # noqa: E402
from hto.kg.text_features import category_means, disease_text, encode_texts, icd_texts  # noqa: E402
from hto.kg.treatment import FEATURES, DrugMapper, build_context, index_candidates  # noqa: E402

M = "http://purl.obolibrary.org/obo/MONDO_"


# ---------- patient graphs: edge_hops ----------

def tiny_global_graph() -> dict:
    ent = pd.DataFrame({"id": [1, 2, 3, 4, 5], "key": ["10_I10", "10_E119", "10_N189", "s1", "s2"],
                        "kind": ["icd", "icd", "icd", "snomed", "snomed"]})
    rel = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["maps_to", "is_a", "inv_maps_to", "inv_is_a"]})
    fwd = [(1, 1, 4), (2, 1, 5), (5, 2, 4), (3, 1, 5)]       # I10->s1, E119->s2, s2 is_a s1, N189->s2
    inv = [(t, r + 2, h) for h, r, t in fwd]
    return {"entities": ent, "relations": rel, "triples": pd.DataFrame(fwd + inv, columns=["h", "r", "t"])}


def test_edge_hops_record_the_shortest_code_to_code_path():
    g = GlobalGraphIndex(tiny_global_graph())
    pg = build_patient_graph({"10_I10": np.array([0.0]), "10_E119": np.array([100.0]),
                              "10_N189": np.array([50.0])}, g, PatientGraphConfig())
    ids = pg["node_ids"]
    hops = {(int(ids[s]), int(r), int(ids[t])): int(h) for s, t, r, h in
            zip(pg["edge_index"][0], pg["edge_index"][1], pg["edge_rel"], pg["edge_hops"])}
    assert hops[(2, 1, 5)] == 2          # E119 -> s2 lies on E119 - s2 - N189 (2 hops)
    assert hops[(5, 2, 4)] == 3          # s2 -> s1 lies on E119 - s2 - s1 - I10 (3 hops)
    assert hops[(4, 4, 5)] == 3          # inverse edges share the forward value
    assert len(pg["edge_hops"]) == len(pg["edge_rel"])


def test_edge_hops_zero_for_one_hop_knowledge_only():
    g = GlobalGraphIndex(tiny_global_graph())
    pg = build_patient_graph({"10_I10": np.array([0.0])}, g, PatientGraphConfig())
    assert (pg["edge_hops"] == 0).all()


# ---------- Stage 1: MONDO / PrimeKG mapping and texts ----------

def mondo_json(tmp_path: Path) -> Path:
    def node(num, lbl, xrefs=(), deprecated=False, definition=None):
        meta = {"xrefs": [{"val": x} for x in xrefs]}
        if deprecated:
            meta["deprecated"] = True
        if definition:
            meta["definition"] = {"val": definition}
        return {"id": f"{M}{num}", "lbl": lbl, "meta": meta}
    nodes = [
        node("0005044", "hypertensive disorder", ["ICD10CM:I10", "ICD9:401.9", "SCTID:38341003"], definition="High BP."),
        node("0005148", "type 2 diabetes mellitus", ["SCTID:44054006"]),
        node("0100000", "rare kidney thing", ["ICD10CM:N18.9"]),
        node("0005300", "chronic kidney disease", ["ICD10CM:N18"]),
        node("0009999", "old term", ["ICD10CM:Z99.9"], deprecated=True),
        node("0001111", "child of htn", ["SCTID:111"]),
    ]
    edges = [{"sub": f"{M}0100000", "pred": "is_a", "obj": f"{M}0005300"},
             {"sub": f"{M}0001111", "pred": "is_a", "obj": f"{M}0005044"}]
    path = tmp_path / "mondo.json"
    path.write_text(json.dumps({"graphs": [{"nodes": nodes, "edges": edges}]}))
    return path


def test_icd_mondo_routes_and_primekg_backoff(tmp_path):
    mondo = load_mondo(mondo_json(tmp_path))
    assert "MONDO:0009999" not in mondo.names
    codes = pd.DataFrame({"icd_version": [10, 9, 10, 10, 9, 10, 10],
                          "icd_code": ["I10", "4019", "E119", "N1830", "40391", "Z999", "R69"],
                          "n_rows_usable": [50, 30, 10, 5, 3, 1, 1]})
    icd_snomed = pd.DataFrame({"icd_version": [10, 9], "icd_code": ["E119", "40391"],
                               "sctid": ["44054006", "222"], "rule_kind": ["unconditional", "unconditional"]})
    parents = {"222": {"111"}}
    gems = pd.DataFrame({"source": ["40391"], "target": ["I129"], "approximate": [False], "no_map": [False]})
    cfg = DiseaseMapConfig(snomed_ancestor_depth=2, primekg_backoff_hops=1)
    m = build_icd_mondo_map(codes, mondo, icd_snomed, parents, gems, cfg)
    route = m.drop_duplicates("diag").set_index("diag")["route"].to_dict()
    assert route == {"10_I10": "icd_xref", "9_4019": "icd_xref", "10_E119": "snomed_xref",
                     "10_N1830": "icd_parent_xref", "9_40391": "snomed_ancestor_xref",
                     "10_Z999": "unmapped", "10_R69": "unmapped"}
    nodes = pd.DataFrame({"node_index": [7, 8, 9], "node_id": ["5044", "1200_5148", "5300"],
                          "node_type": "disease", "node_source": ["MONDO", "MONDO_grouped", "MONDO"]})
    pk = mondo_to_primekg(nodes)
    assert pk["MONDO:0005148"] == 8 and pk["MONDO:0001200"] == 8
    m = attach_primekg(m, mondo, pk, cfg).drop_duplicates("diag").set_index("diag")
    assert m.at["10_I10", "primekg_index"] == 7 and m.at["10_I10", "primekg_hops"] == 0
    assert m.at["10_N1830", "primekg_index"] == 9                    # N18 xref -> its own node
    assert m.at["9_40391", "primekg_index"] == 7 and m.at["9_40391", "primekg_hops"] == 1   # child -> parent node
    rep = coverage_report(attach_primekg(build_icd_mondo_map(codes, mondo, icd_snomed, parents, gems, cfg),
                                         mondo, pk, cfg), codes, cfg)
    assert rep["mondo_usage_weighted"] == pytest.approx(98 / 100)


def test_history_situations_map_through_associated_finding_but_not_family_history(tmp_path):
    mondo = load_mondo(mondo_json(tmp_path))
    rels = pd.DataFrame({"sourceId": ["900", "901", "901"], "typeId": ["246090004", "246090004", "408732007"],
                         "destinationId": ["44054006", "38341003", "444148008"]})
    findings = snomed_associated_findings(rels)
    assert findings == {"900": ["44054006"]}                 # 901 is about a family member
    codes = pd.DataFrame({"icd_version": [10, 10], "icd_code": ["Z8639", "Z8249"]})
    icd_snomed = pd.DataFrame({"icd_version": [10, 10], "icd_code": ["Z8639", "Z8249"], "sctid": ["900", "901"],
                               "rule_kind": ["unconditional", "unconditional"]})
    m = build_icd_mondo_map(codes, mondo, icd_snomed, snomed_findings=findings).set_index("diag")
    assert m.at["10_Z8639", "route"] == "snomed_finding_xref" and m.at["10_Z8639", "mondo_id"] == "MONDO:0005148"
    assert m.at["10_Z8249", "route"] == "unmapped"
    assert is_supplementary(10, "Z8639") and is_supplementary(9, "E8497") and not is_supplementary(9, "4019")


def test_gem_route_for_icd9(tmp_path):
    mondo = load_mondo(mondo_json(tmp_path))
    codes = pd.DataFrame({"icd_version": [9], "icd_code": ["5859"]})
    gems = pd.DataFrame({"source": ["5859"], "target": ["N189"], "approximate": [False], "no_map": [False]})
    m = build_icd_mondo_map(codes, mondo, gems=gems)
    assert m["route"].iloc[0] == "gem>icd_xref" and m["mondo_id"].iloc[0] == "MONDO:0100000"


def test_texts_fall_back_and_stay_code_specific(tmp_path):
    assert disease_text("Essential hypertension", "hypertensive disorder", "headache") == \
        "Disease: Essential hypertension (hypertensive disorder). Symptoms: headache"
    assert disease_text("Foo", "", "", "A definition.") == "Disease: Foo. Definition: A definition."
    mondo = load_mondo(mondo_json(tmp_path))
    codes = pd.DataFrame({"icd_version": [10, 10], "icd_code": ["I10", "R69"], "long_title": ["Essential HTN", "Illness"]})
    m = build_icd_mondo_map(codes, mondo)
    m = attach_primekg(m, mondo, {"MONDO:0005044": 7})
    feats = pd.DataFrame({"node_index": [7], "mayo_symptoms": ["Headache, nosebleeds"],
                          "mondo_definition": [None], "umls_description": [None]})
    t = icd_texts(codes, m, feats, mondo.definitions).set_index("diag")
    assert t.at["10_I10", "source"] == "primekg_symptoms" and "Symptoms: Headache" in t.at["10_I10", "text"]
    assert t.at["10_R69", "source"] == "icd_title" and t.at["10_R69", "text"] == "Disease: Illness."


def test_encode_texts_dedupes_and_keeps_order():
    calls = []

    def enc(batch):
        calls.append(len(batch))
        return np.array([[len(s), 1.0] for s in batch], dtype=np.float32)
    out = encode_texts(["aa", "b", "aa", "cccc"], enc, batch_size=2, log_every=0)
    assert out[:, 0].tolist() == [2, 1, 2, 4] and sum(calls) == 3
    cats = category_means(["10_I10", "10_I119", "9_4019"], np.array([[1.0], [3.0], [5.0]]))
    assert cats["10_cat_I10"][0] == 1.0 and cats["10_cat_I11"][0] == 3.0 and cats["9_cat_401"][0] == 5.0


# ---------- Stage 2: enrichment ----------

def test_code_history_is_strictly_before_t0():
    t0 = pd.Series({1: pd.Timestamp("2150-01-10")})
    dx = pd.DataFrame({"patient_id": [1, 1, 1, 1], "visit_id": [10, 11, 11, 12],
                       "time": pd.to_datetime(["2150-01-01", "2150-01-05", "2150-01-05", "2150-01-10"]),
                       "diag": ["10_I10", "10_I10", "10_E119", "10_N189"]})
    h = code_history(dx, t0).set_index("diag")
    assert set(h.index) == {"10_I10", "10_E119"}                 # the t0 row is dropped
    assert h.at["10_I10", "recency_days"] == 5 and h.at["10_I10", "frequency"] == 2


def test_enrich_and_materialize_dims():
    pytest.importorskip("torch_geometric")
    g = {"node_ids": np.array([1, 4, 5]), "edge_index": np.array([[0, 1, 1], [1, 0, 2]]),
         "edge_rel": np.array([1, 3, 2]), "edge_tau": np.array([0.0, 0.0, 30.0], dtype=np.float32),
         "edge_hops": np.array([0, 0, 3], dtype=np.int16), "code_index": {"10_I10": 0, "10_N189": 2}}
    data, boundary = enrich_graph(g, {"10_I10": (5.0, 2)})
    assert boundary == 1 and data.observed_mask.tolist() == [True, False, False]
    table = torch.randn(6, 768)
    d = materialize(data, table)
    assert d.x.shape == (3, 770) and d.edge_attr.shape == (3, 17)
    assert torch.equal(d.x[1, :768], table[4])
    assert d.x[0, 768] == pytest.approx(np.log1p(5.0)) and d.x[0, 769] == pytest.approx(np.log1p(2))
    assert d.x[1, 768:].abs().sum() == 0
    assert d.edge_attr[2, 16] == pytest.approx(0.75)
    assert torch.allclose(d.edge_attr[0, :16], time_encoding(torch.zeros(1))[0])
    assert not torch.allclose(time_encoding(torch.tensor([30.0])), time_encoding(torch.tensor([31.0])))


def test_enrich_requires_edge_hops():
    pytest.importorskip("torch_geometric")
    with pytest.raises(KeyError):
        enrich_graph({"node_ids": np.array([1]), "edge_index": np.zeros((2, 0)), "edge_rel": np.zeros(0),
                      "edge_tau": np.zeros(0), "code_index": {}}, {})


# ---------- Stage 3: treatment context ----------

def write_rrf(tmp_path: Path) -> Path:
    def conso(rxcui, sab, tty, code, s):
        f = [""] * 18
        f[0], f[1], f[11], f[12], f[13], f[14] = rxcui, "ENG", sab, tty, code, s
        return "|".join(f) + "|"

    def rel(rxcui1, rela, rxcui2):
        f = [""] * 16
        f[0], f[4], f[7], f[10] = rxcui1, rxcui2, rela, "RXNORM"
        return "|".join(f) + "|"

    def sat(rxcui, ndc):
        f = [""] * 13
        f[0], f[8], f[9], f[10] = rxcui, "NDC", "RXNORM", ndc
        return "|".join(f) + "|"
    rows = [conso("29046", "RXNORM", "IN", "29046", "lisinopril"), conso("29046", "DRUGBANK", "IN", "DB00722", "Lisinopril"),
            conso("6918", "RXNORM", "IN", "6918", "metoprolol"), conso("6918", "DRUGBANK", "IN", "DB00264", "Metoprolol"),
            conso("221124", "RXNORM", "PIN", "221124", "metoprolol tartrate"),
            conso("316151", "RXNORM", "SCDC", "316151", "lisinopril 10 MG"),
            conso("314076", "RXNORM", "SCD", "314076", "lisinopril 10 MG Oral Tablet"),
            conso("1191", "RXNORM", "IN", "1191", "aspirin"), conso("1191", "DRUGBANK", "IN", "DB00945", "Aspirin")]
    d = tmp_path / "rrf"
    d.mkdir()
    (d / "RXNCONSO.RRF").write_text("\n".join(rows) + "\n")
    (d / "RXNREL.RRF").write_text("\n".join([rel("29046", "has_ingredient", "316151"),
                                             rel("316151", "consists_of", "314076"),
                                             rel("6918", "form_of", "221124")]) + "\n")
    (d / "RXNSAT.RRF").write_text(sat("314076", "00093111301") + "\n")
    return d


def test_drug_mapper_routes(tmp_path):
    mp = DrugMapper(write_rrf(tmp_path))
    assert mp.map("whatever", "00093-1113-01") == (frozenset({"DB00722"}), "ndc")
    assert mp.map("Metoprolol Tartrate", "0") == (frozenset({"DB00264"}), "name_exact")
    assert mp.map("metoprolol tartrate 25 mg tab", None)[0] == frozenset({"DB00264"})
    assert mp.map("aspirin ec", None) == (frozenset({"DB00945"}), "name_tokens")
    assert mp.map("sodium chloride flush", None) == (frozenset(), "unmapped")


def test_build_context_flags_and_time_filters():
    patients = pd.DataFrame({"patient_id": [1, 2], "index_time": pd.to_datetime(["2150-02-01", "2150-02-01"])})
    dx_pk = pd.DataFrame({"patient_id": [1, 1, 2], "primekg_index": [7, 9, 9],
                          "time": pd.to_datetime(["2150-01-01", "2150-02-01", "2149-01-01"])})
    meds = pd.DataFrame({"patient_id": [1, 1, 1, 2], "drugbank_id": ["DB00945", "DB00264", "DB00722", "DB00945"],
                         "time": pd.to_datetime(["2150-01-25", "2149-01-01", "2150-01-30", "2150-02-01"])})
    rel = pd.DataFrame({"relation": ["indication", "contraindication", "drug_drug", "drug_drug", "drug_drug"],
                        "drug_id": ["DB00722"] * 5, "drug_index": 1,
                        "target_index": [7, 9, 0, 0, 0], "target_id": ["5044", "x", "DB00945", "DB00264", "DB00722"]})
    cands = pd.DataFrame({"ingredient": ["lisinopril"], "drugbank_id": ["DB00722"]})
    feats, matches, counts = build_context(patients, dx_pk, meds, rel, cands)
    f = {name: feats[:, 0, i] for i, name in enumerate(FEATURES)}
    assert f["indication_alignment_flag"].tolist() == [1, 0]
    assert f["contraindication_flag"].tolist() == [0, 1]       # patient 1's node 9 is dated at t0
    assert f["ddi_count"].tolist() == [1, 0]                  # aspirin in window; metoprolol too old; self excluded
    assert f["ddi_fraction_of_active"][0] == pytest.approx(0.5)
    assert f["n_active_meds"].tolist() == [2, 0]              # patient 2's aspirin is at t0
    assert (pd.to_datetime(matches["evidence_time"]) < pd.to_datetime(matches["index_time"])).all()
    assert counts["patients_with_active_meds"] == 1


def test_index_candidates():
    cands = pd.DataFrame({"ingredient": ["lisinopril", "metoprolol", "amlodipine"]})
    got = index_candidates(pd.Series(["metoprolol succinate xl | lisinopril", None]), cands)
    assert got.tolist() == [[True, True, False], [False, False, False]]
