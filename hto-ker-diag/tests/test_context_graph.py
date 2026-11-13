"""Tests for labs / vitals and symptoms in the context graph (build_context_kg.py, measurements, HPO,
Stage 2 value_attr, Stage 3 side effects and measurement matches). Synthetic data only.

    python -m pytest tests/test_context_graph.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hto.kg.context_graph import (build_context_graph, icd_disease_phenotypes, icd_symptom_phenotypes,  # noqa: E402
                                  mondo_disease_names, resolve_panel)
from hto.kg.hpo import hpo_id, load_hpo, resolve  # noqa: E402
from hto.kg.measurements import (add_states, before_t0, current_states, fit_value_scale, last_values,  # noqa: E402
                                 load_measurements, load_panel, measurement_codes, value_features)
from hto.kg.patient_graphs import GlobalGraphIndex, PatientGraphConfig, build_patient_graph  # noqa: E402
from hto.kg.primekg import disease_by_name, phenotype_hpo  # noqa: E402
from hto.kg.text_features import entity_texts  # noqa: E402
from hto.kg.treatment import FEATURES, build_context  # noqa: E402

HP = "http://purl.obolibrary.org/obo/HP_"


def hpo_json(tmp_path: Path) -> Path:
    def node(num, lbl, xrefs=(), syn=(), deprecated=False, definition=None):
        meta = {"xrefs": [{"val": x} for x in xrefs],
                "synonyms": [{"pred": "hasExactSynonym", "val": s} for s in syn]}
        if deprecated:
            meta["deprecated"] = True
        if definition:
            meta["definition"] = {"val": definition}
        return {"id": f"{HP}{num}", "lbl": lbl, "meta": meta}
    nodes = [node("0000118", "Phenotypic abnormality"),
             node("0002153", "Hyperkalemia", ["SNOMEDCT_US:14140009"], ["High blood potassium"], definition="High K."),
             node("0000822", "Hypertension", syn=["High blood pressure"]),
             node("0002315", "Headache", ["SNOMEDCT_US:25064002"]),
             node("0012622", "Chronic kidney disease"),
             node("0100735", "Hypertensive crisis"),
             node("0009999", "Old term", deprecated=True)]
    edges = [{"sub": f"{HP}{c}", "pred": "is_a", "obj": f"{HP}0000118"} for c in ("0002153", "0000822", "0002315")]
    path = tmp_path / "hp.json"
    path.write_text(json.dumps({"graphs": [{"nodes": nodes, "edges": edges}]}))
    return path


def panel_tsv(tmp_path: Path) -> Path:
    rows = [["measure", "source", "keys", "fluid", "unit", "low", "high", "low_phenotype", "high_phenotype",
             "low_disease", "high_disease", "display"],
            ["potassium", "lab", "Potassium", "Blood", "mEq/L", "3.5", "5.0", "Hypokalemia",
             "Hyperkalemia", "", "hyperkalemia", "Potassium"],
            ["sbp", "vital", "bps", "", "mmHg", "", "140", "", "High blood pressure", "", "hypertension",
             "Systolic blood pressure"],
            ["weight", "vital", "weight_lbs", "", "lb", "", "", "", "", "", "", "Weight"]]
    path = tmp_path / "panel.tsv"
    path.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    return path


# ---------- HPO ----------

def test_hpo_load_and_resolve(tmp_path):
    hpo = load_hpo(hpo_json(tmp_path))
    assert "HP:0009999" not in hpo.names and hpo_id(2153) == "HP:0002153"
    assert hpo.parents["HP:0002153"] == {"HP:0000118"}
    assert resolve(hpo, ["Not a term", "high-blood potassium"]) == "HP:0002153"   # exact synonym, normalised
    assert resolve(hpo, ["nothing"]) is None
    assert set(hpo.xrefs["prefix"]) == {"SNOMEDCT_US"}


# ---------- measurements ----------

def test_bundled_panel_loads():
    panel = load_panel()
    assert {"potassium", "egfr", "sbp", "dbp", "bmi"} <= set(panel["measure"])
    assert set(panel["source"]) == {"lab", "vital"} and panel["keys"].map(len).min() >= 1
    assert ((panel["low"] < panel["high"]) | panel["low"].isna() | panel["high"].isna()).all()

def measurement_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": [1, 1, 1, 1, 2, 2],
        "visit_id": [10, None, 11, 12, None, None],
        "time": pd.to_datetime(["2149-01-01", "2150-01-01", "2150-01-20", "2150-02-01", "2148-01-01", "2150-01-15"]),
        "measure": ["potassium", "potassium", "potassium", "potassium", "potassium", "sbp"],
        "value": [4.0, 5.6, 5.4, 9.9, 5.8, 150.0]})


def test_states_codes_and_value_features_use_pre_t0_rows_only(tmp_path):
    panel = load_panel(panel_tsv(tmp_path))
    t0 = pd.Series({1: pd.Timestamp("2150-02-01"), 2: pd.Timestamp("2150-02-01")})
    m = before_t0(add_states(measurement_rows(), panel), t0)
    assert len(m) == 5 and 9.9 not in m["value"].tolist()           # the t0 row is dropped
    assert m["state"].tolist() == ["", "high", "high", "high", "high"]
    codes = measurement_codes(m)
    assert set(codes.loc[codes["patient_id"] == 1, "code"]) == {"meas:potassium", "meas:potassium:high"}
    assert set(codes.loc[codes["patient_id"] == 1, "visit_id"]) == {"10", "d21500101", "11"}   # day fallback
    scale = fit_value_scale(m, patient_ids=[1])
    assert scale["potassium"][0] == pytest.approx(5.4)
    vf = value_features(last_values(m), scale).set_index(["patient_id", "measure"])
    assert vf.loc[(1, "potassium"), "abnormal"] == 1.0 and vf.loc[(1, "potassium"), "has_value"] == 1.0
    s = scale["potassium"][1]
    assert vf.loc[(1, "potassium"), "last_value_z"] == pytest.approx(0.0)
    assert vf.loc[(1, "potassium"), "delta_z"] == pytest.approx((5.4 - 5.6) / s)
    assert vf.loc[(2, "potassium"), "delta_z"] == 0.0                # a single value has no change
    cur = current_states(m, t0, window_days=365)
    assert set(zip(cur["patient_id"], cur["measure"])) == {(1, "potassium"), (2, "sbp")}   # patient 2's K is old


def test_load_measurements_maps_lab_key_and_fluid(tmp_path):
    panel = load_panel(panel_tsv(tmp_path))
    pd.DataFrame({"patient_id": [1, 1, 1], "visit_id": [1, 1, 1],
                  "time": pd.to_datetime(["2150-01-01"] * 3), "lab_key": ["Potassium", "Potassium", "Sodium"],
                  "fluid": ["Blood", "Urine", "Blood"], "lab_value": ["4.1", "30", "140"]}
                 ).to_parquet(tmp_path / "final_labs.parquet")
    pd.DataFrame({"patient_id": [1, 1], "visit_id": [None, None], "time": pd.to_datetime(["2150-01-02"] * 2),
                  "vital_key": ["bps", "bpd"], "vital_value": [130.0, 80.0]}).to_parquet(tmp_path / "final_vitals.parquet")
    m = load_measurements(tmp_path, panel)
    assert sorted(zip(m["measure"], m["value"])) == [("potassium", 4.1), ("sbp", 130.0)]


# ---------- context KG ----------

def base_graph() -> dict:
    ent = pd.DataFrame({"id": [1, 2, 3, 4], "key": ["10_E875", "10_R51", "10_I10", "14140009"],
                        "kind": ["icd", "icd", "icd", "snomed"], "name": ["Hyperkalemia", "Headache", "HTN", "K high"]})
    rel = pd.DataFrame({"id": [1, 2], "name": ["maps_to", "inv_maps_to"]})
    tri = pd.DataFrame({"h": [1, 4], "r": [1, 2], "t": [4, 1]})
    return {"entities": ent, "relations": rel, "triples": tri}


def context_inputs(tmp_path: Path):
    hpo = load_hpo(hpo_json(tmp_path))
    panel = load_panel(panel_tsv(tmp_path))
    nodes = pd.DataFrame({"node_index": [7, 8, 20, 21], "node_id": ["1", "2", "2153", "822"],
                          "node_type": ["disease", "disease", "effect/phenotype", "effect/phenotype"],
                          "node_name": ["hyperkalemia", "hypertension", "Hyperkalemia", "Hypertension"],
                          "node_source": ["MONDO", "MONDO", "HPO", "HPO"]})
    resolved = resolve_panel(panel, hpo, disease_by_name(nodes))
    icd_snomed = pd.DataFrame({"icd_version": 10, "icd_code": ["E875", "N189", "Z87", "I10", "I10", "R51"],
                               "sctid": ["14140009", "709044004", "555", "59621000", "706882009", "25064002"]})
    codes = pd.DataFrame({"icd_version": 10, "icd_code": ["E875", "R51", "N189", "Z87", "I10"],
                          "long_title": ["Hyperkalemia, acute", "Headache, unspecified",
                                         "Chronic kidney disease, stage unspecified", "History of headache",
                                         "Essential (primary) hypertension"]})
    snomed_names = {"14140009": "Acute hyperkalemia (disorder)", "709044004": "Chronic kidney disease (disorder)",
                    "555": "History of headache (procedure)", "59621000": "Essential hypertension (disorder)",
                    "706882009": "Hypertensive crisis (disorder)", "25064002": "Headache (finding)"}
    direct = icd_symptom_phenotypes(icd_snomed, codes, hpo, snomed_names)
    icd_map = pd.DataFrame({"diag": ["10_I10"], "primekg_index": [8.0]})
    dp = pd.DataFrame({"disease_index": [8, 8], "phenotype_index": [21, 20]})
    anchors = set(resolved["phenotype_id"].dropna())
    disease = icd_disease_phenotypes(icd_map, dp, phenotype_hpo(nodes), anchors, cap=1)
    return hpo, panel, resolved, direct, disease, nodes


def test_resolve_panel_and_icd_phenotype_routes(tmp_path):
    _, _, resolved, direct, disease, nodes = context_inputs(tmp_path)
    r = resolved.set_index(["measure", "state"])
    assert r.at[("potassium", "high"), "phenotype_id"] == "HP:0002153"
    assert r.at[("potassium", "high"), "disease_index"] == 7
    assert r.at[("potassium", "low"), "phenotype_id"] is None          # Hypokalemia is not in this HPO
    assert r.at[("sbp", "high"), "phenotype_id"] == "HP:0000822"       # via exact synonym
    assert ("weight", "high") not in r.index                           # no thresholds -> no state
    assert phenotype_hpo(nodes) == {20: "HP:0002153", 21: "HP:0000822"}
    routes = dict(zip(direct["diag"], direct["route"]))
    assert routes == {"10_E875": "snomed_xref", "10_N189": "snomed_name",   # procedures are not matched
                      "10_R51": "name"}                    # title first; ", unspecified" stripped
    assert len(direct[direct["diag"] == "10_R51"]) == 1
    # I10's core concept is "Essential hypertension"; the crisis concept only rolls up into I10
    assert "10_I10" not in routes
    assert disease.to_dict("list") == {"diag": ["10_I10"], "hpo_id": ["HP:0000822"]}   # anchor first, cap 1


def test_mondo_names_back_up_primekg_disease_names(tmp_path):
    nodes = [{"id": "http://purl.obolibrary.org/obo/MONDO_0005015", "lbl": "diabetes mellitus",
              "meta": {"synonyms": [{"pred": "hasExactSynonym", "val": "DM"}]}},
             {"id": "http://purl.obolibrary.org/obo/MONDO_0000001", "lbl": "not in primekg"}]
    path = tmp_path / "mondo.json"
    path.write_text(json.dumps({"graphs": [{"nodes": nodes}]}))
    assert mondo_disease_names(path, {"MONDO:0005015": 42}) == {"dm": 42, "diabetes mellitus": 42}


def test_context_graph_links_abnormal_results_to_diagnoses(tmp_path):
    hpo, panel, resolved, direct, disease, _ = context_inputs(tmp_path)
    graph, texts = build_context_graph(base_graph(), panel, resolved, {"potassium": {"14140009", "999"}},
                                       direct, disease, hpo)
    ent = graph["entities"]
    assert ent["id"].tolist() == list(range(1, len(ent) + 1)) and ent["id"].iloc[:4].tolist() == [1, 2, 3, 4]
    names = dict(zip(graph["relations"]["id"], graph["relations"]["name"]))
    assert list(names.values())[:2] == ["maps_to", "inv_maps_to"] and "inv_indicates" in names.values()
    key = dict(zip(ent["id"], ent["key"]))
    fwd = {(key[h], names[r], key[t]) for h, r, t in graph["triples"].itertuples(index=False)}
    assert ("meas:potassium", "measure_maps_to_snomed", "14140009") in fwd     # 999 is not in the base graph
    assert ("meas:potassium:high", "indicates", "HP:0002153") in fwd
    assert ("HP:0002153", "inv_indicates", "meas:potassium:high") in fwd
    assert ("10_E875", "manifests_as", "HP:0002153") in fwd
    assert ("10_I10", "has_phenotype", "HP:0000822") in fwd
    assert not any(t == "HP:0000118" for _, _, t in fwd)                        # parents only when present
    t = dict(zip(texts["key"], texts["text"]))
    assert t["meas:potassium:high"] == "Abnormal result: high Potassium (> 5 mEq/L), indicating Hyperkalemia."
    assert t["HP:0002153"].startswith("Symptom or sign: Hyperkalemia. Definition:")

    g = GlobalGraphIndex(graph)
    pg = build_patient_graph({"10_E875": np.array([10.0]), "meas:potassium:high": np.array([40.0])}, g,
                             PatientGraphConfig())
    nid = [int(i) for i in pg["node_ids"]]
    eid = dict(zip(ent["key"], ent["id"]))
    hops = {(nid[s], nid[t]): int(h) for s, t, h in zip(pg["edge_index"][0], pg["edge_index"][1], pg["edge_hops"])}
    assert hops[(eid["10_E875"], eid["HP:0002153"])] == 2      # code -> shared phenotype <- abnormal result
    assert hops[(eid["meas:potassium:high"], eid["HP:0002153"])] == 2
    assert pg["n_pairs_connected"] == 1


def test_entity_texts_take_context_texts(tmp_path):
    ent = pd.DataFrame({"id": [1, 2, 3], "key": ["10_I10", "123", "meas:potassium"],
                        "kind": ["icd", "snomed", "measure"], "name": ["HTN", "Hypertension", "Potassium"]})
    pd.DataFrame({"key": ["meas:potassium"], "text": ["Measurement: Potassium (Blood, mEq/L)."],
                  "source": ["context"]}).to_parquet(tmp_path / "t.parquet")
    t = entity_texts(ent, tmp_path / "t.parquet").set_index("diag")
    assert t.at["123", "text"] == "Concept: Hypertension" and "10_I10" not in t.index
    assert t.at["meas:potassium", "source"] == "context"


# ---------- Stage 2: value_attr ----------

def test_enrich_value_attr_only_for_observed_measurements():
    pytest.importorskip("torch_geometric")
    from hto.kg.enrich import enrich_graph
    g = {"node_ids": np.array([1, 2, 3]), "edge_index": np.array([[0, 1], [1, 2]]), "edge_rel": np.array([1, 1]),
         "edge_tau": np.zeros(2, dtype=np.float32), "edge_hops": np.zeros(2, dtype=np.int16),
         "code_index": {"meas:potassium": 0, "10_I10": 1, "meas:sbp": 2}}
    values = {"meas:potassium": np.array([1.0, 0.5, -0.2, 1.0]), "meas:sbp": np.array([1.0, 2.0, 0.0, 1.0])}
    data, _ = enrich_graph(g, {"meas:potassium": (3.0, 2), "10_I10": (5.0, 1)}, values)
    assert data.value_attr.shape == (3, 4)
    assert data.value_attr[0].tolist() == pytest.approx([1.0, 0.5, -0.2, 1.0])
    assert data.value_attr[1:].abs().sum() == 0                     # sbp has no pre-t0 history here
    data, _ = enrich_graph(g, {"10_I10": (5.0, 1)})
    assert getattr(data, "value_attr", None) is None


# ---------- Stage 3: side effects and measurement matches ----------

def test_build_context_side_effects_and_measurement_matches():
    patients = pd.DataFrame({"patient_id": [1, 2], "index_time": pd.to_datetime(["2150-02-01", "2150-02-01"])})
    dx_pk = pd.DataFrame(columns=["patient_id", "time", "primekg_index"])
    meds = pd.DataFrame(columns=["patient_id", "time", "drugbank_id"])
    rel = pd.DataFrame({"relation": ["drug_effect", "contraindication", "indication"],
                        "drug_id": ["DB00722"] * 3, "drug_index": 1,
                        "target_index": [20, 7, 8], "target_id": ["2153", "1", "2"]})
    cands = pd.DataFrame({"ingredient": ["lisinopril"], "drugbank_id": ["DB00722"]})
    symptoms = pd.DataFrame({"patient_id": [2], "time": pd.to_datetime(["2150-03-01"]), "phenotype_index": [20]})
    states = pd.DataFrame({"patient_id": [1, 1, 2], "measure": ["potassium", "sbp", "sbp"],
                           "state": ["high", "high", "high"],
                           "time": pd.to_datetime(["2150-01-20", "2150-01-25", "2150-01-25"]),
                           "phenotype_index": pd.array([20, 21, 21], dtype="Int64"),
                           "disease_index": pd.array([7, 8, 8], dtype="Int64")})
    feats, matches, counts = build_context(patients, dx_pk, meds, rel, cands, symptoms=symptoms,
                                           measurement_states=states)
    f = {name: feats[:, 0, i] for i, name in enumerate(FEATURES)}
    assert f["side_effect_overlap_flag"].tolist() == [1, 0]        # patient 2's symptom is after t0
    assert f["measurement_contraindication_flag"].tolist() == [1, 0]
    assert f["measurement_indication_count"].tolist() == [1, 1]
    assert f["n_abnormal_measurements"].tolist() == [2, 1]
    assert f["n_patient_phenotypes"].tolist() == [2, 1]
    assert f["contraindication_flag"].sum() == 0                   # diagnosis-based flags are unchanged
    src = matches.groupby(["relation", "evidence_source"]).size().to_dict()
    assert src == {("drug_effect", "symptom"): 1, ("contraindication", "measurement"): 1,
                   ("indication", "measurement"): 2}
    assert (pd.to_datetime(matches["evidence_time"]) < pd.to_datetime(matches["index_time"])).all()
    assert counts["patients_with_measurement_diseases"] == 2
