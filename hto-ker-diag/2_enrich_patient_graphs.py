"""Stage 2: enrich the time-aware SNOMED CT patient graphs (g_y) with Stage 1 ClinicalBERT features.

    python 2_enrich_patient_graphs.py --dataset full --kg-tag full \
        --mimic-dir data/processed/mimic-iv-3.1/mimic_data

Reads processed_data/<dataset>/patient_graphs.<kg-tag>.pkl (build_patient_graphs.py; must contain edge_hops),
processed_data/kg/<kg-tag>/entities.tsv, processed_data/kg/node_features_bert.pt (Stage 1), and
<mimic-dir>/final_{diagnoses,patients_table}.parquet for t0, recency and frequency.
Writes processed_data/<dataset>/patient_graphs_enriched.<kg-tag>.pt (+ .stats.json, aggregates only).
Load it with hto.kg.enrich.EnrichedPatientGraphs: each item is a PyG Data with x [n, 770] and
edge_attr [E, 17] (see src/hto/kg/enrich.py).

For a context KG (--kg-tag ctx, with measurement nodes) the panel labs / vitals also give recency and
frequency to the meas:* codes, and each Data gets value_attr [n, 4]. The robust value scale is fitted on the
training split's pre-t0 values (processed_data/<dataset>/split.json; all patients if it is missing).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hto.kg.enrich import MAX_HOPS, TIME_DIM, code_history, enrich_graph, read_parquet_columns
from hto.kg.measurements import (VALUE_FEATURES, add_states, before_t0, fit_value_scale, last_values,
                                 load_measurements, load_panel, measure_key, measurement_codes,
                                 training_patient_ids, value_features)
from hto.kg.text_features import entity_text_table


def measurement_inputs(mimic: Path, index_time: pd.Series, panel_path: str | None, train: set[int] | None):
    """History rows (patient_id, visit_id, time, diag) for the meas:* codes, value_attr per patient and the
    fitted scale."""
    panel = load_panel(Path(panel_path) if panel_path else None)
    m = before_t0(add_states(load_measurements(mimic, panel), panel), index_time)
    rows = measurement_codes(m).drop_duplicates(["patient_id", "visit_id", "code"], keep="last")
    scale = fit_value_scale(m, train)
    vf = value_features(last_values(m), scale)
    values: dict[int, dict[str, np.ndarray]] = {}
    arr = vf[VALUE_FEATURES].to_numpy(np.float32)
    for (pid, meas), v in zip(zip(vf["patient_id"], vf["measure"]), arr):
        values.setdefault(int(pid), {})[measure_key(meas)] = v
    return rows.rename(columns={"code": "diag"}), values, scale, len(m)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="processed_data/<dataset>/")
    ap.add_argument("--kg-tag", required=True)
    ap.add_argument("--mimic-dir", required=True, help="folder with final_diagnoses / final_patients_table")
    ap.add_argument("--patient-graphs", help="default processed_data/<dataset>/patient_graphs.<kg-tag>.pkl")
    ap.add_argument("--kg-dir", help="default processed_data/kg/<kg-tag>/")
    ap.add_argument("--text-features", help="default processed_data/kg/node_features_bert.pt")
    ap.add_argument("--out-dir", help="default processed_data/<dataset>/")
    ap.add_argument("--panel", help="measurement panel TSV for context KGs (default resources/measurement_panel.tsv)")
    args = ap.parse_args()
    data_dir = ROOT / "processed_data" / args.dataset
    pg_path = Path(args.patient_graphs or data_dir / f"patient_graphs.{args.kg_tag}.pkl")
    kg_dir = Path(args.kg_dir or ROOT / "processed_data" / "kg" / args.kg_tag)
    tf_path = Path(args.text_features or ROOT / "processed_data" / "kg" / "node_features_bert.pt")
    out_dir = Path(args.out_dir or data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    mimic = Path(args.mimic_dir).expanduser()
    patients = read_parquet_columns(mimic / "final_patients_table.parquet", ["patient_id", "index_time"])
    index_time = patients.set_index("patient_id")["index_time"]
    dx = read_parquet_columns(mimic / "final_diagnoses.parquet", ["patient_id", "visit_id", "time", "diag"])
    entities = pd.read_csv(kg_dir / "entities.tsv", sep="\t", dtype={"key": str}, keep_default_na=False)
    context = "measure" in set(entities["kind"])
    values, scale, n_meas_rows, split_source = None, None, 0, None
    if context:
        train, split_source = training_patient_ids(data_dir, mimic)
        meas_rows, values, scale, n_meas_rows = measurement_inputs(mimic, index_time, args.panel, train)
        print(f"[meas] {n_meas_rows:,} pre-t0 panel rows; scale fitted on {split_source} "
              f"({time.time() - start:.0f}s)", flush=True)
        dx["visit_id"] = dx["visit_id"].astype("string")
        dx = pd.concat([dx, meas_rows], ignore_index=True)
    hist = code_history(dx, index_time)
    by_patient = {pid: dict(zip(g["diag"], zip(g["recency_days"].astype(float), g["frequency"].astype(int))))
                  for pid, g in hist.groupby("patient_id")}

    with open(pg_path, "rb") as f:
        pg = pickle.load(f)
    if len(entities) != pg["n_entities"]:
        raise SystemExit(f"{pg_path} was built on a graph with {pg['n_entities']} entities, "
                         f"{kg_dir} has {len(entities)}")
    text_table, missing_text = entity_text_table(tf_path, entities)

    patient_ids, graphs, code_index = [], [], {}
    boundary_codes = n_codes = 0
    for pid in sorted(pg["graphs"]):
        g = pg["graphs"][pid]
        if pid not in index_time.index:
            continue
        data, boundary = enrich_graph(g, by_patient.get(pid, {}),
                                      values.get(int(pid), {}) if context else None)
        patient_ids.append(int(pid))
        graphs.append(data)
        code_index[int(pid)] = dict(g["code_index"])
        boundary_codes += boundary
        n_codes += len(g["code_index"])

    used_nodes = torch.unique(torch.cat([d.node_ids for d in graphs])) if graphs else torch.zeros(0, dtype=torch.long)
    zero_rows = (text_table[used_nodes].abs().sum(1) == 0)
    observed_ids = torch.cat([d.node_ids[d.observed_mask] for d in graphs]) if graphs else torch.zeros(0, dtype=torch.long)
    obs_zero = (text_table[observed_ids].abs().sum(1) == 0)
    hops = torch.cat([d.edge_hops for d in graphs]).numpy() if graphs else np.zeros(0)
    rec = torch.cat([d.recency_days[d.observed_mask] for d in graphs]).numpy() if graphs else np.zeros(0)
    pct = lambda a: {q: round(float(np.percentile(a, q)), 1) for q in (10, 50, 90)} if len(a) else {}
    stats = {
        "kg_tag": args.kg_tag, "patients": len(graphs),
        "node_feature_dim": int(text_table.shape[1]) + 2, "edge_feature_dim": TIME_DIM + 1,
        "entities_without_text": missing_text,
        "nodes_in_graphs_with_zero_text": round(float(zero_rows.float().mean()), 4) if len(used_nodes) else None,
        "observed_code_occurrences_with_zero_text": round(float(obs_zero.float().mean()), 4) if len(observed_ids) else None,
        "observed_codes": n_codes - boundary_codes,
        "codes_without_pre_t0_occurrence": boundary_codes,
        "observed_recency_days": pct(rec),
        "edge_hops_share": {int(h): round(float((hops == h).mean()), 3) for h in range(MAX_HOPS + 1)},
    }
    spec = {"x": "[e_text(768), log1p(recency_days), log1p(frequency)]",
            "edge_attr": f"[sin/cos(2*pi*tau/P) for P in geomspace(1, 3650, 8) days, edge_hops/{MAX_HOPS}]"}
    if context:
        kind = entities.set_index("id")["kind"]
        kinds = pd.DataFrame([kind.reindex(d.node_ids.numpy()).value_counts() for d in graphs]).fillna(0)
        has_val = np.array([int((d.value_attr[:, 0] > 0).sum()) for d in graphs])
        spec["value_attr"] = f"{VALUE_FEATURES}: measurement nodes (meas:*), 0 elsewhere; z = (v - median) / (IQR/1.349)"
        stats.update({
            "measurement_rows_pre_t0": n_meas_rows, "value_scale_fitted_on": split_source,
            "patients_with_measurement_values": int((has_val > 0).sum()),
            "measurement_values_per_patient": pct(has_val),
            "median_nodes_by_kind": {k: float(kinds[k].median()) for k in kinds.columns},
            "patients_with_phenotype_nodes": int((kinds.get("phenotype", pd.Series(0)) > 0).sum()),
            "patients_with_abnormal_result_nodes": int((kinds.get("measure_state", pd.Series(0)) > 0).sum()),
        })
    stats["seconds"] = round(time.time() - start, 1)
    out_path = out_dir / f"patient_graphs_enriched.{args.kg_tag}.pt"
    torch.save({"format": "hto-enriched-graphs-v1", "kg_tag": args.kg_tag, "source": str(pg_path),
                "text_features": str(tf_path), "feature_spec": spec,
                "value_scale": scale, "value_features": VALUE_FEATURES if context else None,
                "text_table": text_table, "patient_ids": patient_ids, "graphs": graphs,
                "code_index": code_index}, out_path)
    out_path.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
