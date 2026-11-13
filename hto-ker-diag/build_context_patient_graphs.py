"""Patient context graphs: diagnoses + lab / vital measurements + abnormal results in the context KG.

    python build_context_patient_graphs.py --dataset full_storetime --kg-tag ctx \
        --mimic-dir $P/mimic_data

Observed codes per patient: diagnosis codes (as in build_patient_graphs.py), meas:<measure> for every panel
measurement and meas:<measure>:<state> for every abnormal one. Measurements are used only when strictly
before t0. The graph builder only needs each code's last time, so rows are reduced to one per
(patient, code) first. Writes processed_data/<dataset>/patient_graphs.<kg-tag>.pkl (+ .stats.json).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import fields
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hto.kg.enrich import read_parquet_columns  # noqa: E402
from hto.kg.measurements import add_states, before_t0, load_measurements, load_panel, measurement_codes  # noqa: E402
from hto.kg.patient_graphs import PatientGraphConfig, build_all  # noqa: E402
from hto.kg.snomed_graph import load_graph  # noqa: E402

CONTEXT_DEFAULTS = {"max_codes_per_patient": 192, "max_edges_per_patient": 3000}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--kg-tag", default="ctx")
    ap.add_argument("--mimic-dir", required=True)
    ap.add_argument("--panel", help="measurement panel TSV (default resources/measurement_panel.tsv)")
    for f in fields(PatientGraphConfig):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=int, default=CONTEXT_DEFAULTS.get(f.name, f.default))
    args = ap.parse_args()
    start = time.time()
    kg_dir = ROOT / "processed_data" / "kg" / args.kg_tag
    out_dir = ROOT / "processed_data" / args.dataset
    mimic = Path(args.mimic_dir).expanduser()

    graph = load_graph(kg_dir)
    if "measure" not in set(graph["entities"]["kind"]):
        raise SystemExit(f"{kg_dir} has no measurement nodes; build it with build_context_kg.py")
    patients = read_parquet_columns(mimic / "final_patients_table.parquet", ["patient_id", "index_time"])
    index_time = patients.set_index("patient_id")["index_time"]
    dx = read_parquet_columns(mimic / "final_diagnoses.parquet", ["patient_id", "time", "diag"]).dropna()
    dx = dx[dx["patient_id"].isin(index_time.index)].rename(columns={"diag": "code"})

    panel = load_panel(Path(args.panel) if args.panel else None)
    m = before_t0(add_states(load_measurements(mimic, panel), panel), index_time)
    mc = measurement_codes(m)
    print(f"[meas] {len(m):,} pre-t0 panel rows -> {len(mc):,} code rows ({time.time() - start:.0f}s)", flush=True)
    observed = pd.concat([dx[["patient_id", "time", "code"]], mc[["patient_id", "time", "code"]]], ignore_index=True)
    observed["time"] = pd.to_datetime(observed["time"])
    observed = observed.groupby(["patient_id", "code"], as_index=False)["time"].max()

    cfg = PatientGraphConfig(**{f.name: getattr(args, f.name) for f in fields(PatientGraphConfig)})
    graphs, stats = build_all(observed, graph, cfg, code_col="code", code_kind="mixed")
    kind = dict(zip(graph["entities"]["id"], graph["entities"]["kind"]))
    per = pd.DataFrame([{k: sum(kind[int(i)] == k for i in g["node_ids"]) for k in ("measure", "measure_state", "phenotype")}
                        for g in graphs.values()])
    stats.update(seconds=round(time.time() - start, 1), kg_tag=args.kg_tag, domain="context",
                 measurement_rows_pre_t0=int(len(m)),
                 abnormal_share={k: round(float(v), 4) for k, v in m["state"].value_counts(normalize=True).items()},
                 patients_with_measurement_nodes=int((per["measure"] > 0).sum()) if len(per) else 0,
                 patients_with_phenotype_nodes=int((per["phenotype"] > 0).sum()) if len(per) else 0,
                 median_nodes_by_kind={k: float(per[k].median()) for k in per} if len(per) else {})
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"patient_graphs.{args.kg_tag}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"domain": "context", "kg_tag": args.kg_tag, "n_entities": len(graph["entities"]),
                     "n_relations": len(graph["relations"]), "config": stats["config"], "graphs": graphs}, f)
    out_path.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
