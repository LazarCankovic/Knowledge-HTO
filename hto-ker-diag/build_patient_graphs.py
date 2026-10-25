"""Build per-patient time-aware graphs for htr_former_kg (needs patient data: run where the data is).

    python build_patient_graphs.py --dataset demo --kg-tag demo \
        --mimic-dir data/processed/mimic-iv-demo-2.2/mimic_data

Reads <mimic-dir>/final_diagnoses.parquet (time = when each code became available, all <= index) and
processed_data/kg/<kg-tag>/ (from build_kg.py). Writes processed_data/<dataset>/patient_graphs.<kg-tag>.pkl
plus a .stats.json (aggregate numbers only).
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

from hto.kg.patient_graphs import PatientGraphConfig, build_all
from hto.kg.snomed_graph import load_graph


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="processed_data/<dataset>/ (from build_data.py)")
    ap.add_argument("--kg-tag", required=True, help="processed_data/kg/<kg-tag>/ (from build_kg.py)")
    ap.add_argument("--mimic-dir", required=True, help="folder with final_diagnoses.parquet")
    for f in fields(PatientGraphConfig):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = ap.parse_args()

    kg_dir = ROOT / "processed_data" / "kg" / args.kg_tag
    out_dir = ROOT / "processed_data" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PatientGraphConfig(**{f.name: getattr(args, f.name) for f in fields(PatientGraphConfig)})

    diagnoses = pd.read_parquet(Path(args.mimic_dir).expanduser() / "final_diagnoses.parquet",
                                columns=["patient_id", "time", "diag"])
    graph = load_graph(kg_dir)
    start = time.time()
    graphs, stats = build_all(diagnoses, graph, cfg)
    stats["seconds"] = round(time.time() - start, 1)
    stats["kg_tag"] = args.kg_tag

    out_path = out_dir / f"patient_graphs.{args.kg_tag}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"kg_tag": args.kg_tag, "n_entities": len(graph["entities"]),
                     "n_relations": len(graph["relations"]), "config": stats["config"], "graphs": graphs}, f)
    out_path.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
