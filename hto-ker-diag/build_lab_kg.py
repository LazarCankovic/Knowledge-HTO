"""Build the lab KG joined to the diagnosis SNOMED CT graph, then pretrain TransR.

Example:
  python build_lab_kg.py all --tag lab_snomed --diagnosis-kg-tag full \
    --labitems /path/to/mimiciv/hosp/d_labitems.csv.gz \
    --snomed-dir /path/to/SNOMED_US_release \
    --loinc-extension-dir /path/to/LOINC_RF2_extension

MIMIC-IV 3.1 d_labitems has no loinc_code. Provide --loinc-map, either a
reviewed itemid,loinc_code CSV or the MIT-LCP/mimic-code repository's
mimic-iv/mapping/d_labitems_to_loinc.csv (only LOINC vocabulary rows used).
The LOINC extension's active alternate identifiers resolve LOINC to SNOMED IDs.
The diagnosis graph is copied into the output graph's shared ID space.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hto.kg.lab_snomed_graph import build_lab_snomed_graph
from hto.kg.snomed_graph import load_graph, save_graph


def read_loinc_map(path: str) -> pd.DataFrame:
    mapping = pd.read_csv(path, dtype=str)
    if {"itemid", "loinc_code"} <= set(mapping):
        return mapping[["itemid", "loinc_code"]]
    source = "itemid (omop_source_code)"
    required = {source, "omop_vocabulary_id", "omop_concept_code"}
    if not required <= set(mapping):
        raise ValueError(f"--loinc-map needs itemid,loinc_code or {sorted(required)}")
    loinc = mapping[mapping["omop_vocabulary_id"].str.upper() == "LOINC"].copy()
    return loinc[[source, "omop_concept_code"]].rename(
        columns={source: "itemid", "omop_concept_code": "loinc_code"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["graph", "pretrain", "all"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--diagnosis-kg-tag", help="Existing SNOMED diagnosis graph tag from build_kg.py")
    ap.add_argument("--labitems", help="MIMIC-IV d_labitems.csv exported from BigQuery or raw distribution")
    ap.add_argument("--loinc-map", help="MIT-LCP mapping CSV or reviewed itemid,loinc_code CSV")
    ap.add_argument("--snomed-dir", help="Same SNOMED US release used for the diagnosis graph")
    ap.add_argument("--loinc-extension-dir", help="Compatible LOINC Ontology RF2 release")
    ap.add_argument("--ancestor-depth", type=int, default=4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = ROOT / "processed_data" / "kg" / args.tag
    if args.step in ("graph", "all"):
        required = ("diagnosis_kg_tag", "labitems", "loinc_map", "snomed_dir", "loinc_extension_dir")
        missing = [f"--{key.replace('_', '-')}" for key in required if not getattr(args, key)]
        if missing:
            ap.error("graph/all need " + ", ".join(missing))
        if args.tag == args.diagnosis_kg_tag:
            ap.error("--tag must differ from --diagnosis-kg-tag to protect the diagnosis graph")
        items = pd.read_csv(args.labitems, dtype={"itemid": "Int64"})
        loinc = read_loinc_map(args.loinc_map)
        diag_graph = load_graph(ROOT / "processed_data" / "kg" / args.diagnosis_kg_tag)
        graph = build_lab_snomed_graph(diag_graph, items, loinc, Path(args.snomed_dir),
                                       Path(args.loinc_extension_dir), args.ancestor_depth)
        save_graph(graph, out_dir)
        print(f"[lab graph] {graph['stats']}")
    if args.step in ("pretrain", "all"):
        from hto.kg.transr import TransRConfig, pretrain_transr
        graph = load_graph(out_dir)
        cfg = TransRConfig(dim=args.dim, epochs=args.epochs, batch_size=args.batch_size,
                           lr=args.lr, device=args.device, seed=args.seed)
        pretrain_transr(graph, cfg, out_dir / "transr.pt")
    print(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
