"""Build the SNOMED CT knowledge graph for htr_former_kg (public inputs only, no patient data).

Steps (each writes into processed_data/kg/<tag>/):
  map       ICD codes -> SNOMED concepts            icd_snomed_map.parquet, coverage.json
  graph     global graph                            entities.tsv, relations.tsv, triples.tsv, graph_stats.json
  pretrain  TransR entity/relation embeddings       transr.pt
  all       map + graph + pretrain

--codes takes any CSV with icd_version, icd_code, long_title columns:
  * the extraction's tables/icd_inventory.csv (cohort codes, plus n_rows_usable for usage-weighted coverage)
  * MIMIC's hosp/d_icd_diagnoses.csv(.gz) (every ICD code; a much larger graph)

Example (demo cohort):
  python build_kg.py all --tag demo \
      --codes data/processed/mimic-iv-demo-2.2/tables/icd_inventory.csv \
      --snomed-dir /path/to/SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20260901T120000Z \
      --gems data/external/cms-gems-2018/unzipped/2018_I9gem.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hto.kg.icd_snomed_map import build_icd_map, coverage_report, norm_code
from hto.kg.snomed_graph import IS_A, GraphConfig, build_graph, load_graph, save_graph


def read_codes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"icd_code": str})
    missing = {"icd_version", "icd_code"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns {sorted(missing)}")
    df["icd_code"] = df["icd_code"].map(norm_code)
    return df.drop_duplicates(["icd_version", "icd_code"]).reset_index(drop=True)


def step_map(args, out_dir: Path) -> pd.DataFrame:
    codes = read_codes(Path(args.codes))
    icd_map = build_icd_map(codes, Path(args.snomed_dir), Path(args.gems))
    icd_map.to_parquet(out_dir / "icd_snomed_map.parquet", index=False)
    inventory = codes if "n_rows_usable" in codes.columns else None
    report = coverage_report(icd_map, inventory)
    (out_dir / "coverage.json").write_text(json.dumps(report, indent=2))
    print(f"[map] {len(codes):,} codes -> {icd_map['sctid'].notna().sum():,} code-concept pairs")
    print(json.dumps(report, indent=2))
    return icd_map


def step_graph(args, out_dir: Path) -> None:
    icd_map = pd.read_parquet(out_dir / "icd_snomed_map.parquet")
    codes = read_codes(Path(args.codes))
    titles = {}
    if "long_title" in codes.columns:
        titles = dict(zip(codes["icd_version"].astype(str) + "_" + codes["icd_code"], codes["long_title"].fillna("")))
    cfg = GraphConfig(
        max_concepts_per_code=args.max_concepts_per_code,
        ancestor_depth=args.ancestor_depth,
        max_top_depth=args.max_top_depth,
        hub_max_degree=args.hub_max_degree,
        rewire_seed=args.rewire_seed,
        **({"relations": (IS_A,)} if args.isa_only else {}),
    )
    graph = build_graph(icd_map, titles, Path(args.snomed_dir), cfg)
    save_graph(graph, out_dir)
    s = graph["stats"]
    print(f"[graph] entities={s['entities']:,} (snomed {s['snomed_nodes']:,}) relations={s['relations']} "
          f"triples={s['triples']:,} dropped_hubs={s['dropped_hubs']}")
    print(f"        ICD codes with >= 1 concept: {s['icd_codes_with_concept']:,}/{s['icd_codes']:,}")


def step_pretrain(args, out_dir: Path) -> None:
    from hto.kg.transr import TransRConfig, pretrain_transr
    graph = load_graph(out_dir)
    cfg = TransRConfig(**{f.name: getattr(args, f.name) for f in fields(TransRConfig) if hasattr(args, f.name)})
    pretrain_transr(graph, cfg, out_dir / "transr.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["map", "graph", "pretrain", "all"])
    ap.add_argument("--tag", required=True, help="output folder name under processed_data/kg/")
    ap.add_argument("--codes", help="CSV with icd_version, icd_code[, long_title, n_rows_usable]")
    ap.add_argument("--snomed-dir", help="SNOMED CT RF2 release folder (contains Snapshot/)")
    ap.add_argument("--gems", help="CMS 2018_I9gem.txt (ICD-9-CM -> ICD-10-CM)")
    # graph
    ap.add_argument("--max-concepts-per-code", type=int, default=GraphConfig.max_concepts_per_code)
    ap.add_argument("--ancestor-depth", type=int, default=GraphConfig.ancestor_depth)
    ap.add_argument("--max-top-depth", type=int, default=GraphConfig.max_top_depth)
    ap.add_argument("--hub-max-degree", type=int, default=GraphConfig.hub_max_degree)
    ap.add_argument("--isa-only", action="store_true", help="IS-A edges only (ablation graph)")
    ap.add_argument("--rewire-seed", type=int, default=None,
                    help="degree-preserving random control graph: tails shuffled within each relation")
    # pretrain
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = ROOT / "processed_data" / "kg" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.step in ("map", "graph", "all") and not (args.codes and args.snomed_dir):
        ap.error("map/graph need --codes and --snomed-dir")
    if args.step in ("map", "all") and not args.gems:
        ap.error("map needs --gems")

    if args.step in ("map", "all"):
        step_map(args, out_dir)
    if args.step in ("graph", "all"):
        step_graph(args, out_dir)
    if args.step in ("pretrain", "all"):
        step_pretrain(args, out_dir)
    print(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
