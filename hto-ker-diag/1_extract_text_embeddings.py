"""Stage 1: ClinicalBERT text features for every diagnosis-graph node (public inputs + the code inventory).

    python 1_extract_text_embeddings.py \
        --codes data/processed/mimic-iv-3.1/tables/icd_inventory.csv \
        --mondo /path/to/mondo.json --primekg /path/to/primekg/kg.csv \
        --disease-features /path/to/primekg/disease_features.tab \
        --kg-dir processed_data/kg/full --snomed-dir /path/to/SnomedCT_... \
        --gems data/external/2018_I9gem.txt --device cuda

1. ICD -> MONDO (src/hto/kg/disease_map.py) -> PrimeKG disease node.
2. Text per node: ICD title, MONDO name, PrimeKG Mayo Clinic symptoms (or a definition); for the other
   nodes of --kg-dir/entities.tsv their name (SNOMED, lab metadata) or, for a context KG
   (build_context_kg.py), the texts in --kg-dir/entity_texts.parquet (measurements, states, HPO symptoms).
3. ClinicalBERT (masked mean pooling, 768-d).

Writes to --out-dir (default processed_data/kg/):
  node_features_bert.pt      {"keys", "matrix" [N, 768], "sources", "model", ...}; load_text_features() -> {key: e_text}
  icd_mondo_map.parquet      code -> MONDO -> PrimeKG rows (also used by 3_build_treatment_block.py)
  icd_mondo_coverage.json    aggregate mapping and embedding coverage
  node_texts.parquet         the encoded texts (public terminology only)
  primekg_*.parquet          cached PrimeKG nodes and drug relations
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hto.kg.disease_map import (ASSOCIATED_FINDING, SUBJECT_RELATIONSHIP_CONTEXT, DiseaseMapConfig, attach_primekg,
                                build_icd_mondo_map, coverage_report, load_mondo, snomed_associated_findings)
from hto.kg.icd_snomed_map import load_gems, norm_code
from hto.kg.primekg import extract_primekg, load_disease_features, mondo_to_primekg
from hto.kg.text_features import (DEFAULT_MODEL, category_means, encode_texts, entity_texts, icd_texts,
                                  make_encoder, save_text_features)


def snomed_structure(snomed_dir: Path) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """IS-A parents and situation -> Associated finding links from the SNOMED release."""
    from hto.kg.snomed_graph import IS_A, load_relationships
    rels = load_relationships(snomed_dir, (IS_A, ASSOCIATED_FINDING, SUBJECT_RELATIONSHIP_CONTEXT))
    parents: dict[str, set[str]] = defaultdict(set)
    isa = rels[rels["typeId"] == IS_A]
    for child, parent in isa[["sourceId", "destinationId"]].itertuples(index=False):
        parents[child].add(parent)
    return parents, snomed_associated_findings(rels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", required=True, help="icd_inventory.csv (icd_version, icd_code, long_title, n_rows_usable)")
    ap.add_argument("--mondo", required=True, help="mondo.json")
    ap.add_argument("--primekg", required=True, help="PrimeKG kg.csv or kg_giant.csv")
    ap.add_argument("--disease-features", help="PrimeKG disease_features.tab (Mayo Clinic symptoms, definitions)")
    ap.add_argument("--kg-dir", help="processed_data/kg/<kg-tag>/ from build_kg.py (icd_snomed_map, entities)")
    ap.add_argument("--snomed-dir", help="SNOMED CT RF2 release (IS-A ancestors route)")
    ap.add_argument("--gems", help="CMS 2018_I9gem.txt (ICD-9 -> ICD-10-CM route)")
    ap.add_argument("--out-dir", default=str(ROOT / "processed_data" / "kg"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", help="cuda, cpu, or auto (cuda when available)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    ap.add_argument("--map-only", action="store_true", help="write the map and coverage, skip ClinicalBERT")
    for f in fields(DiseaseMapConfig):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=int, default=f.default)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DiseaseMapConfig(**{f.name: getattr(args, f.name) for f in fields(DiseaseMapConfig)})
    t0 = time.time()

    codes = pd.read_csv(args.codes, dtype={"icd_code": str})
    codes["icd_code"] = codes["icd_code"].map(norm_code)
    codes = codes.drop_duplicates(["icd_version", "icd_code"]).reset_index(drop=True)
    mondo = load_mondo(Path(args.mondo))
    print(f"[mondo] {len(mondo.names):,} terms, xrefs {mondo.xrefs['prefix'].value_counts().to_dict()}")
    nodes, _ = extract_primekg(Path(args.primekg), out_dir)
    mondo_pk = mondo_to_primekg(nodes)
    print(f"[primekg] {int((nodes['node_type'] == 'disease').sum()):,} disease nodes "
          f"covering {len(mondo_pk):,} MONDO terms ({time.time() - t0:.0f}s)")

    kg_dir = Path(args.kg_dir) if args.kg_dir else None
    icd_snomed = pd.read_parquet(kg_dir / "icd_snomed_map.parquet") if kg_dir else None
    if icd_snomed is not None:
        icd_snomed["icd_code"] = icd_snomed["icd_code"].map(norm_code)
    parents, findings = snomed_structure(Path(args.snomed_dir)) if args.snomed_dir else (None, None)
    gems = load_gems(Path(args.gems)) if args.gems else None
    icd_map = build_icd_mondo_map(codes, mondo, icd_snomed, parents, gems, cfg, findings)
    icd_map = attach_primekg(icd_map, mondo, mondo_pk, cfg)
    icd_map.to_parquet(out_dir / "icd_mondo_map.parquet", index=False)
    report = coverage_report(icd_map, codes, cfg)
    report["inputs"] = {"kg_dir": str(kg_dir) if kg_dir else None, "snomed_ancestors": parents is not None,
                        "gems": gems is not None, "disease_features": bool(args.disease_features)}
    print(json.dumps(report, indent=2))
    if args.map_only:
        (out_dir / "icd_mondo_coverage.json").write_text(json.dumps(report, indent=2))
        return

    features = load_disease_features(Path(args.disease_features)) if args.disease_features else None
    texts = icd_texts(codes, icd_map, features, mondo.definitions)
    icd_sources = set(texts["source"])
    if kg_dir:
        ent = pd.read_csv(kg_dir / "entities.tsv", sep="\t", dtype={"key": str}, keep_default_na=False)
        texts = pd.concat([texts, entity_texts(ent, kg_dir / "entity_texts.parquet")], ignore_index=True)
    print(f"[text] {len(texts):,} nodes, sources {texts['source'].value_counts().to_dict()}")
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[bert] {args.model} on {device}", flush=True)
    encode = make_encoder(args.model, device, args.max_length, args.pooling)
    matrix = encode_texts(texts["text"].tolist(), encode, args.batch_size)
    keys, sources = texts["diag"].tolist(), texts["source"].tolist()
    is_icd = [s in icd_sources for s in sources]
    cats = category_means([k for k, i in zip(keys, is_icd) if i], matrix[is_icd])
    keys += list(cats)
    sources += ["category_mean"] * len(cats)
    matrix = np.vstack([matrix, np.stack(list(cats.values()))]) if cats else matrix
    save_text_features(out_dir / "node_features_bert.pt", keys, matrix, sources,
                       {"model": args.model, "pooling": args.pooling, "max_length": args.max_length,
                        "dim": int(matrix.shape[1])})
    texts.to_parquet(out_dir / "node_texts.parquet", index=False)

    nonzero = dict(zip(keys, np.abs(matrix).sum(1) > 0))
    w = codes["n_rows_usable"].astype(float) if "n_rows_usable" in codes else pd.Series(1.0, index=codes.index)
    has = (codes["icd_version"].astype(str) + "_" + codes["icd_code"]).map(nonzero).fillna(False).astype(bool)
    report["embedding"] = {
        "model": args.model, "dim": int(matrix.shape[1]), "nodes": len(keys),
        "sources": {k: int(n) for k, n in pd.Series(sources).value_counts().items()},
        "codes_nonzero_usage_weighted": round(float((w * has).sum() / w.sum()), 4),
        "codes_mondo_and_nonzero_usage_weighted": round(float(
            (w * has * (codes["icd_version"].astype(str) + "_" + codes["icd_code"]).isin(
                set(icd_map.loc[icd_map["mondo_id"].notna(), "diag"]))).sum() / w.sum()), 4),
    }
    report["seconds"] = round(time.time() - t0, 1)
    (out_dir / "icd_mondo_coverage.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["embedding"], indent=2))
    print(f"Wrote {out_dir / 'node_features_bert.pt'}")


if __name__ == "__main__":
    main()
