"""Context KG: the diagnosis + lab SNOMED graph extended with labs / vitals and symptoms (HPO).

    python build_context_kg.py --base-kg-dir processed_data/kg/lab_snomed \
        --icd-snomed-map processed_data/kg/full/icd_snomed_map.parquet \
        --codes $P/tables/icd_inventory.csv --labitems $P/tables/d_labitems.csv \
        --hpo hp.json --primekg kg.csv --mondo mondo.json

Needs the Stage 1 map (--kg-out/icd_mondo_map.parquet). Public inputs only: no patient rows.
Writes --out-dir (default processed_data/kg/ctx/): entities.tsv, relations.tsv, triples.tsv, graph_stats.json,
entity_texts.parquet (texts for the new nodes) and a copy of icd_snomed_map.parquet, plus in --kg-out:
icd_phenotype_map.parquet (code -> HPO, manifests_as) and measurement_panel_resolved.parquet.
See src/hto/kg/context_graph.py for the node and edge types.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hto.kg.context_graph import (ContextGraphConfig, build_context_graph, icd_disease_phenotypes,  # noqa: E402
                                  icd_symptom_phenotypes, mondo_disease_names, resolve_panel)
from hto.kg.hpo import load_hpo  # noqa: E402
from hto.kg.lab_graph import lab_code_key  # noqa: E402
from hto.kg.measurements import load_panel  # noqa: E402
from hto.kg.primekg import (disease_by_name, extract_primekg, load_disease_phenotypes, mondo_to_primekg,  # noqa: E402
                            phenotype_hpo)
from hto.kg.snomed_graph import load_graph, save_graph  # noqa: E402


def measure_snomed_concepts(panel: pd.DataFrame, labitems: pd.DataFrame, base: dict) -> dict[str, set[str]]:
    """measure -> SNOMED keys its MIMIC lab items map to through LOINC (maps_lab_to_snomed)."""
    ent = base["entities"]
    key_of = dict(zip(ent["id"], ent["key"].astype(str)))
    rel = base["relations"]
    rid = rel.loc[rel["name"] == "maps_lab_to_snomed", "id"]
    if rid.empty:
        return {}
    t = base["triples"][base["triples"]["r"] == int(rid.iloc[0])]
    lab_to_sct: dict[str, set[str]] = {}
    for h, s in zip(t["h"].map(key_of), t["t"].map(key_of)):
        lab_to_sct.setdefault(h, set()).add(s)
    out = {}
    for r in panel[panel["source"] == "lab"].itertuples(index=False):
        items = labitems[labitems["label"].isin(r.keys) & (labitems["fluid"] == r.fluid)]["itemid"]
        out[r.measure] = set().union(*[lab_to_sct.get(lab_code_key(i), set()) for i in items]) if len(items) else set()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-kg-dir", required=True, help="build_lab_kg.py output (diagnosis + lab SNOMED graph)")
    ap.add_argument("--icd-snomed-map", required=True, help="icd_snomed_map.parquet of the diagnosis graph")
    ap.add_argument("--codes", required=True, help="icd_inventory.csv (titles for the exact-name route)")
    ap.add_argument("--labitems", required=True, help="d_labitems.csv (itemid, label, fluid, category)")
    ap.add_argument("--hpo", required=True, help="hp.json")
    ap.add_argument("--primekg", help="kg.csv, only needed if the PrimeKG cache in --kg-out is missing or old")
    ap.add_argument("--mondo", help="mondo.json: panel disease names also resolve through MONDO labels / synonyms")
    ap.add_argument("--panel", help="measurement panel TSV (default resources/measurement_panel.tsv)")
    ap.add_argument("--kg-out", default=str(ROOT / "processed_data" / "kg"), help="Stage 1 output folder")
    ap.add_argument("--out-dir", default=str(ROOT / "processed_data" / "kg" / "ctx"))
    ap.add_argument("--max-phenotypes-per-code", type=int, default=ContextGraphConfig.max_phenotypes_per_code)
    args = ap.parse_args()
    start = time.time()
    kg_out, out_dir = Path(args.kg_out), Path(args.out_dir)

    base = load_graph(Path(args.base_kg_dir))
    panel = load_panel(Path(args.panel) if args.panel else None)
    hpo = load_hpo(Path(args.hpo))
    nodes, _ = extract_primekg(Path(args.primekg) if args.primekg else kg_out / "kg.csv", kg_out)
    dis_phen = load_disease_phenotypes(kg_out)
    pk_hpo = phenotype_hpo(nodes)
    print(f"[inputs] base {len(base['entities']):,} entities; HPO {len(hpo.names):,} terms; "
          f"PrimeKG {len(dis_phen):,} disease-phenotype edges ({time.time() - start:.0f}s)", flush=True)

    disease_names = disease_by_name(nodes)
    if args.mondo:
        disease_names = {**mondo_disease_names(Path(args.mondo), mondo_to_primekg(nodes)), **disease_names}
    resolved = resolve_panel(panel, hpo, disease_names)
    icd_snomed = pd.read_parquet(args.icd_snomed_map)
    codes = pd.read_csv(args.codes, dtype={"icd_code": str})
    ent = base["entities"]
    snomed_names = dict(zip(ent.loc[ent["kind"] == "snomed", "key"].astype(str),
                            ent.loc[ent["kind"] == "snomed", "name"].astype(str)))
    direct = icd_symptom_phenotypes(icd_snomed, codes, hpo, snomed_names)
    anchors = set(resolved["phenotype_id"].dropna()) | set(direct["hpo_id"])
    icd_map = pd.read_parquet(kg_out / "icd_mondo_map.parquet", columns=["diag", "primekg_index"])
    disease = icd_disease_phenotypes(icd_map, dis_phen, pk_hpo, anchors, args.max_phenotypes_per_code)
    labitems = pd.read_csv(args.labitems, dtype={"label": str, "fluid": str})
    meas_sct = measure_snomed_concepts(panel, labitems, base)

    graph, texts = build_context_graph(base, panel, resolved, meas_sct, direct, disease, hpo)
    graph["stats"].update({
        "source": f"{args.base_kg_dir} + measurement panel + HPO + PrimeKG disease phenotypes",
        "panel_measures": int(len(panel)), "panel_states": int(len(resolved)),
        "panel_states_without_phenotype": resolved.loc[resolved["phenotype_id"].isna(),
                                                       ["measure", "state", "phenotype_query"]].to_dict("records"),
        "panel_states_without_primekg_disease": resolved.loc[
            resolved["disease_index"].isna() & (resolved["disease_query"] != ""),
            ["measure", "state", "disease_query"]].to_dict("records"),
        "measures_linked_to_snomed": sorted(m for m, s in meas_sct.items() if s),
        "codes_manifesting_as_phenotype": int(direct["diag"].nunique()),
        "manifests_as_routes": {k: int(v) for k, v in direct["route"].value_counts().items()},
        "manifests_as_top_phenotypes": {f"{h} {hpo.names.get(h, '')}": int(n)
                                        for h, n in direct["hpo_id"].value_counts().head(15).items()},
        "codes_with_disease_phenotypes": int(disease["diag"].nunique()),
        "max_phenotypes_per_code": args.max_phenotypes_per_code,
        "seconds": round(time.time() - start, 1)})
    save_graph(graph, out_dir)
    texts.to_parquet(out_dir / "entity_texts.parquet", index=False)
    shutil.copyfile(args.icd_snomed_map, out_dir / "icd_snomed_map.parquet")
    direct.to_parquet(kg_out / "icd_phenotype_map.parquet", index=False)
    resolved.to_parquet(kg_out / "measurement_panel_resolved.parquet", index=False)
    print(json.dumps(graph["stats"], indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
