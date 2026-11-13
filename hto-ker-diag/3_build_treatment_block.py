"""Stage 3: candidate treatment context block (g_a), a 1-hop PrimeKG lookup per (patient, candidate drug).

    python 3_build_treatment_block.py --dataset full \
        --mimic-dir data/processed/mimic-iv-3.1/mimic_data --rxnorm-dir /path/to/rxnorm/rrf

Needs Stage 1 outputs in --kg-out (default processed_data/kg/): icd_mondo_map.parquet and the cached
primekg_{nodes,relations}.parquet (or pass --primekg kg.csv to build the cache).
Writes processed_data/<dataset>/:
  treatment_context.pt               {"patient_ids" [P], "candidates", "feature_names" [F],
                                      "features" [P, C, F], "is_index_candidate" [P, C], ...}
                                      H_treatment(patient_id, drug) = features[pos(patient_id), pos(drug)]
  treatment_context_matches.parquet  every matched contraindication / indication / off-label use / DDI /
                                     side effect with its evidence time and source (patient-level: keep with
                                     the MIMIC data)
  treatment_context.stats.json       aggregates only
When build_context_kg.py has written icd_phenotype_map.parquet and measurement_panel_resolved.parquet to
--kg-out, the block also uses symptoms and labs / vitals (see src/hto/kg/treatment.py), and
treatment_context.pt gets "measurements" [P, M, 4] (value_attr of the M panel measurements, pre-t0) with
"measurement_names". --no-context turns this off.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hto.kg.enrich import read_parquet_columns
from hto.kg.measurements import (VALUE_FEATURES, MeasurementConfig, add_states, before_t0, current_states,
                                 fit_value_scale, last_values, load_measurements, load_panel,
                                 training_patient_ids, value_features)
from hto.kg.primekg import extract_primekg, phenotype_hpo
from hto.kg.treatment import (FEATURES, DrugMapper, TreatmentConfig, build_context, index_candidates,
                              load_candidates, map_medications)


def context_inputs(mimic: Path, kg_out: Path, data_dir: Path, patients: pd.DataFrame, dx: pd.DataFrame,
                   nodes: pd.DataFrame, panel_path: str | None, mcfg: MeasurementConfig):
    """Symptoms, current abnormal results and the per-patient measurement tensor [P, M, 4]."""
    hpo_to_pk = {h: i for i, h in phenotype_hpo(nodes).items()}
    icd_phen = pd.read_parquet(kg_out / "icd_phenotype_map.parquet")
    symptoms = dx.merge(icd_phen[["diag", "hpo_id"]], on="diag")
    symptoms["phenotype_index"] = symptoms["hpo_id"].map(hpo_to_pk).astype("Int64")
    resolved = pd.read_parquet(kg_out / "measurement_panel_resolved.parquet")
    resolved["phenotype_index"] = resolved["phenotype_id"].map(hpo_to_pk).astype("Int64")
    panel = load_panel(Path(panel_path) if panel_path else None)
    index_time = patients.set_index("patient_id")["index_time"]
    m = before_t0(add_states(load_measurements(mimic, panel), panel), index_time)
    states = current_states(m, index_time, mcfg.condition_window_days).merge(
        resolved[["measure", "state", "phenotype_index", "disease_index"]], on=["measure", "state"], how="left")
    train, split_source = training_patient_ids(data_dir, mimic)
    vf = value_features(last_values(m), fit_value_scale(m, train))
    pos = {p: i for i, p in enumerate(patients["patient_id"])}
    mpos = {k: i for i, k in enumerate(panel["measure"])}
    tensor = np.zeros((len(patients), len(panel), len(VALUE_FEATURES)), dtype=np.float32)
    vf = vf[vf["patient_id"].isin(pos)]
    tensor[vf["patient_id"].map(pos).to_numpy(), vf["measure"].map(mpos).to_numpy()] = vf[VALUE_FEATURES].to_numpy()
    info = {"symptom_rows": int(symptoms["phenotype_index"].notna().sum()),
            "measurement_rows_pre_t0": int(len(m)), "current_abnormal_results": int(len(states)),
            "current_abnormal_by_measure": {f"{a}:{b}": int(n) for (a, b), n in
                                            states.groupby(["measure", "state"]).size().items()},
            "value_scale_fitted_on": split_source}
    return symptoms, states, tensor, list(panel["measure"]), info


def load_severity(path: Path) -> dict[frozenset, str]:
    df = pd.read_csv(path, dtype=str)
    return {frozenset((a, b)): str(lv).lower() for a, b, lv in zip(df["drug_a"], df["drug_b"], df["level"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="output folder processed_data/<dataset>/")
    ap.add_argument("--mimic-dir", required=True)
    ap.add_argument("--rxnorm-dir", required=True, help="RxNorm full release rrf/ folder")
    ap.add_argument("--kg-out", default=str(ROOT / "processed_data" / "kg"), help="Stage 1 output folder")
    ap.add_argument("--primekg", help="kg.csv, only needed if the Stage 1 PrimeKG cache is missing")
    ap.add_argument("--candidates", help="TSV ingredient, drug_class, drugbank_id (default: the 24 in resources/)")
    ap.add_argument("--severity", help="optional CSV drug_a, drug_b, level (major / moderate / minor)")
    ap.add_argument("--no-ndc", action="store_true", help="map medications by name only")
    ap.add_argument("--out-dir", help="default processed_data/<dataset>/")
    ap.add_argument("--panel", help="measurement panel TSV (default resources/measurement_panel.tsv)")
    ap.add_argument("--no-context", action="store_true",
                    help="skip symptoms and labs / vitals even if build_context_kg.py outputs exist")
    for f in fields(TreatmentConfig) + fields(MeasurementConfig):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=float, default=f.default)
    args = ap.parse_args()
    cfg = TreatmentConfig(**{f.name: getattr(args, f.name) for f in fields(TreatmentConfig)})
    mcfg = MeasurementConfig(**{f.name: getattr(args, f.name) for f in fields(MeasurementConfig)})
    kg_out = Path(args.kg_out)
    out_dir = Path(args.out_dir or ROOT / "processed_data" / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    nodes, relations = extract_primekg(Path(args.primekg) if args.primekg else kg_out / "kg.csv", kg_out)
    names = dict(zip(nodes["node_index"].astype(int), nodes["node_name"]))
    names.update(dict(zip(nodes.loc[nodes["node_type"] == "drug", "node_id"],
                          nodes.loc[nodes["node_type"] == "drug", "node_name"])))
    candidates = load_candidates(Path(args.candidates) if args.candidates else None)
    pk_drugs = set(nodes.loc[nodes["node_type"] == "drug", "node_id"])
    missing = sorted(set(candidates["drugbank_id"]) - pk_drugs)
    if missing:
        print(f"[warn] candidates without a PrimeKG drug node: {missing}")

    mimic = Path(args.mimic_dir).expanduser()
    patients = read_parquet_columns(mimic / "final_patients_table.parquet",
                                    ["patient_id", "index_time", "index_hto_drugs"]).sort_values("patient_id")
    dx = read_parquet_columns(mimic / "final_diagnoses.parquet", ["patient_id", "time", "diag"])
    icd_map = pd.read_parquet(kg_out / "icd_mondo_map.parquet", columns=["diag", "primekg_index"])
    dx_pk = dx.merge(icd_map.dropna(subset=["primekg_index"]).drop_duplicates(), on="diag", how="inner")

    meds = read_parquet_columns(mimic / "final_medications.parquet", ["patient_id", "time", "medication", "ndc"])
    print(f"[meds] {len(meds):,} rows, {meds['medication'].nunique():,} names; loading RxNorm ...", flush=True)
    mapper = DrugMapper(Path(args.rxnorm_dir), use_ndc=not args.no_ndc)
    meds_db = map_medications(meds, mapper)
    row_route = meds_db.groupby(level=0)["drug_route"].first()
    (meds_db[["medication", "ndc", "drugbank_ids", "drug_route"]].drop_duplicates(["medication", "ndc"])
     .assign(drugbank_ids=lambda d: d["drugbank_ids"].map(list)).to_parquet(out_dir / "medication_drugbank_map.parquet", index=False))
    med_cov = {k: round(float(n), 4) for k, n in row_route.value_counts(normalize=True).items()}
    print(f"[meds] mapping routes (row share): {med_cov} ({time.time() - start:.0f}s)", flush=True)

    severity = load_severity(Path(args.severity)) if args.severity else None
    use_context = not args.no_context and (kg_out / "measurement_panel_resolved.parquet").exists()
    symptoms = states = meas_tensor = meas_names = None
    context_info = {}
    if use_context:
        symptoms, states, meas_tensor, meas_names, context_info = context_inputs(
            mimic, kg_out, out_dir, patients, dx, nodes, args.panel, mcfg)
        print(f"[context] {json.dumps({k: v for k, v in context_info.items() if k != 'current_abnormal_by_measure'})} "
              f"({time.time() - start:.0f}s)", flush=True)
    feats, matches, counts = build_context(patients, dx_pk, meds_db, relations, candidates, cfg, severity, names,
                                           symptoms, states)
    is_index = index_candidates(patients["index_hto_drugs"], candidates)

    col = {f: i for i, f in enumerate(FEATURES)}
    any_ctx = (feats[..., col["matched_contraindication_count"]] + feats[..., col["matched_indication_count"]]
               + feats[..., col["ddi_count"]]) > 0
    any_ctx_offlabel = any_ctx | (feats[..., col["matched_off_label_count"]] > 0)
    any_ctx_full = any_ctx | ((feats[..., col["matched_side_effect_count"]]
                               + feats[..., col["matched_measurement_contraindication_count"]]
                               + feats[..., col["measurement_indication_count"]]) > 0)
    rel = relations
    per_candidate = []
    for ci, (ing, cid) in enumerate(zip(candidates["ingredient"], candidates["drugbank_id"])):
        per_candidate.append({
            "ingredient": ing, "drugbank_id": cid,
            "primekg_edges": {r: int(((rel["drug_id"] == cid) & (rel["relation"] == r)).sum())
                              for r in ("contraindication", "indication", "off-label use", "drug_drug",
                                        "drug_effect")},
            "contraindication_rate": round(float(feats[:, ci, col["contraindication_flag"]].mean()), 4),
            "indication_rate": round(float(feats[:, ci, col["indication_alignment_flag"]].mean()), 4),
            "ddi_rate": round(float((feats[:, ci, col["ddi_count"]] > 0).mean()), 4),
            "side_effect_overlap_rate": round(float(feats[:, ci, col["side_effect_overlap_flag"]].mean()), 4),
            "measurement_contraindication_rate": round(float(
                feats[:, ci, col["measurement_contraindication_flag"]].mean()), 4),
            "empty_context_rate": round(float(1 - any_ctx[:, ci].mean()), 4),
            "index_patients": int(is_index[:, ci].sum()),
        })
    stats = {
        "config": asdict(cfg), "patients": int(len(patients)), "candidates": int(len(candidates)),
        "feature_names": FEATURES, **counts,
        "medication_row_mapping": med_cov,
        "empty_context_rate": round(float(1 - any_ctx.mean()), 4),
        "empty_context_rate_with_off_label": round(float(1 - any_ctx_offlabel.mean()), 4),
        "empty_context_rate_index_pairs": round(float(1 - any_ctx[is_index].mean()), 4) if is_index.any() else None,
        "empty_context_rate_with_symptoms_and_measurements": round(float(1 - any_ctx_full.mean()), 4),
        "patients_with_index_candidate": int(is_index.any(1).sum()),
        "severity_source": args.severity or "none (PrimeKG DDIs are unrated)",
        "context": {"used": use_context, "measurement_config": asdict(mcfg), **context_info},
        "matches_by_source": {k: int(n) for k, n in matches["evidence_source"].value_counts().items()},
        "per_candidate": per_candidate,
        "seconds": round(time.time() - start, 1),
    }
    payload = {"format": "hto-treatment-context-v2", "patient_ids": torch.as_tensor(patients["patient_id"].to_numpy()),
               "candidates": candidates.to_dict("records"), "feature_names": FEATURES,
               "features": torch.as_tensor(feats), "is_index_candidate": torch.as_tensor(is_index),
               "config": asdict(cfg)}
    if use_context:
        payload.update({"measurements": torch.as_tensor(meas_tensor), "measurement_names": meas_names,
                        "measurement_features": VALUE_FEATURES, "measurement_config": asdict(mcfg)})
    torch.save(payload, out_dir / "treatment_context.pt")
    matches.to_parquet(out_dir / "treatment_context_matches.parquet", index=False)
    (out_dir / "treatment_context.stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "per_candidate"}, indent=2))
    print(f"Wrote {out_dir / 'treatment_context.pt'}")


if __name__ == "__main__":
    main()
