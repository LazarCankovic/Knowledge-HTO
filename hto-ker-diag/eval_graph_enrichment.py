"""Hand-off checks for the enriched graphs (Stages 1-3) before modelling.

    python eval_graph_enrichment.py --dataset full --kg-tag full \
        --mimic-dir data/processed/mimic-iv-3.1/mimic_data --codes data/processed/mimic-iv-3.1/tables/icd_inventory.csv

  1. 0% temporal leakage: every observed code in g_y has an occurrence strictly before t0 and a positive
     recency; every matched diagnosis / medication / symptom / measurement in g_a is dated strictly before
     t0 (re-checked against the source parquet). For context graphs (build_context_kg.py) the allowed
     codes include the measurement codes of lab / vital rows strictly before t0.
  2. >= 90% code mapping coverage: share of diagnosis occurrences (icd_inventory n_rows_usable) whose code
     maps to a MONDO id and has a non-zero ClinicalBERT vector.
  3. < 20% empty context: share of (patient, candidate) pairs in g_a with no contraindication, indication
     or DDI match (the original definition; the rate counting side effects and measurement matches too is
     reported alongside).
Also checks the materialised feature shapes (x: 770, edge_attr: 17; context graphs: value_attr: 4) and
reports symptom / measurement coverage. Writes aggregates only to
processed_data/<dataset>/graph_enrichment_eval.<kg-tag>.json; exits 1 if a check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hto.kg.disease_map import is_supplementary
from hto.kg.enrich import EnrichedPatientGraphs, read_parquet_columns
from hto.kg.icd_snomed_map import norm_code
from hto.kg.measurements import VALUE_DIM, add_states, before_t0, load_measurements, load_panel, measurement_codes
from hto.kg.treatment import FEATURES


def check_gy(path: Path, dx: pd.DataFrame, t0: pd.Series, n_sample: int,
             measurement_rows: pd.DataFrame | None = None) -> dict:
    ds = EnrichedPatientGraphs(path)
    pre = dx[pd.to_datetime(dx["time"]) < pd.to_datetime(dx["patient_id"].map(t0))]
    if measurement_rows is not None:
        pre = pd.concat([pre[["patient_id", "diag"]], measurement_rows.rename(columns={"code": "diag"})])
    pre_codes = pre.groupby("patient_id")["diag"].agg(set).to_dict()
    observed = violations = nonpositive_recency = 0
    kinds = {"diagnosis": 0, "measurement": 0, "abnormal_result": 0}
    for pid, g in zip(ds.patient_ids, ds.graphs):
        local_to_code = {li: c for c, li in ds.code_index[pid].items()}
        obs = torch.nonzero(g.observed_mask).flatten().tolist()
        observed += len(obs)
        allowed = pre_codes.get(pid, set())
        violations += sum(local_to_code.get(li) not in allowed for li in obs)
        nonpositive_recency += int((g.recency_days[g.observed_mask] <= 0).sum())
        for li in obs:
            c = str(local_to_code.get(li, ""))
            kinds["diagnosis" if not c.startswith("meas:") else
                  "abnormal_result" if c.count(":") == 2 else "measurement"] += 1
    rng = np.random.default_rng(0)
    shapes, value_dims, finite = set(), set(), True
    for i in rng.choice(len(ds), size=min(n_sample, len(ds)), replace=False):
        d = ds[int(i)]
        shapes.add((d.x.shape[1], d.edge_attr.shape[1] if d.edge_attr.numel() else 17))
        finite &= bool(torch.isfinite(d.x).all() and torch.isfinite(d.edge_attr).all())
        if getattr(d, "value_attr", None) is not None:
            value_dims.add(int(d.value_attr.shape[1]))
            finite &= bool(torch.isfinite(d.value_attr).all())
    return {"patients": len(ds), "observed_codes": observed, "observed_codes_by_kind": kinds,
            "observed_codes_without_pre_t0_row": violations,
            "observed_codes_nonpositive_recency": nonpositive_recency,
            "leakage_rate": round((violations + nonpositive_recency) / max(observed, 1), 6),
            "feature_dims": sorted(shapes), "value_attr_dims": sorted(value_dims), "features_finite": finite}


def check_ga(ctx_path: Path, matches_path: Path, t0: pd.Series) -> dict:
    ctx = torch.load(ctx_path, map_location="cpu", weights_only=False)
    feats = ctx["features"].numpy()
    col = {f: i for i, f in enumerate(ctx["feature_names"])}
    any_ctx = (feats[..., col["matched_contraindication_count"]] + feats[..., col["matched_indication_count"]]
               + feats[..., col["ddi_count"]]) > 0
    extra = [c for c in ("matched_side_effect_count", "matched_measurement_contraindication_count",
                         "measurement_indication_count") if c in col]
    any_full = any_ctx | (feats[..., [col[c] for c in extra]].sum(-1) > 0) if extra else any_ctx
    m = pd.read_parquet(matches_path)
    m_t0 = pd.to_datetime(m["patient_id"].map(t0))
    late = int((pd.to_datetime(m["evidence_time"]) >= m_t0).sum())
    stored_t0_mismatch = int((pd.to_datetime(m["index_time"]) != m_t0).sum())
    is_index = ctx["is_index_candidate"].numpy()
    out = {"pairs": int(any_ctx.size), "matches": int(len(m)),
           "matches_at_or_after_t0": late, "matches_with_wrong_t0": stored_t0_mismatch,
           "leakage_rate": round(late / max(len(m), 1), 6),
           "empty_context_rate": round(float(1 - any_ctx.mean()), 4),
           "empty_context_rate_index_pairs": round(float(1 - any_ctx[is_index].mean()), 4) if is_index.any() else None,
           "empty_context_rate_with_symptoms_and_measurements": round(float(1 - any_full.mean()), 4),
           "matches_by_relation": {k: int(n) for k, n in m["relation"].value_counts().items()},
           "feature_names_match": ctx["feature_names"] == FEATURES}
    if "evidence_source" in m:
        out["matches_by_source"] = {k: int(n) for k, n in m["evidence_source"].value_counts().items()}
    if "measurements" in ctx:
        mt = ctx["measurements"]
        out["measurements"] = {"shape": list(mt.shape), "finite": bool(torch.isfinite(mt).all()),
                               "patients_with_any_value": int((mt[..., 0] > 0).any(1).sum()),
                               "coverage_by_measure": {k: round(float(v), 4) for k, v in
                                                       zip(ctx["measurement_names"], (mt[..., 0] > 0).float().mean(0))}}
    return out


def check_context(codes_path: Path, kg_out: Path, ctx_kg_dir: Path) -> dict:
    """Symptom (HPO) coverage of diagnosis occurrences and the context KG build report."""
    inv = pd.read_csv(codes_path, dtype={"icd_code": str})
    inv["diag"] = inv["icd_version"].astype(str) + "_" + inv["icd_code"].map(norm_code)
    w = inv["n_rows_usable"].astype(float)
    phen = pd.read_parquet(kg_out / "icd_phenotype_map.parquet")
    has = inv["diag"].isin(set(phen["diag"]))
    symptom_chapter = [(v == 10 and str(c).startswith("R")) or (v == 9 and str(c)[:3].isdigit() and 780 <= int(str(c)[:3]) <= 799)
                       for v, c in zip(inv["icd_version"].astype(int), inv["icd_code"])]
    sc = pd.Series(symptom_chapter, index=inv.index)
    stats_path = ctx_kg_dir / "graph_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    return {"diagnosis_rows_with_hpo_phenotype": round(float((w * has).sum() / w.sum()), 4),
            "symptom_chapter_rows_with_hpo_phenotype": round(float((w * has)[sc].sum() / max(w[sc].sum(), 1)), 4),
            "codes_with_hpo_phenotype": int(has.sum()),
            "context_kg": {k: v for k, v in stats.items() if not isinstance(v, (list, dict)) or k.startswith("unresolved")
                           or k.endswith("routes")}}


def check_coverage(codes_path: Path, icd_map_path: Path, text_path: Path) -> dict:
    inv = pd.read_csv(codes_path, dtype={"icd_code": str})
    inv["diag"] = inv["icd_version"].astype(str) + "_" + inv["icd_code"].map(norm_code)
    icd_map = pd.read_parquet(icd_map_path)
    tf = torch.load(text_path, map_location="cpu", weights_only=False)
    nonzero = set(k for k, z in zip(tf["keys"], (tf["matrix"].abs().sum(1) > 0).tolist()) if z)
    mondo = set(icd_map.loc[icd_map["mondo_id"].notna(), "diag"])
    pk = set(icd_map.loc[icd_map["primekg_index"].notna(), "diag"])
    w = inv["n_rows_usable"].astype(float)
    share = lambda mask: round(float((w * mask).sum() / w.sum()), 4)
    disease = ~pd.Series([is_supplementary(v, norm_code(c)) for v, c in zip(inv["icd_version"], inv["icd_code"])],
                         index=inv.index)
    wd = w[disease]
    return {"rows": int(w.sum()), "codes": int(len(inv)),
            "mondo": share(inv["diag"].isin(mondo)),
            "mondo_and_nonzero_embedding": share(inv["diag"].isin(mondo) & inv["diag"].isin(nonzero)),
            "supplementary_code_rows_share": round(float(1 - wd.sum() / w.sum()), 4),
            "mondo_and_nonzero_embedding_disease_codes": round(float(
                (wd * (inv["diag"].isin(mondo) & inv["diag"].isin(nonzero))[disease]).sum() / wd.sum()), 4),
            "primekg_node": share(inv["diag"].isin(pk)),
            "nonzero_embedding": share(inv["diag"].isin(nonzero)),
            "embedding_dim": int(tf["matrix"].shape[1])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--kg-tag", required=True)
    ap.add_argument("--mimic-dir", required=True)
    ap.add_argument("--codes", required=True, help="icd_inventory.csv")
    ap.add_argument("--kg-out", default=str(ROOT / "processed_data" / "kg"), help="Stage 1 output folder")
    ap.add_argument("--data-dir", help="default processed_data/<dataset>/ (Stage 2 and 3 outputs)")
    ap.add_argument("--n-sample", type=int, default=200, help="graphs materialised for the shape check")
    ap.add_argument("--panel", help="measurement panel TSV for context graphs (default resources/measurement_panel.tsv)")
    args = ap.parse_args()
    data_dir = Path(args.data_dir or ROOT / "processed_data" / args.dataset)
    kg_out = Path(args.kg_out)
    kg_dir = kg_out / args.kg_tag
    mimic = Path(args.mimic_dir).expanduser()
    t0 = read_parquet_columns(mimic / "final_patients_table.parquet",
                              ["patient_id", "index_time"]).set_index("patient_id")["index_time"]
    dx = read_parquet_columns(mimic / "final_diagnoses.parquet", ["patient_id", "time", "diag"])
    entities = pd.read_csv(kg_dir / "entities.tsv", sep="\t", dtype={"key": str}, keep_default_na=False)
    context = bool((entities["kind"] == "measure").any()) if "kind" in entities else False
    meas_rows = None
    if context:
        panel = load_panel(Path(args.panel) if args.panel else None)
        meas_rows = measurement_codes(before_t0(add_states(load_measurements(mimic, panel), panel), t0))

    report = {
        "coverage": check_coverage(Path(args.codes), kg_out / "icd_mondo_map.parquet",
                                   kg_out / "node_features_bert.pt"),
        "g_y": check_gy(data_dir / f"patient_graphs_enriched.{args.kg_tag}.pt", dx, t0, args.n_sample,
                        meas_rows[["patient_id", "code"]].drop_duplicates() if context else None),
        "g_a": check_ga(data_dir / "treatment_context.pt", data_dir / "treatment_context_matches.parquet", t0),
        "vitals_used": context,
    }
    if context:
        report["context"] = check_context(Path(args.codes), kg_out, kg_dir)
    checks = {
        "temporal_leakage_0pct": report["g_y"]["leakage_rate"] == 0 and report["g_a"]["leakage_rate"] == 0
                                 and report["g_a"]["matches_with_wrong_t0"] == 0,
        "code_mapping_coverage_ge_90pct": report["coverage"]["mondo_and_nonzero_embedding"] >= 0.90,
        "empty_context_lt_20pct": report["g_a"]["empty_context_rate"] < 0.20,
        "feature_dims_770_17": report["g_y"]["feature_dims"] == [(770, 17)] and report["g_y"]["features_finite"],
    }
    if context:
        checks["value_attr_dim_4"] = report["g_y"]["value_attr_dims"] == [VALUE_DIM]
        checks["measurements_tensor_finite"] = bool(report["g_a"].get("measurements", {}).get("finite", False))
    report["checks"] = checks
    report["passed"] = all(checks.values())
    out = data_dir / f"graph_enrichment_eval.{args.kg_tag}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    for k, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {k}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
