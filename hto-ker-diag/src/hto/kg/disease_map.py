"""ICD-9-CM / ICD-10-CM -> MONDO -> PrimeKG disease nodes.

MONDO links few ICD codes directly (about 2k ICD-10-CM and 6k ICD-9 xrefs) but about 9k SNOMED CT concepts,
and every cohort code already has SNOMED concepts from icd_snomed_map. A code takes the first route below
that gives at least one MONDO term, and keeps at most max_per_code of them:

  icd_xref               MONDO xref to the code itself (ICD10CM:... for ICD-10, ICD9:... for ICD-9)
  gem>icd_xref           ICD-9 only: CMS GEM targets (ICD-10-CM) with a MONDO xref
  snomed_xref            the code's SNOMED concepts (icd_snomed_map) with a MONDO SCTID xref
  icd_parent_xref        the nearest parent code (one character shorter at a time, down to 3) with a xref
  snomed_ancestor_xref   the nearest SNOMED IS-A ancestors (up to snomed_ancestor_depth levels) with a xref
  snomed_finding_xref    for SNOMED situations ("Personal history of pulmonary embolism"), the disorder in
                         their Associated finding attribute, then its ancestors. Situations about a family
                         member are excluded upstream (snomed_associated_findings).

Status, encounter and external-cause codes (ICD-10 V-Z, ICD-9 V and E) have no disease to map to; the
coverage report also gives the share over the remaining disease codes.

A MONDO term absent from PrimeKG resolves to its nearest MONDO is_a ancestor that is a PrimeKG node, up to
primekg_backoff_hops levels; primekg_hops records how many levels were climbed.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from hto.kg.icd_snomed_map import norm_code

MONDO_URL = "http://purl.obolibrary.org/obo/MONDO_"
ROUTES = ("icd_xref", "gem>icd_xref", "snomed_xref", "icd_parent_xref", "snomed_ancestor_xref",
          "snomed_finding_xref")
ASSOCIATED_FINDING = "246090004"
SUBJECT_RELATIONSHIP_CONTEXT = "408732007"
SUBJECT_OF_RECORD = "410604004"


def is_supplementary(icd_version: int, icd_code: str) -> bool:
    """Status, encounter and external-cause codes: ICD-10 chapters V-Z, ICD-9 V and E codes."""
    return icd_code[:1] in (("V", "W", "X", "Y", "Z") if int(icd_version) == 10 else ("V", "E"))


def snomed_associated_findings(rels: pd.DataFrame) -> dict[str, list[str]]:
    """Situation concept -> its Associated finding concepts, skipping situations whose subject is not the
    patient (family history). rels: sourceId, destinationId, typeId (inferred SNOMED relationships)."""
    ctx = rels[rels["typeId"] == SUBJECT_RELATIONSHIP_CONTEXT]
    other_subject = set(ctx.loc[ctx["destinationId"] != SUBJECT_OF_RECORD, "sourceId"])
    af = rels[(rels["typeId"] == ASSOCIATED_FINDING) & ~rels["sourceId"].isin(other_subject)]
    af = af.sort_values(["sourceId", "destinationId"])
    return {s: list(g["destinationId"]) for s, g in af.groupby("sourceId")}


@dataclass
class DiseaseMapConfig:
    max_per_code: int = 3
    snomed_ancestor_depth: int = 3
    primekg_backoff_hops: int = 2


@dataclass
class Mondo:
    names: dict[str, str]
    definitions: dict[str, str]
    parents: dict[str, set[str]]
    xrefs: pd.DataFrame          # mondo_id, prefix, code (ICD codes without dots, SCTIDs as digits)


def _curie(url: str) -> str | None:
    return f"MONDO:{url[len(MONDO_URL):]}" if url.startswith(MONDO_URL) else None


def load_mondo(path: Path) -> Mondo:
    """mondo.json (OBO Graphs). Deprecated terms are dropped."""
    with open(path, encoding="utf-8") as f:
        graph = json.load(f)["graphs"][0]
    names, definitions, rows = {}, {}, []
    for node in graph["nodes"]:
        mid = _curie(node.get("id", ""))
        meta = node.get("meta", {}) or {}
        if mid is None or meta.get("deprecated"):
            continue
        names[mid] = node.get("lbl", "") or ""
        if (meta.get("definition") or {}).get("val"):
            definitions[mid] = meta["definition"]["val"]
        for x in meta.get("xrefs", []) or []:
            val = (x.get("val", "") if isinstance(x, dict) else str(x)).replace("\\:", ":")
            prefix, _, code = val.partition(":")
            if prefix in ("ICD10CM", "ICD9") and code and "-" not in code:
                rows.append((mid, prefix, norm_code(code)))
            elif prefix == "SCTID" and code.isdigit():
                rows.append((mid, prefix, code))
    parents: dict[str, set[str]] = defaultdict(set)
    for e in graph.get("edges", []):
        if e.get("pred") != "is_a":
            continue
        sub, obj = _curie(e.get("sub", "")), _curie(e.get("obj", ""))
        if sub in names and obj in names:
            parents[sub].add(obj)
    xrefs = pd.DataFrame(rows, columns=["mondo_id", "prefix", "code"]).drop_duplicates()
    return Mondo(names, definitions, dict(parents), xrefs.reset_index(drop=True))


def _index(xrefs: pd.DataFrame, prefix: str) -> dict[str, list[str]]:
    sub = xrefs[xrefs["prefix"] == prefix].sort_values(["code", "mondo_id"])
    return {k: list(g["mondo_id"]) for k, g in sub.groupby("code")}


def _ordered_unique(items) -> list:
    return list(dict.fromkeys(items))


def build_icd_mondo_map(codes: pd.DataFrame, mondo: Mondo, icd_snomed: pd.DataFrame | None = None,
                        snomed_parents: dict[str, set[str]] | None = None, gems: pd.DataFrame | None = None,
                        cfg: DiseaseMapConfig = DiseaseMapConfig(),
                        snomed_findings: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """codes: icd_version, icd_code. icd_snomed: build_icd_map output (icd_version, icd_code, sctid,
    rule_kind). gems: load_gems output. snomed_findings: snomed_associated_findings output. Returns one row
    per (code, mondo_id), with route 'unmapped' and mondo_id None for codes no route reaches."""
    by_icd = {10: _index(mondo.xrefs, "ICD10CM"), 9: _index(mondo.xrefs, "ICD9")}
    by_sct = _index(mondo.xrefs, "SCTID")
    sct_of: dict[tuple[int, str], list[str]] = {}
    if icd_snomed is not None:
        m = icd_snomed.dropna(subset=["sctid"]).copy()
        m["strength"] = (m["rule_kind"] != "unconditional").astype(int)
        m = m.sort_values(["icd_version", "icd_code", "strength", "sctid"])
        sct_of = {(int(v), str(c)): list(g["sctid"].astype(str))
                  for (v, c), g in m.groupby(["icd_version", "icd_code"])}
    gem_of: dict[str, list[str]] = {}
    if gems is not None:
        g = gems[~gems["no_map"]] if "no_map" in gems else gems
        gem_of = {k: list(v["target"]) for k, v in g.groupby("source")}

    def via_snomed_ancestors(sctids: list[str]) -> list[str]:
        if not snomed_parents:
            return []
        seen, frontier = set(sctids), list(sctids)
        for _ in range(cfg.snomed_ancestor_depth):
            frontier = _ordered_unique(p for s in frontier for p in sorted(snomed_parents.get(s, ())) if p not in seen)
            seen.update(frontier)
            hits = _ordered_unique(mid for s in frontier for mid in by_sct.get(s, []))
            if hits:
                return hits
        return []

    def via_findings(sctids: list[str]) -> list[str]:
        findings = _ordered_unique(f for s in sctids for f in (snomed_findings or {}).get(s, []))
        direct = _ordered_unique(mid for f in findings for mid in by_sct.get(f, []))
        return direct or via_snomed_ancestors(findings)

    rows = []
    for version, code in zip(codes["icd_version"].astype(int), codes["icd_code"].map(norm_code)):
        idx = by_icd.get(version, {})
        sctids = sct_of.get((version, code), [])
        candidates = [
            ("icd_xref", lambda: idx.get(code, [])),
            ("gem>icd_xref", lambda: _ordered_unique(
                mid for t in gem_of.get(code, []) for mid in by_icd[10].get(t, [])) if version == 9 else []),
            ("snomed_xref", lambda: _ordered_unique(mid for s in sctids for mid in by_sct.get(s, []))),
            ("icd_parent_xref", lambda: next(
                (idx[code[:n]] for n in range(len(code) - 1, 2, -1) if code[:n] in idx), [])),
            ("snomed_ancestor_xref", lambda: via_snomed_ancestors(sctids)),
            ("snomed_finding_xref", lambda: via_findings(sctids)),
        ]
        found, route = [], "unmapped"
        for name, fn in candidates:
            found = fn()
            if found:
                route = name
                break
        if not found:
            rows.append((version, code, None, route, None))
            continue
        for rank, mid in enumerate(found[: cfg.max_per_code]):
            rows.append((version, code, mid, route, rank))
    out = pd.DataFrame(rows, columns=["icd_version", "icd_code", "mondo_id", "route", "rank"])
    out["diag"] = out["icd_version"].astype(str) + "_" + out["icd_code"]
    out["mondo_name"] = out["mondo_id"].map(mondo.names)
    return out


def resolve_primekg(mondo_ids, mondo: Mondo, mondo_pk: dict[str, int],
                    max_hops: int) -> dict[str, tuple[int, int]]:
    """MONDO id -> (PrimeKG node_index, hops climbed). Ties at the same level take the smallest index."""
    out = {}
    for mid in set(mondo_ids):
        frontier, seen = [mid], {mid}
        for hop in range(max_hops + 1):
            hits = sorted(mondo_pk[m] for m in frontier if m in mondo_pk)
            if hits:
                out[mid] = (hits[0], hop)
                break
            frontier = [p for m in frontier for p in mondo.parents.get(m, ()) if p not in seen]
            seen.update(frontier)
    return out


def attach_primekg(icd_mondo: pd.DataFrame, mondo: Mondo, mondo_pk: dict[str, int],
                   cfg: DiseaseMapConfig = DiseaseMapConfig()) -> pd.DataFrame:
    res = resolve_primekg(icd_mondo["mondo_id"].dropna(), mondo, mondo_pk, cfg.primekg_backoff_hops)
    out = icd_mondo.copy()
    out["primekg_index"] = out["mondo_id"].map(lambda m: res[m][0] if m in res else None).astype("Int64")
    out["primekg_hops"] = out["mondo_id"].map(lambda m: res[m][1] if m in res else None).astype("Int64")
    return out


def coverage_report(icd_map: pd.DataFrame, inventory: pd.DataFrame | None, cfg: DiseaseMapConfig) -> dict:
    per_code = (icd_map.assign(mondo=icd_map["mondo_id"].notna(), pk=icd_map["primekg_index"].notna())
                .groupby(["icd_version", "icd_code"])
                .agg(mondo=("mondo", "any"), primekg=("pk", "any"), route=("route", "first"))
                .reset_index())
    report = {"config": asdict(cfg), "codes": int(len(per_code)),
              "codes_mapped_mondo": int(per_code["mondo"].sum()),
              "codes_mapped_primekg": int(per_code["primekg"].sum()),
              "routes": {k: int(n) for k, n in per_code["route"].value_counts().items()}}
    if inventory is not None and "n_rows_usable" in inventory:
        inv = inventory.assign(icd_code=inventory["icd_code"].map(norm_code))
        merged = inv.merge(per_code, on=["icd_version", "icd_code"], how="left").fillna(
            {"mondo": False, "primekg": False})
        w = merged["n_rows_usable"].astype(float)
        report["rows"] = int(w.sum())
        report["mondo_usage_weighted"] = round(float((w * merged["mondo"]).sum() / w.sum()), 4)
        report["primekg_usage_weighted"] = round(float((w * merged["primekg"]).sum() / w.sum()), 4)
        disease = ~merged.apply(lambda r: is_supplementary(r["icd_version"], r["icd_code"]), axis=1)
        wd = w[disease]
        report["supplementary_code_rows_share"] = round(float(1 - wd.sum() / w.sum()), 4)
        report["mondo_usage_weighted_disease_codes"] = round(
            float((wd * merged.loc[disease, "mondo"]).sum() / wd.sum()), 4) if wd.sum() > 0 else None
        report["by_version"] = {
            f"icd{v}": {"mondo_usage_weighted": round(float((g["n_rows_usable"] * g["mondo"]).sum()
                                                            / g["n_rows_usable"].sum()), 4),
                        "primekg_usage_weighted": round(float((g["n_rows_usable"] * g["primekg"]).sum()
                                                              / g["n_rows_usable"].sum()), 4)}
            for v, g in merged.groupby("icd_version")}
        report["route_usage_weighted"] = {
            k: round(float(g["n_rows_usable"].sum() / w.sum()), 4)
            for k, g in merged.fillna({"route": "missing"}).groupby("route")}
    return report
