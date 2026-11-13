"""Context knowledge graph: the diagnosis + lab SNOMED graph extended with measurements and symptoms.

The base graph (build_lab_kg.py output, e.g. processed_data/kg/lab_snomed) is kept verbatim, ids included.
New nodes and edges (each relation also gets its inverse "inv_<name>"):
  measure        meas:potassium          -measure_maps_to_snomed->  SNOMED concepts its lab items map to (LOINC)
  measure_state  meas:potassium:high     -state_of->  meas:potassium
                                         -indicates-> HP:0002153 (Hyperkalemia)
  phenotype      HP:0002153              -phenotype_is_a-> HPO parent (when both are in the graph)
  icd            10_E875                 -manifests_as->   HPO term named like the code's title, else HPO
                                                           terms of its core SNOMED concepts (those named
                                                           most like its title; HPO SNOMEDCT_US xref or
                                                           exact name)
                                         -has_phenotype->  phenotypes of the code's PrimeKG disease(s)
                                                           (disease_phenotype_positive), at most
                                                           max_phenotypes_per_code: phenotypes that a
                                                           measurement state or symptom code can reach
                                                           first, then the most specific ones
A measurement state and a diagnosis meet at a shared phenotype (state -indicates-> P <-manifests_as- code),
so the patient graph builder links observed abnormal results to observed diagnoses with the usual τ.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hto.kg.hpo import Hpo, norm_name, resolve
from hto.kg.icd_snomed_map import norm_code

NEW_RELATIONS = ("measure_maps_to_snomed", "state_of", "indicates", "manifests_as", "has_phenotype",
                 "phenotype_is_a")
TITLE_SUFFIX = re.compile(r"\s+(unspecified|not elsewhere classified|nos|nec)$")
SNOMED_TAG = r"\s*\((?:finding|disorder)\)$"
TITLE_STOPWORDS = {"of", "the", "and", "or", "with", "without", "in", "to", "due", "unspecified", "nos", "nec",
                   "other", "primary", "specified"}
CORE_MIN_OVERLAP = 0.6


@dataclass
class ContextGraphConfig:
    max_phenotypes_per_code: int = 10


def resolve_panel(panel: pd.DataFrame, hpo: Hpo, disease_names: dict[str, int]) -> pd.DataFrame:
    """One row per (measure, state) with a threshold: the HPO term and PrimeKG disease it indicates."""
    rows = []
    for r in panel.itertuples(index=False):
        for state in ("low", "high"):
            thr = getattr(r, state)
            if pd.isna(thr):
                continue
            phen_q, dis_q = getattr(r, f"{state}_phenotype"), getattr(r, f"{state}_disease")
            dis = next((disease_names[d.casefold()] for d in dis_q if d.casefold() in disease_names), None)
            rows.append((r.measure, state, float(thr), resolve(hpo, phen_q) if phen_q else None,
                         "|".join(phen_q), dis, "|".join(dis_q)))
    out = pd.DataFrame(rows, columns=["measure", "state", "threshold", "phenotype_id", "phenotype_query",
                                      "disease_index", "disease_query"])
    out["disease_index"] = out["disease_index"].astype("Int64")
    return out


def mondo_disease_names(mondo_json: Path, mondo_pk: dict[str, int]) -> dict[str, int]:
    """Lower-cased MONDO label or exact synonym -> PrimeKG disease node_index (labels win)."""
    with open(mondo_json, encoding="utf-8") as f:
        nodes = json.load(f)["graphs"][0]["nodes"]
    labels, synonyms = {}, {}
    for n in nodes:
        url = n.get("id", "")
        mid = "MONDO:" + url.rsplit("MONDO_", 1)[1] if "MONDO_" in url else None
        if mid not in mondo_pk:
            continue
        if n.get("lbl"):
            labels.setdefault(n["lbl"].casefold().strip(), mondo_pk[mid])
        for s in (n.get("meta") or {}).get("synonyms", []) or []:
            if s.get("pred") == "hasExactSynonym" and s.get("val"):
                synonyms.setdefault(s["val"].casefold().strip(), mondo_pk[mid])
    return {**synonyms, **labels}


def _tokens(s: str) -> set[str]:
    return {t for t in norm_name(s).split() if t not in TITLE_STOPWORDS}


def core_concepts(m: pd.DataFrame, titles: dict[str, str], snomed_names: dict[str, str],
                  min_overlap: float = CORE_MIN_OVERLAP) -> pd.DataFrame:
    """Rows of m (diag, sctid) whose SNOMED name best matches the code's title (token Jaccard, at least
    min_overlap). The ICD -> SNOMED map is many-to-one (every specific concept that rolls up into a code
    maps to it), so only these core concepts say what the code itself denotes."""
    rows = []
    for diag, sctid in zip(m["diag"], m["sctid"]):
        a, b = _tokens(titles.get(diag, "")), _tokens(re.sub(r"\s*\([^)]*\)$", "", snomed_names.get(sctid, "")))
        rows.append(len(a & b) / len(a | b) if a and b else 0.0)
    m = m.assign(overlap=rows)
    best = m.groupby("diag")["overlap"].transform("max")
    return m[(m["overlap"] >= min_overlap) & (m["overlap"] == best)].drop(columns="overlap")


def icd_symptom_phenotypes(icd_snomed: pd.DataFrame, codes: pd.DataFrame, hpo: Hpo,
                           snomed_names: dict[str, str] | None = None) -> pd.DataFrame:
    """diag, hpo_id, route: HPO terms a code itself denotes, by the first route that finds one:
    name (exact ICD title), snomed_xref (HPO SNOMEDCT_US xref of the code's SNOMED concepts; older HPO
    releases), snomed_name (exact name of the code's core SNOMED finding / disorder concept).
    icd_snomed: icd_version, icd_code, sctid. codes: icd_version, icd_code, long_title.
    snomed_names: sctid -> SNOMED name with its semantic tag; when given, only core concepts are used."""
    c = codes.copy()
    c["diag"] = c["icd_version"].astype(int).astype(str) + "_" + c["icd_code"].map(norm_code)
    titles = c["long_title"].fillna("").map(lambda t: TITLE_SUFFIX.sub("", norm_name(t)).strip())
    found = [pd.DataFrame({"diag": c["diag"], "hpo_id": titles.map(hpo.lookup)}).dropna().assign(route="name")]
    sx = hpo.xrefs[hpo.xrefs["prefix"] == "SNOMEDCT_US"][["code", "hpo_id"]].rename(columns={"code": "sctid"})
    m = icd_snomed.dropna(subset=["sctid"]).copy()
    m["sctid"] = m["sctid"].astype(str)
    m["diag"] = m["icd_version"].astype(int).astype(str) + "_" + m["icd_code"].map(norm_code)
    m = m.loc[~m["diag"].isin(set(found[0]["diag"])), ["diag", "sctid"]].drop_duplicates()
    if snomed_names:
        m = core_concepts(m, dict(zip(c["diag"], c["long_title"].fillna(""))), snomed_names)
    found.append(m.merge(sx, on="sctid")[["diag", "hpo_id"]].assign(route="snomed_xref"))
    if snomed_names:
        rest = m[~m["diag"].isin(set(found[1]["diag"]))]
        names = rest["sctid"].map(snomed_names).fillna("")
        tagged = names.str.contains(SNOMED_TAG, regex=True)
        hid = names[tagged].str.replace(SNOMED_TAG, "", regex=True).map(norm_name).map(hpo.lookup)
        found.append(pd.DataFrame({"diag": rest.loc[tagged, "diag"], "hpo_id": hid}).dropna()
                     .assign(route="snomed_name"))
    return pd.concat(found, ignore_index=True).drop_duplicates(["diag", "hpo_id"]).reset_index(drop=True)


def icd_disease_phenotypes(icd_map: pd.DataFrame, disease_phenotypes: pd.DataFrame, pk_hpo: dict[int, str],
                           anchors: set[str], cap: int) -> pd.DataFrame:
    """diag, hpo_id: phenotypes of each code's PrimeKG disease(s), anchors first, then most specific."""
    m = icd_map.dropna(subset=["primekg_index"])[["diag", "primekg_index"]].drop_duplicates()
    m["primekg_index"] = m["primekg_index"].astype("int64")
    dp = disease_phenotypes.assign(hpo_id=disease_phenotypes["phenotype_index"].map(pk_hpo)).dropna(subset=["hpo_id"])
    n_dis = dp.groupby("hpo_id")["disease_index"].nunique()
    j = m.merge(dp, left_on="primekg_index", right_on="disease_index")[["diag", "hpo_id"]].drop_duplicates()
    j = j.assign(not_anchor=~j["hpo_id"].isin(anchors), n_diseases=j["hpo_id"].map(n_dis))
    j = j.sort_values(["diag", "not_anchor", "n_diseases", "hpo_id"]).groupby("diag").head(cap)
    return j[["diag", "hpo_id"]].reset_index(drop=True)


def build_context_graph(base: dict, panel: pd.DataFrame, panel_resolved: pd.DataFrame,
                        measure_snomed: dict[str, set[str]], icd_direct: pd.DataFrame,
                        icd_disease: pd.DataFrame, hpo: Hpo) -> tuple[dict, pd.DataFrame]:
    """Returns the extended graph (entities, relations, triples, stats) and texts for the new nodes."""
    ent = base["entities"]
    base_keys = set(ent["key"].astype(str))
    nodes: dict[str, tuple[str, str]] = {}
    edges: set[tuple[str, str, str]] = set()
    texts: dict[str, str] = {}

    def phenotype(hid: str) -> str:
        if hid not in nodes:
            name = hpo.names.get(hid, hid)
            nodes[hid] = ("phenotype", name)
            d = hpo.definitions.get(hid, "")
            texts[hid] = f"Symptom or sign: {name}." + (f" Definition: {d}" if d else "")
        return hid

    p = panel.set_index("measure")
    for m in panel["measure"]:
        key = f"meas:{m}"
        row = p.loc[m]
        nodes[key] = ("measure", row["display"])
        where = row["fluid"] or "vital sign"
        texts[key] = f"Measurement: {row['display']} ({where}, {row['unit']})."
        for sct in sorted(measure_snomed.get(m, set()) & base_keys):
            edges.add((key, "measure_maps_to_snomed", sct))
    for r in panel_resolved.itertuples(index=False):
        row = p.loc[r.measure]
        key = f"meas:{r.measure}:{r.state}"
        nodes[key] = ("measure_state", f"{r.state} {row['display']}")
        sign = "<" if r.state == "low" else ">"
        text = f"Abnormal result: {r.state} {row['display']} ({sign} {r.threshold:g} {row['unit']})"
        edges.add((key, "state_of", f"meas:{r.measure}"))
        if r.phenotype_id:
            edges.add((key, "indicates", phenotype(r.phenotype_id)))
            text += f", indicating {hpo.names.get(r.phenotype_id, '')}"
        texts[key] = text + "."
    for rel, df in (("manifests_as", icd_direct), ("has_phenotype", icd_disease)):
        for diag, hid in df[["diag", "hpo_id"]].itertuples(index=False):
            if diag in base_keys and hid in hpo.names:
                edges.add((diag, rel, phenotype(hid)))
    phen = {k for k, (kind, _) in nodes.items() if kind == "phenotype"}
    for hid in phen:
        for parent in hpo.parents.get(hid, ()):
            if parent in phen:
                edges.add((hid, "phenotype_is_a", parent))

    new_keys = sorted(nodes)
    entities = pd.concat([ent, pd.DataFrame(
        [(len(ent) + i, k, *nodes[k]) for i, k in enumerate(new_keys, 1)],
        columns=["id", "key", "kind", "name"])], ignore_index=True)
    names = list(base["relations"]["name"])
    names += [r for r in NEW_RELATIONS if r not in names] + [f"inv_{r}" for r in NEW_RELATIONS if f"inv_{r}" not in names]
    relations = pd.DataFrame({"id": range(1, len(names) + 1), "name": names})
    eid = dict(zip(entities["key"].astype(str), entities["id"]))
    rid = dict(zip(relations["name"], relations["id"]))
    new = [(eid[h], rid[r], eid[t]) for h, r, t in sorted(edges)] + \
          [(eid[t], rid[f"inv_{r}"], eid[h]) for h, r, t in sorted(edges)]
    triples = pd.concat([base["triples"][["h", "r", "t"]], pd.DataFrame(new, columns=["h", "r", "t"])],
                        ignore_index=True).drop_duplicates().astype(int).reset_index(drop=True)
    rel_counts = pd.Series([r for _, r, _ in edges]).value_counts().to_dict() if edges else {}
    kinds = pd.Series([k for k, _ in nodes.values()]).value_counts().to_dict() if nodes else {}
    stats = {"base_entities": len(ent), "entities": len(entities), "relations": len(relations),
             "triples": len(triples), "new_nodes": {k: int(v) for k, v in kinds.items()},
             "new_forward_edges": {k: int(v) for k, v in rel_counts.items()}}
    text_df = pd.DataFrame({"key": list(texts), "text": list(texts.values()), "source": "context"})
    return {"entities": entities, "relations": relations, "triples": triples, "stats": stats}, text_df
