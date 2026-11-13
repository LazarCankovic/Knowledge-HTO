"""Candidate treatment context block (g_a): 1-hop PrimeKG safety / interaction lookup per (patient, drug).

For patient p with index time t0 and candidate drug T (a DrugBank ingredient):
  V_p       PrimeKG disease nodes of p's diagnoses dated strictly before t0 (whole history)
  M_active  DrugBank ingredients of p's prescriptions started in [t0 - active_window_days, t0)
  contraindications  (T -contraindication-> disease) ∩ V_p
  indications        (T -indication-> disease) ∩ V_p
  off-label uses     (T -off-label use-> disease) ∩ V_p
  DDIs               (T -drug_drug-> drug) ∩ (M_active \\ {T})
With the context inputs (build_context_kg.py + the measurement panel):
  P_p       the patient's observed symptoms before t0: HPO terms of diagnosis codes (manifests_as) and of
            current abnormal results (last value within the window is low / high)
  D_meas    PrimeKG diseases indicated by current abnormal results (e.g. potassium > 5.0 -> hyperkalemia)
  side effects       (T -drug_effect-> phenotype) ∩ P_p    (the drug may worsen a problem the patient has)
  measurement contraindications / indications  (T -contraindication / indication-> disease) ∩ D_meas

PrimeKG's drug_drug edges carry no severity. Without a severity file every DDI is 'unrated' and weighs
unrated_severity; with one (drug_a, drug_b, level in major / moderate / minor, DrugBank ids) rated pairs
weigh SEVERITY[level].

Medications map to DrugBank through RxNorm: the NDC (RXNSAT) or the drug name (RXNCONSO, RxNorm and
DrugBank names and synonyms) gives an RxNorm concept, which is walked to its ingredients (TTY IN) through
RXNREL; each ingredient's DrugBank atom gives the id.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RESOURCES = Path(__file__).resolve().parent / "resources"
SEVERITY = {"major": 1.0, "moderate": 0.66, "minor": 0.33}
FEATURES = [
    "contraindication_flag", "matched_contraindication_count",
    "indication_alignment_flag", "matched_indication_count",
    "off_label_flag", "matched_off_label_count",
    "ddi_count", "ddi_fraction_of_active",
    "ddi_severity_max", "ddi_severity_mean",
    "ddi_n_major", "ddi_n_moderate", "ddi_n_minor", "ddi_n_unrated",
    "n_active_meds", "n_patient_diseases",
    "side_effect_overlap_flag", "matched_side_effect_count",
    "measurement_contraindication_flag", "matched_measurement_contraindication_count",
    "measurement_indication_count", "n_patient_phenotypes", "n_abnormal_measurements",
]
# RXNREL row (RXCUI1, RELA, RXCUI2) reads "RXCUI2 RELA RXCUI1"; these lead from a product to its ingredients
TOWARD_INGREDIENT = {"tradename_of", "consists_of", "has_ingredient", "has_precise_ingredient", "form_of",
                     "contains", "has_ingredients", "has_part", "quantified_form_of"}
NAME_TTYS = {"IN", "PIN", "MIN", "BN", "SCD", "SBD", "SCDC", "SBDC", "SY", "TMSY", "PSN", "GPCK", "BPCK"}
STOP_TOKENS = {
    "hcl", "hydrochloride", "sodium", "potassium", "calcium", "magnesium", "chloride", "sulfate", "phosphate",
    "acetate", "tartrate", "succinate", "besylate", "maleate", "citrate", "bromide", "mesylate", "fumarate",
    "flush", "bag", "premix", "solution", "tablet", "tab", "capsule", "oral", "injection", "drip", "dose",
    "extended", "release", "delayed", "xl", "er", "sr", "dr", "cr", "la", "in", "and", "with", "dextrose",
    "water", "ns", "d5w", "liquid", "susp", "suspension", "syrup", "vial", "pf", "desensitization",
}


@dataclass
class TreatmentConfig:
    active_window_days: float = 30.0
    unrated_severity: float = 1.0


def load_candidates(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or RESOURCES / "antihypertensive_candidates.tsv", sep="\t", dtype=str)


def _norm_name(s) -> str:
    s = str(s).lower()
    s = re.sub(r"\d+(\.\d+)?\s*(%|mg|mcg|g|ml|meq|units?|unit)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _read_rrf(path: Path, cols: dict[int, str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", header=None, usecols=sorted(cols), dtype=str, quoting=3, keep_default_na=False)
    return df.rename(columns=cols)


def _norm_ndc(s) -> str | None:
    d = re.sub(r"\D", "", str(s or ""))
    return d.zfill(11) if d.strip("0") else None


class DrugMapper:
    """Medication name / NDC -> DrugBank ingredient ids via RxNorm RRF files."""

    def __init__(self, rrf_dir: Path, use_ndc: bool = True):
        rrf_dir = Path(rrf_dir)
        conso = _read_rrf(rrf_dir / "RXNCONSO.RRF", {0: "rxcui", 1: "lat", 11: "sab", 12: "tty", 13: "code", 14: "str"})
        conso = conso[(conso["lat"] == "ENG") & conso["sab"].isin(["RXNORM", "DRUGBANK"])]
        rx = conso[conso["sab"] == "RXNORM"]
        prio = {t: i for i, t in enumerate(["IN", "PIN", "MIN", "BN", "SCDC", "SBDC", "SCD", "SBD", "GPCK", "BPCK"])}
        best = (rx[rx["tty"].isin(prio)].assign(p=lambda d: d["tty"].map(prio))
                .sort_values("p").drop_duplicates("rxcui"))
        self.tty = dict(zip(best["rxcui"], best["tty"]))
        db = conso[conso["sab"] == "DRUGBANK"].sort_values("code")
        self.drugbank = db.drop_duplicates("rxcui").set_index("rxcui")["code"].to_dict()
        names = pd.concat([rx[rx["tty"].isin(NAME_TTYS)], db])
        self.by_name: dict[str, set[str]] = defaultdict(set)
        for c, s in zip(names["rxcui"], names["str"]):
            self.by_name[_norm_name(s)].add(c)
        rel = _read_rrf(rrf_dir / "RXNREL.RRF", {0: "rxcui1", 4: "rxcui2", 7: "rela", 10: "sab"})
        rel = rel[(rel["sab"] == "RXNORM") & rel["rela"].isin(TOWARD_INGREDIENT)]
        self.up: dict[str, list[str]] = defaultdict(list)
        for a, b in zip(rel["rxcui2"], rel["rxcui1"]):
            self.up[a].append(b)
        self.by_ndc: dict[str, str] = {}
        if use_ndc:
            sat = _read_rrf(rrf_dir / "RXNSAT.RRF", {0: "rxcui", 8: "atn", 9: "sab", 10: "atv"})
            sat = sat[(sat["atn"] == "NDC") & (sat["sab"] == "RXNORM")]
            self.by_ndc = dict(zip(sat["atv"].map(_norm_ndc), sat["rxcui"]))
        self._ing_cache: dict[str, frozenset[str]] = {}

    def ingredients(self, rxcui: str, depth: int = 5) -> frozenset[str]:
        """DrugBank ids of the RxNorm ingredients (TTY IN) reachable from rxcui."""
        if rxcui in self._ing_cache:
            return self._ing_cache[rxcui]
        seen, frontier, ins = {rxcui}, [rxcui], set()
        for _ in range(depth + 1):
            nxt = []
            for c in frontier:
                if self.tty.get(c) == "IN" or (c in self.drugbank and not self.up.get(c)):
                    ins.add(c)
                    continue
                nxt += [p for p in self.up.get(c, ()) if p not in seen]
            seen.update(nxt)
            frontier = nxt
        out = frozenset(self.drugbank[c] for c in ins if c in self.drugbank)
        self._ing_cache[rxcui] = out
        return out

    def _by_text(self, text: str) -> frozenset[str]:
        found = set()
        for c in self.by_name.get(text, ()):
            found |= self.ingredients(c)
        return frozenset(found)

    def map_name(self, name) -> tuple[frozenset[str], str]:
        text = _norm_name(name)
        if not text:
            return frozenset(), "unmapped"
        hit = self._by_text(text)
        if hit:
            return hit, "name_exact"
        tokens = text.split()
        core = " ".join(t for t in tokens if t not in STOP_TOKENS)
        if core and core != text:
            hit = self._by_text(core)
            if hit:
                return hit, "name_core"
        # longest runs of tokens first; a matched run consumes its tokens
        found, used = set(), [False] * len(tokens)
        for n in range(min(3, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                if any(used[i:i + n]):
                    continue
                run = tokens[i:i + n]
                if n == 1 and (run[0] in STOP_TOKENS or len(run[0]) < 4):
                    continue
                hit = self._by_text(" ".join(run))
                if hit:
                    found |= hit
                    used[i:i + n] = [True] * n
        return (frozenset(found), "name_tokens") if found else (frozenset(), "unmapped")

    def map(self, name, ndc=None) -> tuple[frozenset[str], str]:
        code = _norm_ndc(ndc)
        if code and code in self.by_ndc:
            hit = self.ingredients(self.by_ndc[code])
            if hit:
                return hit, "ndc"
        return self.map_name(name)


def map_medications(meds: pd.DataFrame, mapper: DrugMapper) -> pd.DataFrame:
    """meds: medication, ndc (+ any columns). Adds drugbank_ids (tuple) and drug_route, exploding to one row per
    (medication row, DrugBank id); unmapped rows keep drugbank_id = None."""
    pairs = meds[["medication", "ndc"]].drop_duplicates()
    res = [mapper.map(m, n) for m, n in zip(pairs["medication"], pairs["ndc"])]
    pairs = pairs.assign(drugbank_ids=[tuple(sorted(r[0])) for r in res], drug_route=[r[1] for r in res])
    out = meds.merge(pairs, on=["medication", "ndc"], how="left")
    out["drugbank_id"] = out["drugbank_ids"].map(lambda t: list(t) if t else [None])
    return out.explode("drugbank_id")


def index_candidates(index_drugs: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
    """[P, C] bool: candidate ingredient named in the patient's index prescription string."""
    pats = [re.compile(rf"\b{re.escape(i)}\b") for i in candidates["ingredient"]]
    s = index_drugs.fillna("").str.lower()
    return np.stack([s.map(lambda x, p=p: bool(p.search(x))).to_numpy() for p in pats], axis=1)


def _pre_t0_sets(df: pd.DataFrame, t0: pd.Series, key: str) -> dict:
    """patient -> {key value: last time} for rows strictly before the patient's t0."""
    if df is None or df.empty:
        return {}
    d = df.dropna(subset=[key]).copy()
    d = d[pd.to_datetime(d["time"]) < pd.to_datetime(d["patient_id"].map(t0))]
    d[key] = d[key].astype(int)
    return {p: g.groupby(key)["time"].max().to_dict() for p, g in d.groupby("patient_id")}


def build_context(patients: pd.DataFrame, dx_pk: pd.DataFrame, meds_db: pd.DataFrame, relations: pd.DataFrame,
                  candidates: pd.DataFrame, cfg: TreatmentConfig = TreatmentConfig(),
                  severity: dict[frozenset, str] | None = None, names: dict | None = None,
                  symptoms: pd.DataFrame | None = None, measurement_states: pd.DataFrame | None = None
                  ) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """patients: patient_id, index_time. dx_pk: patient_id, time, primekg_index. meds_db: patient_id, time,
    drugbank_id. relations: extract_primekg relations. Optional context: symptoms (patient_id, time,
    phenotype_index) and measurement_states (patient_id, time, measure, state, phenotype_index,
    disease_index; current abnormal results). Returns features [P, C, F], the matched evidence
    (patient-level, with evidence_source) and aggregate counts."""
    names = names or {}
    t0 = patients.set_index("patient_id")["index_time"]
    v_p = _pre_t0_sets(dx_pk, t0, "primekg_index")
    md = meds_db.dropna(subset=["drugbank_id"]).copy()
    md["t0"] = md["patient_id"].map(t0)
    tm, t0m = pd.to_datetime(md["time"]), pd.to_datetime(md["t0"])
    md = md[(tm < t0m) & (tm >= t0m - pd.Timedelta(days=cfg.active_window_days))]
    m_act = {p: g.groupby("drugbank_id")["time"].max().to_dict() for p, g in md.groupby("patient_id")}
    ms = measurement_states if measurement_states is not None else pd.DataFrame(
        columns=["patient_id", "time", "phenotype_index", "disease_index"])
    phen_rows = [ms[["patient_id", "time", "phenotype_index"]]]
    if symptoms is not None:
        phen_rows.append(symptoms[["patient_id", "time", "phenotype_index"]])
    p_phen = _pre_t0_sets(pd.concat(phen_rows, ignore_index=True), t0, "phenotype_index")
    d_meas = _pre_t0_sets(ms[["patient_id", "time", "disease_index"]], t0, "disease_index")
    n_abn = _pre_t0_sets(ms.assign(row=np.arange(len(ms))), t0, "row") if len(ms) else {}

    rel_sets = {}
    for rel in ("contraindication", "indication", "off-label use", "drug_effect"):
        r = relations[relations["relation"] == rel]
        rel_sets[rel] = {d: set(g["target_index"].astype(int)) for d, g in r.groupby("drug_id")}
    dd = relations[relations["relation"] == "drug_drug"]
    ddi_sets = {d: set(g["target_id"]) for d, g in dd.groupby("drug_id")}

    pids = patients["patient_id"].tolist()
    cand_ids = candidates["drugbank_id"].tolist()
    feats = np.zeros((len(pids), len(cand_ids), len(FEATURES)), dtype=np.float32)
    col = {f: i for i, f in enumerate(FEATURES)}
    matches = []
    for pi, pid in enumerate(pids):
        dis, act = v_p.get(pid, {}), m_act.get(pid, {})
        phen, mdis = p_phen.get(pid, {}), d_meas.get(pid, {})
        for ci, cid in enumerate(cand_ids):
            f = feats[pi, ci]
            f[col["n_active_meds"]] = len(act)
            f[col["n_patient_diseases"]] = len(dis)
            f[col["n_patient_phenotypes"]] = len(phen)
            f[col["n_abnormal_measurements"]] = len(n_abn.get(pid, {}))
            for rel, flag, cnt in (("contraindication", "contraindication_flag", "matched_contraindication_count"),
                                   ("indication", "indication_alignment_flag", "matched_indication_count"),
                                   ("off-label use", "off_label_flag", "matched_off_label_count")):
                hit = rel_sets[rel].get(cid, set()) & dis.keys()
                f[col[cnt]] = len(hit)
                f[col[flag]] = float(bool(hit))
                matches += [(pid, cid, rel, str(h), names.get(h, ""), dis[h], "diagnosis") for h in hit]
            hit = rel_sets["drug_effect"].get(cid, set()) & phen.keys()
            f[col["matched_side_effect_count"]] = len(hit)
            f[col["side_effect_overlap_flag"]] = float(bool(hit))
            matches += [(pid, cid, "drug_effect", str(h), names.get(h, ""), phen[h], "symptom") for h in hit]
            hit = rel_sets["contraindication"].get(cid, set()) & mdis.keys()
            f[col["matched_measurement_contraindication_count"]] = len(hit)
            f[col["measurement_contraindication_flag"]] = float(bool(hit))
            matches += [(pid, cid, "contraindication", str(h), names.get(h, ""), mdis[h], "measurement") for h in hit]
            hit = rel_sets["indication"].get(cid, set()) & mdis.keys()
            f[col["measurement_indication_count"]] = len(hit)
            matches += [(pid, cid, "indication", str(h), names.get(h, ""), mdis[h], "measurement") for h in hit]
            partners = (ddi_sets.get(cid, set()) & act.keys()) - {cid}
            if partners:
                levels = [severity.get(frozenset((cid, d))) if severity else None for d in partners]
                w = [SEVERITY.get(lv, cfg.unrated_severity) if lv else cfg.unrated_severity for lv in levels]
                f[col["ddi_count"]] = len(partners)
                f[col["ddi_fraction_of_active"]] = len(partners) / max(len(act), 1)
                f[col["ddi_severity_max"]] = max(w)
                f[col["ddi_severity_mean"]] = float(np.mean(w))
                for lv in ("major", "moderate", "minor"):
                    f[col[f"ddi_n_{lv}"]] = sum(x == lv for x in levels)
                f[col["ddi_n_unrated"]] = sum(x is None for x in levels)
                matches += [(pid, cid, "drug_drug", d, names.get(d, ""), act[d], "medication") for d in partners]
    match_df = pd.DataFrame(matches, columns=["patient_id", "candidate_drugbank_id", "relation", "target",
                                              "target_name", "evidence_time", "evidence_source"])
    match_df["index_time"] = match_df["patient_id"].map(t0)
    counts = {"patients_with_mapped_diseases": len(v_p), "patients_with_active_meds": len(m_act),
              "patients_with_symptoms": len(p_phen), "patients_with_measurement_diseases": len(d_meas)}
    return feats, match_df, counts
