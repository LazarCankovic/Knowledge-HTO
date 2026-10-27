"""ICD-9-CM / ICD-10-CM -> SNOMED CT concept map.

ICD-10-CM: the SNOMED CT -> ICD-10-CM extended map (US edition refset 6011000124106), read in reverse:
           a code maps to every active concept whose active map row targets it.
ICD-9-CM:  CMS 2018 GEMs (ICD-9-CM -> ICD-10-CM), then the ICD-10-CM route.

A code with no exact target falls back to its more specific codes ("children") and then to its
parent codes (one character shorter at a time, down to the 3-character category).

Codes are handled without dots and upper-cased, the way MIMIC stores them ("4019", "I10", "E1122").
"""
from __future__ import annotations

import bisect
import csv
import re
from pathlib import Path

import pandas as pd

ICD10CM_REFSET = "6011000124106"
UNCONDITIONAL_RULES = {"TRUE", "OTHERWISE TRUE"}
INFERRED = "900000000000011006"


def norm_code(code) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(code).upper())


def rf2_file(snomed_dir: Path, subdir: str, prefix: str) -> Path:
    """The single Snapshot file under <release>/Snapshot/<subdir>/ whose name starts with prefix."""
    hits = sorted((Path(snomed_dir) / "Snapshot" / subdir).glob(f"{prefix}*.txt"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {prefix}*.txt in {snomed_dir}/Snapshot/{subdir}, found {hits}")
    return hits[0]


def read_rf2(path: Path, usecols: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, usecols=usecols,
                       quoting=csv.QUOTE_NONE, keep_default_na=False)


def load_active_concepts(snomed_dir: Path) -> set[str]:
    df = read_rf2(rf2_file(snomed_dir, "Terminology", "sct2_Concept_Snapshot"), ["id", "active"])
    return set(df.loc[df["active"] == "1", "id"])


def load_icd10cm_targets(snomed_dir: Path, active_concepts: set[str] | None = None) -> pd.DataFrame:
    """One row per (icd10, sctid): every active concept whose active ICD-10-CM map row targets the code.
    rule_kind is 'unconditional' when any such row is TRUE / OTHERWISE TRUE, else 'conditional'."""
    df = read_rf2(rf2_file(snomed_dir, "Refset/Map", "der2_iisssccRefset_ExtendedMapSnapshot"),
                  ["active", "refsetId", "referencedComponentId", "mapGroup", "mapRule", "mapTarget"])
    df = df[(df["active"] == "1") & (df["refsetId"] == ICD10CM_REFSET) & (df["mapTarget"] != "")]
    if active_concepts is not None:
        df = df[df["referencedComponentId"].isin(active_concepts)]
    out = pd.DataFrame({
        "icd10": df["mapTarget"].map(norm_code),
        "sctid": df["referencedComponentId"],
        "rule_kind": df["mapRule"].isin(UNCONDITIONAL_RULES).map({True: "unconditional", False: "conditional"}),
        "map_group": df["mapGroup"].astype(int),
    })
    # keep the strongest row per pair: unconditional before conditional, then the lowest map group
    out = out.sort_values(["icd10", "sctid", "rule_kind", "map_group"], ascending=[True, True, False, True])
    return out.drop_duplicates(["icd10", "sctid"]).reset_index(drop=True)


def load_gems(path: Path) -> pd.DataFrame:
    """CMS GEM file rows: source, target, approximate, no_map (flags are the 5-digit field)."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            src, tgt, flags = parts
            rows.append((norm_code(src), norm_code(tgt), flags[0] == "1", flags[1] == "1"))
    return pd.DataFrame(rows, columns=["source", "target", "approximate", "no_map"])


class PrefixIndex:
    """code -> values, with exact, more-specific (prefix) and parent lookups."""

    def __init__(self, mapping: dict[str, list]):
        self.mapping = mapping
        self.keys = sorted(mapping)

    def children(self, code: str) -> list[str]:
        i = bisect.bisect_left(self.keys, code)
        out = []
        while i < len(self.keys) and self.keys[i].startswith(code):
            if self.keys[i] != code:
                out.append(self.keys[i])
            i += 1
        return out

    def resolve(self, code: str, min_len: int = 3) -> tuple[list[str], str]:
        """Codes in the index that stand for `code`, and how they were found."""
        if code in self.mapping:
            return [code], "direct"
        kids = self.children(code)
        if kids:
            return kids, "via_children"
        for n in range(len(code) - 1, min_len - 1, -1):
            if code[:n] in self.mapping:
                return [code[:n]], "via_parent"
        return [], "unmapped"


def build_icd_map(codes: pd.DataFrame, snomed_dir: Path, gems_path: Path) -> pd.DataFrame:
    """codes: DataFrame with icd_version (9/10) and icd_code.

    Returns one row per (icd_version, icd_code, sctid) with
      route      direct | via_children | via_parent for ICD-10; gem>... for ICD-9
      rule_kind  unconditional | conditional (from the SNOMED map rule)
      approximate  True when the GEM step is flagged approximate (ICD-9 only)
    Unmapped codes get one row with sctid = None and route 'unmapped'.
    """
    active = load_active_concepts(snomed_dir)
    targets = load_icd10cm_targets(snomed_dir, active)
    by_icd10 = {k: g[["sctid", "rule_kind"]].to_records(index=False).tolist()
                for k, g in targets.groupby("icd10")}
    icd10_index = PrefixIndex(by_icd10)

    gems = load_gems(gems_path)
    gems = gems[~gems["no_map"]]
    by_icd9 = {k: g[["target", "approximate"]].to_records(index=False).tolist()
               for k, g in gems.groupby("source")}
    icd9_index = PrefixIndex(by_icd9)

    def icd10_rows(code: str) -> tuple[list[tuple[str, str]], str]:
        keys, route = icd10_index.resolve(code)
        return [pair for k in keys for pair in by_icd10[k]], route

    rows = []
    for version, code in zip(codes["icd_version"].astype(int), codes["icd_code"].map(norm_code)):
        if version == 10:
            pairs, route = icd10_rows(code)
            found = [(sctid, route, kind, False) for sctid, kind in pairs]
        else:
            keys, gem_route = icd9_index.resolve(code)
            best: dict[str, tuple[str, str, bool]] = {}
            for k in keys:
                for target, approximate in by_icd9[k]:
                    pairs, sub = icd10_rows(target)
                    for sctid, kind in pairs:
                        prev = best.get(sctid)
                        # keep the strongest evidence per concept
                        if prev is None or (prev[1] == "conditional" and kind == "unconditional"):
                            best[sctid] = (f"gem_{gem_route}>{sub}", kind, bool(approximate))
            found = [(sctid, r, kind, approx) for sctid, (r, kind, approx) in best.items()]
        if not found:
            found = [(None, "unmapped", None, False)]
        rows += [(version, code, *f) for f in found]

    out = pd.DataFrame(rows, columns=["icd_version", "icd_code", "sctid", "route", "rule_kind", "approximate"])
    return out.drop_duplicates(["icd_version", "icd_code", "sctid"]).reset_index(drop=True)


def coverage_report(icd_map: pd.DataFrame, inventory: pd.DataFrame | None = None) -> dict:
    """Share of codes with at least one concept; with an inventory (icd_version, icd_code, n_rows_usable),
    also the share of diagnosis rows they account for."""
    per_code = (icd_map.assign(mapped=icd_map["sctid"].notna())
                .groupby(["icd_version", "icd_code"])
                .agg(mapped=("mapped", "any"), n_concepts=("sctid", "count"), route=("route", "first"))
                .reset_index())
    report = {}
    for v, g in per_code.groupby("icd_version"):
        report[f"icd{v}"] = {
            "codes": int(len(g)),
            "mapped_codes": int(g["mapped"].sum()),
            "mapped_share": round(float(g["mapped"].mean()), 4),
            "median_concepts_per_mapped_code": float(g.loc[g["mapped"], "n_concepts"].median()) if g["mapped"].any() else 0.0,
            "routes": {k: int(n) for k, n in g["route"].value_counts().items()},
        }
    if inventory is not None:
        inv = inventory.assign(icd_code=inventory["icd_code"].map(norm_code))
        merged = inv.merge(per_code, on=["icd_version", "icd_code"], how="left").fillna({"mapped": False})
        for v, g in merged.groupby("icd_version"):
            w = g["n_rows_usable"].astype(float)
            report[f"icd{v}"]["mapped_share_usage_weighted"] = (
                round(float((w * g["mapped"]).sum() / w.sum()), 4) if w.sum() > 0 else None)
        w = merged["n_rows_usable"].astype(float)
        report["all_usage_weighted"] = round(float((w * merged["mapped"]).sum() / w.sum()), 4) if w.sum() > 0 else None
    return report
