"""Human Phenotype Ontology (hp.json, OBO Graphs): names, exact synonyms, definitions, IS-A parents, xrefs.

PrimeKG phenotype nodes are HPO terms with x_id the HPO number without padding ("2153" = HP:0002153).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HP_URL = "http://purl.obolibrary.org/obo/HP_"


@dataclass
class Hpo:
    names: dict[str, str]                 # "HP:0002153" -> "Hyperkalemia"
    definitions: dict[str, str]
    parents: dict[str, set[str]]
    xrefs: pd.DataFrame                   # hpo_id, prefix, code ("SNOMEDCT_US", "123456")
    lookup: dict[str, str]                # normalised label or exact synonym -> id (labels win)


def norm_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", str(s).casefold())
    return re.sub(r"\s+", " ", s).strip()


def hpo_id(num) -> str:
    return f"HP:{int(num):07d}"


def load_hpo(path: Path) -> Hpo:
    with open(path, encoding="utf-8") as f:
        graph = json.load(f)["graphs"][0]
    names, definitions, rows, synonyms = {}, {}, [], []
    for node in graph["nodes"]:
        url = node.get("id", "")
        if not url.startswith(HP_URL):
            continue
        meta = node.get("meta", {}) or {}
        if meta.get("deprecated"):
            continue
        hid = f"HP:{url[len(HP_URL):]}"
        names[hid] = node.get("lbl", "") or ""
        if (meta.get("definition") or {}).get("val"):
            definitions[hid] = meta["definition"]["val"]
        for x in meta.get("xrefs", []) or []:
            prefix, _, code = (x.get("val", "") if isinstance(x, dict) else str(x)).partition(":")
            if code:
                rows.append((hid, prefix, code))
        for syn in meta.get("synonyms", []) or []:
            if syn.get("pred") == "hasExactSynonym" and syn.get("val"):
                synonyms.append((norm_name(syn["val"]), hid))
    parents: dict[str, set[str]] = defaultdict(set)
    for e in graph.get("edges", []):
        if e.get("pred") == "is_a" and e["sub"].startswith(HP_URL) and e["obj"].startswith(HP_URL):
            child, parent = f"HP:{e['sub'][len(HP_URL):]}", f"HP:{e['obj'][len(HP_URL):]}"
            if child in names and parent in names:
                parents[child].add(parent)
    lookup = {}
    for n, hid in synonyms:
        lookup.setdefault(n, hid)
    lookup.update({norm_name(v): k for k, v in names.items() if v})
    xrefs = pd.DataFrame(rows, columns=["hpo_id", "prefix", "code"]).drop_duplicates().reset_index(drop=True)
    return Hpo(names, definitions, dict(parents), xrefs, lookup)


def resolve(hpo: Hpo, candidates: list[str]) -> str | None:
    """First candidate name (label or exact synonym) found in HPO."""
    for c in candidates:
        hid = hpo.lookup.get(norm_name(c))
        if hid:
            return hid
    return None
