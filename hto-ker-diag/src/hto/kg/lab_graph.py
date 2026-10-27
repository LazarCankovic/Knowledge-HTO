"""Lab metadata graph with one distinct node per MIMIC-IV d_labitems itemid.

The graph uses item descriptions, specimen fluid and category. An optional
itemid-to-LOINC table adds standardized concept nodes. It contains no patient
rows, outcomes, lab values or timestamps.
"""
from __future__ import annotations

import pandas as pd

from hto.kg.snomed_graph import rewire_tails


def lab_code_key(itemid: int) -> str:
    return f"lab:{int(itemid)}"


def _metadata_key(kind: str, value) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    return f"{kind}:{str(value).strip().casefold()}"


def build_lab_graph(items: pd.DataFrame, loinc_map: pd.DataFrame | None = None,
                    rewire_seed: int | None = None) -> dict:
    """Create the entities/relations/triples tables consumed by the existing TransR pipeline.

    ``items`` has itemid, label, fluid and category columns from d_labitems.
    ``loinc_map`` optionally has itemid and loinc_code columns. Unmapped items
    retain their metadata edges and remain usable by the model.
    """
    required = {"itemid", "label", "fluid", "category"}
    if required - set(items):
        raise ValueError(f"d_labitems is missing {sorted(required - set(items))}")
    items = items.dropna(subset=["itemid"]).copy()
    items["itemid"] = items["itemid"].astype(int)
    if items["itemid"].duplicated().any():
        raise ValueError("d_labitems has duplicate itemid values")

    nodes: dict[str, tuple[str, str]] = {}
    edges: set[tuple[str, str, str]] = set()
    for row in items.itertuples(index=False):
        key = lab_code_key(row.itemid)
        nodes[key] = ("lab", str(row.label) if pd.notna(row.label) else "")
        for column, kind, relation in (
            ("label", "lab_label", "has_label"),
            ("fluid", "fluid", "has_fluid"),
            ("category", "lab_category", "has_category"),
        ):
            value = getattr(row, column)
            target = _metadata_key(kind, value)
            if target is not None:
                nodes[target] = (kind, str(value).strip())
                edges.add((key, relation, target))

    mapped_items: set[int] = set()
    if loinc_map is not None:
        if {"itemid", "loinc_code"} - set(loinc_map):
            raise ValueError("LOINC map needs itemid and loinc_code columns")
        for itemid, code in loinc_map[["itemid", "loinc_code"]].dropna().itertuples(index=False):
            source = lab_code_key(itemid)
            target = f"loinc:{str(code).strip()}"
            if source in nodes and str(code).strip():
                nodes[target] = ("loinc", str(code).strip())
                edges.add((source, "maps_to_loinc", target))
                mapped_items.add(int(itemid))

    entities = pd.DataFrame(
        [(i, key, *nodes[key]) for i, key in enumerate(sorted(nodes), start=1)],
        columns=["id", "key", "kind", "name"],
    )
    eid = dict(zip(entities["key"], entities["id"]))
    forward_names = sorted({relation for _, relation, _ in edges})
    relation_names = forward_names + [f"inv_{name}" for name in forward_names]
    relations = pd.DataFrame(
        {"id": range(1, len(relation_names) + 1), "name": relation_names}
    )
    rid = dict(zip(relations["name"], relations["id"]))
    forward = pd.DataFrame(
        [(eid[src], rid[rel], eid[dst]) for src, rel, dst in sorted(edges)],
        columns=["h", "r", "t"],
    )
    if forward.empty:
        raise ValueError("Lab metadata yielded no graph edges")
    if rewire_seed is not None and not forward.empty:
        keep = {rid["has_label"]} if "has_label" in rid else set()
        forward = rewire_tails(forward, keep, rewire_seed)
    inverse = pd.DataFrame({"h": forward["t"], "r": forward["r"] + len(forward_names),
                            "t": forward["h"]})
    triples = pd.concat([forward, inverse], ignore_index=True).drop_duplicates()
    triples = triples.astype(int).reset_index(drop=True)
    stats = {
        "source": "MIMIC-IV d_labitems metadata",
        "lab_codes": len(items),
        "lab_codes_with_loinc": len(mapped_items),
        "entities": len(entities),
        "relations": len(relations),
        "triples": len(triples),
        "rewire_seed": rewire_seed,
    }
    return {"entities": entities, "relations": relations, "triples": triples, "stats": stats}
