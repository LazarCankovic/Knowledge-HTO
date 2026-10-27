"""Join MIMIC lab items to the diagnosis SNOMED graph via the LOINC Ontology RF2 release.

Identifiers are authoritative RF2 alternate identifiers, never label matches.
The resulting graph retains the diagnosis graph and adds lab metadata, mapped
SNOMED concepts, their ancestors, and the LOINC extension's defining edges.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from hto.kg.icd_snomed_map import INFERRED, load_active_concepts, read_rf2, rf2_file
from hto.kg.lab_graph import build_lab_graph, lab_code_key
from hto.kg.snomed_graph import IS_A, RELATIONS, load_fsn, load_relationships

LOINC_SCHEME = "30051010000102"


def loinc_snomed_identifiers(extension_dir: Path) -> pd.DataFrame:
    """Active LOINC alternate identifiers and active SCTIDs from the same extension."""
    path = rf2_file(extension_dir, "Terminology", "sct2_Identifier_Snapshot")
    df = read_rf2(path, ["alternateIdentifier", "active", "identifierSchemeId",
                         "referencedComponentId"])
    df = df[(df["active"] == "1") & (df["identifierSchemeId"] == LOINC_SCHEME)]
    active = load_active_concepts(extension_dir)
    df = df[df["referencedComponentId"].isin(active)]
    out = df[["alternateIdentifier", "referencedComponentId"]].rename(
        columns={"alternateIdentifier": "loinc_code", "referencedComponentId": "sctid"})
    out = out.drop_duplicates()
    if out["loinc_code"].duplicated().any():
        raise ValueError("Active LOINC identifiers resolve to multiple SNOMED concepts")
    return out.reset_index(drop=True)


def _extension_relationships(extension_dir: Path) -> pd.DataFrame:
    df = read_rf2(rf2_file(extension_dir, "Terminology", "sct2_Relationship_Snapshot"),
                  ["active", "sourceId", "destinationId", "typeId", "characteristicTypeId"])
    df = df[(df["active"] == "1") & (df["characteristicTypeId"] == INFERRED)]
    return df[["sourceId", "destinationId", "typeId"]].drop_duplicates()


def build_lab_snomed_graph(diag_graph: dict, items: pd.DataFrame, item_loinc: pd.DataFrame,
                           snomed_dir: Path, extension_dir: Path,
                           ancestor_depth: int = 4) -> dict:
    """Produce one ID space containing the diagnosis graph and SNOMED-linked labs.

    ``item_loinc`` is a reviewed itemid,loinc_code CSV; do not derive it from
    lab labels. No patient rows or outcome labels enter this global graph.
    """
    if ancestor_depth < 1:
        raise ValueError("ancestor_depth must be at least 1")
    if {"itemid", "loinc_code"} - set(item_loinc):
        raise ValueError("Item-to-LOINC mapping needs itemid and loinc_code")
    mapping = item_loinc[["itemid", "loinc_code"]].dropna().copy()
    mapping["itemid"] = mapping["itemid"].astype(int)
    mapping["loinc_code"] = mapping["loinc_code"].astype(str).str.strip()
    mapping = mapping[mapping["loinc_code"] != ""].drop_duplicates()
    if mapping["itemid"].duplicated().any():
        raise ValueError("Review item-to-LOINC mapping: one itemid has multiple LOINC codes")
    lab = build_lab_graph(items, mapping)
    identifiers = loinc_snomed_identifiers(extension_dir)
    mapped = mapping.merge(identifiers, on="loinc_code", how="inner")
    known_labs = set(lab["entities"].loc[lab["entities"]["kind"] == "lab", "key"])
    mapped = mapped[mapped["itemid"].map(lab_code_key).isin(known_labs)]
    if mapped.empty:
        raise ValueError("No lab items map to active SNOMED concepts through the LOINC extension")

    base_entities = diag_graph["entities"].copy()
    base_relations = diag_graph["relations"].copy()
    base_entity = dict(zip(base_entities["id"], base_entities["key"].astype(str)))
    base_relation = dict(zip(base_relations["id"], base_relations["name"]))
    if not {"icd", "snomed"} <= set(base_entities["kind"]):
        raise ValueError("Expected the diagnosis ICD/SNOMED graph from build_kg.py")
    if set(lab["entities"]["key"]) & set(base_entities["key"].astype(str)):
        raise ValueError("Lab node keys collide with the diagnosis graph")

    # The existing diagnosis graph is retained verbatim, including its IDs.
    nodes = dict(zip(base_entities["key"].astype(str),
                     zip(base_entities["kind"], base_entities["name"])))
    nodes.update(zip(lab["entities"]["key"],
                     zip(lab["entities"]["kind"], lab["entities"]["name"])))
    lab_entity = dict(zip(lab["entities"]["id"], lab["entities"]["key"]))
    lab_relation = dict(zip(lab["relations"]["id"], lab["relations"]["name"]))
    edges: set[tuple[str, str, str]] = set()
    for h, r, t in diag_graph["triples"][["h", "r", "t"]].itertuples(index=False):
        edges.add((base_entity[h], base_relation[r], base_entity[t]))
    for h, r, t in lab["triples"][["h", "r", "t"]].itertuples(index=False):
        edges.add((lab_entity[h], lab_relation[r], lab_entity[t]))

    core = load_relationships(snomed_dir, tuple(RELATIONS))
    extension = _extension_relationships(extension_dir)
    rels = pd.concat([core, extension], ignore_index=True).drop_duplicates()
    parents: dict[str, set[str]] = defaultdict(set)
    for child, parent in rels.loc[rels["typeId"] == IS_A,
                                  ["sourceId", "destinationId"]].itertuples(index=False):
        parents[child].add(parent)
    seeds = set(mapped["sctid"])
    snomed_nodes = set(seeds)
    frontier = seeds.copy()
    for _ in range(ancestor_depth):
        frontier = {parent for child in frontier for parent in parents[child]} - snomed_nodes
        snomed_nodes.update(frontier)
    # Defining attributes of the LOINC concepts connect observations to
    # clinical components, specimens, properties and other SNOMED concepts.
    attributes = extension[(extension["typeId"] != IS_A) &
                           (extension["sourceId"].isin(seeds))]
    snomed_nodes.update(attributes["destinationId"])
    base_snomed = set(base_entities.loc[base_entities["kind"] == "snomed", "key"].astype(str))
    snomed_nodes.update(base_snomed)
    names = load_fsn(snomed_dir, snomed_nodes)
    names.update(load_fsn(extension_dir, snomed_nodes))
    for sctid in snomed_nodes:
        nodes.setdefault(sctid, ("snomed", names.get(sctid, "")))

    selected = rels[rels["sourceId"].isin(snomed_nodes) &
                    rels["destinationId"].isin(snomed_nodes)]
    # Keep the teammate's filtered diagnosis subgraph intact. Only new
    # concepts and their edges extend it; do not restore pruned base edges.
    selected = selected[~(selected["sourceId"].isin(base_snomed) &
                          selected["destinationId"].isin(base_snomed))]
    for source, target, rel in selected.itertuples(index=False):
        relation = RELATIONS.get(rel, f"snomed_{rel}")
        edges.add((source, relation, target))
        edges.add((target, f"inv_{relation}", source))
    for itemid, loinc_code, sctid in mapped[["itemid", "loinc_code", "sctid"]].itertuples(index=False):
        key = lab_code_key(itemid)
        loinc_key = f"loinc:{loinc_code}"
        edges.add((loinc_key, "loinc_snomed_identifier", sctid))
        edges.add((sctid, "inv_loinc_snomed_identifier", loinc_key))
        edges.add((key, "maps_lab_to_snomed", sctid))
        edges.add((sctid, "inv_maps_lab_to_snomed", key))

    extra = sorted(set(nodes) - set(base_entities["key"].astype(str)))
    entities = pd.concat([base_entities, pd.DataFrame(
        [(len(base_entities) + i, key, *nodes[key]) for i, key in enumerate(extra, 1)],
        columns=["id", "key", "kind", "name"])], ignore_index=True)
    relation_names = list(base_relations["name"])
    relation_names.extend(sorted({r for _, r, _ in edges} - set(relation_names)))
    relations = pd.DataFrame({"id": range(1, len(relation_names) + 1),
                              "name": relation_names})
    eid = dict(zip(entities["key"].astype(str), entities["id"]))
    rid = dict(zip(relations["name"], relations["id"]))
    triples = pd.DataFrame([(eid[h], rid[r], eid[t]) for h, r, t in sorted(edges)],
                           columns=["h", "r", "t"]).astype(int)
    mapped_to_base = set(mapped["sctid"]) & set(
        base_entities.loc[base_entities["kind"] == "snomed", "key"].astype(str))
    return {"entities": entities, "relations": relations, "triples": triples,
            "stats": {"source": "diagnosis graph + MIMIC labs + LOINC/SNOMED extension",
                      "diagnosis_snomed_nodes": int((base_entities["kind"] == "snomed").sum()),
                      "lab_codes": len(known_labs),
                      "lab_codes_with_loinc": lab["stats"]["lab_codes_with_loinc"],
                      "lab_codes_with_active_snomed": int(mapped["itemid"].nunique()),
                      "mapped_snomed_concepts_already_in_diagnosis_graph": len(mapped_to_base),
                      "entities": len(entities), "relations": len(relations),
                      "triples": len(triples), "ancestor_depth": ancestor_depth}}
