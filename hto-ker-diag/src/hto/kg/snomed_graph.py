"""Global knowledge graph for the diagnosis tower.

Nodes
  icd      one per ICD code, keyed like the raw diagnosis strings ("9_4019", "10_I10")
  icd_cat  ICD category per code ("9_cat_401", "10_cat_I10"); connects codes in the same category
  snomed   concepts the codes map to (at most max_concepts_per_code per code), their IS-A ancestors
           up to ancestor_depth levels, and the targets of their clinical attributes
Edges
  maps_to (icd -> snomed), icd_parent (icd -> icd_cat), and the SNOMED relations in RELATIONS
  among the selected concepts. Every relation also gets an inverse ("inv_<name>"), so messages flow
  both ways.
Filters
  Generic upper-level concepts (IS-A depth <= max_top_depth below the SNOMED root) and non-seed hubs
  (more than hub_max_degree incident SNOMED edges) are dropped.

Ids are 1-based; 0 is reserved for padding in both the entity and the relation table.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hto.kg.icd_snomed_map import INFERRED, read_rf2, rf2_file

SNOMED_ROOT = "138875005"
IS_A = "116680003"
FSN = "900000000000003001"

# inferred relationship types kept (typeId -> name). Laterality, occurrence, clinical course etc. are
# left out: their targets ("Left", "Chronic") are generic values shared by thousands of concepts.
RELATIONS = {
    IS_A:        "is_a",
    "363698007": "finding_site",
    "116676008": "associated_morphology",
    "370135005": "pathological_process",
    "42752001":  "due_to",
    "246075003": "causative_agent",
    "363705008": "has_definitional_manifestation",
    "47429007":  "associated_with",
    "255234002": "after",
    "363714003": "interprets",
}


@dataclass
class GraphConfig:
    max_concepts_per_code: int = 10
    ancestor_depth: int = 2
    max_top_depth: int = 2
    hub_max_degree: int = 300
    relations: tuple[str, ...] = tuple(RELATIONS)   # typeIds; e.g. (IS_A,) for an IS-A-only graph
    rewire_seed: int | None = None   # degree-preserving random control graph (see rewire_tails)


def rewire_tails(forward: pd.DataFrame, keep_relations: set[int], seed: int) -> pd.DataFrame:
    """Control graph: within each relation, permute the tails of its triples, so which concept is linked
    to which is random. Each node keeps its number of outgoing and incoming edges per relation, except for
    the self-loops and duplicates the shuffle creates, which are dropped (~0.1% of edges on the demo)."""
    rng = np.random.default_rng(seed)
    parts = []
    for r, g in forward.groupby("r"):
        if r in keep_relations:
            parts.append(g)
        else:
            parts.append(g.assign(t=rng.permutation(g["t"].to_numpy())))
    out = pd.concat(parts, ignore_index=True)
    return out[out["h"] != out["t"]].drop_duplicates().reset_index(drop=True)


def icd_category(diag: str) -> str:
    version, code = diag.split("_", 1)
    n = 4 if (version == "9" and code.startswith("E")) else 3
    return f"{version}_cat_{code[:n]}"


def load_relationships(snomed_dir: Path, type_ids: tuple[str, ...]) -> pd.DataFrame:
    df = read_rf2(rf2_file(snomed_dir, "Terminology", "sct2_Relationship_Snapshot"),
                  ["active", "sourceId", "destinationId", "typeId", "characteristicTypeId"])
    df = df[(df["active"] == "1") & (df["characteristicTypeId"] == INFERRED) & (df["typeId"].isin(type_ids))]
    return df[["sourceId", "destinationId", "typeId"]].drop_duplicates().reset_index(drop=True)


def load_fsn(snomed_dir: Path, concepts: set[str]) -> dict[str, str]:
    df = read_rf2(rf2_file(snomed_dir, "Terminology", "sct2_Description_Snapshot"),
                  ["active", "conceptId", "typeId", "term"])
    df = df[(df["active"] == "1") & (df["typeId"] == FSN) & (df["conceptId"].isin(concepts))]
    return dict(zip(df["conceptId"], df["term"]))


def isa_depth(rels: pd.DataFrame) -> dict[str, int]:
    """Shortest IS-A distance from the SNOMED root for every concept reachable from it."""
    children = defaultdict(list)
    for child, parent in rels.loc[rels["typeId"] == IS_A, ["sourceId", "destinationId"]].itertuples(index=False):
        children[parent].append(child)
    depth = {SNOMED_ROOT: 0}
    queue = deque([SNOMED_ROOT])
    while queue:
        node = queue.popleft()
        for c in children[node]:
            if c not in depth:
                depth[c] = depth[node] + 1
                queue.append(c)
    return depth


def build_graph(icd_map: pd.DataFrame, icd_titles: dict[str, str], snomed_dir: Path,
                cfg: GraphConfig = GraphConfig()) -> dict:
    """icd_map: output of build_icd_map. icd_titles: diag key -> ICD long title.
    Returns {"entities", "relations", "triples", "stats"} DataFrames/dict (see save_graph)."""
    rels_all = load_relationships(snomed_dir, tuple(set(cfg.relations) | {IS_A}))
    depth = isa_depth(rels_all)
    rels = rels_all[rels_all["typeId"].isin(cfg.relations)]

    # 1. seed concepts: the strongest few per ICD code (unconditional first, then the more general)
    m = icd_map.dropna(subset=["sctid"]).copy()
    m["diag"] = m["icd_version"].astype(str) + "_" + m["icd_code"]
    m["depth"] = m["sctid"].map(depth).fillna(99)
    m = (m.sort_values(["diag", "rule_kind", "depth", "sctid"], ascending=[True, False, True, True])
          .groupby("diag").head(cfg.max_concepts_per_code))
    seeds = set(m["sctid"])

    # 2. neighbourhood: IS-A ancestors up to ancestor_depth + attribute targets of the seeds
    parents = defaultdict(set)
    for child, parent in rels_all.loc[rels_all["typeId"] == IS_A, ["sourceId", "destinationId"]].itertuples(index=False):
        parents[child].add(parent)
    nodes = set(seeds)
    frontier = set(seeds)
    for _ in range(cfg.ancestor_depth):
        frontier = {p for c in frontier for p in parents[c]} - nodes
        nodes |= frontier
    attr = rels[(rels["typeId"] != IS_A) & rels["sourceId"].isin(seeds)]
    nodes |= set(attr["destinationId"])

    # 3. drop generic upper-level concepts, then non-seed hubs
    nodes = {n for n in nodes if n in seeds or depth.get(n, 99) > cfg.max_top_depth}
    edges = rels[rels["sourceId"].isin(nodes) & rels["destinationId"].isin(nodes)]
    degree = pd.concat([edges["sourceId"], edges["destinationId"]]).value_counts()
    hubs = {n for n, d in degree.items() if d > cfg.hub_max_degree and n not in seeds}
    nodes -= hubs
    edges = edges[edges["sourceId"].isin(nodes) & edges["destinationId"].isin(nodes)]

    # 4. entity table
    diags = sorted(set(icd_map["icd_version"].astype(str) + "_" + icd_map["icd_code"]))
    cats = sorted({icd_category(d) for d in diags})
    snomed_nodes = sorted(nodes)
    names = load_fsn(snomed_dir, set(snomed_nodes))
    entities = pd.DataFrame(
        [(d, "icd", icd_titles.get(d, "")) for d in diags]
        + [(c, "icd_cat", "") for c in cats]
        + [(s, "snomed", names.get(s, "")) for s in snomed_nodes],
        columns=["key", "kind", "name"],
    )
    entities.insert(0, "id", range(1, len(entities) + 1))
    eid = dict(zip(entities["key"], entities["id"]))

    # 5. relation table (+ inverses) and triples
    rel_names = ["maps_to", "icd_parent"] + [RELATIONS[t] for t in cfg.relations]
    rel_names = rel_names + [f"inv_{r}" for r in rel_names]
    relations = pd.DataFrame({"id": range(1, len(rel_names) + 1), "name": rel_names})
    rid = dict(zip(relations["name"], relations["id"]))

    m_kept = m[m["sctid"].isin(nodes)]
    forward = pd.concat([
        pd.DataFrame({"h": m_kept["diag"].map(eid), "r": rid["maps_to"], "t": m_kept["sctid"].map(eid)}),
        pd.DataFrame({"h": [eid[d] for d in diags], "r": rid["icd_parent"],
                      "t": [eid[icd_category(d)] for d in diags]}),
        pd.DataFrame({"h": edges["sourceId"].map(eid), "r": edges["typeId"].map(lambda t: rid[RELATIONS[t]]),
                      "t": edges["destinationId"].map(eid)}),
    ], ignore_index=True)
    if cfg.rewire_seed is not None:
        # ICD category edges are ICD structure, not SNOMED knowledge: they stay real
        forward = rewire_tails(forward, {rid["icd_parent"]}, cfg.rewire_seed)
    n_fwd = len(rel_names) // 2
    inverse = pd.DataFrame({"h": forward["t"], "r": forward["r"] + n_fwd, "t": forward["h"]})
    triples = pd.concat([forward, inverse], ignore_index=True).drop_duplicates().astype(int).reset_index(drop=True)

    diag_with_concept = set(m_kept["diag"])
    stats = {
        "config": asdict(cfg),
        "icd_codes": len(diags),
        "icd_codes_with_concept": len(diag_with_concept),
        "seed_concepts": len(seeds & nodes),
        "snomed_nodes": len(snomed_nodes),
        "entities": len(entities),
        "relations": len(relations),
        "triples": len(triples),
        "dropped_hubs": len(hubs),
        "triples_by_relation": {r: int(n) for r, n in
                                triples["r"].map(dict(zip(relations["id"], relations["name"]))).value_counts().items()},
        "top_snomed_degree": [
            {"name": names.get(n, n), "degree": int(d)}
            for n, d in degree[degree.index.isin(nodes)].head(10).items()
        ],
    }
    return {"entities": entities, "relations": relations, "triples": triples, "stats": stats}


def save_graph(graph: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    graph["entities"].to_csv(out_dir / "entities.tsv", sep="\t", index=False)
    graph["relations"].to_csv(out_dir / "relations.tsv", sep="\t", index=False)
    graph["triples"].to_csv(out_dir / "triples.tsv", sep="\t", index=False)
    (out_dir / "graph_stats.json").write_text(json.dumps(graph["stats"], indent=2))


def load_graph(kg_dir: Path) -> dict:
    kg_dir = Path(kg_dir)
    return {
        "entities": pd.read_csv(kg_dir / "entities.tsv", sep="\t", dtype={"key": str}, keep_default_na=False),
        "relations": pd.read_csv(kg_dir / "relations.tsv", sep="\t"),
        "triples": pd.read_csv(kg_dir / "triples.tsv", sep="\t"),
    }
