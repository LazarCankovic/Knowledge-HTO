"""PrimeKG (Chandak et al., 2023) nodes and the drug relations used by the treatment context block.

kg.csv stores every edge in both directions with columns
    relation, display_relation, x_index, x_id, x_type, x_name, x_source, y_index, y_id, y_type, y_name, y_source
Disease nodes come from MONDO: x_source "MONDO" with x_id the MONDO number without zero padding ("5044"
for MONDO:0005044), or "MONDO_grouped" with several numbers joined by "_" ("1200_1134_15512"). Drug nodes
come from DrugBank (x_id "DB00722"); phenotype nodes from HPO (x_id "2153" for HP:0002153).

extract_primekg scans kg.csv once and caches three small parquet files:
    primekg_nodes.parquet              node_index, node_id, node_type, node_name, node_source
                                       (disease, drug and effect/phenotype nodes)
    primekg_relations.parquet          relation, drug_index, drug_id, target_index, target_id (drug on the x side;
                                       drug_effect targets are phenotype nodes = side effects)
    primekg_disease_phenotype.parquet  disease_index, phenotype_index (disease_phenotype_positive)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DRUG_RELATIONS = ("indication", "contraindication", "off-label use", "drug_drug", "drug_effect")
NODE_TYPES = ("disease", "drug", "effect/phenotype")
KG_COLUMNS = ["relation", "x_index", "x_id", "x_type", "x_name", "x_source",
              "y_index", "y_id", "y_type", "y_name", "y_source"]
CACHE = ("primekg_nodes.parquet", "primekg_relations.parquet", "primekg_disease_phenotype.parquet")


def _cache_complete(cache_dir: Path) -> bool:
    if not all((cache_dir / f).exists() for f in CACHE):
        return False
    rel = pd.read_parquet(cache_dir / CACHE[1], columns=["relation"])
    return "drug_effect" in set(rel["relation"])


def extract_primekg(kg_csv: Path, cache_dir: Path, chunksize: int = 2_000_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    nodes_path, rel_path, dp_path = (cache_dir / f for f in CACHE)
    if _cache_complete(cache_dir):
        return pd.read_parquet(nodes_path), pd.read_parquet(rel_path)
    nodes, rels, dps = [], [], []
    for chunk in pd.read_csv(kg_csv, usecols=KG_COLUMNS, dtype=str, chunksize=chunksize, low_memory=False):
        keep = chunk["x_type"].isin(NODE_TYPES)
        nodes.append(chunk.loc[keep, ["x_index", "x_id", "x_type", "x_name", "x_source"]].drop_duplicates())
        r = chunk[chunk["relation"].isin(DRUG_RELATIONS) & (chunk["x_type"] == "drug")]
        rels.append(r[["relation", "x_index", "x_id", "y_index", "y_id"]])
        d = chunk[(chunk["relation"] == "disease_phenotype_positive") & (chunk["x_type"] == "disease")]
        dps.append(d[["x_index", "y_index"]])
    node_df = (pd.concat(nodes, ignore_index=True).drop_duplicates("x_index")
               .rename(columns={"x_index": "node_index", "x_id": "node_id", "x_type": "node_type",
                                "x_name": "node_name", "x_source": "node_source"}))
    node_df["node_index"] = node_df["node_index"].astype("int64")
    rel_df = (pd.concat(rels, ignore_index=True).drop_duplicates()
              .rename(columns={"x_index": "drug_index", "x_id": "drug_id",
                               "y_index": "target_index", "y_id": "target_id"}))
    rel_df[["drug_index", "target_index"]] = rel_df[["drug_index", "target_index"]].astype("int64")
    dp_df = (pd.concat(dps, ignore_index=True).drop_duplicates()
             .rename(columns={"x_index": "disease_index", "y_index": "phenotype_index"}).astype("int64"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    node_df.sort_values("node_index").to_parquet(nodes_path, index=False)
    rel_df.to_parquet(rel_path, index=False)
    dp_df.to_parquet(dp_path, index=False)
    return node_df, rel_df


def load_disease_phenotypes(cache_dir: Path) -> pd.DataFrame:
    """disease_index, phenotype_index (run extract_primekg first)."""
    return pd.read_parquet(Path(cache_dir) / CACHE[2])


def phenotype_hpo(nodes: pd.DataFrame) -> dict[int, str]:
    """PrimeKG phenotype node_index -> 'HP:0002153'."""
    ph = nodes[(nodes["node_type"] == "effect/phenotype") & nodes["node_id"].str.isdigit()]
    return {int(i): f"HP:{int(n):07d}" for i, n in zip(ph["node_index"], ph["node_id"])}


def mondo_to_primekg(nodes: pd.DataFrame) -> dict[str, int]:
    """'MONDO:0005044' -> PrimeKG node_index. A MONDO term with its own node wins over a grouped node."""
    out: dict[str, int] = {}
    dis = nodes[nodes["node_type"] == "disease"]
    for grouped in (True, False):          # standalone nodes are written last, so they overwrite
        sub = dis[(dis["node_source"] == "MONDO_grouped") == grouped]
        for idx, node_id in zip(sub["node_index"], sub["node_id"]):
            for num in str(node_id).split("_"):
                if num.isdigit():
                    out[f"MONDO:{int(num):07d}"] = int(idx)
    return out


def disease_by_name(nodes: pd.DataFrame) -> dict[str, int]:
    """Lower-cased PrimeKG disease name -> node_index (standalone nodes win over grouped ones)."""
    dis = nodes[nodes["node_type"] == "disease"].sort_values("node_source", kind="stable")   # MONDO first
    out: dict[str, int] = {}
    for name, idx in zip(dis["node_name"], dis["node_index"]):
        out.setdefault(str(name).casefold().strip(), int(idx))
    return out


def load_disease_features(path: Path) -> pd.DataFrame:
    """PrimeKG disease_features.tab, one row per disease node_index (first non-empty value per column)."""
    cols = ["node_index", "mondo_name", "mondo_definition", "umls_description", "mayo_symptoms"]
    df = pd.read_csv(path, sep="\t", dtype=str, usecols=lambda c: c in cols, keep_default_na=False)
    df["node_index"] = df["node_index"].astype("int64")
    df = df.replace("", pd.NA)
    return df.groupby("node_index", as_index=False).first()
