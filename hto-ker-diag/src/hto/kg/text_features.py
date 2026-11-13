"""ClinicalBERT text features for knowledge-graph nodes.

Every node of the diagnosis graph gets a text and a 768-d embedding, keyed like entities.tsv:
  icd      "9_4019"      "Disease: <ICD long title> (<MONDO name>). Symptoms: <Mayo Clinic symptoms>."
                         The MONDO / PrimeKG part is added when the code maps; a code without symptoms gets
                         the MONDO (or UMLS) definition instead. Unmapped codes keep their ICD title, so no
                         observed code has a zero vector.
  snomed   "38341003"    "Concept: <SNOMED fully specified name>"
  icd_cat  "9_cat_401"   no text: the mean of its codes' embeddings
  lab nodes and the context KG's measurement, state and HPO phenotype nodes: see entity_texts
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from hto.kg.snomed_graph import icd_category

DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


def _clean(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)) or text is pd.NA:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().strip('"').strip()


def disease_text(title: str, mondo_name: str = "", symptoms: str = "", definition: str = "") -> str:
    title, mondo_name, symptoms, definition = map(_clean, (title, mondo_name, symptoms, definition))
    name = title or mondo_name
    if mondo_name and title and mondo_name.lower() != title.lower():
        name = f"{title} ({mondo_name})"
    text = f"Disease: {name}."
    if symptoms:
        text += f" Symptoms: {symptoms}"
    elif definition:
        text += f" Definition: {definition}"
    return text


def icd_texts(codes: pd.DataFrame, icd_map: pd.DataFrame, features: pd.DataFrame | None,
              mondo_definitions: dict[str, str]) -> pd.DataFrame:
    """codes: icd_version, icd_code, long_title. icd_map: attach_primekg output (rank 0 = best MONDO term).
    features: load_disease_features output. Returns diag, text, source."""
    best = icd_map[icd_map["rank"] == 0].drop_duplicates("diag").set_index("diag")
    feats = features.set_index("node_index") if features is not None else pd.DataFrame()
    titles = codes["long_title"] if "long_title" in codes else [""] * len(codes)
    rows = []
    for v, c, title in zip(codes["icd_version"].astype(int), codes["icd_code"], titles):
        diag = f"{v}_{c}"
        mondo_name = symptoms = definition = ""
        source = "icd_title"
        if diag in best.index and pd.notna(best.at[diag, "mondo_id"]):
            mid = best.at[diag, "mondo_id"]
            mondo_name = best.at[diag, "mondo_name"] or ""
            definition = mondo_definitions.get(mid, "")
            source = "mondo"
            pk = best.at[diag, "primekg_index"]
            if pd.notna(pk) and int(pk) in feats.index:
                f = feats.loc[int(pk)]
                symptoms = _clean(f.get("mayo_symptoms"))
                definition = definition or _clean(f.get("mondo_definition")) or _clean(f.get("umls_description"))
                source = "primekg_symptoms" if symptoms else "primekg"
        rows.append((diag, disease_text(title, mondo_name, symptoms, definition), source))
    return pd.DataFrame(rows, columns=["diag", "text", "source"])


KIND_PREFIX = {"snomed": "Concept", "lab": "Lab test", "lab_label": "Lab test name", "fluid": "Specimen",
               "lab_category": "Lab category", "loinc": "Lab code LOINC"}


def entity_texts(entities: pd.DataFrame, context_texts: Path | None = None) -> pd.DataFrame:
    """diag, text, source for the non-ICD nodes of a graph: '<kind prefix>: <name>' from entities.tsv,
    overridden by context_texts (key, text) when the file exists. ICD codes and categories are left out."""
    ent = entities[~entities["kind"].isin(["icd", "icd_cat"]) & (entities["name"].astype(str) != "")]
    ent = ent[ent["kind"].isin(KIND_PREFIX)]
    out = pd.DataFrame({"diag": ent["key"].astype(str),
                        "text": ent["kind"].map(KIND_PREFIX) + ": " + ent["name"].astype(str),
                        "source": ent["kind"].map(lambda k: "snomed_name" if k == "snomed" else f"{k}_name")})
    if context_texts is not None and Path(context_texts).exists():
        ctx = pd.read_parquet(context_texts).rename(columns={"key": "diag"})
        out = pd.concat([out[~out["diag"].isin(set(ctx["diag"]))], ctx[["diag", "text", "source"]]],
                        ignore_index=True)
    return out.reset_index(drop=True)


def make_encoder(model_name: str = DEFAULT_MODEL, device: str = "cpu", max_length: int = 256,
                 pooling: str = "mean") -> Callable[[list[str]], np.ndarray]:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def encode(texts: list[str]) -> np.ndarray:
        enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state
        if pooling == "cls":
            out = hidden[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            out = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return out.float().cpu().numpy()

    return encode


def encode_texts(texts: list[str], encode: Callable[[list[str]], np.ndarray], batch_size: int = 128,
                 log_every: int = 50) -> np.ndarray:
    """Encode unique texts once, longest first (less padding per batch)."""
    uniq = list(dict.fromkeys(texts))
    order = sorted(range(len(uniq)), key=lambda i: -len(uniq[i]))
    vecs: np.ndarray | None = None
    for b, start in enumerate(range(0, len(order), batch_size)):
        idx = order[start:start + batch_size]
        out = encode([uniq[i] for i in idx])
        if vecs is None:
            vecs = np.zeros((len(uniq), out.shape[1]), dtype=np.float32)
        vecs[idx] = out
        if log_every and b % log_every == 0:
            print(f"  encoded {min(start + batch_size, len(order)):,}/{len(order):,}", flush=True)
    pos = {t: i for i, t in enumerate(uniq)}
    return vecs[[pos[t] for t in texts]] if vecs is not None else np.zeros((0, 0), dtype=np.float32)


def category_means(keys: list[str], matrix: np.ndarray) -> dict[str, np.ndarray]:
    """ICD category key -> mean embedding of its codes."""
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        if "_cat_" not in k:
            groups.setdefault(icd_category(k), []).append(i)
    return {cat: matrix[idx].mean(0) for cat, idx in groups.items()}


def save_text_features(path: Path, keys: list[str], matrix: np.ndarray, sources: list[str], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"keys": keys, "matrix": torch.as_tensor(matrix, dtype=torch.float32),
                "sources": sources, **meta}, path)


def load_text_features(path: Path) -> dict[str, torch.Tensor]:
    """{key: 768-d tensor} from node_features_bert.pt."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {k: data["matrix"][i] for i, k in enumerate(data["keys"])}


def entity_text_table(path: Path, entities: pd.DataFrame) -> tuple[torch.Tensor, int]:
    """Text features aligned to graph entity ids: row i = entity with id i, row 0 = padding (zeros).
    Returns the table and the number of entities without a feature (left at zero)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    pos = {k: i for i, k in enumerate(data["keys"])}
    table = torch.zeros(int(entities["id"].max()) + 1, data["matrix"].shape[1])
    missing = 0
    for eid, key in zip(entities["id"], entities["key"]):
        i = pos.get(str(key))
        if i is None:
            missing += 1
        else:
            table[int(eid)] = data["matrix"][i]
    return table, missing
