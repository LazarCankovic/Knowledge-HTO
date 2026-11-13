"""Patient disease graphs (g_y) enriched with ClinicalBERT node features, as PyTorch Geometric Data.

Node features (770-d)   x_v  = [ e_text(v) (768) | log1p(recency_days) | log1p(frequency) ]
Edge features (17-d)    e_ij = [ f_tau(tau_ij) (16) | snomed_distance_ij ]

  e_text         Stage 1 ClinicalBERT vector of the node's text (ICD code, SNOMED concept or ICD category)
  recency_days   days from the code's most recent occurrence strictly before t0 to t0 (observed codes only)
  frequency      number of distinct pre-t0 visits with the code (observed codes only)
  f_tau          fixed sinusoidal encoding of the edge's day gap tau: sin/cos at 8 periods, 1 day..10 years
  snomed_distance  edge_hops / 4: the shortest observed-code-to-observed-code path the edge lies on
                   (0 for an edge that is only part of one code's own 1-hop knowledge)

Unobserved nodes (SNOMED concepts, ICD categories, codes seen only at t0) have recency = frequency = 0
and observed_mask = False.

Context graphs (kg tag ctx, build_context_patient_graphs.py) also observe lab / vital measurements
(meas:<measure>) and abnormal results (meas:<measure>:<state>), and contain HPO symptom nodes. Their
measurement values are a separate channel so x keeps its 770 dims:
  value_attr [n, 4] = [has_value, z(last value), z(last - previous), abnormal -1/0/+1], 0 for other nodes
(src/hto/kg/measurements.py; values strictly before t0).

Stored form. Materialising x for every node of every patient would take tens of GB (the same SNOMED
concept recurs in thousands of patients), so each stored Data keeps global node ids plus the raw
per-node / per-edge quantities, and the 768-d text vectors live once in a table indexed by entity id.
materialize() returns the exact x [n, 770] and edge_attr [E, 17]; EnrichedPatientGraphs does it on access.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

TIME_DIM = 16
MAX_HOPS = 4
TIME_PERIODS_DAYS = np.geomspace(1.0, 3650.0, TIME_DIM // 2)


def read_parquet_columns(path, columns: list[str]) -> pd.DataFrame:
    """Read columns by Arrow type, ignoring stored pandas metadata: the MIMIC exports declare BigQuery's
    'dbdate' extension dtype, which plain pandas cannot resolve."""
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns).replace_schema_metadata(None).to_pandas()


def time_encoding(tau: torch.Tensor) -> torch.Tensor:
    """[E] days -> [E, 16]: sin and cos at 8 log-spaced periods."""
    w = torch.as_tensor(2 * math.pi / TIME_PERIODS_DAYS, dtype=torch.float32)
    ang = tau.float().unsqueeze(-1) * w
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def code_history(diagnoses: pd.DataFrame, index_time: pd.Series) -> pd.DataFrame:
    """Per (patient_id, diag): last occurrence strictly before t0, recency in days, distinct visits.
    diagnoses: patient_id, visit_id, time, diag. index_time: patient_id -> t0."""
    df = diagnoses.dropna(subset=["diag", "time"]).copy()
    df["t0"] = df["patient_id"].map(index_time)
    df = df[pd.to_datetime(df["time"]) < pd.to_datetime(df["t0"])]
    g = df.groupby(["patient_id", "diag"])
    out = g.agg(last_time=("time", "max"), t0=("t0", "first"),
                frequency=("visit_id", "nunique")).reset_index()
    out["recency_days"] = (pd.to_datetime(out["t0"]) - pd.to_datetime(out["last_time"])) / pd.Timedelta(days=1)
    return out


def enrich_graph(g: dict, history: dict[str, tuple[float, int]],
                 values: dict[str, np.ndarray] | None = None):
    """One patient's graph (build_patient_graph output) -> compact torch_geometric Data.
    history: code -> (recency_days, frequency) for codes seen strictly before t0.
    values: code -> value_attr (4,) for measurement codes; when given, Data gets value_attr [n, 4]."""
    from torch_geometric.data import Data
    n = len(g["node_ids"])
    recency = torch.zeros(n)
    freq = torch.zeros(n)
    observed = torch.zeros(n, dtype=torch.bool)
    boundary = 0
    for code, li in g["code_index"].items():
        if code in history:
            recency[li], freq[li] = history[code]
            observed[li] = True
        else:
            boundary += 1
    if "edge_hops" not in g:
        raise KeyError("patient graph has no edge_hops; rebuild it with build_patient_graphs.py")
    data = Data(
        node_ids=torch.as_tensor(g["node_ids"], dtype=torch.long),
        edge_index=torch.as_tensor(g["edge_index"], dtype=torch.long),
        edge_rel=torch.as_tensor(g["edge_rel"], dtype=torch.long),
        edge_tau=torch.as_tensor(g["edge_tau"], dtype=torch.float32),
        edge_hops=torch.as_tensor(g["edge_hops"], dtype=torch.int16),
        recency_days=recency, frequency=freq, observed_mask=observed,
        num_nodes=n,
    )
    if values is not None:
        va = torch.zeros(n, 4)
        for code, li in g["code_index"].items():
            if code in values and code in history:
                va[li] = torch.as_tensor(values[code], dtype=torch.float32)
        data.value_attr = va
    return data, boundary


def materialize(data, text_table: torch.Tensor):
    """Add x [n, 770] and edge_attr [E, 17] to a stored Data (returns a copy)."""
    out = data.clone()
    out.x = torch.cat([text_table[data.node_ids],
                       torch.log1p(data.recency_days.clamp(min=0)).unsqueeze(-1),
                       torch.log1p(data.frequency).unsqueeze(-1)], dim=-1)
    out.edge_attr = torch.cat([time_encoding(data.edge_tau),
                               (data.edge_hops.float() / MAX_HOPS).unsqueeze(-1)], dim=-1)
    return out


class EnrichedPatientGraphs(torch.utils.data.Dataset):
    """patient_graphs_enriched.<kg>.pt -> materialised Data per patient (ds[i] or ds.get(patient_id))."""

    def __init__(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = {k: v for k, v in payload.items() if k not in ("graphs", "text_table", "code_index")}
        self.text_table = payload["text_table"]
        self.patient_ids: list[int] = payload["patient_ids"]
        self.graphs = payload["graphs"]
        self.code_index: dict[int, dict[str, int]] = payload["code_index"]
        self._pos = {p: i for i, p in enumerate(self.patient_ids)}

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, i: int):
        d = materialize(self.graphs[i], self.text_table)
        d.patient_id = self.patient_ids[i]
        return d

    def get(self, patient_id: int):
        return self[self._pos[int(patient_id)]] if int(patient_id) in self._pos else None
