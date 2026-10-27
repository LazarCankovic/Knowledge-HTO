"""Per-patient time-aware graphs (KerPrint's local KG, adapted to ICD nodes + SNOMED).

For a patient whose diagnosis codes (ICD nodes) were observed at known times:
  * every observed code keeps its own 1-hop knowledge: maps_to concepts and its ICD category (tau = 0)
  * every pair of observed codes (s, t) that the global graph connects within 2 + 2 hops through a
    shared "meeting" node m contributes the edges on s -> m -> t, for up to max_meetings_per_pair of
    the closest meeting nodes. Other ICD nodes are never used as intermediates.
  * tau for path edges = the gap in days between the most recent visit with s and the most recent visit
    with t (KerPrint: "the time interval between the two concepts in the real visit sequence"). The most
    recent occurrence is used, not the closest pair of occurrences: chronic codes recur at nearly every
    visit, so the closest-pair gap is ~0 for almost every pair and carries no time signal. An edge on
    several paths keeps its smallest tau.
Edges are stored in both directions (forward relation and its inverse), as in the global graph.

Output per patient (numpy arrays, local node indices):
  node_ids    global entity ids of the local nodes
  edge_index  [2, E] (src, dst) local indices
  edge_rel    [E] relation ids
  edge_tau    [E] days (float32)
  code_index  {diag key -> local index} for the observed codes
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class PatientGraphConfig:
    max_meetings_per_pair: int = 3
    max_codes_per_patient: int = 128      # most recently observed codes used for pairing
    max_edges_per_patient: int = 2000     # cap on forward path edges; edges on the longest paths go first


class GlobalGraphIndex:
    def __init__(self, graph: dict, code_kind: str = "icd"):
        ent = graph["entities"]
        self.code_kind = code_kind
        self.key_to_id = dict(zip(ent["key"], ent["id"]))
        self.kind = dict(zip(ent["id"], ent["kind"]))
        rel = graph["relations"]
        names = dict(zip(rel["id"], rel["name"]))
        by_name = {v: k for k, v in names.items()}
        self.inverse = {rid: by_name[f"inv_{n}"] if not n.startswith("inv_") else by_name[n[4:]]
                        for rid, n in names.items()}
        self.neighbors: dict[int, list[tuple[int, int]]] = defaultdict(list)   # node -> [(rel, nbr)]
        for h, r, t in graph["triples"][["h", "r", "t"]].itertuples(index=False):
            self.neighbors[h].append((r, t))
        self._ball_cache: dict[int, dict[int, tuple[int, list[tuple[int, int, int]]]]] = {}

    def ball(self, src: int, depth: int = 2) -> dict[int, tuple[int, list[tuple[int, int, int]]]]:
        """Nodes within `depth` hops of src -> (distance, path edges src..node). ICD nodes other than
        src are not expanded (they may only be path endpoints)."""
        if src in self._ball_cache:
            return self._ball_cache[src]
        out = {src: (0, [])}
        frontier = [src]
        for d in range(1, depth + 1):
            nxt = []
            for u in frontier:
                # An unobserved code from another tower must never become a
                # shortcut through the global graph.
                if u != src and self.kind[u] in {"icd", "lab"}:
                    continue
                for r, v in self.neighbors[u]:
                    if v not in out:
                        out[v] = (d, out[u][1] + [(u, r, v)])
                        nxt.append(v)
            frontier = nxt
        self._ball_cache[src] = out
        return out


def build_patient_graph(codes_times: dict[str, np.ndarray], g: GlobalGraphIndex,
                        cfg: PatientGraphConfig, path_time_priority: bool = False) -> dict:
    """codes_times: diag key -> array of observation times in days."""
    observed = {k: t for k, t in codes_times.items() if k in g.key_to_id}
    # most recently observed codes first, capped
    keys = sorted(observed, key=lambda k: -observed[k].max())[: cfg.max_codes_per_patient]
    ids = [g.key_to_id[k] for k in keys]

    edges: dict[tuple[int, int, int], float] = {}     # (u, r, v) forward edge -> tau

    def add(u: int, r: int, v: int, tau: float) -> None:
        if (u, r, v) not in edges or tau < edges[(u, r, v)]:
            edges[(u, r, v)] = tau

    # 1. each code's own 1-hop knowledge
    for s in ids:
        for r, v in g.neighbors[s]:
            add(s, r, v, 0.0)

    # 2. paths between pairs of observed codes through shared meeting nodes
    # meeting node -> [(source, dist)]; an observed code is its own meeting node at distance 0, so direct
    # paths s -> ... -> t count; unobserved ICD nodes are never meeting nodes
    observed_ids = set(ids)
    reach: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for s in ids:
        for m, (d, _) in g.ball(s).items():
            if g.kind[m] in {"icd", "lab"} and m not in observed_ids:
                continue
            reach[m].append((s, d))
    candidates: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for m, sources in reach.items():
        if len(sources) < 2:
            continue
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                (s, ds), (t, dt) = sources[i], sources[j]
                pair = (s, t) if s < t else (t, s)
                candidates[pair].append((ds + dt, m))
    last_seen = {g.key_to_id[k]: float(observed[k].max()) for k in keys}
    path_edges: dict[tuple[int, int, int], float] = {}
    path_len: dict[tuple[int, int, int], int] = {}     # shortest s..t path the edge lies on
    for (s, t), cands in candidates.items():
        tau = abs(last_seen[s] - last_seen[t])
        for dist, m in sorted(cands)[: cfg.max_meetings_per_pair]:
            for u, r, v in g.ball(s)[m][1] + g.ball(t)[m][1]:
                key = (u, r, v)
                if key not in path_edges or tau < path_edges[key]:
                    path_edges[key] = tau
                path_len[key] = min(path_len.get(key, dist), dist)
    if len(path_edges) > cfg.max_edges_per_patient:
        # keep edges on the shortest paths (cutting by tau would bias the graph towards tau = 0)
        keep = sorted(path_edges, key=lambda k: (path_len[k], k))[: cfg.max_edges_per_patient]
        path_edges = {k: path_edges[k] for k in keep}
    for (u, r, v), tau in path_edges.items():
        # In a shallow metadata graph, a path edge is often also a code's
        # one-hop knowledge edge. The default diagnosis behaviour keeps 0;
        # labs retain the observed pair gap so temporal attention can use it.
        if path_time_priority:
            edges[(u, r, v)] = tau
        else:
            add(u, r, v, tau)
    path_taus = np.asarray(list(path_edges.values()), dtype=np.float32)

    # 3. both directions, local indexing
    both: dict[tuple[int, int, int], float] = {}
    for (u, r, v), tau in edges.items():
        for key in ((u, r, v), (v, g.inverse[r], u)):
            if key not in both or tau < both[key]:
                both[key] = tau
    nodes = sorted(set(ids) | {u for u, _, _ in both} | {v for _, _, v in both})
    local = {n: i for i, n in enumerate(nodes)}
    triples = sorted(both)
    return {
        "node_ids": np.asarray(nodes, dtype=np.int64),
        "edge_index": np.asarray([[local[u] for u, _, _ in triples], [local[v] for _, _, v in triples]],
                                 dtype=np.int64).reshape(2, -1),
        "edge_rel": np.asarray([r for _, r, _ in triples], dtype=np.int64),
        "edge_tau": np.asarray([both[t] for t in triples], dtype=np.float32),
        "code_index": {k: local[g.key_to_id[k]] for k in keys},
        "n_pairs_connected": len(candidates),
        "n_cross_domain_pairs_connected": sum(g.kind[s] != g.kind[t] for s, t in candidates),
        "path_taus": path_taus,     # forward path edges only (for stats)
    }


def build_all(diagnoses: pd.DataFrame, graph: dict, cfg: PatientGraphConfig = PatientGraphConfig(),
              code_col: str = "diag", code_kind: str = "icd",
              path_time_priority: bool = False) -> tuple[dict, dict]:
    """Build patient graphs from (patient_id, time, code_col) observations."""
    g = GlobalGraphIndex(graph, code_kind=code_kind)
    df = diagnoses[["patient_id", "time", code_col]].dropna()
    # unit-safe (parquet timestamps can come back as us or ns resolution)
    df = df.assign(day=(pd.to_datetime(df["time"]) - pd.Timestamp(0)) / pd.Timedelta(days=1))
    out = {}
    for pid, grp in df.groupby("patient_id"):
        codes_times = {k: v.to_numpy() for k, v in grp.groupby(code_col)["day"]}
        out[int(pid)] = build_patient_graph(codes_times, g, cfg,
                                            path_time_priority=path_time_priority)

    n_obs = np.array([len(p["code_index"]) for p in out.values()])
    n_edges = np.array([len(p["edge_rel"]) for p in out.values()])
    n_nodes = np.array([len(p["node_ids"]) for p in out.values()])
    taus = np.concatenate([p["edge_tau"] for p in out.values()]) if out else np.zeros(0)
    path_taus = np.concatenate([p.pop("path_taus") for p in out.values()]) if out else np.zeros(0)
    pct = lambda a: {q: round(float(np.percentile(a, q)), 1) for q in (10, 50, 90, 99)} if len(a) else {}
    stats = {
        "config": asdict(cfg),
        "path_time_priority": path_time_priority,
        ({"icd": "patients_with_diagnoses", "lab": "patients_with_labs"}
         .get(code_kind, "patients_with_events")):
            int(df["patient_id"].nunique()),
        "patients_with_graph_code": int((n_obs > 0).sum()),
        "patients_with_cross_code_path": int(sum(p["n_pairs_connected"] > 0 for p in out.values())),
        "patients_with_lab_diagnosis_path": int(sum(
            p["n_cross_domain_pairs_connected"] > 0 for p in out.values())),
        "observed_codes_per_patient": pct(n_obs),
        "nodes_per_patient": pct(n_nodes),
        "edges_per_patient": pct(n_edges),
        "edge_tau_days": pct(taus),
        "share_edges_tau_zero": round(float((taus == 0).mean()), 3) if len(taus) else None,
        "path_edge_tau_days": pct(path_taus),
        "share_path_edges_tau_zero": round(float((path_taus == 0).mean()), 3) if len(path_taus) else None,
    }
    return out, stats
