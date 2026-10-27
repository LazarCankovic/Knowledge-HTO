"""Shared batch builders for the multidomain_visit variant.

Two entry points:
  - build_flat_batch:    one flat token stream per patient with domain/day/
                          delta/position per token. For BERT-style baselines.
  - build_visit_batch:   [B, V, L] token lattice with per-visit day/delta and
                          per-token domain/numeric. For hierarchical baselines.

Both prepend a [CLS] token at id=2 (specials: PAD=0 UNK=1 CLS=2 SEP=3 MASK=4).
"""
from __future__ import annotations

from typing import List

import torch

CLS_ID, SEP_ID = 2, 3


def build_flat_batch(items: List[dict], max_len: int = 1024) -> dict[str, torch.Tensor]:
    B = len(items)
    L = max_len

    ids = torch.zeros((B, L), dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    dom = torch.zeros((B, L), dtype=torch.long)
    day = torch.zeros((B, L), dtype=torch.float32)
    dlt = torch.zeros((B, L), dtype=torch.float32)
    vis = torch.zeros((B, L), dtype=torch.long)
    num = torch.zeros((B, L), dtype=torch.float32)
    nm  = torch.zeros((B, L), dtype=torch.float32)
    mask = torch.zeros((B, L), dtype=torch.float32)

    for i, it in enumerate(items):
        ft, fd, f_day, f_dlt, f_vi, f_num, f_nm = (
            it["flat_tokens"], it["flat_domain"], it["flat_day"],
            it["flat_delta"], it["flat_visit_idx"], it["flat_numeric"], it["flat_num_mask"],
        )
        # prepend CLS — truncate to L-1 from the back
        if len(ft) >= L:
            start = len(ft) - (L - 1)
            ft = ft[start:]; fd = fd[start:]; f_day = f_day[start:]
            f_dlt = f_dlt[start:]; f_vi = f_vi[start:]; f_num = f_num[start:]; f_nm = f_nm[start:]
        n = len(ft) + 1
        ids[i, 0]   = CLS_ID
        pos[i, 0]   = 0
        dom[i, 0]   = 4  # DOMAIN_STRUCT
        day[i, 0]   = 0.0
        dlt[i, 0]   = 0.0
        vis[i, 0]   = 0
        mask[i, 0]  = 1.0
        ids[i, 1:n]  = torch.tensor(ft, dtype=torch.long)
        pos[i, 1:n]  = torch.arange(1, n, dtype=torch.long)
        dom[i, 1:n]  = torch.tensor(fd, dtype=torch.long)
        day[i, 1:n]  = torch.tensor(f_day, dtype=torch.float32)
        dlt[i, 1:n]  = torch.tensor(f_dlt, dtype=torch.float32)
        vis[i, 1:n]  = torch.tensor(f_vi, dtype=torch.long)
        num[i, 1:n]  = torch.tensor(f_num, dtype=torch.float32)
        nm[i, 1:n]   = torch.tensor(f_nm, dtype=torch.float32)
        mask[i, 1:n] = 1.0

    labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
    patient_ids = torch.tensor([int(it["patient_id"]) for it in items], dtype=torch.long)
    target_ids = torch.tensor([int(it.get("target_med_token_id", 1)) for it in items], dtype=torch.long)
    return {
        "ids": ids, "pos": pos, "domain_id": dom, "day": day, "delta": dlt,
        "visit_idx": vis, "num": num, "num_mask": nm, "mask": mask,
        "labels": labels, "patient_ids": patient_ids, "target_med_id": target_ids,
    }


def build_visit_batch(items: List[dict], max_visits: int = 64, max_codes: int = 96) -> dict[str, torch.Tensor]:
    B = len(items)
    V, L = max_visits, max_codes

    ids    = torch.zeros((B, V, L), dtype=torch.long)
    dom    = torch.full((B, V, L), 4, dtype=torch.long)  # default = STRUCT
    num    = torch.zeros((B, V, L), dtype=torch.float32)
    nm     = torch.zeros((B, V, L), dtype=torch.float32)
    cmask  = torch.zeros((B, V, L), dtype=torch.float32)
    vmask  = torch.zeros((B, V),    dtype=torch.float32)
    v_day  = torch.zeros((B, V),    dtype=torch.float32)
    v_dlt  = torch.zeros((B, V),    dtype=torch.float32)

    for i, it in enumerate(items):
        visits = it["visits"][-V:]
        for j, v in enumerate(visits):
            blocks   = v["blocks"]
            block_num = v["block_numeric"]
            # assemble per-visit flat with domain/numeric parallel arrays
            tok_list: list[int] = []
            dom_list: list[int] = []
            num_list: list[float] = []
            nm_list:  list[int] = []

            def _push(domain_id: int, tokens, numerics=None):
                for k, tok in enumerate(tokens):
                    tok_list.append(int(tok))
                    # last token of the block is the terminator (EOD/EOM/EOL/EOVI) -> STRUCT
                    is_terminator = (k == len(tokens) - 1)
                    dom_list.append(4 if is_terminator else domain_id)
                    if numerics is not None and k < len(numerics) and not is_terminator:
                        num_list.append(float(numerics[k])); nm_list.append(1)
                    else:
                        num_list.append(0.0); nm_list.append(0)

            _push(0, blocks["diag"])
            _push(1, blocks["med"])
            _push(2, blocks["lab"],  block_num["lab"])
            _push(3, blocks["vital"], block_num["vital"])

            if not tok_list:
                continue
            tok_list = tok_list[:L]; dom_list = dom_list[:L]
            num_list = num_list[:L]; nm_list  = nm_list[:L]
            n = len(tok_list)
            ids[i, j, :n]   = torch.tensor(tok_list, dtype=torch.long)
            dom[i, j, :n]   = torch.tensor(dom_list, dtype=torch.long)
            num[i, j, :n]   = torch.tensor(num_list, dtype=torch.float32)
            nm[i,  j, :n]   = torch.tensor(nm_list,  dtype=torch.float32)
            cmask[i, j, :n] = 1.0
            vmask[i, j]     = 1.0
            v_day[i, j]     = float(v["day_offset"])
            v_dlt[i, j]     = float(v["delta_prev_days"])

    labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
    patient_ids = torch.tensor([int(it["patient_id"]) for it in items], dtype=torch.long)
    target_ids = torch.tensor([int(it.get("target_med_token_id", 1)) for it in items], dtype=torch.long)
    return {
        "ids": ids, "domain": dom, "num": num, "num_mask": nm,
        "cmask": cmask, "vmask": vmask, "v_day": v_day, "v_delta": v_dlt,
        "labels": labels, "patient_ids": patient_ids, "target_med_id": target_ids,
    }
