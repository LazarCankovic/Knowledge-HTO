from __future__ import annotations

from typing import List, Dict, Any
import torch


def pad_code_visits(patient_visits_list, max_visits: int, max_codes: int):
    B = len(patient_visits_list)

    ids = torch.zeros((B, max_visits, max_codes), dtype=torch.long)
    mask = torch.zeros((B, max_visits, max_codes), dtype=torch.float32)
    vmask = torch.zeros((B, max_visits), dtype=torch.float32)

    for i, visits in enumerate(patient_visits_list):
        visits = visits[-max_visits:]

        for j, codes in enumerate(visits):
            vmask[i, j] = 1.0
            codes = codes[:max_codes]
            if len(codes) > 0:
                ids[i, j, :len(codes)] = torch.tensor(codes, dtype=torch.long)
                mask[i, j, :len(codes)] = 1.0

    return ids, mask, vmask


def build_visit_batch(
    items: List[Dict[str, Any]],
    max_visits: int = 64,
    max_diag_codes: int = 64,
    max_med_codes: int = 64,
    n_lab: int = 6,
    n_vit: int = 33,
):
    """
    Uses your actual multidomain_visit format.

    Each patient item has:
      patient_id
      label
      visits = [
        {
          "day_offset": float,
          "delta_prev_days": float,
          "blocks": {
              "diag": [...],
              "lab": [...],
              "med": [...],
              "vital": [...]
          },
          "block_numeric": {
              "lab": [...],
              "vital": [...]
          }
        },
        ...
      ]
    """
    B = len(items)

    diag_visits_all = []
    med_visits_all = []

    for it in items:
        visits = it["visits"][-max_visits:]
        diag_visits = [visit["blocks"].get("diag", []) for visit in visits]
        med_visits = [visit["blocks"].get("med", []) for visit in visits]

        diag_visits_all.append(diag_visits)
        med_visits_all.append(med_visits)

    diag_ids, diag_mask, diag_vmask = pad_code_visits(
        diag_visits_all, max_visits=max_visits, max_codes=max_diag_codes
    )
    med_ids, med_mask, med_vmask = pad_code_visits(
        med_visits_all, max_visits=max_visits, max_codes=max_med_codes
    )

    vmask = torch.maximum(diag_vmask, med_vmask)

    lab_vals = torch.zeros((B, max_visits, n_lab), dtype=torch.float32)
    lab_mask = torch.zeros((B, max_visits, n_lab), dtype=torch.float32)

    vit_vals = torch.zeros((B, max_visits, n_vit), dtype=torch.float32)
    vit_mask = torch.zeros((B, max_visits, n_vit), dtype=torch.float32)

    days = torch.zeros((B, max_visits), dtype=torch.float32)
    labels = torch.zeros((B,), dtype=torch.long)
    patient_ids = torch.zeros((B,), dtype=torch.long)

    for i, it in enumerate(items):
        labels[i] = int(it["label"])
        patient_ids[i] = int(it["patient_id"])

        visits = it["visits"][-max_visits:]

        for j, visit in enumerate(visits):
            vmask[i, j] = 1.0
            days[i, j] = float(visit["day_offset"]) / 365.0

            labs = visit["block_numeric"].get("lab", [])
            vits = visit["block_numeric"].get("vital", [])

            if len(labs) > 0:
                use_lab = min(len(labs), n_lab)
                lab_vals[i, j, :use_lab] = torch.tensor(labs[:use_lab], dtype=torch.float32)
                lab_mask[i, j, :use_lab] = 1.0

            if len(vits) > 0:
                use_vit = min(len(vits), n_vit)
                vit_vals[i, j, :use_vit] = torch.tensor(vits[:use_vit], dtype=torch.float32)
                vit_mask[i, j, :use_vit] = 1.0

    return {
        "diag_ids": diag_ids,
        "diag_mask": diag_mask,
        "med_ids": med_ids,
        "med_mask": med_mask,
        "lab_vals": lab_vals,
        "lab_mask": lab_mask,
        "vit_vals": vit_vals,
        "vit_mask": vit_mask,
        "days": days,
        "vmask": vmask,
        "labels": labels,
        "patient_ids": patient_ids,
    }


def visit_collate_fn(max_visits=64, max_diag_codes=64, max_med_codes=64, n_lab=6, n_vit=33):
    def _fn(items):
        return build_visit_batch(
            items,
            max_visits=max_visits,
            max_diag_codes=max_diag_codes,
            max_med_codes=max_med_codes,
            n_lab=n_lab,
            n_vit=n_vit,
        )
    return _fn