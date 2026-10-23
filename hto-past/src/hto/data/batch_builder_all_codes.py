from __future__ import annotations

from typing import List, Dict, Any
import json
import torch

from hto.data.lab_vital_tokenizer import build_lab_sequence, build_vital_sequence

VOCAB_PATH = "/fs/atipa/data/nsf-jg/hto/hto2/processed_data/full_80_20/allcodes_timeaware.vocab.json"


with open(VOCAB_PATH, "r") as f:
    VOCAB = json.load(f)

TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}


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


def ids_from_tokens(tokens: List[str]) -> List[int]:
    unk_id = TOKEN_TO_ID.get("[UNK]", 1)
    return [TOKEN_TO_ID.get(tok, unk_id) for tok in tokens]


def build_allcodes_batch(
    items: List[Dict[str, Any]],
    max_visits: int = 64,
    max_diag_codes: int = 64,
    max_med_codes: int = 64,
    max_lab_codes: int = 128,
    max_vit_codes: int = 128,
):
    B = len(items)

    diag_visits_all = []
    med_visits_all = []
    lab_visits_all = []
    vit_visits_all = []

    labels = torch.zeros((B,), dtype=torch.long)
    patient_ids = torch.zeros((B,), dtype=torch.long)
    days = torch.zeros((B, max_visits), dtype=torch.float32)

    for i, it in enumerate(items):
        labels[i] = int(it["label"])
        patient_ids[i] = int(it["patient_id"])

        visits = it["visits"][-max_visits:]

        diag_visits = []
        med_visits = []
        lab_visits = []
        vit_visits = []

        for j, visit in enumerate(visits):
            days[i, j] = float(visit["day_offset"]) / 365.0

            # diagnoses and meds stay as normal ids
            diag_visits.append(list(map(int, visit["blocks"].get("diag", []))))
            med_visits.append(list(map(int, visit["blocks"].get("med", []))))

            # labs become token sequences
            lab_token_ids = visit["blocks"].get("lab", [])
            lab_values = visit["block_numeric"].get("lab", [])
            lab_code_tokens = [VOCAB[t] for t in lab_token_ids if 0 <= t < len(VOCAB)]
            lab_seq = build_lab_sequence(lab_code_tokens, lab_values)
            lab_visits.append(ids_from_tokens(lab_seq))

            # vitals become token sequences
            vit_token_ids = visit["blocks"].get("vital", [])
            vit_values = visit["block_numeric"].get("vital", [])
            vit_code_tokens = [VOCAB[t] for t in vit_token_ids if 0 <= t < len(VOCAB)]
            vit_seq = build_vital_sequence(vit_code_tokens, vit_values)
            vit_visits.append(ids_from_tokens(vit_seq))

        diag_visits_all.append(diag_visits)
        med_visits_all.append(med_visits)
        lab_visits_all.append(lab_visits)
        vit_visits_all.append(vit_visits)

    diag_ids, diag_mask, diag_vmask = pad_code_visits(diag_visits_all, max_visits, max_diag_codes)
    med_ids, med_mask, med_vmask = pad_code_visits(med_visits_all, max_visits, max_med_codes)
    lab_ids, lab_mask, lab_vmask = pad_code_visits(lab_visits_all, max_visits, max_lab_codes)
    vit_ids, vit_mask, vit_vmask = pad_code_visits(vit_visits_all, max_visits, max_vit_codes)

    vmask = torch.maximum(diag_vmask, med_vmask)
    vmask = torch.maximum(vmask, lab_vmask)
    vmask = torch.maximum(vmask, vit_vmask)

    return {
        "diag_ids": diag_ids,
        "diag_mask": diag_mask,
        "med_ids": med_ids,
        "med_mask": med_mask,
        "lab_ids": lab_ids,
        "lab_mask": lab_mask,
        "vit_ids": vit_ids,
        "vit_mask": vit_mask,
        "days": days,
        "vmask": vmask,
        "labels": labels,
        "patient_ids": patient_ids,
    }


def allcodes_collate_fn(
    max_visits=64,
    max_diag_codes=64,
    max_med_codes=64,
    max_lab_codes=128,
    max_vit_codes=128,
):
    def _fn(items):
        return build_allcodes_batch(
            items,
            max_visits=max_visits,
            max_diag_codes=max_diag_codes,
            max_med_codes=max_med_codes,
            max_lab_codes=max_lab_codes,
            max_vit_codes=max_vit_codes,
        )
    return _fn
