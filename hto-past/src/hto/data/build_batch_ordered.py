from __future__ import annotations

from typing import List, Dict, Any
import torch


def build_ordered_batch(
    items: List[Dict[str, Any]],
    max_events: int = 512,
):
    B = len(items)

    event_ids = torch.zeros((B, max_events), dtype=torch.long)
    domain_ids = torch.zeros((B, max_events), dtype=torch.long)
    times = torch.zeros((B, max_events), dtype=torch.float32)
    positions = torch.zeros((B, max_events), dtype=torch.long)
    values = torch.zeros((B, max_events), dtype=torch.float32)
    has_value = torch.zeros((B, max_events), dtype=torch.float32)
    mask = torch.zeros((B, max_events), dtype=torch.float32)

    labels = torch.zeros((B,), dtype=torch.long)
    patient_ids = torch.zeros((B,), dtype=torch.long)

    for i, it in enumerate(items):
        labels[i] = int(it["label"])
        patient_ids[i] = int(it["patient_id"])

        toks = it["flat_tokens"][-max_events:]
        doms = it["flat_domain"][-max_events:]
        days = it["flat_day"][-max_events:]
        nums = it["flat_numeric"][-max_events:]
        num_masks = it["flat_num_mask"][-max_events:]

        for j in range(len(toks)):
            event_ids[i, j] = int(toks[j])
            domain_ids[i, j] = int(doms[j])
            times[i, j] = float(days[j]) / 365.0
            positions[i, j] = j

            # convert numeric side-channel into a single scalar summary
            # if it is vector-valued, mean over observed dimensions
            x = nums[j]
            m = num_masks[j]

            if not isinstance(x, (list, tuple)):
                x = [x]
            if not isinstance(m, (list, tuple)):
                m = [m]

            observed = [float(a) for a, b in zip(x, m) if float(b) > 0]
            if len(observed) > 0:
                values[i, j] = sum(observed) / len(observed)
                has_value[i, j] = 1.0
            else:
                values[i, j] = 0.0
                has_value[i, j] = 0.0

            mask[i, j] = 1.0

    return {
        "event_ids": event_ids,
        "domain_ids": domain_ids,
        "times": times,
        "positions": positions,
        "values": values,
        "has_value": has_value,
        "mask": mask,
        "labels": labels,
        "patient_ids": patient_ids,
    }


def ordered_collate_fn(max_events=512):
    def _fn(items):
        return build_ordered_batch(items, max_events=max_events)
    return _fn