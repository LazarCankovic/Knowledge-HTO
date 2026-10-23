from __future__ import annotations

from typing import List, Dict, Any
import torch


def build_ordered_batch(
    items: List[Dict[str, Any]],
    max_events: int = 512,
):
    """
    Uses your actual flattened fields:
      flat_tokens
      flat_domain
      flat_day
      flat_numeric
      flat_num_mask
      flat_visit_idx
    """
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
            times[i, j] = float(days[j])  # already looks normalized in your sample
            positions[i, j] = j

            x = nums[j]
            m = num_masks[j]

            # flat_numeric sample showed scalar values, but support list too
            if isinstance(x, (list, tuple)):
                if isinstance(m, (list, tuple)):
                    observed = [float(a) for a, b in zip(x, m) if float(b) > 0]
                else:
                    observed = list(map(float, x)) if float(m) > 0 else []
            else:
                observed = [float(x)] if float(m) > 0 else []

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