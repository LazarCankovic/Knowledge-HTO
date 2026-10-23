from __future__ import annotations

import json
import pickle
import torch


def load_serialized_items(path: str):
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if isinstance(data, dict) and "items" in data:
        data = data["items"]

    if not isinstance(data, list):
        raise ValueError(f"Expected list of patient items, got {type(data)}")

    return data
