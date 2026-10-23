from __future__ import annotations

from typing import List, Dict, Any
from torch.utils.data import Dataset

from hto.data.data_utils import load_serialized_items


class PatientDataset(Dataset):
    def __init__(self, path: str):
        self.items: List[Dict[str, Any]] = load_serialized_items(path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]
