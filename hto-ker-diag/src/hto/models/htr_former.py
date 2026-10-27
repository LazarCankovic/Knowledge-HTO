from __future__ import annotations
from hto.models.four_towers_timeaware_all_codes import FourTowersTimeAwareAllCodes
from hto.data.batch_builder_all_codes_local import build_allcodes_batch

class HTRFormer(FourTowersTimeAwareAllCodes):
    """Registry-compatible wrapper for HTR-Former."""
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_visit_layers: int = 2, n_patient_layers: int = 2,
                 max_visits: int = 64, dropout: float = 0.1,
                 n_classes: int = 2, time_fn: str = "log", **kwargs):
        super().__init__(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_visit_layers=n_visit_layers, n_patient_layers=n_patient_layers,
            max_visits=max_visits, dropout=dropout,
            n_classes=n_classes, time_fn=time_fn,
        )

    def build_batch(self, items):
        return build_allcodes_batch(
            items,
            max_visits=64,
            max_diag_codes=64,
            max_med_codes=64,
            max_lab_codes=128,
            max_vit_codes=128,
        )
