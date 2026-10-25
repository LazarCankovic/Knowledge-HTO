"""Baseline registry: name -> (class, required variant, constructor factory)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# existing (untouched) baselines
from .models.behrt import BEHRT
from .models.cehrbert import CEHRBEHRT
from .models.medbert import MedBERT
from .models.multimodal_behrt import MultimodalBEHRT
from .models.time2vec import Time2VecModel
from .models.tlstm import TLSTM
from .models.transformehr import TransformEHR

# new multidomain baselines
from .models.behrt_md import BEHRT_MD
from .models.cehrbert_md import CEHRBEHRT_MD
from .models.medbert_md import MedBERT_MD
from .models.md_hier import MultimodalBEHRT_MD, TLSTM_MD, TransformEHR_MD
from .models.mtan import MTAN
from .models.strats import STraTS
from .models.htr_former import HTRFormer
from .models.a1_time_former import A1TimeFormer
from .models.htr_former_cross_attn import HTRFormerCrossAttn
from .models.htr_former_kg import HTRFormerKG
from .models.a2_no_time_former import A2NoTimeFormer
from .models.time2vec_md import Time2Vec_MD


@dataclass
class Entry:
    name: str
    variant: str
    cls: type
    factory: Callable[[int, dict[str, Any]], Any]


REGISTRY: dict[str, Entry] = {
    # --- original 7 (unchanged) ---
    "medbert":          Entry("medbert",          "codes_flat",       MedBERT,          lambda v, x: MedBERT(v, **x)),
    "tlstm":            Entry("tlstm",            "codes_visit",      TLSTM,            lambda v, x: TLSTM(v, **x)),
    "time2vec":         Entry("time2vec",         "codes_flat",       Time2VecModel,    lambda v, x: Time2VecModel(v, **x)),
    "transformehr":     Entry("transformehr",     "codes_visit",      TransformEHR,     lambda v, x: TransformEHR(v, **x)),
    "behrt":            Entry("behrt",            "codes_visit",      BEHRT,            lambda v, x: BEHRT(v, **x)),
    "cehrbert":         Entry("cehrbert",         "codes_visit",      CEHRBEHRT,        lambda v, x: CEHRBEHRT(v, **x)),
    "multimodal_behrt": Entry("multimodal_behrt", "multimodal",       MultimodalBEHRT,  lambda v, x: MultimodalBEHRT(v, **x)),

    # --- multidomain_visit versions ---
    "medbert_md":          Entry("medbert_md",          "multidomain_visit", MedBERT_MD,          lambda v, x: MedBERT_MD(v, **x)),
    "tlstm_md":            Entry("tlstm_md",            "multidomain_visit", TLSTM_MD,            lambda v, x: TLSTM_MD(v, **x)),
    "time2vec_md":         Entry("time2vec_md",         "multidomain_visit", Time2Vec_MD,         lambda v, x: Time2Vec_MD(v, **x)),
    "transformehr_md":     Entry("transformehr_md",     "multidomain_visit", TransformEHR_MD,     lambda v, x: TransformEHR_MD(v, **x)),
    "behrt_md":            Entry("behrt_md",            "multidomain_visit", BEHRT_MD,            lambda v, x: BEHRT_MD(v, **x)),
    "cehrbert_md":         Entry("cehrbert_md",         "multidomain_visit", CEHRBEHRT_MD,        lambda v, x: CEHRBEHRT_MD(v, **x)),
    "multimodal_behrt_md": Entry("multimodal_behrt_md", "multidomain_visit", MultimodalBEHRT_MD,  lambda v, x: MultimodalBEHRT_MD(v, **x)),

    # --- new irregular-time-series baselines ---
    "htr_former": Entry("htr_former", "allcodes_timeaware", HTRFormer,
               lambda v, x: HTRFormer(v, **x)),
    "htr_former_ca": Entry("htr_former_ca", "allcodes_timeaware", HTRFormerCrossAttn,
               lambda v, x: HTRFormerCrossAttn(v, **x)),
    "htr_former_kg": Entry("htr_former_kg", "allcodes_timeaware", HTRFormerKG,
               lambda v, x: HTRFormerKG(v, **x)),
    "a1_time":    Entry("a1_time",    "allcodes_timeaware", A1TimeFormer,    lambda v, x: A1TimeFormer(v, **x)),
    "a2_no_time": Entry("a2_no_time", "allcodes_timeaware", A2NoTimeFormer, lambda v, x: A2NoTimeFormer(v, **x)),
    "mtan":    Entry("mtan",    "multidomain_visit", MTAN,    lambda v, x: MTAN(v, **x)),
    "strats":  Entry("strats",  "multidomain_visit", STraTS,  lambda v, x: STraTS(v, **x)),
}


def get(name: str) -> Entry:
    if name not in REGISTRY:
        raise KeyError(f"Unknown baseline '{name}'. Known: {list(REGISTRY)}")
    return REGISTRY[name]


ORIGINAL_BASELINES = ["medbert", "tlstm", "time2vec", "transformehr", "behrt", "cehrbert", "multimodal_behrt"]
MD_BASELINES       = ["medbert_md", "tlstm_md", "time2vec_md", "transformehr_md", "behrt_md", "cehrbert_md", "multimodal_behrt_md"]
NEW_BASELINES      = ["mtan", "strats"]
