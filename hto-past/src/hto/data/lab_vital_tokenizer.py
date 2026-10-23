from __future__ import annotations

from typing import List


def sanitize_token(text: str) -> str:
    return str(text).replace(" ", "_")


def value_to_bin_token(value: float, prefix: str, scale: float = 100.0, clip_min: int = -5000, clip_max: int = 50000) -> str:
    """
    Example:
      0.23  -> LVAL_BIN_23
      146.0 -> LVAL_BIN_14600
      -1.2  -> LVAL_BIN_NEG_120
    """
    bin_id = int(round(value * scale))
    bin_id = max(clip_min, min(clip_max, bin_id))
    if bin_id >= 0:
        return f"{prefix}_BIN_{bin_id}"
    return f"{prefix}_BIN_NEG_{abs(bin_id)}"


def build_lab_sequence(lab_code_tokens: List[str], lab_values: List[float]) -> List[str]:
    seq = []
    for code_tok, val in zip(lab_code_tokens, lab_values):
        seq.extend(["[LCODE]", sanitize_token(code_tok), "[LVAL]", value_to_bin_token(float(val), "LVAL")])
    return seq


def build_vital_sequence(vital_code_tokens: List[str], vital_values: List[float]) -> List[str]:
    seq = []
    for code_tok, val in zip(vital_code_tokens, vital_values):
        seq.extend(["[VCODE]", sanitize_token(code_tok), "[VVAL]", value_to_bin_token(float(val), "VVAL")])
    return seq
