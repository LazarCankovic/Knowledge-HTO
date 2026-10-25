"""Per-baseline dataset variants derived from canonical records.

Each variant is a list of per-patient dicts ready to be batched. A separate
`Vocab` object is fit on the training patients (passed in) and persisted so
the same vocab is used at test time.

The variants we expose:

- "codes_visit":  diagnosis+medication codes per visit. Used by Med-BERT,
                  T-LSTM, BEHRT, TransformEHR, CEHR-BEHRT. Emits for each
                  patient a list-of-visits with (code_ids, day_offset).
- "codes_flat":   flat ordered code stream with per-token day_offset (good
                  for Time2Vec).
- "multimodal":   codes_visit PLUS per-visit numeric lab/vital vectors
                  (aggregated to mean). Used by Multimodal-BEHRT.
- "text_sequence": pre-tokenized text from final_sequences.parquet, BPE-style
                   whitespace split. Used as an optional Med-BERT variant.

All variants share the same `Vocab`. Target label is carried as `label`.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
SPECIALS = [PAD, UNK, CLS, SEP, MASK]

# Multidomain-variant extra specials (mirror final_sequences.parquet grammar).
# These are reserved at fixed ids near the top of the vocab.
MD_SPECIALS = [
    "[HT]", "[DEMO]", "[BOH]", "[EOH]",
    "[EOV]", "[EOD]", "[EOM]", "[EOL]", "[EOVI]",
]

# Four semantic domains + one "structural" domain used for boundary tokens.
DOMAIN_DIAG, DOMAIN_MED, DOMAIN_LAB, DOMAIN_VITAL, DOMAIN_STRUCT = 0, 1, 2, 3, 4
N_DOMAINS = 5


class Vocab:
    def __init__(self, tokens: list[str]):
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = tokens

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.token_to_id[UNK]
        return [self.token_to_id.get(t, unk) for t in tokens]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.id_to_token, f)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def fit(cls, token_iter, min_count: int = 2, cap: int | None = 30000) -> "Vocab":
        c = Counter()
        for toks in token_iter:
            c.update(toks)
        items = [(t, n) for t, n in c.items() if n >= min_count]
        items.sort(key=lambda x: -x[1])
        if cap is not None:
            items = items[: max(0, cap - len(SPECIALS))]
        return cls(SPECIALS + [t for t, _ in items])


# ---------- token emission helpers ----------

def _diag_token(code: str) -> str:
    return "D_" + code


def _med_token(name: str) -> str:
    return "M_" + name.lower().strip().replace(" ", "_")[:48]


def _lab_bin_token(key: str, value: float) -> str:
    # Coarse tertile binning per lab is tricky without precomputed stats, so
    # use a simple log-magnitude binning. This is only used when labs are
    # turned into tokens (codes_flat, Med-BERT).
    if value is None:
        return f"L_{key}_NA"
    sign = "n" if value < 0 else "p"
    mag = int(np.clip(np.log10(abs(value) + 1.0) * 2, 0, 6))
    return f"L_{key}_{sign}{mag}"


def _vital_bin_token(key: str, value: float) -> str:
    if value is None:
        return f"V_{key}_NA"
    mag = int(np.clip(value / 20.0, 0, 15))  # good enough for bps/bpd/bmi ranges
    return f"V_{key}_{mag}"


# ---------- variants ----------

def iter_patient_visit_tokens(rec: dict, include_labs: bool = False, include_vitals: bool = False) -> list[list[str]]:
    """Per-visit lists of tokens (diag + med [+ lab/vital bins])."""
    out = []
    for v in rec["visits"]:
        toks = []
        toks += [_diag_token(c) for c in v["diag_codes"]]
        toks += [_med_token(m) for m in v["med_codes"]]
        if include_labs:
            toks += [_lab_bin_token(k, val) for (k, val) in v["lab_num"]]
        if include_vitals:
            toks += [_vital_bin_token(k, val) for (k, val) in v["vital_num"]]
        if toks:
            out.append(toks)
    return out


def make_codes_visit(records: list[dict], vocab: Vocab | None = None, max_visits: int = 64, max_codes_per_visit: int = 64) -> tuple[list[dict], Vocab]:
    """Per-visit code sequences."""
    visits_per_patient = [iter_patient_visit_tokens(r, include_labs=True, include_vitals=True) for r in records]

    if vocab is None:
        token_iter = (tok for vps in visits_per_patient for visit in vps for tok in visit)
        # Flatten once per-patient for .fit
        def gen():
            for vps in visits_per_patient:
                flat = []
                for visit in vps:
                    flat += visit
                yield flat
        vocab = Vocab.fit(gen())

    out = []
    for rec, visits in zip(records, visits_per_patient):
        # truncate to most recent `max_visits`
        visits = visits[-max_visits:]
        code_ids = [vocab.encode(v[:max_codes_per_visit]) for v in visits]
        day_offsets = []
        # day_offset per visit from canonical record
        for i, v_src in enumerate(rec["visits"][-max_visits:]):
            day_offsets.append(float(v_src["day_offset"]))
        # if no visits survived (shouldn't happen), append placeholder
        if not code_ids:
            code_ids = [[vocab.token_to_id[CLS]]]
            day_offsets = [0.0]
        out.append({
            "patient_id": rec["patient_id"],
            "label": rec["label"],
            "visits": code_ids,             # list[list[int]]
            "day_offsets": day_offsets,     # list[float]
            "target_med": rec["target_med"],
        })
    return out, vocab


def make_codes_flat(records: list[dict], vocab: Vocab | None = None, max_len: int = 512) -> tuple[list[dict], Vocab]:
    """Flat ordered code stream with per-token day_offset."""
    flats = []
    offs = []
    for r in records:
        toks, dos = [], []
        for v in r["visits"]:
            vtoks = [_diag_token(c) for c in v["diag_codes"]] + [_med_token(m) for m in v["med_codes"]]
            vtoks += [_lab_bin_token(k, val) for (k, val) in v["lab_num"]]
            vtoks += [_vital_bin_token(k, val) for (k, val) in v["vital_num"]]
            for t in vtoks:
                toks.append(t)
                dos.append(float(v["day_offset"]))
        # keep last max_len
        flats.append(toks[-max_len:])
        offs.append(dos[-max_len:])

    if vocab is None:
        vocab = Vocab.fit(iter(flats))

    out = []
    for r, toks, dos in zip(records, flats, offs):
        ids = vocab.encode(toks) if toks else [vocab.token_to_id[CLS]]
        if not toks:
            dos = [0.0]
        out.append({
            "patient_id": r["patient_id"],
            "label": r["label"],
            "token_ids": ids,
            "day_offsets": dos,
        })
    return out, vocab


# Numeric feature keys chosen for multimodal. Keep small + clinically relevant.
LAB_KEYS = ["Glucose", "Creatinine", "Sodium", "Potassium", "BUN",
            "Magnesium", "Calcium", "Cholesterol", "HDL", "LDL", "TSH", "Triglycerides"]
VITAL_KEYS = ["bps", "bpd", "bmi"]


def make_multimodal(records: list[dict], vocab: Vocab | None = None, max_visits: int = 64, max_codes_per_visit: int = 64) -> tuple[list[dict], Vocab]:
    """codes_visit + per-visit numeric lab & vital vectors."""
    cv, vocab = make_codes_visit(records, vocab=vocab, max_visits=max_visits, max_codes_per_visit=max_codes_per_visit)

    lab_idx = {k: i for i, k in enumerate(LAB_KEYS)}
    vit_idx = {k: i for i, k in enumerate(VITAL_KEYS)}
    n_lab = len(LAB_KEYS)
    n_vit = len(VITAL_KEYS)

    out = []
    for rec, cv_item in zip(records, cv):
        visits_src = rec["visits"][-max_visits:]
        lab_vecs = []
        lab_masks = []
        vit_vecs = []
        vit_masks = []
        for v in visits_src:
            lv = np.zeros(n_lab, dtype=np.float32)
            lm = np.zeros(n_lab, dtype=np.float32)
            cnt = np.zeros(n_lab, dtype=np.float32)
            for (k, val) in v["lab_num"]:
                if k in lab_idx:
                    lv[lab_idx[k]] += float(val)
                    cnt[lab_idx[k]] += 1.0
                    lm[lab_idx[k]] = 1.0
            nz = cnt > 0
            lv[nz] = lv[nz] / cnt[nz]
            vv = np.zeros(n_vit, dtype=np.float32)
            vm = np.zeros(n_vit, dtype=np.float32)
            cnt2 = np.zeros(n_vit, dtype=np.float32)
            for (k, val) in v["vital_num"]:
                if k in vit_idx:
                    vv[vit_idx[k]] += float(val)
                    cnt2[vit_idx[k]] += 1.0
                    vm[vit_idx[k]] = 1.0
            nz2 = cnt2 > 0
            vv[nz2] = vv[nz2] / cnt2[nz2]
            lab_vecs.append(lv)
            lab_masks.append(lm)
            vit_vecs.append(vv)
            vit_masks.append(vm)

        if not lab_vecs:
            lab_vecs = [np.zeros(n_lab, dtype=np.float32)]
            lab_masks = [np.zeros(n_lab, dtype=np.float32)]
            vit_vecs = [np.zeros(n_vit, dtype=np.float32)]
            vit_masks = [np.zeros(n_vit, dtype=np.float32)]

        out.append({
            "patient_id": rec["patient_id"],
            "label": rec["label"],
            "visits": cv_item["visits"],
            "day_offsets": cv_item["day_offsets"],
            "lab_vecs": [lv.tolist() for lv in lab_vecs],
            "lab_masks": [lm.tolist() for lm in lab_masks],
            "vital_vecs": [vv.tolist() for vv in vit_vecs],
            "vital_masks": [vm.tolist() for vm in vit_masks],
            "lab_keys": LAB_KEYS,
            "vital_keys": VITAL_KEYS,
        })
    return out, vocab


def make_text_sequence(records: list[dict], raw_sequences: dict[int, str], vocab: Vocab | None = None, max_len: int = 512) -> tuple[list[dict], Vocab]:
    """Use the pre-tokenized text-style sequences from final_sequences.parquet.

    Tokenize by whitespace and by bracket boundary; then fit a vocab.
    """
    import re

    def tok(s: str) -> list[str]:
        # whitespace split but keep bracket tokens as single units
        parts = re.findall(r"\[[A-Z]+\]|\S+", s)
        return parts

    tokenized = []
    for r in records:
        s = raw_sequences.get(int(r["patient_id"]), "")
        toks = tok(s)[-max_len:] if s else [CLS]
        tokenized.append(toks)

    if vocab is None:
        vocab = Vocab.fit(iter(tokenized), min_count=2, cap=50000)

    out = []
    for r, toks in zip(records, tokenized):
        ids = vocab.encode(toks)
        out.append({
            "patient_id": r["patient_id"],
            "label": r["label"],
            "token_ids": ids,
        })
    return out, vocab


# ---------- multidomain_visit variant ----------

def _md_vocab_fit(records: list[dict], cap: int | None = 30000, min_count: int = 2) -> Vocab:
    """Vocab with SPECIALS + MD_SPECIALS reserved at the top, followed by
    fitted code tokens. Reserved-id order is deterministic and matches the
    final_sequences.parquet grammar, so models can reference tokens by id."""
    c = Counter()
    for r in records:
        for v in r["visits"]:
            for code in v["diag_codes"]:
                c[_diag_token(code)] += 1
            for m in v["med_codes"]:
                c[_med_token(m)] += 1
            for (k, val) in v["lab_num"]:
                c[_lab_bin_token(k, val)] += 1
            for (k, val) in v["vital_num"]:
                c[_vital_bin_token(k, val)] += 1
    items = [(t, n) for t, n in c.items() if n >= min_count]
    items.sort(key=lambda x: -x[1])
    reserved = SPECIALS + MD_SPECIALS
    remaining = max(0, (cap or 10**9) - len(reserved))
    items = items[:remaining]
    tokens = reserved + [t for t, _ in items]
    return Vocab(tokens)


def md_reserved(vocab: Vocab) -> dict[str, int]:
    """Return a dict of reserved token -> id for a multidomain vocab."""
    names = SPECIALS + MD_SPECIALS
    return {n: vocab.token_to_id[n] for n in names}


def make_multidomain_visit(
    records: list[dict],
    vocab: Vocab | None = None,
    max_visits: int = 64,
    max_per_domain: int = 32,
    max_flat_len: int = 1024,
) -> tuple[list[dict], Vocab]:
    """Domain-delimited per-visit representation.

    For every patient we emit:
      visits: list of
        { "day_offset": float,
          "delta_prev_days": float,
          "blocks": {                                  # per-domain token ids
              "diag":  [int, ..., EOD_id],
              "med":   [int, ..., EOM_id],
              "lab":   [int, ..., EOL_id],
              "vital": [int, ..., EOVI_id],
          },
          "block_numeric": {                           # values for lab/vital
              "lab":   [float, ..., 0.0],             # 0.0 for EOL terminator
              "vital": [float, ..., 0.0],
          },
        }
      flat: a flattened concatenation across visits with EOV separators.
        "flat_tokens":   [int],
        "flat_domain":   [int] (0..4),
        "flat_day":      [float] (day_offset of the event's visit)
        "flat_delta":    [float] (delta_prev_days of the event's visit)
        "flat_visit_idx":[int]   (0-based visit index within the sequence)
        "flat_numeric":  [float] (0 for non-numeric tokens)
        "flat_num_mask": [0|1]
      target_med_token_id: int (one of the antihypertensive M_* code ids, if known).
    """
    if vocab is None:
        vocab = _md_vocab_fit(records)

    R = md_reserved(vocab)
    out: list[dict] = []

    for rec in records:
        visits_src = rec["visits"][-max_visits:]
        visits: list[dict] = []
        flat_tokens:   list[int]   = []
        flat_domain:   list[int]   = []
        flat_day:      list[float] = []
        flat_delta:    list[float] = []
        flat_visit:    list[int]   = []
        flat_numeric:  list[float] = []
        flat_num_mask: list[int]   = []

        prev_day = None
        for vi, v in enumerate(visits_src):
            day = float(v["day_offset"])
            delta = 0.0 if prev_day is None else max(0.0, day - prev_day)
            prev_day = day

            diag_tokens = [vocab.encode([_diag_token(c)])[0] for c in v["diag_codes"]][:max_per_domain]
            # raw codes aligned with diag_tokens (the KG tower needs codes the vocab maps to UNK)
            diag_raw    = [str(c) for c in v["diag_codes"]][:max_per_domain]
            med_tokens  = [vocab.encode([_med_token(m)])[0]  for m in v["med_codes"]][:max_per_domain]
            lab_pairs   = [(vocab.encode([_lab_bin_token(k, val)])[0], float(val)) for (k, val) in v["lab_num"]][:max_per_domain]
            vit_pairs   = [(vocab.encode([_vital_bin_token(k, val)])[0], float(val)) for (k, val) in v["vital_num"]][:max_per_domain]

            blocks = {"diag": [], "med": [], "lab": [], "vital": []}
            block_num = {"lab": [], "vital": []}

            def _push_block(name: str, domain_id: int, tokens: list[int], terminator: int,
                            values: list[float] | None = None):
                if not tokens:
                    return
                for k, tok in enumerate(tokens):
                    val = float(values[k]) if values is not None else 0.0
                    m = 1 if values is not None else 0
                    blocks[name].append(tok)
                    if values is not None:
                        block_num[name].append(val)
                    flat_tokens.append(tok); flat_domain.append(domain_id)
                    flat_day.append(day);   flat_delta.append(delta); flat_visit.append(vi)
                    flat_numeric.append(val); flat_num_mask.append(m)
                blocks[name].append(terminator)
                if values is not None:
                    block_num[name].append(0.0)
                flat_tokens.append(terminator); flat_domain.append(DOMAIN_STRUCT)
                flat_day.append(day); flat_delta.append(delta); flat_visit.append(vi)
                flat_numeric.append(0.0); flat_num_mask.append(0)

            _push_block("diag",  DOMAIN_DIAG,  diag_tokens, R["[EOD]"])
            _push_block("med",   DOMAIN_MED,   med_tokens,  R["[EOM]"])
            _push_block("lab",   DOMAIN_LAB,   [t for (t, _) in lab_pairs], R["[EOL]"],  [val for (_, val) in lab_pairs])
            _push_block("vital", DOMAIN_VITAL, [t for (t, _) in vit_pairs], R["[EOVI]"], [val for (_, val) in vit_pairs])

            # visit terminator
            flat_tokens.append(R["[EOV]"]); flat_domain.append(DOMAIN_STRUCT)
            flat_day.append(day); flat_delta.append(delta); flat_visit.append(vi)
            flat_numeric.append(0.0); flat_num_mask.append(0)

            visits.append({
                "day_offset": day,
                "delta_prev_days": delta,
                "blocks": blocks,
                "block_numeric": block_num,
                "blocks_raw": {"diag": diag_raw},
            })

        # truncate flat to max_flat_len (keep most recent events)
        if len(flat_tokens) > max_flat_len:
            keep_from = len(flat_tokens) - max_flat_len
            flat_tokens = flat_tokens[keep_from:]
            flat_domain = flat_domain[keep_from:]
            flat_day    = flat_day[keep_from:]
            flat_delta  = flat_delta[keep_from:]
            flat_visit  = flat_visit[keep_from:]
            flat_numeric = flat_numeric[keep_from:]
            flat_num_mask = flat_num_mask[keep_from:]

        # empty fallback
        if not flat_tokens:
            cls_id = vocab.token_to_id[CLS]
            flat_tokens = [cls_id]; flat_domain = [DOMAIN_STRUCT]
            flat_day = [0.0]; flat_delta = [0.0]; flat_visit = [0]
            flat_numeric = [0.0]; flat_num_mask = [0]
            visits = [{"day_offset": 0.0, "delta_prev_days": 0.0,
                       "blocks": {"diag": [], "med": [], "lab": [], "vital": []},
                       "block_numeric": {"lab": [], "vital": []},
                       "blocks_raw": {"diag": []}}]

        target_med_token = _med_token(rec["target_med"]) if rec.get("target_med") else None
        target_id = vocab.token_to_id.get(target_med_token, vocab.token_to_id[UNK]) if target_med_token else vocab.token_to_id[UNK]

        out.append({
            "patient_id": rec["patient_id"],
            "label": rec["label"],
            "target_med": rec["target_med"],
            "target_med_token_id": target_id,
            "visits": visits,
            "flat_tokens":   flat_tokens,
            "flat_domain":   flat_domain,
            "flat_day":      flat_day,
            "flat_delta":    flat_delta,
            "flat_visit_idx":flat_visit,
            "flat_numeric":  flat_numeric,
            "flat_num_mask": flat_num_mask,
        })
    return out, vocab


# ---------- persistence helpers ----------

def save_variant(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_variant(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)
