from __future__ import annotations

import json
import pickle
from pathlib import Path

from hto.data.lab_vital_tokenizer import build_lab_sequence, build_vital_sequence

TRAIN_PATH = "/fs/atipa/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.train.pkl"
OLD_VOCAB_PATH = "/fs/atipa/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.vocab.json"
NEW_VOCAB_PATH = "/fs/atipa/data/nsf-jg/hto/hto2/processed_data/full_80_20/allcodes_timeaware.vocab.json"


def extend_vocab(data: list[dict], vocab: list[str]) -> list[str]:
    """multidomain_visit vocab + the [LCODE]/[LVAL]/[VCODE]/[VVAL] code-value tokens seen in `data`.
    Existing ids are unchanged; new tokens are appended in sorted order."""
    vocab_set = set(vocab)

    # special tokens for lab/vital code-value format
    vocab_set.update(["[LCODE]", "[LVAL]", "[VCODE]", "[VVAL]"])

    for item in data:
        for visit in item["visits"]:
            lab_token_ids = visit["blocks"].get("lab", [])
            lab_values = visit["block_numeric"].get("lab", [])

            vit_token_ids = visit["blocks"].get("vital", [])
            vit_values = visit["block_numeric"].get("vital", [])

            lab_code_tokens = [vocab[t] for t in lab_token_ids if 0 <= t < len(vocab)]
            vit_code_tokens = [vocab[t] for t in vit_token_ids if 0 <= t < len(vocab)]

            lab_seq = build_lab_sequence(lab_code_tokens, lab_values)
            vit_seq = build_vital_sequence(vit_code_tokens, vit_values)

            vocab_set.update(lab_seq)
            vocab_set.update(vit_seq)

    new_vocab = list(vocab)
    seen = set(new_vocab)

    for tok in sorted(vocab_set):
        if tok not in seen:
            new_vocab.append(tok)
            seen.add(tok)

    return new_vocab


def main():
    with open(TRAIN_PATH, "rb") as f:
        data = pickle.load(f)

    with open(OLD_VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    new_vocab = extend_vocab(data, vocab)

    with open(NEW_VOCAB_PATH, "w") as f:
        json.dump(new_vocab, f)

    print("Saved:", NEW_VOCAB_PATH)
    print("Old vocab size:", len(vocab))
    print("New vocab size:", len(new_vocab))


if __name__ == "__main__":
    main()
