"""End-to-end data builder.

Steps:
  1. Build canonical records from mimic_data/*.parquet (or reuse cached pickle).
  2. Build the stable 85/15 patient split from the canonical records.
  3. For each variant, fit a vocab on TRAIN records and encode TRAIN + TEST.
  4. Persist per-variant pickles and vocabs to processed_data/.

Usage:
    python build_data.py --n-patients 500              # smoke-size build
    python build_data.py                               # full build

Output layout under processed_data/<tag>/:
    canonical.pkl
    split.json
    codes_visit.train.pkl
    codes_visit.test.pkl
    codes_visit.vocab.json
    codes_flat.{train,test}.pkl   codes_flat.vocab.json
    multimodal.{train,test}.pkl   multimodal.vocab.json
    multidomain_visit.{train,val,test}.pkl  multidomain_visit.vocab.json
    allcodes_timeaware.{train,val,test}.pkl allcodes_timeaware.vocab.json   (htr_former*, a1/a2)
    text_sequence.{train,test}.pkl text_sequence.vocab.json

Using the extraction output (extraction/MIMIC_IV_Extraction.ipynb):
    python build_data.py --tag demo --skip-text \
        --mimic-dir data/processed/mimic-iv-demo-2.2/mimic_data \
        --split-csv data/processed/mimic-iv-demo-2.2/tables/splits.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import hto.data.build as build_module
from hto.build_allcodes_vocab import extend_vocab
from hto.data.build import PROCESSED_DIR, build_canonical, load_canonical
from hto.data.splits import make_split, load_split, split_from_csv
from hto.data.variants import (
    Vocab,
    make_codes_flat,
    make_codes_visit,
    make_multidomain_visit,
    make_multimodal,
    make_text_sequence,
    save_variant,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None,
                    help="Subdir name under processed_data/. Default: smoke_<N> or full.")
    ap.add_argument("--reuse-canonical", action="store_true",
                    help="If the canonical pickle already exists, skip rebuilding.")
    ap.add_argument("--max-visits", type=int, default=64)
    ap.add_argument("--max-codes-per-visit", type=int, default=64)
    ap.add_argument("--max-flat-len", type=int, default=512)
    ap.add_argument("--max-text-len", type=int, default=512)
    ap.add_argument("--max-md-flat-len", type=int, default=1024,
                    help="Max flat length for the multidomain_visit variant.")
    ap.add_argument("--max-md-per-domain", type=int, default=32,
                    help="Max events per domain per visit in multidomain_visit.")
    ap.add_argument("--skip-text", action="store_true",
                    help="Skip text_sequence variant (it is large and optional).")
    ap.add_argument("--mimic-dir", type=str, default=None,
                    help="Folder with the final_*.parquet files. Default: mimic_data/ next to this script.")
    ap.add_argument("--split-csv", type=str, default=None,
                    help="CSV with patient_id,split (train/val/test) to use instead of a fresh 70/10/20 split, "
                         "e.g. the extraction's tables/splits.csv.")
    args = ap.parse_args()

    if args.mimic_dir:
        build_module.MIMIC_DIR = Path(args.mimic_dir).expanduser().resolve()
    MIMIC_DIR = build_module.MIMIC_DIR
    print(f"[source] {MIMIC_DIR}")

    tag = args.tag or (f"smoke_{args.n_patients}" if args.n_patients else "full")
    out_dir = PROCESSED_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = out_dir / "canonical.pkl"

    # 1. canonical
    if args.reuse_canonical and canonical_path.exists():
        print(f"[canonical] reusing {canonical_path}")
        records = load_canonical(canonical_path)
    else:
        print(f"[canonical] building -> {canonical_path}")
        records = build_canonical(n_patients=args.n_patients, out_path=canonical_path, seed=args.seed)
    print(f"[canonical] {len(records)} patients, mean visits={np.mean([len(r['visits']) for r in records]):.1f}")

    # 2. split
    split_path = out_dir / "split.json"
    if args.split_csv:
        split_from_csv(
            Path(args.split_csv).expanduser(),
            patient_ids=[r["patient_id"] for r in records],
            out_path=split_path,
        )
    else:
        make_split(
            patient_ids=[r["patient_id"] for r in records],
            labels=[r["label"] for r in records],
            out_path=split_path,
            seed=args.seed,
        )
    split = load_split(split_path)
    train_records = [r for r in records if r["patient_id"] in split["train_ids"]]
    val_records   = [r for r in records if r["patient_id"] in split["val_ids"]]
    test_records  = [r for r in records if r["patient_id"] in split["test_ids"]]
    print(f"[split] train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    # 3. variants

    print("[variant] codes_visit")
    cv_train, cv_vocab = make_codes_visit(
        train_records, vocab=None,
        max_visits=args.max_visits, max_codes_per_visit=args.max_codes_per_visit,
    )
    cv_test, _ = make_codes_visit(
        test_records, vocab=cv_vocab,
        max_visits=args.max_visits, max_codes_per_visit=args.max_codes_per_visit,
    )
    cv_val, _ = make_codes_visit(val_records, vocab=cv_vocab, max_visits=args.max_visits, max_codes_per_visit=args.max_codes_per_visit)
    save_variant(out_dir / "codes_visit.train.pkl", cv_train)
    save_variant(out_dir / "codes_visit.val.pkl", cv_val)
    save_variant(out_dir / "codes_visit.test.pkl", cv_test)
    cv_vocab.save(out_dir / "codes_visit.vocab.json")
    print(f"  vocab={len(cv_vocab)}")

    print("[variant] codes_flat")
    cf_train, cf_vocab = make_codes_flat(train_records, vocab=None, max_len=args.max_flat_len)
    cf_test, _ = make_codes_flat(test_records, vocab=cf_vocab, max_len=args.max_flat_len)
    cf_val, _ = make_codes_flat(val_records, vocab=cf_vocab, max_len=args.max_flat_len)
    save_variant(out_dir / "codes_flat.train.pkl", cf_train)
    save_variant(out_dir / "codes_flat.val.pkl", cf_val)
    save_variant(out_dir / "codes_flat.test.pkl", cf_test)
    cf_vocab.save(out_dir / "codes_flat.vocab.json")
    print(f"  vocab={len(cf_vocab)}")

    print("[variant] multimodal")
    mm_train, mm_vocab = make_multimodal(
        train_records, vocab=None,
        max_visits=args.max_visits, max_codes_per_visit=args.max_codes_per_visit,
    )
    mm_test, _ = make_multimodal(
        test_records, vocab=mm_vocab,
        max_visits=args.max_visits, max_codes_per_visit=args.max_codes_per_visit,
    )
    mm_val, _ = make_multimodal(val_records, vocab=mm_vocab, max_visits=args.max_visits, max_codes_per_visit=args.max_codes_per_visit)
    save_variant(out_dir / "multimodal.train.pkl", mm_train)
    save_variant(out_dir / "multimodal.val.pkl", mm_val)
    save_variant(out_dir / "multimodal.test.pkl", mm_test)
    mm_vocab.save(out_dir / "multimodal.vocab.json")
    print(f"  vocab={len(mm_vocab)}")

    print("[variant] multidomain_visit")
    md_train, md_vocab = make_multidomain_visit(
        train_records, vocab=None,
        max_visits=args.max_visits,
        max_per_domain=args.max_md_per_domain,
        max_flat_len=args.max_md_flat_len,
    )
    md_test, _ = make_multidomain_visit(
        test_records, vocab=md_vocab,
        max_visits=args.max_visits,
        max_per_domain=args.max_md_per_domain,
        max_flat_len=args.max_md_flat_len,
    )
    md_val, _ = make_multidomain_visit(val_records, vocab=md_vocab, max_visits=args.max_visits, max_per_domain=args.max_md_per_domain, max_flat_len=args.max_md_flat_len)
    save_variant(out_dir / "multidomain_visit.train.pkl", md_train)
    save_variant(out_dir / "multidomain_visit.val.pkl", md_val)
    save_variant(out_dir / "multidomain_visit.test.pkl", md_test)
    md_vocab.save(out_dir / "multidomain_visit.vocab.json")
    print(f"  vocab={len(md_vocab)}")

    # allcodes_timeaware = the multidomain_visit items + a vocab extended with the lab/vital
    # code-value tokens seen in TRAIN (previously done by hand with build_allcodes_vocab.py on the HPC)
    print("[variant] allcodes_timeaware")
    ac_vocab = Vocab(extend_vocab(md_train, md_vocab.id_to_token))
    save_variant(out_dir / "allcodes_timeaware.train.pkl", md_train)
    save_variant(out_dir / "allcodes_timeaware.val.pkl", md_val)
    save_variant(out_dir / "allcodes_timeaware.test.pkl", md_test)
    ac_vocab.save(out_dir / "allcodes_timeaware.vocab.json")
    print(f"  vocab={len(ac_vocab)}")

    if not args.skip_text:
        print("[variant] text_sequence")
        # load the big final_sequences and pull just the rows for our patients
        kept_ids = {r["patient_id"] for r in records}
        seq_df = pd.read_parquet(MIMIC_DIR / "final_sequences.parquet")
        seq_df = seq_df[seq_df["patient_id"].isin(kept_ids)]
        raw_seq = {int(pid): str(s) for pid, s in zip(seq_df["patient_id"], seq_df["sequence"])}
        ts_train, ts_vocab = make_text_sequence(train_records, raw_sequences=raw_seq, vocab=None, max_len=args.max_text_len)
        ts_test, _ = make_text_sequence(test_records, raw_sequences=raw_seq, vocab=ts_vocab, max_len=args.max_text_len)
        save_variant(out_dir / "text_sequence.train.pkl", ts_train)
        save_variant(out_dir / "text_sequence.test.pkl", ts_test)
        ts_vocab.save(out_dir / "text_sequence.vocab.json")
        print(f"  vocab={len(ts_vocab)}")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
