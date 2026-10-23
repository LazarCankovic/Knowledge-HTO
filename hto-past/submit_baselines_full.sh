#!/usr/bin/env python3
"""
collect_results.py
------------------
Reads predictions/full_80_20/<baseline>/metrics.json for every baseline
and prints a ranked markdown + TSV summary.

Usage (from hto2 root):
    python collect_results.py
    python collect_results.py --pred-dir predictions/full_80_20 --out results_full.tsv
"""

import argparse
import json
import os
import sys

BASELINES = [
    "medbert", "tlstm", "time2vec", "transformehr",
    "behrt", "cehrbert", "multimodal_behrt",
    "medbert_md", "tlstm_md", "time2vec_md", "transformehr_md",
    "behrt_md", "cehrbert_md", "multimodal_behrt_md",
    "mtan", "strats",
]

# keys we want from metrics.json (add/remove as needed)
METRIC_KEYS = ["auroc", "auprc", "accuracy", "f1", "brier", "ece"]


def load_metrics(pred_dir: str, baseline: str) -> dict | None:
    path = os.path.join(pred_dir, baseline, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(v, pct=False) -> str:
    if v is None:
        return "-"
    if pct:
        return f"{v:.1%}"
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="predictions/full_80_20",
                    help="path to predictions/<dataset> directory")
    ap.add_argument("--out", default=None,
                    help="optional TSV output path")
    args = ap.parse_args()

    rows = []
    missing = []

    for b in BASELINES:
        m = load_metrics(args.pred_dir, b)
        if m is None:
            missing.append(b)
            continue
        row = {"baseline": b}
        for k in METRIC_KEYS:
            # handle both lower and upper key names
            row[k] = m.get(k, m.get(k.upper(), None))
        rows.append(row)

    if not rows:
        print(f"No metrics.json files found under {args.pred_dir}.")
        print("Make sure the baselines have finished running.")
        sys.exit(1)

    # sort by auroc descending
    rows.sort(key=lambda r: r.get("auroc") or 0, reverse=True)

    # ── markdown table ─────────────────────────────────────────────────────────
    header = f"{'Baseline':<25} {'AUROC':>7} {'AUPRC':>7} {'ACC':>7} {'F1':>7} {'Brier':>7} {'ECE':>7}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['baseline']:<25}"
            f" {fmt(r.get('auroc')):>7}"
            f" {fmt(r.get('auprc')):>7}"
            f" {fmt(r.get('accuracy')):>7}"
            f" {fmt(r.get('f1')):>7}"
            f" {fmt(r.get('brier')):>7}"
            f" {fmt(r.get('ece')):>7}"
        )
    print(sep)

    if missing:
        print(f"\nNot yet finished (no metrics.json): {', '.join(missing)}")

    # ── TSV output ─────────────────────────────────────────────────────────────
    if args.out:
        import csv
        fields = ["baseline"] + METRIC_KEYS
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t",
                               extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in fields})
        print(f"\nTSV written to {args.out}")


if __name__ == "__main__":
    main()
