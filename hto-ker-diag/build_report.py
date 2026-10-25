"""Aggregate per-run metrics.json files into one report.

Usage:
    python build_report.py [--root predictions] [--out reports/summary.md]

Scans predictions/<dataset>/<baseline>/metrics.json and produces:
    - reports/summary.tsv     (machine-readable)
    - reports/summary.md      (human-readable)
    - reports/auc_bar.png     (AUC by baseline × dataset)
    - reports/f1_bar.png      (F1 by baseline × dataset)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for bl_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            mf = bl_dir / "metrics.json"
            if not mf.exists():
                continue
            m = json.loads(mf.read_text())
            extra = m.get("extra", {})
            rows.append({
                "dataset": ds_dir.name,
                "baseline": bl_dir.name,
                "variant": extra.get("variant", "-"),
                "pe_parts": extra.get("pe_parts", "-"),
                "rel_day_bias": extra.get("rel_day_bias", "-"),
                "n_params": extra.get("n_params", None),
                "auc": m.get("auc"),
                "auprc": m.get("auprc"),
                "brier": m.get("brier"),
                "ece": m.get("ece"),
                "accuracy": m.get("accuracy"),
                "f1": m.get("f1"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "n": m.get("n"),
                "n_pos": m.get("n_pos"),
                "n_neg": m.get("n_neg"),
                "path": str(bl_dir),
            })
    return pd.DataFrame(rows)


def _bar(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(index="baseline", columns="dataset", values=metric)
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(pivot)), 4.2))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No runs found._\n"
    order = ["dataset", "baseline", "variant", "pe_parts", "rel_day_bias",
             "auc", "auprc", "brier", "ece", "accuracy", "f1",
             "n", "n_pos", "n_params"]
    cols = [c for c in order if c in df.columns]
    d = df[cols].copy()
    for c in ("auc", "auprc", "brier", "ece", "accuracy", "f1"):
        if c in d.columns:
            d[c] = d[c].map(lambda v: "-" if v is None or pd.isna(v) else f"{v:.3f}")
    return d.to_markdown(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="predictions")
    ap.add_argument("--out", default="reports/summary.md")
    args = ap.parse_args()

    root = Path(args.root)
    df = collect(root)
    reports_dir = Path(args.out).resolve().parent
    reports_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = reports_dir / "summary.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)

    _bar(df, "auc", reports_dir / "auc_bar.png", "Test AUC by baseline × dataset")
    _bar(df, "f1",  reports_dir / "f1_bar.png",  "Test F1 by baseline × dataset")

    md = "# HTO baseline results\n\n"
    md += f"_{len(df)} runs across {df['dataset'].nunique() if not df.empty else 0} datasets._\n\n"
    md += to_markdown(df) + "\n"
    Path(args.out).write_text(md)
    print(f"Wrote {args.out}, {tsv_path}")
    if not df.empty:
        print(df.groupby("dataset")["auc"].describe())


if __name__ == "__main__":
    main()
