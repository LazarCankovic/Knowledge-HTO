"""Paired repeated k-fold comparison of baselines on a small cohort (e.g. the 100-patient demo).

    python cv_compare.py --dataset demo --kg-tag demo --repeats 3 --folds 5 --epochs 5

Every config sees the same folds and seeds. Per fold: a vocab is fitted on the training patients
only, a stratified validation slice of them drives early stopping (train.py), and the test fold is
predicted once. Out-of-fold predictions are pooled, so each repeat gives one prediction per patient
and one AUC over the whole cohort. Differences vs the reference config get a paired patient-level
bootstrap 95% CI (on predictions averaged over repeats).

Configs are name=baseline[:run.py-style KG flags], e.g.
    kg=htr_former_kg   kg_no_time=htr_former_kg:--kg-time off   kg_rewired=htr_former_kg:--kg-tag demo_rewired

Outputs: reports/cv_<dataset>/{summary.md, summary.json, oof_predictions.csv}
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CONFIGS = [
    "baseline=htr_former_ca",
    "kg=htr_former_kg",
    "kg_add=htr_former_kg:--kg-fusion add",
    "kg_no_time=htr_former_kg:--kg-time off",
    "kg_tau_shuffled=htr_former_kg:--kg-tau shuffled",
    "kg_rewired=htr_former_kg:--kg-tag {kg}_rewired",
    "kg_isa=htr_former_kg:--kg-tag {kg}_isa",
]


def parse_config(spec: str, kg_tag: str) -> dict:
    name, rest = spec.split("=", 1)
    baseline, _, flags = rest.partition(":")
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-tag", default=kg_tag)
    ap.add_argument("--kg-layers", type=int, default=2)
    ap.add_argument("--kg-time", default="on", choices=["on", "off"])
    ap.add_argument("--kg-tau", default="real", choices=["real", "shuffled"])
    ap.add_argument("--kg-no-transr", action="store_true")
    ap.add_argument("--kg-fusion", default="gate", choices=["gate", "add"])
    ap.add_argument("--kg-scale-lr-mult", type=float, default=30.0)
    kg = ap.parse_args(shlex.split(flags.format(kg=kg_tag)))
    return {"name": name, "baseline": baseline, "kg": vars(kg)}


def model_kwargs(cfg: dict, dataset: str, d_model: int, entry) -> dict:
    import inspect
    extra = {"d_model": d_model}
    if "kg_patient_graphs" in inspect.signature(entry.cls.__init__).parameters:
        kg = cfg["kg"]
        graphs = ROOT / "processed_data" / dataset / f"patient_graphs.{kg['kg_tag']}.pkl"
        if not graphs.exists():
            raise SystemExit(f"missing {graphs}: run build_patient_graphs.py --kg-tag {kg['kg_tag']}")
        extra.update(
            kg_patient_graphs=str(graphs),
            kg_transr=None if kg["kg_no_transr"] else str(ROOT / "processed_data" / "kg" / kg["kg_tag"] / "transr.pt"),
            kg_layers=kg["kg_layers"], kg_time=kg["kg_time"] == "on", kg_tau_mode=kg["kg_tau"],
            kg_fusion=kg["kg_fusion"], kg_scale_lr_mult=kg["kg_scale_lr_mult"],
        )
    return extra


def run_fold(job: dict) -> dict:
    """One (config, repeat, fold) training run; returns test-fold predictions."""
    import torch
    from hto.build_allcodes_vocab import extend_vocab
    from hto.data import batch_builder_all_codes_local as bb
    from hto.data.build import load_canonical
    from hto.data.variants import Vocab, make_multidomain_visit
    from hto.registry import get as get_baseline
    from hto.train import TrainConfig, train_and_eval

    torch.set_num_threads(job["threads"])
    records = {r["patient_id"]: r for r in load_canonical(ROOT / "processed_data" / job["dataset"] / "canonical.pkl")}
    pick = lambda ids: [records[i] for i in ids]

    md_train, md_vocab = make_multidomain_visit(pick(job["train_ids"]))
    md_val, _ = make_multidomain_visit(pick(job["val_ids"]), vocab=md_vocab)
    md_test, _ = make_multidomain_visit(pick(job["test_ids"]), vocab=md_vocab)
    vocab = Vocab(extend_vocab(md_train, md_vocab.id_to_token))
    bb.set_vocab(vocab.id_to_token)

    cfg = job["config"]
    entry = get_baseline(cfg["baseline"])
    torch.manual_seed(job["seed"])
    model = entry.factory(len(vocab), model_kwargs(cfg, job["dataset"], job["d_model"], entry))
    tcfg = TrainConfig(epochs=job["epochs"], batch_size=job["batch_size"], lr=job["lr"],
                       seed=job["seed"], device="cpu")
    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        out = train_and_eval(model, md_train, md_test, tcfg, val_items=md_val)
    p = out["predictions"]
    return {"config": cfg["name"], "repeat": job["repeat"], "fold": job["fold"],
            "patient_id": p["patient_ids"].tolist(), "y_true": p["y_true"].tolist(),
            "y_prob": p["y_prob"].tolist(), "best_epoch": out["best_epoch"], "seconds": time.time() - start}


def make_jobs(args, configs, ids, labels) -> list[dict]:
    from sklearn.model_selection import StratifiedKFold, train_test_split
    jobs = []
    for rep in range(args.repeats):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + rep)
        for fold, (tr, te) in enumerate(skf.split(ids, labels)):
            tr_ids, va_ids = train_test_split(ids[tr], test_size=args.val_frac, stratify=labels[tr],
                                              random_state=args.seed + rep * 100 + fold)
            for cfg in configs:
                jobs.append({"config": cfg, "repeat": rep, "fold": fold, "dataset": args.dataset,
                             "train_ids": tr_ids.tolist(), "val_ids": va_ids.tolist(),
                             "test_ids": ids[te].tolist(), "seed": args.seed + rep * 100 + fold,
                             "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                             "d_model": args.d_model, "threads": args.threads})
    return jobs


def summarize(oof: pd.DataFrame, reference: str, n_boot: int, seed: int) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    rng = np.random.default_rng(seed)
    per_repeat = (oof.groupby(["config", "repeat"])
                  .apply(lambda g: pd.Series({"auc": roc_auc_score(g.y_true, g.y_prob),
                                              "auprc": average_precision_score(g.y_true, g.y_prob),
                                              "brier": brier_score_loss(g.y_true, g.y_prob)}),
                         include_groups=False)
                  .reset_index())
    mean_prob = oof.groupby(["config", "patient_id"]).agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
    ref = mean_prob.loc[reference]
    y = ref["y_true"].to_numpy()
    boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    boot_idx = [b for b in boot_idx if 0 < y[b].sum() < len(b)]

    def auc(yt, yp):
        return roc_auc_score(yt, yp)

    out = {}
    for name, g in per_repeat.groupby("config"):
        mp = mean_prob.loc[name].reindex(ref.index)
        row = {k: {"mean": float(g[k].mean()), "sd": float(g[k].std(ddof=1)) if len(g) > 1 else 0.0}
               for k in ("auc", "auprc", "brier")}
        boots = [auc(y[b], mp["y_prob"].to_numpy()[b]) for b in boot_idx]
        row["auc_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        if name != reference:
            d = [auc(y[b], mp["y_prob"].to_numpy()[b]) - auc(y[b], ref["y_prob"].to_numpy()[b]) for b in boot_idx]
            row["delta_auc_vs_ref"] = float(auc(y, mp["y_prob"]) - auc(y, ref["y_prob"]))
            row["delta_auc_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        out[name] = row
    return out


def write_report(summary: dict, args, n_patients: int, n_pos: int, seconds: float, out_dir: Path) -> None:
    lines = [
        f"# Paired {args.repeats}x{args.folds}-fold comparison on `{args.dataset}`",
        "",
        f"{n_patients} patients ({n_pos} controlled). Each fold: {args.epochs} epochs, d_model {args.d_model}, "
        f"early stopping on a {args.val_frac:.0%} validation slice of the training patients. "
        f"AUC per repeat on pooled out-of-fold predictions; CIs are a paired patient bootstrap "
        f"({args.n_boot} resamples) on repeat-averaged predictions. Total {seconds / 60:.1f} min.",
        "",
    ]
    if n_patients < 200:
        lines += [f"> **Pipeline dry run, not evidence.** With {n_patients} patients the CIs below span most of "
                  "the AUC range; no difference here says anything about whether the KG helps.", ""]
    lines += ["| config | AUC (mean ± sd over repeats) | AUC 95% CI | ΔAUC vs baseline [95% CI] | AUPRC | Brier |",
              "|---|---|---|---|---|---|"]
    for name, r in summary.items():
        delta = (f"{r['delta_auc_vs_ref']:+.3f} [{r['delta_auc_ci95'][0]:+.3f}, {r['delta_auc_ci95'][1]:+.3f}]"
                 if "delta_auc_vs_ref" in r else "reference")
        lines.append(f"| {name} | {r['auc']['mean']:.3f} ± {r['auc']['sd']:.3f} | "
                     f"[{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}] | {delta} | "
                     f"{r['auprc']['mean']:.3f} | {r['brier']['mean']:.3f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="processed_data/<dataset>/canonical.pkl")
    ap.add_argument("--kg-tag", required=True)
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from hto.data.build import load_canonical
    records = load_canonical(ROOT / "processed_data" / args.dataset / "canonical.pkl")
    ids = np.array([r["patient_id"] for r in records])
    labels = np.array([r["label"] for r in records])
    configs = [parse_config(c, args.kg_tag) for c in args.configs]
    if args.reference not in {c["name"] for c in configs}:
        raise SystemExit(f"--reference {args.reference!r} is not one of the configs")

    jobs = make_jobs(args, configs, ids, labels)
    print(f"{len(records)} patients ({labels.sum()} controlled) | {len(configs)} configs x "
          f"{args.repeats} repeats x {args.folds} folds = {len(jobs)} runs on {args.workers} workers")
    start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(run_fold, jobs), 1):
            results.append(res)
            if i % max(1, len(jobs) // 10) == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} runs done ({(time.time() - start) / 60:.1f} min)", flush=True)

    oof = pd.concat([pd.DataFrame({k: r[k] for k in ("patient_id", "y_true", "y_prob")})
                     .assign(config=r["config"], repeat=r["repeat"], fold=r["fold"]) for r in results])
    counts = oof.groupby(["config", "repeat"])["patient_id"].nunique()
    assert (counts == len(ids)).all(), "every patient must get exactly one out-of-fold prediction per repeat"

    out_dir = ROOT / "reports" / f"cv_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    summary = summarize(oof, args.reference, args.n_boot, args.seed)
    seconds = time.time() - start
    (out_dir / "summary.json").write_text(json.dumps(
        {"args": vars(args), "n_patients": len(ids), "n_controlled": int(labels.sum()),
         "best_epochs": {c: [r["best_epoch"] for r in results if r["config"] == c]
                         for c in summary}, "results": summary}, indent=2))
    write_report(summary, args, len(ids), int(labels.sum()), seconds, out_dir)
    print((out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
