"""Single-run CLI: train a baseline on a dataset variant, save predictions + metrics.

Usage:
    python run.py --baseline medbert --dataset smoke_200
    python run.py --baseline multimodal_behrt --dataset smoke_200 --epochs 5

Expects processed_data/<dataset>/ to have been built via build_data.py. The
baseline's required variant is discovered from registry and the matching
{<variant>.train.pkl, <variant>.test.pkl, <variant>.vocab.json} files are
loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hto.data.variants import Vocab, load_variant
from hto.eval.metrics import write_report_dir
from hto.registry import get as get_baseline
from hto.train import TrainConfig, train_and_eval


def detect_device(override: str | None) -> str:
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "backends") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _factory_params(entry) -> set[str]:
    """Return the kwarg names that entry.cls.__init__ accepts."""
    import inspect
    try:
        return set(inspect.signature(entry.cls.__init__).parameters.keys())
    except Exception:
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--baseline", required=True)
    ap.add_argument("--dataset", required=True,
                    help="Subdir under processed_data/")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)

    # NEW
    ap.add_argument("--time-fn", type=str, default="log",
                    choices=["log", "exp_30", "exp_90", "exp_180"],
                    help="Temporal weighting function for TransformEHR.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--max-train-batches", type=int, default=None,
                    help="Cap steps per epoch (smoke tests).")

    ap.add_argument("--max-eval-batches", type=int, default=None)

    ap.add_argument("--out-root", type=str, default=None,
                    help="Parent dir for predictions. Default: predictions/")

    ap.add_argument("--pe-parts", type=str, default=None,
                    help="CompositePE ablation: comma-separated subset of "
                         "{abs,gap,domain,interaction}. Empty string disables all. "
                         "Only used by *_md / mtan / strats baselines.")

    ap.add_argument("--rel-day-bias", type=str, default=None,
                    choices=["on", "off"],
                    help="Relative-day attention bias. Only used by *_md baselines.")

    ap.add_argument("--run-suffix", type=str, default=None,
                    help="Optional suffix appended to predictions/<dataset>/<baseline><suffix>/")

    args = ap.parse_args()

    entry = get_baseline(args.baseline)
    variant = entry.variant

    data_dir = ROOT / "processed_data" / args.dataset

    if not data_dir.exists():
        raise SystemExit(
            f"Dataset dir not found: {data_dir}. Run build_data.py first."
        )

    train_path = data_dir / f"{variant}.train.pkl"
    val_path   = data_dir / f"{variant}.val.pkl"
    test_path  = data_dir / f"{variant}.test.pkl"
    vocab_path = data_dir / f"{variant}.vocab.json"

    for p in (train_path, test_path, vocab_path):
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

    vocab = Vocab.load(vocab_path)
    train_items = load_variant(train_path)
    val_items   = load_variant(val_path) if val_path.exists() else None
    test_items  = load_variant(test_path)

    print(
        f"[data] variant={variant} "
        f"train={len(train_items)} "
        f"test={len(test_items)} "
        f"vocab={len(vocab)}"
    )

    # model-specific kwargs
    extra: dict = {"d_model": args.d_model}

    # NEW
    if args.baseline == "transformehr":
        extra["time_fn"] = args.time_fn

    if variant == "multimodal" and args.baseline == "multimodal_behrt":
        n_lab = len(train_items[0]["lab_keys"]) if "lab_keys" in train_items[0] else 12
        n_vit = len(train_items[0]["vital_keys"]) if "vital_keys" in train_items[0] else 3
        extra["n_lab"] = n_lab
        extra["n_vit"] = n_vit

    # PE / rel-day-bias flags
    accepted = _factory_params(entry)

    if args.pe_parts is not None and "pe_parts" in accepted:
        extra["pe_parts"] = args.pe_parts

    if args.rel_day_bias is not None and "rel_day_bias" in accepted:
        extra["rel_day_bias"] = (args.rel_day_bias == "on")

    model = entry.factory(len(vocab), extra)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"[model] {args.baseline} params={n_params:,}")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=detect_device(args.device),
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )

    print(f"[train] {asdict(cfg)}")

    t0 = time.time()

    out = train_and_eval(model, train_items, test_items, cfg, val_items=val_items)

    print(f"[done] total_sec={time.time()-t0:.1f}")

    out_root = Path(args.out_root) if args.out_root else (ROOT / "predictions")

    suffix = f"_{args.run_suffix}" if args.run_suffix else ""

    run_dir = out_root / args.dataset / f"{args.baseline}{suffix}"

    run_dir.mkdir(parents=True, exist_ok=True)

    preds = out["predictions"]

    extra_info = {
        "variant": variant,
        "n_params": n_params,
        "history": out["history"],
        "config": out["config"],
        "val_best_auc":  out.get("val_best_auc"),
        "best_epoch":    out.get("best_epoch"),
        "stopped_epoch": out.get("stopped_epoch"),
    }

    if "pe_parts" in extra:
        extra_info["pe_parts"] = extra["pe_parts"]

    if "rel_day_bias" in extra:
        extra_info["rel_day_bias"] = extra["rel_day_bias"]

    if "time_fn" in extra:
        extra_info["time_fn"] = extra["time_fn"]

    metrics = write_report_dir(
        run_dir,
        run_name=f"{args.baseline}/{args.dataset}{suffix}",
        y_true=preds["y_true"],
        y_prob=preds["y_prob"],
        patient_ids=preds["patient_ids"],
        logits=preds["logits"],
        target_meds=preds.get("target_meds"),
        extra=extra_info,
    )

    print(
        f"[save] {run_dir}/metrics.json  "
        f"auc={metrics['auc']:.3f}  "
        f"auprc={metrics['auprc']:.3f}  "
        f"f1={metrics['f1']:.3f}  "
        f"acc={metrics['accuracy']:.3f}  "
        f"brier={metrics['brier']:.3f}"
    )


if __name__ == "__main__":
    main()
