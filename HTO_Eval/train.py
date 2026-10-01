#!/usr/bin/env python3
"""Main entrypoint: split -> fit -> (optional threshold tuning) -> eval -> log.

Usage:
    python train.py --config configs/baseline_logistic_regression.yaml
    python train.py --config configs/baseline_logistic_regression.yaml --model lstm
    python train.py --list-models

This is the only script you run. Point config['data_path'] at your real
dataset and config['model_type'] at any model in MODEL_REGISTRY, and the
same script produces a fully-logged run every time -- nothing else in the
project needs to change between a Logistic Regression baseline and a
future LSTM/BERT run.

`--model` overrides whatever model_type is set in the config file, so you
can evaluate one specific model (e.g. one you just added to bp_eval/models/)
without editing or duplicating a config each time.
"""
import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bp_eval.data.loader import load_dataset
from bp_eval.evaluation.metrics import compute_metrics, find_best_threshold, plot_confusion_matrix, plot_roc_pr_curves
from bp_eval.models import MODEL_REGISTRY, build_model
from bp_eval.splits.split import apply_split, get_or_create_split
from bp_eval.tracking.env_capture import capture_environment, get_git_commit, get_run_command
from bp_eval.tracking.factory import get_tracker
from bp_eval.utils.class_weights import maybe_compute_class_weights
from bp_eval.utils.seed import set_global_seed


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_experiment_id(model_type: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{model_type}_{ts}_{short_uuid}"


def main(config_path: str, model_override: str = None) -> None:
    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    split_seed = int(config.get("split_seed", 42))

    set_global_seed(seed)

    if model_override is not None:
        if model_override not in MODEL_REGISTRY:
            raise ValueError(
                f"--model '{model_override}' is not registered. Available: "
                f"{list(MODEL_REGISTRY.keys())}. Register new models in "
                f"bp_eval/models/__init__.py::MODEL_REGISTRY."
            )
        config["model_type"] = model_override
        print(f"[run] model_type overridden via --model -> {model_override}")

    model_type = config["model_type"]
    experiment_id = make_experiment_id(model_type)
    run_command = get_run_command()
    git_commit = get_git_commit(config.get("repo_dir", "."))
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"[run] experiment_id={experiment_id}")
    print(f"[run] git_commit={git_commit}")
    print(f"[run] command={run_command}")

    tracker = get_tracker(config)
    tracker.start_run(experiment_id=experiment_id, run_name=config.get("run_name", experiment_id))
    run_dir = Path(tracker.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(msg: str) -> None:
        print(f"[run] {msg}")
        log_lines.append(msg)

    # ---- 1. Load + validate data ----
    load_result = load_dataset(config["data_path"])
    df = load_result.df
    for w in load_result.warnings:
        log(f"WARNING: {w}")

    excluded_path = run_dir / "excluded_patients.csv"
    load_result.excluded.to_csv(excluded_path, index=False)

    data_version = config.get("data_version", Path(config["data_path"]).name)

    # ---- 2. Split (created once, reused forever after) ----
    split = get_or_create_split(
        df,
        split_dir=config.get("split_dir", "splits"),
        val_size=config.get("val_size", 0.10),
        test_size=config.get("test_size", 0.10),
        random_state=split_seed,
    )
    train_df, val_df, test_df = apply_split(df, split)
    log(f"split_id={split.split_id} | n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            "One of train/val/test is empty after applying the saved split to "
            "the current dataset. Check that config['data_path'] matches the "
            "dataset the split was created from."
        )

    # ---- 3. Class weights (train split only) ----
    class_weight = maybe_compute_class_weights(train_df, enabled=config.get("class_weighted", False))
    if class_weight is not None:
        log(f"class_weighted=True, weights from train only: {class_weight}")

    # ---- 4. Fit model (train only; val used internally for early stopping if the model supports it) ----
    model = build_model(model_type, **config.get("model_params", {}))
    model.fit(train_df, val_df=val_df, class_weight=class_weight)

    # ---- 5. Decision threshold: fixed, or tuned on val only ----
    if config.get("tune_threshold", False):
        val_probs = model.predict_proba(val_df)
        threshold = find_best_threshold(
            val_df["label"].values, val_probs, metric=config.get("threshold_metric", "f1_label1")
        )
        log(f"decision_threshold tuned on validation set -> {threshold:.3f}")
    else:
        threshold = float(config.get("decision_threshold", 0.5))
        log(f"decision_threshold fixed at {threshold}")

    # ---- 6. Evaluate on test set ONCE ----
    test_probs = model.predict_proba(test_df)
    metrics = compute_metrics(test_df["label"].values, test_probs, threshold=threshold)
    units = metrics.pop("units")
    cm_info = metrics.pop("confusion_matrix")

    log(f"TEST METRICS: { {k: v for k, v in metrics.items() if k != 'decision_threshold'} }")

    # ---- 7. Save artifacts ----
    predictions_path = run_dir / "predictions.csv"
    test_df[["patient_id", "label"]].assign(
        y_prob=test_probs, y_pred=(test_probs >= threshold).astype(int)
    ).rename(columns={"label": "y_true"}).to_csv(predictions_path, index=False)

    import numpy as np
    cm_array = np.array(cm_info["matrix"])

    cm_png_path = run_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm_array, cm_info["labels"], str(cm_png_path))

    cm_csv_path = run_dir / "confusion_matrix.csv"
    with open(cm_csv_path, "w") as f:
        f.write(",".join([""] + [f"pred_{l}" for l in cm_info["labels"]]) + "\n")
        for label, row in zip(cm_info["labels"], cm_info["matrix"]):
            f.write(",".join([f"true_{label}"] + [str(v) for v in row]) + "\n")

    if config.get("save_roc_pr_curves", True):
        plot_roc_pr_curves(test_df["label"].values, test_probs, str(run_dir / "test"))

    model_path = run_dir / f"model.{('joblib' if model_type == 'logistic_regression' else 'bin')}"
    model.save(str(model_path))

    env_path = run_dir / "environment.txt"
    env_reference = capture_environment(str(env_path))

    # ---- 8. Log everything to the tracker ----
    full_config = dict(config)
    full_config["resolved_seed"] = seed
    full_config["resolved_split_seed"] = split_seed
    full_config["model_hyperparams"] = model.get_params()
    tracker.log_params(full_config)

    metrics_for_tracker = dict(metrics)
    metrics_for_tracker["confusion_matrix"] = cm_info
    tracker.log_metrics(metrics_for_tracker, units=units)

    tracker.set_tags(
        {
            "timestamp": timestamp,
            "git_commit": git_commit,
            "run_command": run_command,
            "data_version": data_version,
            "split_id": split.split_id,
            "seed": seed,
            "model_type": model_type,
            "class_weighted": bool(config.get("class_weighted", False)),
            "env_reference": env_reference,
            "has_warnings": bool(load_result.warnings),
        }
    )

    run_meta = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "run_command": run_command,
        "data_version": data_version,
        "split_id": split.split_id,
        "seed": seed,
        "split_seed": split_seed,
        "env_reference": env_reference,
    }
    run_meta_path = run_dir / "run_meta.json"
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    log_lines.append(f"n_excluded_patients={len(load_result.excluded)}")
    if len(load_result.excluded) > 0:
        log_lines.append("see excluded_patients.csv for patient_id + reason")
    run_log_path = run_dir / "run.log"
    run_log_path.write_text("\n".join(log_lines) + "\n")

    for path in [predictions_path, cm_png_path, cm_csv_path, model_path,
                 env_path, run_meta_path, run_log_path, excluded_path]:
        tracker.log_artifact(str(path))

    tracker.end_run()

    print(f"\n[run] DONE. experiment_id={experiment_id}")
    print(f"[run] artifacts written to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train + evaluate a BP-stability model.")
    parser.add_argument("--config", required=False, help="Path to a YAML config file.")
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help=(
            "Optional: name of a model in bp_eval/models/MODEL_REGISTRY to run "
            "instead of the config's model_type (e.g. --model lstm). Useful for "
            "running just the one model you added without editing the config."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print every model_type registered in MODEL_REGISTRY and exit.",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Registered models (usable as model_type in a config, or via --model):")
        for name in MODEL_REGISTRY:
            print(f"  - {name}")
        sys.exit(0)

    if not args.config:
        parser.error("--config is required unless --list-models is passed.")

    main(args.config, model_override=args.model)
