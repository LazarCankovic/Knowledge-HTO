"""Shared evaluation: metrics, plots, raw prediction persistence."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error — standard equal-width binning."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if hi == 1.0:
            m = m | (y_prob == 1.0)
        if m.sum() == 0:
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    target_meds: np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    both_classes = len(np.unique(y_true)) > 1
    auc   = float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan")
    auprc = float(average_precision_score(y_true, y_prob)) if both_classes else float("nan")
    brier = float(brier_score_loss(y_true, y_prob))
    ece   = _ece(y_true, y_prob)

    out = {
        "auc": auc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    if target_meds is not None:
        target_meds = np.asarray(target_meds)
        by_target: dict[str, dict] = {}
        for tm in sorted(set(map(str, target_meds.tolist()))):
            sel = target_meds == tm
            if sel.sum() == 0:
                continue
            y_t = y_true[sel]
            y_p = y_prob[sel]
            if len(np.unique(y_t)) > 1:
                sub_auc   = float(roc_auc_score(y_t, y_p))
                sub_auprc = float(average_precision_score(y_t, y_p))
            else:
                sub_auc = sub_auprc = float("nan")
            by_target[tm] = {
                "n": int(sel.sum()),
                "n_pos": int(y_t.sum()),
                "n_neg": int(sel.sum() - y_t.sum()),
                "auc": sub_auc,
                "auprc": sub_auprc,
                "accuracy": float(accuracy_score(y_t, (y_p >= threshold).astype(int))),
                "f1": float(f1_score(y_t, (y_p >= threshold).astype(int), zero_division=0)),
            }
        out["by_target"] = by_target
    return out


def save_predictions(
    out_dir: Path,
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    logits: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.npz"
    np.savez(
        path,
        patient_ids=np.asarray(patient_ids, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=np.int64),
        logits=np.asarray(logits, dtype=np.float32),
        y_prob=np.asarray(y_prob, dtype=np.float32),
        y_pred=np.asarray(y_pred, dtype=np.int64),
    )
    return path


def plot_confusion(cm: list[list[int]], out_path: Path, title: str) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(rec, prec, label=f"AUPRC={ap:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_prob_hist(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="true 0")
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="true 1")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_report_dir(
    out_dir: Path,
    run_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: np.ndarray,
    logits: np.ndarray,
    extra: dict | None = None,
    target_meds: np.ndarray | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_prob, target_meds=target_meds)
    if extra:
        metrics["extra"] = extra
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    save_predictions(out_dir, patient_ids, y_true, logits, y_prob, y_pred)
    plot_confusion(metrics["confusion"], out_dir / "confusion.png", f"{run_name} — confusion")
    plot_roc(y_true, y_prob, out_dir / "roc.png", f"{run_name} — ROC")
    plot_pr(y_true, y_prob, out_dir / "pr.png", f"{run_name} — PR")
    plot_prob_hist(y_true, y_prob, out_dir / "prob_hist.png", f"{run_name} — probs")
    return metrics
