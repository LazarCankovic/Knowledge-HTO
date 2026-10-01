"""Evaluation metrics.

Accuracy alone is misleading against a 72/28 baseline (predicting all 0s
already scores ~72%), so every run reports the full set below plus the
confusion matrix, and callers are expected to look at f1_label1 / pr_auc /
balanced_accuracy, not accuracy, when judging a model.
"""
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe for servers/CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_UNITS = {
    "accuracy": "proportion_correct",
    "precision_label1": "proportion",
    "recall_label1": "proportion",
    "f1_label1": "f1_score",
    "macro_f1": "f1_score",
    "balanced_accuracy": "proportion",
    "roc_auc": "auc",
    "pr_auc": "average_precision",
    "decision_threshold": "probability_cutoff",
}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute the full metric set at a given decision threshold.

    Returns a dict of {metric_name: value} plus a nested confusion_matrix,
    ready to be written to metrics.json with METRIC_UNITS supplying the
    unit for each metric (Part 4: "metric name, value, and unit").
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_label1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_label1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_label1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "decision_threshold": float(threshold),
    }

    # ROC-AUC / PR-AUC need both classes present in y_true to be defined.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    metrics["confusion_matrix"] = {
        "labels": [0, 1],
        "matrix": cm.tolist(),  # rows = true, cols = predicted
    }
    metrics["units"] = METRIC_UNITS
    return metrics


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1_label1",
                         grid: Optional[np.ndarray] = None) -> float:
    """Sweep thresholds on VALIDATION data only and return the one that
    maximizes the chosen metric. Never call this with test data.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    best_threshold, best_score = 0.5, -1.0
    for t in grid:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        score = m.get(metric)
        if score is not None and score > best_score:
            best_score, best_threshold = score, float(t)
    return best_threshold


def plot_confusion_matrix(cm: np.ndarray, labels, out_path: str, title: str = "Confusion Matrix") -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, out_path_prefix: str) -> None:
    """Bonus diagnostic figures (not required by Part 4 but cheap and
    useful alongside roc_auc/pr_auc): saves <prefix>_roc.png and
    <prefix>_pr.png.
    """
    from sklearn.metrics import precision_recall_curve, roc_curve

    if len(np.unique(y_true)) != 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(f"{out_path_prefix}_roc.png", dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(f"{out_path_prefix}_pr.png", dpi=150)
    plt.close(fig)
