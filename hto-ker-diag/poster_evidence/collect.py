"""Generate aggregate poster evidence on the HPC; never copies patient-level predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


RUNS = {
    "htr_former_ca": ("96731", "C1"),
    "htr_former_lab_kg": ("96730", "C2"),
    "htr_former_lab_kg_add_s42": ("96828", "C3"),
    "htr_former_kg": ("96829", "C4"),
}
METRICS = ("auc", "auprc", "f1", "accuracy", "brier", "ece", "precision", "recall")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def code_manifest(root: Path, out: Path) -> None:
    def git(*args: str) -> str:
        p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else "unavailable"

    files = sorted(root.rglob("*.py")) + [root / "requirements.txt"]
    excludes = {"predictions", "processed_data", "data", ".venv", "__pycache__", "reports"}
    lines = [
        "Source directory: " + str(root),
        "Git HEAD: " + git("rev-parse", "HEAD"),
        "Git branch: " + git("branch", "--show-current"),
        "Git status (including untracked source):",
        git("status", "--short"),
        "SHA256 of source files present when evidence was collected:",
    ]
    for p in files:
        if p.is_file() and not set(p.relative_to(root).parts).intersection(excludes):
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "poster_evidence").resolve()
    diag = out / "diagnostics"

    reports = {}
    evidence, errors, groups, stress, threshold_rows, paired, prototypes = ([] for _ in range(7))
    for name, (job, claim) in RUNS.items():
        rel = Path("predictions/full_storetime") / name / "metrics.json"
        path = root / rel
        if not path.exists():
            raise SystemExit(f"Missing {path}; collect from the HPC after the full run completes")
        reports[name] = json.loads(path.read_text())
        m = reports[name]
        if m.get("n") != 5754:
            raise SystemExit(f"Unexpected test sample count for {name}: {m.get('n')}")
        for key in METRICS:
            evidence.append(dict(claim_id=claim, run_id=job, model=name, dataset="full_storetime",
                                 split="test", metric=key.upper() if key in ("auc", "auprc", "f1", "ece") else key,
                                 value=m[key], n=m["n"], n_pos=m["n_pos"], n_neg=m["n_neg"],
                                 source_file=str(rel), evidence_precision="exact from metrics.json",
                                 limitations="One seed; exploratory test inspection; no repeat-seed inference"))
        extra = m.get("extra", {})
        evidence.append(dict(claim_id=claim, run_id=job, model=name, dataset="full_storetime",
                             split="validation", metric="AUROC_best", value=extra.get("val_best_auc"), n=5755,
                             n_pos="", n_neg="", source_file=str(rel), evidence_precision="exact from metrics.json",
                             limitations="Best checkpoint selected by validation AUROC"))
        cm = m["confusion"]
        for category, count in zip(("TN", "FP", "FN", "TP"), (cm[0][0], cm[0][1], cm[1][0], cm[1][1])):
            errors.append(dict(model=name, category=category, count=count, test_n=m["n"], threshold=m["threshold"]))
        for target, values in m.get("by_target", {}).items():
            # Tiny target groups are unstable and can reveal unnecessarily fine counts.
            if values["n"] >= 30 and values["n_pos"] >= 10 and values["n_neg"] >= 10:
                groups.append(dict(model=name, target_med=target, **values))
        for key, value in extra.get("kg_diagnostics", {}).items():
            stress.append(dict(model=name, diagnostic=key, value=value, source_file=str(rel)))

    # Clinical data stay on the HPC. Only aggregated categories and uncertainty leave this step.
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    predictions = {}
    for name in RUNS:
        path = root / "predictions/full_storetime" / name / "predictions.npz"
        if not path.exists():
            raise SystemExit(f"Missing {path}; do not substitute predictions from a different cohort")
        with np.load(path, allow_pickle=False) as p:
            ids = p["patient_ids"].astype(np.int64)
            if len(np.unique(ids)) != len(ids):
                raise SystemExit(f"Duplicate patient identifiers in {path}")
            predictions[name] = (ids, p["y_true"].astype(int), p["y_prob"].astype(float))

    ref_ids, ref_y, ref_prob = predictions["htr_former_ca"]
    for name, (ids, y, prob) in predictions.items():
        if set(ids) != set(ref_ids):
            raise SystemExit(f"Patient IDs differ for {name}; no paired analysis produced")
        order = np.argsort(ids)
        idx = np.searchsorted(ids[order], ref_ids)
        y, prob = y[order][idx], prob[order][idx]
        if not np.array_equal(y, ref_y):
            raise SystemExit(f"Labels differ for {name}; no paired analysis produced")
        predictions[name] = (ref_ids, y, prob)
        for th in (0.3, 0.5, 0.7):
            pred = prob >= th
            tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
            fn, tn = int(((~pred) & (y == 1)).sum()), int(((~pred) & (y == 0)).sum())
            threshold_rows.append(dict(model=name, threshold=th, n=len(y), TP=tp, FP=fp, FN=fn,
                                       TN=tn, f1=2 * tp / max(1, 2 * tp + fp + fn)))
        pred = prob >= 0.5
        for label, mask in (("TN", (y == 0) & ~pred), ("FP", (y == 0) & pred),
                            ("FN", (y == 1) & ~pred), ("TP", (y == 1) & pred)):
            if mask.any():
                q = np.quantile(prob[mask], (0.25, 0.5, 0.75))
                prototypes.append(dict(model=name, category=label, n=int(mask.sum()),
                                       p25=q[0], median=q[1], p75=q[2],
                                       description="Aggregate probability profile; no individual case data"))

    rng = np.random.default_rng(20261027)
    for name in RUNS:
        if name == "htr_former_ca":
            continue
        _, y, prob = predictions[name]
        for metric, func in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
            point = float(func(y, prob) - func(ref_y, ref_prob))
            deltas = []
            for _ in range(args.bootstrap):
                idx = rng.integers(0, len(y), len(y))
                if len(np.unique(y[idx])) == 2:
                    deltas.append(func(y[idx], prob[idx]) - func(ref_y[idx], ref_prob[idx]))
            if not deltas:
                raise SystemExit("No valid bootstrap resamples")
            lo, hi = np.quantile(deltas, (0.025, 0.975))
            paired.append(dict(model=name, baseline="htr_former_ca", metric=metric, delta=point,
                               ci_low=lo, ci_high=hi, n=len(y), resamples=len(deltas), seed=20261027,
                               limitations="Descriptive CI; test was inspected during model iteration; training-seed uncertainty omitted"))

    write_csv(out / "evidence_table.csv", ["claim_id", "run_id", "model", "dataset", "split", "metric", "value",
                                         "n", "n_pos", "n_neg", "source_file", "evidence_precision", "limitations"], evidence)
    write_csv(diag / "error_categories.csv", ["model", "category", "count", "test_n", "threshold"], errors)
    write_csv(diag / "subgroups_by_target.csv", ["model", "target_med", "n", "n_pos", "n_neg", "auc", "auprc", "accuracy", "f1"], groups)
    write_csv(diag / "graph_stress.csv", ["model", "diagnostic", "value", "source_file"], stress)
    write_csv(diag / "threshold_sensitivity.csv", ["model", "threshold", "n", "TP", "FP", "FN", "TN", "f1"], threshold_rows)
    write_csv(diag / "error_prototypes.csv", ["model", "category", "n", "p25", "median", "p75", "description"], prototypes)
    write_csv(diag / "paired_bootstrap.csv", ["model", "baseline", "metric", "delta", "ci_low", "ci_high", "n", "resamples", "seed", "limitations"], paired)
    code_manifest(root, out / "code_manifest.txt")
    print(f"Wrote aggregate evidence to {out}; predictions.npz stayed in {root / 'predictions'}")


if __name__ == "__main__":
    main()
