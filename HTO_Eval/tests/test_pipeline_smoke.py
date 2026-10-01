"""End-to-end smoke test: run the whole pipeline on synthetic data and
check the expected artifacts exist and the split is stable across runs.

Run with: PYTHONPATH=. python3 tests/test_pipeline_smoke.py
(kept dependency-free / not pytest-specific so it runs anywhere python3 runs)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, f"Command failed: {cmd}"
    return result.stdout


def main():
    # Clean slate
    for d in ["runs", "splits"]:
        shutil.rmtree(ROOT / d, ignore_errors=True)

    if not (ROOT / "sample_data" / "synthetic_patients.csv").exists():
        run([sys.executable, "scripts/generate_synthetic_data.py"])

    env = {"PYTHONPATH": "."}
    import os
    full_env = {**os.environ, **env}

    def run_train():
        result = subprocess.run(
            [sys.executable, "train.py", "--config", "configs/baseline_logistic_regression.yaml"],
            cwd=ROOT, capture_output=True, text=True, env=full_env,
        )
        print(result.stdout)
        assert result.returncode == 0, result.stderr
        return result.stdout

    out1 = run_train()
    out2 = run_train()

    split_id_1 = [l for l in out1.splitlines() if "split_id=" in l][0]
    split_id_2 = [l for l in out2.splitlines() if "split_id=" in l][0]
    assert split_id_1 == split_id_2, "Split was not reused across runs!"
    print("PASS: split reused across runs ->", split_id_1)

    runs_dir = ROOT / "runs"
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    assert len(run_dirs) == 2, f"Expected 2 run dirs, found {len(run_dirs)}"

    required_files = [
        "config.json", "run_meta.json", "environment.txt", "metrics.json",
        "predictions.csv", "confusion_matrix.png", "confusion_matrix.csv",
        "run.log", "excluded_patients.csv", "model.joblib", "tags.json",
    ]
    for rd in run_dirs:
        for fname in required_files:
            assert (rd / fname).exists(), f"Missing artifact {fname} in {rd}"
    print(f"PASS: all {len(required_files)} required artifacts present in both run dirs")

    registry_path = runs_dir / "registry.csv"
    assert registry_path.exists()
    with open(registry_path) as f:
        lines = f.readlines()
    assert len(lines) == 3, f"Expected header + 2 rows in registry.csv, got {len(lines)}"
    print("PASS: registry.csv has one row per run")

    metrics = json.loads((run_dirs[0] / "metrics.json").read_text())
    for key in ["accuracy", "precision_label1", "recall_label1", "f1_label1",
                "macro_f1", "balanced_accuracy", "roc_auc", "pr_auc", "confusion_matrix"]:
        assert key in metrics, f"Missing metric {key}"
    print("PASS: all required metrics present")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
