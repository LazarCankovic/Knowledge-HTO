"""The experiment tracker: zero extra dependencies, plain files.

Every run gets its own folder under runs_dir/<experiment_id>/ holding
config.json, run_meta.json, environment.txt, metrics.json, the model,
predictions.csv, confusion matrix figure/table, run.log and
excluded_patients.csv. One row is also appended to runs_dir/registry.csv
as the master ledger (the "CSV template" Part 4 calls out directly).

Requires nothing beyond this repo's own requirements.txt.
"""
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from bp_eval.tracking.base_tracker import ExperimentTracker

REGISTRY_FIELDS = [
    "experiment_id",
    "timestamp",
    "run_name",
    "git_commit",
    "run_command",
    "data_version",
    "split_id",
    "seed",
    "model_type",
    "class_weighted",
    "decision_threshold",
    "accuracy",
    "precision_label1",
    "recall_label1",
    "f1_label1",
    "macro_f1",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "has_warnings",
    "run_dir",
]


class CSVTracker(ExperimentTracker):
    def __init__(self, runs_dir: str):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.runs_dir / "registry.csv"
        self._experiment_id = None
        self._run_name = None
        self._run_path = None
        self._params: Dict = {}
        self._metrics: Dict = {}
        self._units: Dict = {}
        self._tags: Dict = {}

    def start_run(self, experiment_id: str, run_name: Optional[str] = None) -> None:
        self._experiment_id = experiment_id
        self._run_name = run_name or experiment_id
        self._run_path = self.runs_dir / experiment_id
        self._run_path.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> str:
        if self._run_path is None:
            raise RuntimeError("start_run() must be called before run_dir is accessed.")
        return str(self._run_path)

    def log_params(self, params: Dict) -> None:
        self._params.update(params)
        with open(self._run_path / "config.json", "w") as f:
            json.dump(self._params, f, indent=2, default=str)

    def log_metrics(self, metrics: Dict, units: Optional[Dict] = None) -> None:
        self._metrics.update(metrics)
        if units:
            self._units.update(units)
        payload = dict(self._metrics)
        payload["units"] = self._units
        with open(self._run_path / "metrics.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def log_artifact(self, local_path: str) -> None:
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Cannot log artifact, file does not exist: {local_path}")
        dst = self._run_path / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    def set_tags(self, tags: Dict) -> None:
        self._tags.update(tags)
        with open(self._run_path / "tags.json", "w") as f:
            json.dump(self._tags, f, indent=2, default=str)

    def end_run(self) -> None:
        self._append_to_registry()

    def _append_to_registry(self) -> None:
        row = {field: "" for field in REGISTRY_FIELDS}
        row["experiment_id"] = self._experiment_id
        row["run_name"] = self._run_name
        row["run_dir"] = str(self._run_path)

        for key in ("timestamp", "git_commit", "run_command", "data_version",
                    "split_id", "seed", "model_type", "class_weighted"):
            if key in self._tags:
                row[key] = self._tags[key]
            elif key in self._params:
                row[key] = self._params[key]

        for key in ("decision_threshold", "accuracy", "precision_label1",
                    "recall_label1", "f1_label1", "macro_f1",
                    "balanced_accuracy", "roc_auc", "pr_auc"):
            if key in self._metrics:
                row[key] = self._metrics[key]

        row["has_warnings"] = self._tags.get("has_warnings", False)

        file_exists = self.registry_path.exists()
        with open(self.registry_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
