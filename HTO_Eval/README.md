# BP-Stability Evaluation Pipeline

A model-agnostic training + evaluation harness for the "BP stable post-HTO"
prediction task. This is a **skeleton**: it runs end-to-end right now on
synthetic data, and is built so that plugging in your real data and a real
git repo (and later, real sequence models) requires no changes to the
harness itself — only config.

## Task

One row per patient: `patient_id`, `sequence` (DX_*/MED_* tokens ordered by
visit, ending at the last observed HTO cutoff), `label` (1 if median SBP <
120 AND median DBP < 80 during the 1-90 day post-HTO window, else 0).
Class balance is ~72% label 0 / 28% label 1, which is why accuracy alone is
never trusted as a metric here.

## Quickstart (works today, on synthetic data)

```bash
pip install -r requirements.txt
python3 scripts/generate_synthetic_data.py       # writes sample_data/synthetic_patients.csv
PYTHONPATH=. python3 train.py --config configs/baseline_logistic_regression.yaml
```

That single command: builds (or reuses) the patient-level stratified
80/10/10 split, trains a TF-IDF + Logistic Regression baseline on train
only, optionally tunes the decision threshold on validation, evaluates
once on test, and writes a full run record under `runs/<experiment_id>/`.

Verify everything works with the smoke test:

```bash
PYTHONPATH=. python3 tests/test_pipeline_smoke.py
```

## Switching to your real data

1. Put your real dataframe (CSV/parquet/pickle, same three columns) anywhere.
2. Edit `configs/baseline_logistic_regression.yaml` (or copy it to a new
   config file): set `data_path` to your file and bump `data_version` to
   something meaningful (e.g. `2026-09-28_export` or a data hash).
3. **Delete `splits/` before the very first real-data run** if it still
   contains the synthetic split — otherwise `get_or_create_split` will try
   to reuse a split built from synthetic patient IDs, notice they don't
   match, and exclude everyone. After that first real run, never delete
   `splits/` again — it's the single source of truth every future model
   run must share.
4. Run `python3 train.py --config <your config>`.

## Switching to a real git repo

Once you `git init` (or clone this into an existing repo) and make a
commit, `train.py` automatically captures the real commit hash via
`bp_eval/tracking/env_capture.py::get_git_commit()` — no code change
needed. Until then, the run record honestly logs `git_commit: no-git-repo`
rather than a fake value. If you run with uncommitted changes, the commit
hash is suffixed `-DIRTY` so a run is never silently misattributed to a
clean commit it doesn't match.

## Adding a new model (LSTM, BERT, anything else)

1. Implement `bp_eval/models/base.py::BaseBPModel` (`fit`, `predict_proba`,
   `save`, `load`) in a new file under `bp_eval/models/`. Stubs with an
   implementation sketch already exist for `lstm` and `bert` in
   `lstm_stub.py` / `bert_stub.py` — fill those in, or add your own.
2. Register it in `bp_eval/models/__init__.py::MODEL_REGISTRY`.
3. Set `model_type: <your_key>` in a config file.

Nothing in `train.py`, the split logic, metrics, or tracking changes.

## Experiment tracking: CSV

Every run writes a full record to `runs/<experiment_id>/` plus one row in
`runs/registry.csv` (the master ledger). This needs zero extra
dependencies beyond `requirements.txt`. The tracker sits behind a small
`ExperimentTracker` interface (`bp_eval/tracking/base_tracker.py`), so a
different backend could be added later as a new implementation without
touching `train.py` — but there's currently only the one, `CSVTracker`.

## What every run saves, and why (mapped to the record-keeping requirement)

| Requirement | Where it lives |
|---|---|
| Unique experiment ID + timestamp | `run_meta.json`, `registry.csv` row |
| Code commit + exact run command | `run_meta.json` (`git_commit`, `run_command`) |
| Data version + split identifier | `data_version` tag + `split_id` (fingerprint of `splits/*_ids.json`) |
| Configuration + random seed | `config.json` (full resolved config, including model hyperparams and `seed`) |
| Environment/dependency reference | `environment.txt` (`pip freeze` snapshot at run time) |
| Metric name, value, unit | `metrics.json` (accuracy, precision/recall/F1 for label 1, macro-F1, balanced accuracy, ROC-AUC, PR-AUC, confusion matrix, each with a unit) |
| Warnings/failures/exclusions/unresolved questions | `run.log` (free-text) + `excluded_patients.csv` (patient_id + reason) |
| Model / predictions / figure / table / log (not a screenshot) | `model.joblib`, `predictions.csv`, `confusion_matrix.png` (+ optional `test_roc.png`/`test_pr.png`), `confusion_matrix.csv`, `run.log` |


## Design rules baked into the code (don't work around these)

- **Split is created once, then only ever reused.** `get_or_create_split`
  refuses to regenerate a split that already exists; it warns instead if
  the current dataset and the saved split have diverged.
- **Class weights are computed from `train_df` only**
  (`bp_eval/utils/class_weights.py`) — never pass `val_df`/`test_df`/the
  full frame into that function.
- **Threshold tuning only touches validation.**
  `find_best_threshold` is only ever called on `val_df` in `train.py`;
  `test_df` is scored exactly once, at the fixed/tuned threshold, and
  never used to pick anything.
- **Every load-time exclusion is recorded**, never silently dropped —
  see `bp_eval/data/loader.py::load_dataset`.

## Project layout

```
bp_eval/
  data/loader.py          # load + validate + record exclusions
  splits/split.py         # patient-level stratified 80/10/10, persisted
  models/                 # BaseBPModel + registry (LR baseline, LSTM/BERT stubs)
  evaluation/metrics.py   # accuracy, precision/recall/F1(1), macro-F1,
                           # balanced accuracy, ROC-AUC, PR-AUC, confusion matrix
  tracking/               # CSVTracker, behind a common ExperimentTracker interface
  utils/                  # seeding, class weights
train.py                  # the one script you run
configs/                  # one YAML per experiment
scripts/generate_synthetic_data.py
sample_data/synthetic_patients.csv   # delete once you have real data
tests/test_pipeline_smoke.py
runs/                      # created at run time — one folder per experiment_id
splits/                    # created on first run — the persisted patient split
```
