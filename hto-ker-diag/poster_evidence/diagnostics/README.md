# Diagnostic evidence

Run `python poster_evidence/collect.py` on the HPC from `hto-ker-diag/` after the four `metrics.json` and `predictions.npz` files exist. It writes aggregate-only files here:

- `error_categories.csv`: TP, FP, FN and TN counts for each model at the fixed 0.5 threshold.
- `subgroups_by_target.csv`: sample sizes and metrics per target medication with at least 30 test patients and at least 10 patients in each outcome class, read from `metrics.json`. Smaller groups are omitted, and even included groups are exploratory.
- `paired_bootstrap.csv`: same-patient differences from the plain baseline, with percentile 95% intervals from 2,000 paired patient resamples.
- `threshold_sensitivity.csv`: how error counts and F1 change at thresholds 0.3, 0.5 and 0.7. This is **diagnostic**, not a test-set threshold search.
- `graph_stress.csv`: measured KG coverage, KG-removal and time-removal prediction changes; the maximum probability change does not measure effect on AUROC.
- `error_prototypes.csv`: aggregate median and interquartile predicted probabilities within each error category. These are illustrative group summaries, **not individual clinical case examples**.

Clinical case examples with event histories have not been reviewed and are **pending**. Do not invent cases or upload raw patient rows or `predictions.npz` for the poster. If examples are required, review and prepare deidentified, approved examples in the authorized data environment, then document the selection rule and source separately.

The poster should report diagnostic limitations: one seed per model; the additive lab KG was tried after inspecting the gated model's test performance; only 6,958 patients had a lab-to-diagnosis path; model comparisons need fixed-split repeat-seed and graph-content controls. All claims about gain should use the `full_storetime` plain model, not the older `full` dataset.
