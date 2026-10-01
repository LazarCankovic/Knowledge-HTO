Final cohort: 38,364 patients

Split:
80% train
10% validation
10% test

Train:      30,690
Validation:  3,837
Test:        3,837

Split strategy:
patient-level stratified split

split_seed = 42

Frozen split ID:
6441f1d0b131

What each one of them does:
TRAIN
- fit vocabulary
- fit model
- calculate class weights

VALIDATION
- early stopping
- model selection
- decision-threshold tuning

TEST
- final evaluation only



METRICS:

Accuracy
Precision (label 1)
Recall (label 1)
F1 (label 1)
Macro-F1
Balanced Accuracy
ROC-AUC
Average Precision / PR-AUC
Confusion Matrix

Since label 1 represents around 28% of the cohort, we chose not to interpret accuracy only.
