# Review of the full_storetime poster package

Four exact 5,754-patient test results were checked against the uploaded evidence table. Positive cases: 1,616; negative cases: 4,138.

| Model | Validation AUROC | Test AUROC | Test AUPRC | Test F1 | Accuracy | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain CA | 0.709828 | 0.707960 | 0.495953 | 0.503678 | 0.648245 | 0.215724 |
| Lab KG, gated | 0.708301 | 0.705628 | 0.505412 | 0.496177 | 0.690824 | 0.205773 |
| Lab KG, additive | 0.706973 | 0.717433 | 0.511801 | 0.517385 | 0.638165 | 0.223807 |
| Diagnosis KG | 0.702076 | 0.706198 | 0.504899 | 0.505549 | 0.620612 | 0.228301 |

The additive lab test AUROC exceeds plain CA by 0.009473. Its paired patient bootstrap percentile 95% interval is [0.001888, 0.016724] with 2,000 resamples. The diagnosis KG difference is -0.001762, interval [-0.009533, 0.006351]. These intervals are descriptive: the additive run followed inspection of gated lab test results, and no repeat-seed or graph-content control was performed. Baseline validation AUROC was higher than additive lab validation AUROC. Do not claim confirmed superiority for either KG model.

The additive lab model's max probability change when KG was removed was 0.03477; the diagnosis KG's was 0.15581; the gated lab model's was 0.0000504. This shows KG affects predictions in the first two models, not that graph content improves outcome performance. Time-removal effects were much smaller.

This reviewed copy removed target-medication groups with fewer than 30 test patients from the shareable subgroup CSV. There were 330 target groups per model, 306 below 30 (244 below 5); 23 per model remain. These subgroup analyses are exploratory. Clinical case examples and explicit shuffled-graph stress tests are still pending. The included code snapshot covers the training, graph and BigQuery export Python files, but omits the locally modified extraction notebook. `code_manifest.txt` records the HPC source at collection time; the reviewed `collect.py` adds the n>=30 filtering after collection (SHA256: 05c77a45b1e0b076ce59d20dde1b3db95200d0ffe36cbbafcb6c5168efb9c30c).
