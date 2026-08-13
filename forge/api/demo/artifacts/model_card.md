# Model Card: lightgbm

**Generated:** 2026-08-13 19:49 UTC

## Model Details
- **Model:** lightgbm
- **Target:** `churned`
- **Task:** Predict which customers will churn
- **Type:** classification

## Training Data
- **Rows:** 1,600
- **Columns:** 9
- **Quality Score:** 96.7/100

## Performance Metrics
- **accuracy:** 0.9469
- **balanced_accuracy:** 0.9295
- **f1_macro:** 0.9311
- **f1_weighted:** 0.9468
- **precision_macro:** 0.9328
- **recall_macro:** 0.9295
- **mcc:** 0.8623
- **cohen_kappa:** 0.8623
- **roc_auc:** 0.9746
- **pr_auc:** 0.9571
- **brier_score:** 0.0409
- **log_loss:** 0.1535
- **ece:** 0.0198
- **f1:** 0.8982

## Performance vs Baseline
| Metric | Model | Baseline (majority-class / mean) |
|---|---|---|
| roc_auc | 0.9746 | 0.5000 |
| f1 | 0.8982 | 0.0000 |
| balanced_accuracy | 0.9295 | 0.5000 |
| accuracy | 0.9469 | 0.7375 |

## Features
- **Input columns:** 8
- **Engineered features:** 6

## Fairness Assessment
No sensitive attributes detected.

## Known Limitations
- None identified

## Intended Use
- Batch and real-time prediction for the task described above
- Decision support — not autonomous decision-making without human review

## Out of Scope
- Data distributions significantly different from training data
- Features not present during training
- Adversarial or manipulated inputs

## Ethical Considerations
Dataset quality score 96.7/100. Average missing rate 0.0%. Task: Predict which customers will churn.
