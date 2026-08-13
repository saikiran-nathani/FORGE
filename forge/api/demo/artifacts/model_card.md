# Model Card: random_forest

**Generated:** 2026-08-13 20:11 UTC

## Model Details
- **Model:** random_forest
- **Target:** `churned`
- **Task:** Predict which customers will churn so the retention team can intervene. Missing a churner costs about 5x a false alarm, and scoring must respond in under 50ms.
- **Type:** classification

## Training Data
- **Rows:** 1,600
- **Columns:** 9
- **Quality Score:** 96.7/100

## Performance Metrics
- **accuracy:** 0.9500
- **balanced_accuracy:** 0.9431
- **f1_macro:** 0.9364
- **f1_weighted:** 0.9504
- **precision_macro:** 0.9303
- **recall_macro:** 0.9431
- **mcc:** 0.8733
- **cohen_kappa:** 0.8728
- **roc_auc:** 0.9813
- **pr_auc:** 0.9649
- **brier_score:** 0.0441
- **log_loss:** 0.1552
- **ece:** 0.0486
- **f1:** 0.9070

## Performance vs Baseline
| Metric | Model | Baseline (majority-class / mean) |
|---|---|---|
| roc_auc | 0.9813 | 0.5000 |
| f1 | 0.9070 | 0.0000 |
| balanced_accuracy | 0.9431 | 0.5000 |
| accuracy | 0.9500 | 0.7375 |

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
Dataset quality score 96.7/100. Average missing rate 0.0%. Task: Predict which customers will churn so the retention team can intervene. Missing a churner costs about 5x a false alarm, and scoring must respond in under 50ms..
