# Model Card: random_forest

**Generated:** 2026-08-13 22:39 UTC

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
- **accuracy:** 0.9437
- **balanced_accuracy:** 0.9350
- **f1_macro:** 0.9284
- **f1_weighted:** 0.9442
- **precision_macro:** 0.9224
- **recall_macro:** 0.9350
- **mcc:** 0.8573
- **cohen_kappa:** 0.8569
- **roc_auc:** 0.9815
- **pr_auc:** 0.9650
- **brier_score:** 0.0440
- **log_loss:** 0.1563
- **ece:** 0.0442
- **f1:** 0.8953

## Performance vs Baseline
| Metric | Model | Baseline (majority-class / mean) |
|---|---|---|
| roc_auc | 0.9815 | 0.5000 |
| f1 | 0.8953 | 0.0000 |
| balanced_accuracy | 0.9350 | 0.5000 |
| accuracy | 0.9437 | 0.7375 |

## Features
- **Input columns:** 8
- **Engineered features:** 8

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
