# Model Card: voting_ensemble

**Generated:** 2026-06-30 19:24 UTC

## Model Details
- **Model:** voting_ensemble
- **Target:** `churned`
- **Task:** Predict which customers will churn
- **Type:** classification

## Training Data
- **Rows:** 1,600
- **Columns:** 9
- **Quality Score:** 93.5/100

## Performance Metrics
- **accuracy:** 0.9375
- **f1_macro:** 0.9193
- **f1_weighted:** 0.9375
- **precision_macro:** 0.9193
- **recall_macro:** 0.9193
- **mcc:** 0.8386
- **cohen_kappa:** 0.8386
- **roc_auc:** 0.9766
- **pr_auc:** 0.9579
- **brier_score:** 0.0430
- **log_loss:** 0.1568
- **ece:** 0.0365
- **f1:** 0.8810

## Features
- **Input columns:** 8
- **Engineered features:** 3

## Fairness Assessment
No fairness issues flagged.

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
Dataset quality score 93.5/100. Average missing rate 0.0%. Task: Predict which customers will churn.
