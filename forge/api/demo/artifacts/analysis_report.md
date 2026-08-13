# FORGE Analysis Report

## Executive Summary
The pipeline trained **random_forest** for task: _Predict which customers will churn so the retention team can intervene. Missing a churner costs about 5x a false alarm, and scoring must respond in under 50ms._.
Primary metrics indicate strong performance.

## Model Metrics
- accuracy: 0.9437
- balanced_accuracy: 0.9350
- f1_macro: 0.9284
- f1_weighted: 0.9442
- precision_macro: 0.9224
- recall_macro: 0.9350
- mcc: 0.8573
- cohen_kappa: 0.8569
- roc_auc: 0.9815
- pr_auc: 0.9650
- brier_score: 0.0440
- log_loss: 0.1563
- ece: 0.0442
- f1: 0.8953

## Key Feature Insights
- **logins_last_30d** (SHAP: 0.2685)
- **tenure_months_x_monthly_charges** (SHAP: 0.0682)
- **monthly_charges_log1p** (SHAP: 0.0577)
- **tenure_months_ratio_monthly_charges** (SHAP: 0.0408)
- **tenure_months_log1p** (SHAP: 0.0355)

## Error Patterns
- No significant underperforming slices detected



## Recommendations
1. Review underperforming slices and consider targeted feature engineering
2. Monitor top SHAP features for drift in production
3. Validate fairness metrics if sensitive attributes are present
4. Retrain when data quality score drops below 80
5. Consider ensemble methods if single-model variance is high

## Deployment Considerations
- Serialize preprocessing pipeline alongside the model
- Set up drift monitoring on top 5 SHAP features
- Track latency for real-time inference requirements
