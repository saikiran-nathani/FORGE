# FORGE Analysis Report

## Executive Summary
The pipeline trained **random_forest** for task: _Predict which customers will churn so the retention team can intervene. Missing a churner costs about 5x a false alarm, and scoring must respond in under 50ms._.
Primary metrics indicate strong performance.

## Model Metrics
- accuracy: 0.9500
- balanced_accuracy: 0.9431
- f1_macro: 0.9364
- f1_weighted: 0.9504
- precision_macro: 0.9303
- recall_macro: 0.9431
- mcc: 0.8733
- cohen_kappa: 0.8728
- roc_auc: 0.9813
- pr_auc: 0.9649
- brier_score: 0.0441
- log_loss: 0.1552
- ece: 0.0486
- f1: 0.9070

## Key Feature Insights
- **logins_last_30d** (SHAP: 0.2720)
- **tenure_months_x_monthly_charges** (SHAP: 0.0674)
- **monthly_charges_log** (SHAP: 0.0571)
- **tenure_months_ratio_monthly_charges** (SHAP: 0.0416)
- **tenure_months** (SHAP: 0.0344)

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
