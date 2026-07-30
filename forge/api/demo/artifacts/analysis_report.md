# FORGE Analysis Report

## Executive Summary
The pipeline trained **lightgbm** for task: _Predict which customers will churn_.
Primary metrics indicate strong performance.

## Model Metrics
- accuracy: 0.9469
- balanced_accuracy: 0.9295
- f1_macro: 0.9311
- f1_weighted: 0.9468
- precision_macro: 0.9328
- recall_macro: 0.9295
- mcc: 0.8623
- cohen_kappa: 0.8623
- roc_auc: 0.9746
- pr_auc: 0.9571
- brier_score: 0.0409
- log_loss: 0.1535
- ece: 0.0198
- f1: 0.8982

## Key Feature Insights
- **logins_last_30d** (SHAP: 2.0481)
- **tenure_months_x_monthly_charges** (SHAP: 1.0290)
- **monthly_charges_log** (SHAP: 0.7116)
- **tenure_months_ratio_monthly_charges** (SHAP: 0.6827)
- **tenure_months** (SHAP: 0.2403)

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
