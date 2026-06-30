# FORGE Analysis Report

## Executive Summary
The pipeline trained **voting_ensemble** for task: _Predict which customers will churn_.
Primary metrics indicate strong performance.

## Model Metrics
- accuracy: 0.9375
- f1_macro: 0.9193
- f1_weighted: 0.9375
- precision_macro: 0.9193
- recall_macro: 0.9193
- mcc: 0.8386
- cohen_kappa: 0.8386
- roc_auc: 0.9766
- pr_auc: 0.9579
- brier_score: 0.0430
- log_loss: 0.1568
- ece: 0.0365
- f1: 0.8810

## Key Feature Insights
- **tenure_months** (SHAP: 0.1294)
- **monthly_charges** (SHAP: 0.1294)

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
