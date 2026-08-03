# FORGE Benchmark

Reproducible run across **6 datasets** with all model families (classical + MLP / TabTransformer / **FT-Transformer**) under Optuna Bayesian (TPE) HPO.

- **Datasets:** 6  ·  **Model instances trained:** 94  ·  **Optuna trials:** 770  ·  **wall-clock:** 33.9 min
- **HPO budget:** 10 trials/model (deep models capped at 5)  ·  seed 42  ·  offline (bundled sklearn + synthetic + demo churn)
- Reproduce: `python scripts/benchmark.py --trials 10`

## Summary

| Dataset | Task | Rows | Feats | Best model | CV score | Metric | Models | Time (s) |
|---|---|--:|--:|---|--:|---|--:|--:|
| breast_cancer | binary_classification | 569 | 30 | **logistic_regression** | 0.9877 | f1 | 16 | 132.0 |
| wine | multiclass_classification | 178 | 13 | **knn** | 0.9933 | f1_macro | 16 | 133.4 |
| digits | multiclass_classification | 800 | 64 | **svm** | 0.9859 | f1_macro | 16 | 837.9 |
| diabetes | regression | 442 | 10 | **lasso** | 55.7369 | rmse | 15 | 247.7 |
| synthetic_reg | regression | 800 | 12 | **lasso** | 30.571 | rmse | 15 | 507.0 |
| churn | binary_classification | 800 | 8 | **catboost** | 0.8989 | f1 | 16 | 175.1 |

## Per-dataset leaderboards (top 5)

### breast_cancer — binary_classification
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | logistic_regression | 0.9877 | 0.15 |
| 2 | voting_ensemble | 0.9876 | 0.52 |
| 3 | sgd | 0.9859 | 0.15 |
| 4 | stacking_ensemble | 0.9789 | 1.49 |
| 5 | catboost | 0.9771 | 0.13 |

### wine — multiclass_classification
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | knn | 0.9933 | 0.42 |
| 2 | extra_trees | 0.986 | 5.84 |
| 3 | stacking_ensemble | 0.986 | 7.71 |
| 4 | lightgbm | 0.9853 | 0.46 |
| 5 | naive_bayes | 0.9798 | 0.14 |

### digits — multiclass_classification
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | svm | 0.9859 | 1.47 |
| 2 | stacking_ensemble | 0.983 | 15.12 |
| 3 | voting_ensemble | 0.9829 | 2.57 |
| 4 | catboost | 0.9767 | 0.5 |
| 5 | logistic_regression | 0.9723 | 0.19 |

### diabetes — regression
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | lasso | 55.7369 | 0.13 |
| 2 | elastic_net | 55.7604 | 0.13 |
| 3 | ridge | 55.7925 | 0.13 |
| 4 | sgd | 55.8374 | 0.13 |
| 5 | xgboost | 57.0751 | 0.55 |

### synthetic_reg — regression
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | lasso | 30.571 | 0.14 |
| 2 | elastic_net | 30.573 | 0.14 |
| 3 | ridge | 30.5754 | 0.14 |
| 4 | sgd | 30.6058 | 0.14 |
| 5 | mlp | 35.3943 | 0.15 |

### churn — binary_classification
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | catboost | 0.8989 | 0.14 |
| 2 | voting_ensemble | 0.8989 | 9.62 |
| 3 | lightgbm | 0.8974 | 0.46 |
| 4 | stacking_ensemble | 0.8931 | 49.96 |
| 5 | extra_trees | 0.8859 | 9.24 |

## Reading the results

- **No single model wins everywhere.** Across 6 datasets the best model was one of **5 different algorithms** (catboost, knn, lasso, logistic_regression, svm) — the core case for comparing a broad panel per dataset rather than committing to one.
- **Simple models won the small, clean datasets** (linear / KNN / SVM); gradient-boosted trees topped only **1/6** (the larger, noisier set). Which family wins is itself the diagnostic about the data's shape.
- **Deep models (MLP / TabTransformer / FT-Transformer) were trained and compared on every dataset but did not top a classical model here** (best FT-Transformer rank across datasets: 10). Expected at this sample size — GBMs/classical dominate small tabular data, consistent with the tabular-DL literature.
- Latency spread across each leaderboard is the accuracy/latency trade-off the Pareto frontier surfaces.