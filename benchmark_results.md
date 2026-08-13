# FORGE Benchmark

Reproducible run across **6 datasets** with all model families (classical + MLP / TabTransformer / **FT-Transformer**) under Optuna Bayesian (TPE) HPO.

- **Datasets:** 6  ·  **Model instances trained:** 94  ·  **Optuna trials:** 770  ·  **wall-clock:** 34.7 min
- **HPO budget:** 10 trials/model (deep models capped at 5)  ·  seed 42  ·  offline (bundled sklearn + synthetic + demo churn)
- Reproduce: `python scripts/benchmark.py --trials 10`

## Summary

| Dataset | Task | Rows | Feats | Best model | CV score | Metric | Models | Time (s) |
|---|---|--:|--:|---|--:|---|--:|--:|
| breast_cancer | binary_classification | 569 | 30 | **logistic_regression** | 0.9877 | f1 | 16 | 167.3 |
| wine | multiclass_classification | 178 | 13 | **knn** | 0.9933 | f1_macro | 16 | 132.1 |
| digits | multiclass_classification | 800 | 64 | **svm** | 0.9875 | f1_macro | 16 | 834.5 |
| diabetes | regression | 442 | 10 | **lasso** | 55.7497 | rmse | 15 | 249.6 |
| synthetic_reg | regression | 800 | 12 | **lasso** | 30.7307 | rmse | 15 | 501.7 |
| churn | binary_classification | 800 | 8 | **lightgbm** | 0.91 | f1 | 16 | 194.7 |

## Per-dataset leaderboards (top 5)

### breast_cancer — binary_classification
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | logistic_regression | 0.9877 | 0.19 |
| 2 | voting_ensemble | 0.986 | 0.81 |
| 3 | sgd | 0.9843 | 0.16 |
| 4 | stacking_ensemble | 0.9825 | 1.83 |
| 5 | lightgbm | 0.9772 | 0.45 |

### wine — multiclass_classification
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | knn | 0.9933 | 0.46 |
| 2 | extra_trees | 0.986 | 5.88 |
| 3 | naive_bayes | 0.9798 | 0.15 |
| 4 | voting_ensemble | 0.9798 | 6.56 |
| 5 | sgd | 0.9798 | 0.15 |

### digits — multiclass_classification
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | svm | 0.9875 | 1.48 |
| 2 | stacking_ensemble | 0.9815 | 13.62 |
| 3 | voting_ensemble | 0.9813 | 2.2 |
| 4 | catboost | 0.9732 | 0.34 |
| 5 | logistic_regression | 0.9674 | 0.21 |

### diabetes — regression
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | lasso | 55.7497 | 0.14 |
| 2 | elastic_net | 55.7768 | 0.13 |
| 3 | ridge | 55.8054 | 0.13 |
| 4 | sgd | 55.8387 | 0.13 |
| 5 | xgboost | 57.0751 | 0.53 |

### synthetic_reg — regression
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | lasso | 30.7307 | 0.14 |
| 2 | elastic_net | 30.7326 | 0.14 |
| 3 | ridge | 30.7342 | 0.14 |
| 4 | sgd | 30.7675 | 0.14 |
| 5 | mlp | 35.749 | 0.15 |

### churn — binary_classification
| # | Model | CV score | Latency (ms) |
|--:|---|--:|--:|
| 1 | lightgbm | 0.91 | 0.53 |
| 2 | voting_ensemble | 0.9003 | 9.57 |
| 3 | stacking_ensemble | 0.8988 | 46.91 |
| 4 | catboost | 0.8986 | 0.15 |
| 5 | extra_trees | 0.8914 | 8.62 |

## Reading the results

- **No single model wins everywhere.** Across 6 datasets the best model was one of **5 different algorithms** (knn, lasso, lightgbm, logistic_regression, svm) — the core case for comparing a broad panel per dataset rather than committing to one.
- **Simple models won the small, clean datasets** (linear / KNN / SVM); gradient-boosted trees topped only **1/6** (the larger, noisier set). Which family wins is itself the diagnostic about the data's shape.
- **Deep models (MLP / TabTransformer / FT-Transformer) were trained and compared on every dataset but did not top a classical model here** (best FT-Transformer rank across datasets: 10). Expected at this sample size — GBMs/classical dominate small tabular data, consistent with the tabular-DL literature.
- Latency spread across each leaderboard is the accuracy/latency trade-off the Pareto frontier surfaces.