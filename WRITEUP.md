# FORGE — Engineering Writeup

*A reference for talking about this project: the non-obvious decisions, the trade-offs,
and the parts that were actually hard. Each section is something you can defend in a
conversation.*

FORGE turns a raw CSV plus a natural-language goal into a deployed prediction API —
profiling, feature engineering, training/tuning 14+ model families, explainability,
and a live serving endpoint. ~5K LOC Python + a React/TypeScript dashboard.

---

## 1. One HPO loop across very different model families

**Problem.** Logistic regression, random forests, XGBoost/LightGBM/CatBoost, and
PyTorch MLP/TabTransformer have nothing in common in their APIs or hyperparameters.
Tuning them with separate code paths would be unmaintainable.

**Approach.** A single `BaseModel` interface exposes `get_search_space(trial)` and
`build_model(params)`; one Optuna optimizer drives all of them with the same
cross-validated objective, and records CV score *and* inference latency per model.

**Trade-off.** The abstraction adds a layer, but it means adding a 15th model is one
small class, and the HPO/selection/evaluation code never changes.

## 2. Model selection on a Pareto frontier, not "highest accuracy"

**Problem.** The most accurate model is often the wrong one to ship — a 0.3% accuracy
gain that costs 200× the inference latency is a bad production trade.

**Approach.** After training, FORGE computes the accuracy-vs-latency **Pareto frontier**
and surfaces which models are Pareto-optimal, so the choice is explicit rather than
blindly "max CV score."

**Talking point.** This is the difference between a Kaggle mindset and a production
mindset, and it's visible in the demo leaderboard (RandomForest is accurate but ~26 ms
vs. sub-millisecond for the boosters).

## 3. Running LLM-generated feature code without getting burned

**Problem.** The LLM proposes pandas feature-engineering code. Running arbitrary
generated code is a real risk (data exfiltration, target leakage, runaway loops).

**Approach.** A validation layer (`CodeSandbox`) parses the code with `ast`, blocks
imports/calls outside an allow-list, forbids referencing the target column (leakage),
and validates the output (row count preserved, no all-NaN/constant columns). Heuristic
fallbacks run when no LLM key is configured, so the pipeline never hard-depends on an API.

**Honest framing (this is the strongest interview answer).** It's a *validation layer,
not a true sandbox.* Python denylists are bypassable via attribute traversal, and the
timeout is checked after execution rather than interrupting it. I know those limits; for
real production I'd execute in a subprocess with `resource` limits or a container. Showing
you understand *why* your own mitigation is incomplete reads far better than claiming it's
airtight.

## 4. Replaying training transforms at inference (a real bug I fixed)

**Problem.** Serving a prediction means reproducing the *exact* training pipeline —
median/most-frequent imputation, outlier clipping, one-hot encoding, scaling, and feature
selection — from a serialized bundle, on a single raw input row.

**The bug.** Feature selection kept a 3-column subset, so the bundle's `feature_names`
was that subset — but the fitted `ColumnTransformer` still emits all 22 encoded columns.
Naively zipping 22 values to 3 names threw a shape error and broke every prediction.

**Fix.** Reconstruct the *full* output names from the fitted preprocessor
(`get_feature_names_out` per transformer), build the frame, *then* subset to the selected
features. Works for both the selection and no-selection paths, and fixes already-trained
bundles without retraining.

## 5. A lean, split deployment

**Problem.** `torch` was a ~2 GB hard dependency, but the deployed API always runs in
fast mode and never trains a neural net — pure dead weight that bloated the image and
risked OOM on small container tiers.

**Approach.** Moved torch to an optional `[deep-learning]` extra and made the pipeline
lazy-import the DL models (graceful skip if torch is absent). The frontend (static, on
Netlify) is decoupled from the backend (container host) via a configurable `VITE_API_URL`.

**Result.** The backend image installs none of PyTorch; verified torch isn't imported on
the API path at all. Frontend and backend scale and deploy independently.

## 6. A demo that's believable *and* reliable

**Problem.** A public "live training" demo is fragile: cold starts, multi-minute waits,
OOM, and an abuse/cost surface — exactly when a recruiter clicks it.

**Approach.** Ship a **pre-trained experiment** that the API loads on startup and
auto-deploys, so the dashboard opens instantly with a real leaderboard + SHAP + reports,
and the Playground makes live predictions with zero training. Live training still exists;
the demo just doesn't depend on it being fast.

**Trade-off.** Less "watch it train live," but it never embarrasses you with a 502.

---

## Demo snapshot (synthetic churn dataset, 1,600 rows, fast mode)

| Model | CV score | Latency |
|---|---|---|
| **voting_ensemble ★** | 0.904 | ~0 ms |
| stacking_ensemble | 0.903 | ~0 ms |
| catboost | 0.887 | 0.1 ms |
| lightgbm | 0.869 | 0.6 ms |
| random_forest | 0.859 | 26.9 ms |
| xgboost | 0.848 | 0.6 ms |
| logistic_regression | 0.754 | 0.1 ms |

Best model on hold-out: **accuracy 0.94, ROC-AUC 0.98, MCC 0.84**, plus calibration
(Brier 0.043, ECE 0.037). Ensembles beat every base model and the metrics are internally
consistent — the signature of a real pipeline, not hardcoded numbers.

## Likely interview questions (and where to point)

- *"How do you pick the best model?"* → §2, Pareto frontier.
- *"You let an LLM write code that you execute?"* → §3, and be honest about the limits.
- *"What was the hardest bug?"* → §4, the inference feature-name mismatch.
- *"How would this scale / what would you change for production?"* → real sandbox isolation
  (§3), persistent experiment store + object storage (currently in-memory), auth + rate
  limiting on the public API, and locking CORS to the frontend origin.
