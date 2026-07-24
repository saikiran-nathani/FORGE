 # FORGE — Complete Codebase Guide

> A read-through reference for explaining the entire FORGE codebase to a senior developer.
> Covers the architecture, every directory, and every code file with its classes/functions and
> what they're for — plus the data flows, cross-cutting patterns, and the subtle bugs/trade-offs
> worth naming out loud. Read §1–§6 for the mental model and flows; use §7 as the per-file map;
> §9–§10 are your interview ammunition.

---

## 0. How to use this document
- **To get the big picture fast:** §1 (what it is) → §2 (architecture) → §5 (end-to-end flows).
- **To answer "what does file X do?":** jump to §7, organized by subsystem, then file.
- **To sound senior:** §8 (patterns), §9 (known bugs & trade-offs), §10 (5-minute script + Q&A).
- Every `path` is relative to the repo root. Function signatures are given as they appear in code.

---

## 1. What FORGE is

**One line:** an LLM-assisted AutoML platform — you give it a CSV and a plain-English goal, and it
profiles the data, engineers features, trains & tunes ~16 model families, explains the winner, and
serves it as a live prediction API.

**The mental model — a 5-stage assembly line ("the forge"):**
```
raw CSV + task text
      │
  1. PROFILE      statistical + LLM-semantic read of every column → ProfileReport, task type, metric
  2. ENGINEER     LLM/heuristic feature generation (sandboxed) → preprocessing → feature selection
  3. TRAIN        ~16 model families, each Bayesian-tuned (Optuna) w/ CV → + voting/stacking ensembles
  4. EVALUATE     metrics (incl. calibration) + SHAP/LIME/PDP + error analysis + fairness audit
  5. DEPLOY       serialize best model → FastAPI endpoint + model card + drift monitoring
```

**Two halves of the codebase:**
1. **The pipeline library** (`forge/` minus `api/`) — pure Python, runnable via CLI, produces
   artifacts on disk. This is the ML engine.
2. **The web app** (`forge/api/` + `frontend/`) — a FastAPI backend that runs the pipeline
   asynchronously and streams progress, plus a React/TypeScript UI ("the Foundry").

Everything in half 2 is a thin orchestration layer over half 1.

---

## 2. Architecture at a glance

```
┌─────────────────────────── FRONTEND (React/TS, Vite) ───────────────────────────┐
│  LandingPage · HomePage(yard) · NewExperimentPage(anvil) · ExperimentPage        │
│  DeployPage · PlaygroundPage · MonitoringPage      — talk to backend via api.ts  │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                         │  HTTP /api/v1/*
┌───────────────────────────────────────▼──────────────────────────────────────────┐
│  FastAPI app (forge/api/app.py)                                                    │
│    ExperimentStore ──run_async──► background thread ──► ForgePipeline.run()        │
│    DeploymentService ──► DeployManager + ModelServer + DriftMonitor + PerfTracker  │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                         │  in-process calls
┌───────────────────────────────────────▼──────────────────────────────────────────┐
│  ForgePipeline (forge/pipeline.py) — orchestrates the 5 stages                     │
│   profiling/ · feature_engineering/ · training/ · evaluation/ · deployment/ ·      │
│   monitoring/ · llm/                                                               │
└────────────────────────────────────────────────────────────────────────────────────┘
```

**Key contract that ties it together:** the **model bundle** (`best_model.joblib`) — a dict holding
the fitted model + preprocessor + metadata. The pipeline writes it; `ModelServer` reads it to serve
predictions. See §6 and §8.

---

## 3. Tech stack
- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch (optional), Optuna (HPO), SHAP, LIME, MLflow (tracking), Evidently (optional drift).
- **Backend:** FastAPI + Uvicorn, Pydantic, Typer (CLI), Rich (CLI output), pandas/numpy.
- **Frontend:** React 18 + TypeScript (strict) + Vite + Tailwind + react-router.
- **LLM:** OpenAI or Anthropic via a unified client, with heuristic fallback when no key.
- **Packaging/deploy:** Docker (multi-stage), GitHub Actions CI, Netlify (frontend), container host (backend).

---

## 4. Directory map (annotated)
```
forge/
├── __init__.py              version constant
├── config.py                ForgeConfig — the single run-config dataclass (the on/off switchboard)
├── cli.py                   Typer CLI: run · serve api · deploy · profile
├── pipeline.py              ForgePipeline — the 5-stage orchestrator (the heart)
├── llm/
│   └── client.py            LLMClient — OpenAI/Anthropic wrapper + no-key fallback
├── profiling/
│   ├── models.py            ColumnType enum, ColumnProfile, ProfileReport dataclasses
│   ├── statistical_profiler.py   type detection, per-column stats, quality score, metric pick
│   ├── semantic_profiler.py      LLM (or heuristic) column meaning/importance/sensitivity
│   └── eda_report.py             self-contained Plotly HTML EDA report
├── feature_engineering/
│   ├── cleaner.py           FeaturePipeline — impute/encode/scale + inference-time transform
│   ├── code_sandbox.py      CodeSandbox — validate+exec LLM-written pandas (denylist)
│   ├── llm_engineer.py      LLMFeatureEngineer — generate features per column + heuristics
│   └── feature_selector.py  correlation → mutual-info → SHAP → RFECV selection
├── training/
│   ├── base_model.py        BaseModel ABC (uniform interface for all models)
│   ├── task_router.py       TaskType enum + TaskRouter.detect()
│   ├── classical/           14 model wrappers + BASE_MODELS/FAST_MODELS registries + ensembles
│   ├── deep_learning/       PyTorch MLP + TabTransformer, sklearn wrappers, trainer
│   └── hpo/                 optuna_optimizer (CV+HPO+latency) + pareto (frontier)
├── evaluation/
│   ├── metrics_calculator.py   classification/regression metrics incl. calibration
│   ├── explainability.py       SHAP/LIME/PDP/permutation → JSON+PNG artifacts
│   ├── error_analysis.py       worst predictions, confusion pairs, weak slices, residuals
│   ├── fairness_auditor.py     subgroup metrics + disparate-impact flags
│   └── llm_report_generator.py markdown analysis report (LLM or heuristic)
├── deployment/
│   ├── model_exporter.py    copy artifact whitelist → deployment dir + manifest
│   ├── api_generator.py     emit standalone serve.py + Dockerfile + requirements + compose
│   ├── batch_pipeline.py    emit an Airflow DAG for scheduled batch scoring
│   ├── model_card_generator.py  markdown model card
│   ├── deploy_manager.py    DeployManager — orchestrates the above (one-click deploy)
│   └── inference.py         ModelServer — loads the bundle, serves predictions
├── monitoring/
│   ├── drift_monitor.py     PSI (or Evidently) feature-drift vs baseline
│   └── performance_tracker.py  latency/volume/error/prediction-distribution
└── api/
    ├── app.py               FastAPI routes + CORS + lifespan (demo seed)
    ├── schemas.py           Pydantic request/response models
    ├── demo_loader.py       load the committed pre-trained demo on startup
    ├── demo/                committed seed (artifacts + result.json + meta.json)
    └── services/
        ├── experiment_service.py   ExperimentStore + async pipeline runner + progress
        └── deployment_service.py   deploy/predict/monitoring per experiment

frontend/src/                React app (main, App, pages/*, services/api.ts, hooks/useReveal.ts)
tests/                       unit (profiler, cleaner, sandbox, pareto, mlp, semantic) + integration (deployment)
benchmarks/run_all_benchmarks.py   runs the pipeline over benchmark datasets
main.py                      python main.py → forge CLI
```

---

## 5. End-to-end flows (the "what calls what")

### 5.1 CLI training run — `forge run data.csv --target y --task "..."`
1. `cli.run()` builds a `ForgeConfig` and calls `ForgePipeline(config).run(path)`.
2. `run()` executes the 5 stages into `outputs/<dataset_stem>/`:
   - **Profile:** `_load_dataset` → `_validate_target` (friendly error if the column is missing) → `StatisticalProfiler.profile` → `SemanticProfiler.profile` → `TaskRouter.detect` → `EDAReportGenerator.generate` (HTML).
   - **Engineer:** if `enable_llm`, `LLMFeatureEngineer.engineer` adds columns; then `FeaturePipeline.fit_transform` splits + imputes/encodes/scales; then (if >5 features) `FeatureSelector.select` prunes.
   - **Train:** `_model_registry` picks the model classes; for each, `OptunaOptimizer.optimize_model` runs CV + Bayesian HPO and returns a `ModelResult` (with measured latency); optional `EnsembleBuilder` adds voting/stacking; `_pick_best` selects the winner; `ParetoAnalyzer` computes the accuracy/latency frontier. All under one `mlflow.start_run`.
   - **Evaluate:** predict on the test split → `MetricsCalculator.compute` → optional `ExplainabilityEngine.explain` (SHAP/LIME/PDP), `ErrorAnalyzer.analyze`, `FairnessAuditor.audit` → optional `LLMReportGenerator.generate`.
   - **Finalize:** write `reference_data.parquet`, the `best_model.joblib` **bundle**, JSON artifacts, and `ModelCardGenerator.generate`.
3. Returns a `PipelineResult`; the CLI prints the artifact dir.

### 5.2 API experiment (async + live progress)
1. `POST /api/v1/experiments` (multipart upload) → `create_experiment` validates the file, `store.create(...)` mints an 8-char id + output dir, writes the dataset, then `store.run_async(...)` **spawns a daemon thread** and returns immediately (status `pending`/`running`).
2. In the thread, `_run_pipeline` builds a `ForgeConfig` and runs `ForgePipeline(config, progress_cb=on_progress)`. The pipeline calls `_progress(stage, msg)` at each stage and **per model** during training; `on_progress` writes `exp.stage`/`exp.progress` and appends to `exp.progress_log` (capped at 16).
3. The frontend `ExperimentPage` **polls** `GET /experiments/{id}` every 2s, renders the `RunningView` stage-stepper from `exp.stage` + `progress_log`, and stops polling when status becomes `completed`/`failed`.
4. On success, `exp.result` is populated (best model, metrics, model list, pareto, shap, errors, fairness, artifact paths); on error, `exp.error = str(exc)` (e.g. the friendly missing-target message).

### 5.3 Deploy — `POST /experiments/{id}/deploy`
`deploy_experiment` → `deployment_service.deploy(id, artifact_dir, ...)` → `DeployManager.deploy`
which: exports the artifact whitelist, `copytree`s the full artifacts, generates `serve.py`/Docker/compose,
an Airflow DAG, and a model card, and (if `reference_data.parquet` exists) sets a drift baseline.
The service then constructs a `ModelServer` + `DriftMonitor` + `PerformanceTracker` and registers a
`DeploymentState` in `_active[id]`.

### 5.4 Predict (Playground) — `POST /experiments/{id}/predict`
`PlaygroundPage` first calls `getModelInfo` to learn `input_columns`, renders one field each, then
`predict(id, body)` → `deployment_service.predict` → `ModelServer.predict`: builds a DataFrame,
`FeaturePipeline.transform_raw` reconstructs the exact training features from the bundle, `model.predict`
runs, `_decode_prediction` maps back to the original label; latency + result are recorded by the tracker.

### 5.5 Frontend routing (`App.tsx`)
`/` → Landing (full-bleed) · `/experiments` → Home (the yard) · `/new` → NewExperiment (the anvil) ·
`/experiments/:id` → Experiment (progress + results) · `/experiments/:id/{deploy,playground,monitoring}`.
The seeded demo lives at `/experiments/demo`.

---

## 6. Core data structures (cheat sheet)

| Type | File | What it carries |
|---|---|---|
| **ForgeConfig** (dataclass) | config.py | target_column, task_description, output_dir, random_state, test_size, cv_folds, hpo_trials_per_model, mlflow_*, `enable_*` toggles (llm, feature_selection, shap, lime, pdp, ensembles, deep_learning, error_analysis, fairness, llm_report), fast_mode |
| **ColumnType** (str Enum) | profiling/models.py | NUMERICAL, CATEGORICAL, DATETIME, TEXT, BINARY, ID |
| **ProfileReport** (dataclass) | profiling/models.py | n_rows/n_cols, column_types, column_profiles, target_analysis, correlations, missing_analysis, outlier_analysis, quality_score, recommended_metric; `to_dict()` serializes a JSON-safe subset |
| **SemanticProfile** (dataclass) | profiling/semantic_profiler.py | columns{meaning/importance/sensitive/transforms}, data_quality_summary, key_interactions, leakage_risks, `source` ("llm"/"heuristic") |
| **FeaturePipelineResult** (dataclass) | feature_engineering/cleaner.py | X_train/X_test, y_train/y_test, feature_names, fitted `preprocessor` (ColumnTransformer), label_encoder, metadata{numerical_cols, categorical_cols, dropped_id_cols, input_columns, clip_bounds} |
| **ModelResult** (dataclass) | training/hpo/optuna_optimizer.py | model_name, best_params, cv_score, cv_score_std, training_time, inference_latency_ms, fitted_model, metric_name |
| **ParetoPoint** (dataclass) | training/hpo/pareto.py | model_name, cv_score, latency_ms, is_pareto_optimal |
| **PipelineResult** (dataclass) | pipeline.py | profile, semantic_profile, task_type, model_results, pareto_frontier, best_model_name, best_metrics, generated_features, feature_selection, shap_summary, error_analysis, fairness_report, llm_report_path, output_dir |
| **the model bundle** (dict) | written in pipeline.py, read by inference.py | `model`, `preprocessor`, `label_encoder`, `feature_names`, `model_name`, `params`, `metadata`, `profile`, `task_type`, `target_column`, `selected_features` |
| **Experiment** (dataclass) | api/services/experiment_service.py | id, name, task_description, target_column, status(ExperimentStatus), created_at, dataset_path, output_dir, error, result, progress, **stage**, **progress_log** |

---

## 7. Module-by-module reference

### 7.1 Entry & orchestration

**`forge/__init__.py`** — declares `__version__`. No logic.

**`main.py`** — `if __name__ == "__main__": app()` where `app` is the Typer CLI; makes `python main.py …` = the `forge` command.

**`forge/config.py`** — `ForgeConfig` (dataclass, see §6). `ensure_output_dir()` creates and returns `output_dir`. Every optional stage is gated by an `enable_*` flag here — it's the central switchboard. `fast_mode` uses the 5-model fast subset and skips deep learning.

**`forge/cli.py`** — Typer app (`console` = Rich). Commands:
- `run(dataset, target, task, output, trials, test_size, fast, no_llm, no_shap)` — validates the file, builds `ForgeConfig` (maps `no_llm/no_shap` to `enable_*` negations), runs the pipeline.
- `serve_api(host, port, reload)` (`serve api`) — lazily imports uvicorn, serves `forge.api.app:app`.
- `deploy(artifacts, experiment_id, target, task)` — loads the joblib bundle, resolves target, optionally loads `reference_data.parquet`, calls `DeployManager().deploy(...)`.
- `profile(dataset, target)` — runs statistical + (heuristic) semantic profile, prints combined JSON.
- *Notable:* heavy deps imported lazily per command; error paths print red + `raise typer.Exit(1)`.

**`forge/pipeline.py`** — `ForgePipeline`, the orchestrator.
- `__init__(config, progress_cb=None)` — holds `TaskRouter`, `MetricsCalculator`, `LLMClient`, and the progress callback.
- `_progress(stage, message)` — fires `progress_cb` (exception-safe); stages: `profiling`, `feature_engineering`, `training`, `evaluation`, `finalizing`.
- `run(dataset_path) -> PipelineResult` — the 5-stage flow (see §5.1). Writes all artifacts + the bundle + model card.
- `_model_registry(task_type, config)` — starts from `FAST_MODELS`/`BASE_MODELS`; for regression removes LogisticRegression/NaiveBayes, for classification removes Ridge/Lasso/ElasticNet; appends DL models only if `enable_deep_learning and not fast_mode` via `_load_dl_models()`.
- `_load_dl_models()` (module fn) — imports MLP/TabTransformer lazily; returns `[]` if torch missing (keeps torch out of the default path).
- `_pick_best(results, metric)` — `min` by cv_score for `"rmse"`, else `max`.
- `_validate_target(df)` — raises `ValueError` with a `difflib` "did you mean" suggestion if the target column is absent.
- `_save_artifacts`, `_serialize_results`, `_load_dataset` (csv/parquet/json), `_print_results` (Rich table).
- *DL trials are capped:* `dl_trials = min(hpo_trials_per_model, 5)`, DL detected via `family == "deep_learning"`.

**`forge/llm/client.py`** — `LLMClient`.
- `__init__(provider="openai", model=None, api_key=None)` — model from arg/`FORGE_LLM_MODEL`/`gpt-4o-mini`; key from arg/`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`.
- `available` (property) — `True` iff a key is set.
- `complete(system, user, json_mode=False)` — raises if no key; routes OpenAI/Anthropic.
- `complete_json(system, user)` — **returns `{}` when unavailable** (the graceful-degradation seam callers rely on).
- `_openai_complete` / `_anthropic_complete` (lazy SDK imports; Anthropic hard-falls-back to `claude-3-5-haiku` if model name lacks "claude"); `_parse_json` strips code fences.

**`forge/training/task_router.py`** — `TaskType(str, Enum)` (binary/multiclass/regression/time_series) + `TaskRouter`.
- `detect(profile, task_description, override=None)` — override wins; else regression from target_analysis; else by cardinality (2 → binary, 3–50 → multiclass); time-series only on forecast keywords; default multiclass.
- `is_classification(task_type)`.

### 7.2 Profiling

**`profiling/models.py`** — `ColumnType`, `ColumnProfile` (name, detected_type, statistics, missing_pct, n_unique), `ProfileReport` (see §6). `ProfileReport.to_dict()` omits column_profiles/correlations/outlier_analysis.

**`profiling/statistical_profiler.py`** — `StatisticalProfiler(target_column, task_description)`.
- `profile(df) -> ProfileReport` — orchestrates type detection, per-column stats, target/correlation/missing/outlier analysis, quality score, metric recommendation.
- Type detection order: bool/{0,1} → BINARY; ID; DATETIME; object → TEXT (avg len>50) else CATEGORICAL; numeric → CATEGORICAL (int, <20 uniques) else NUMERICAL.
- `_infer_target_type` decides regression vs classification by cardinality/unique-ratio (see thresholds in code).
- `_compute_quality_score` — weighted blend (completeness .3, uniqueness .2, type_consistency .2, validity .2, timeliness .1), 0–100.
- `_recommend_metric` — regression→rmse; binary→roc_auc (if imbalanced) else f1; else f1_macro.
- Per-type stat helpers (`_numerical_stats` incl. skew/kurtosis/outliers/shapiro; `_categorical_stats` incl. entropy/rare cats; `_datetime_stats`; `_text_stats`).

**`profiling/semantic_profiler.py`** — `SemanticProfiler(llm=None)` + `SemanticProfile`.
- `profile(report, task_description, target_column)` — LLM path if `llm.available`, wrapped in try/except → falls back to `_heuristic_profile` (silent; `source` records which ran).
- Heuristic importance by name substrings (`IMPORTANCE_HINTS`), sensitivity by `SENSITIVE_KEYWORDS`, leakage = columns whose name contains the target's.

**`profiling/eda_report.py`** — `EDAReportGenerator.generate(df, profile, semantic, target, output_path)` writes a dark-themed self-contained HTML report (Plotly fragments + one CDN script): overview, target dist, feature dists (first 8), correlation heatmap, missing bars, optional LLM insights.

### 7.3 Feature engineering

**`feature_engineering/cleaner.py`** — `FeaturePipeline(target_column, random_state=42)`.
- `fit_transform(df, profile, test_size=0.2) -> FeaturePipelineResult` — drops target-NaN rows + ID columns; label-encodes y for classification; stratified split (only if min class ≥2); numeric/datetime → median-impute + StandardScaler; categorical/binary/text → most-frequent-impute + OneHotEncoder(handle_unknown=ignore); `remainder="drop"`.
- `transform_raw(df, bundle) -> DataFrame` — **inference-time**: rebuilds a ProfileReport from `bundle["profile"]`, applies the saved preprocessor, and — crucially — reconstructs the **full** output feature names via `_get_feature_names` (NOT `bundle["feature_names"]`, which may be a post-selection subset), then subsets to `selected_features`. (This is the bug we fixed earlier this session.)
- Helpers: `_apply_clip_bounds`, `_handle_missing_indicators`, `_clip_outliers`, `_get_feature_names`.

**`feature_engineering/code_sandbox.py`** — `CodeSandbox(timeout_seconds=30, max_nan_pct=0.5)` + `SandboxResult`.
- `execute(code, df, target_column)` — extract fenced code → `_validate_code` (denylist regex + AST import allow-list + rejects target-name substring) → `exec(compile(...))` **in-process** → `_validate_output` (row count preserved, ≥1 new column, not target-named, NaN% ≤ max, not constant).
- *Security reality:* denylist only, no process isolation, `__builtins__` not stripped, and the "timeout" is checked **after** exec (can't stop an infinite loop). It's a validator, not a sandbox.

**`feature_engineering/llm_engineer.py`** — `LLMFeatureEngineer(target_column, task_description, llm=None, max_retries=3)` + `FeatureEngineeringResult`.
- `engineer(df, profile, semantic=None)` — per non-target/non-ID column: `_generate_code` (LLM if available else heuristic) → `_execute_with_retry` (sandbox + up to N LLM repair attempts) → collect; then `_apply_heuristic_features` adds a product + ratio of the first two numeric columns.
- `_heuristic_code` templates by type (datetime → dow/month/is_weekend; numeric → log/zscore/rank; categorical → frequency).

**`feature_engineering/feature_selector.py`** — `FeatureSelector(task_type, ...)` + `FeatureSelectionResult`.
- `select(X_train, X_test, y_train, skip_rfe=False)` — 4 stages: `_correlation_filter` (drop the lower-target-corr of each |corr|>0.95 pair) → `_mutual_info_filter` (MI ≥ 0.01) → `_shap_importance` (LightGBM+SHAP, drop < 1% of max; skipped if import/fit fails) → `_rfe_filter` (RFECV, only if >10 features; no-op on exception). Returns reduced frames + `removed_features` reasons + `importance_scores`.

### 7.4 Training

**`training/base_model.py`** — `BaseModel(ABC)`; class attrs `name`, `family`. Abstract `get_search_space(trial)` + `build_model(params)`; concrete `train(X,y,params)`, `predict`, `predict_proba` (None if unsupported), `get_feature_importance` (feature_importances_ → coef_ fallback; positional `feature_i` keys). **Every model conforms to this** so the optimizer/ensembles treat them uniformly.

**`training/classical/__init__.py`** — `BASE_MODELS` (14: logistic, ridge, lasso, elastic_net, sgd, decision_tree, random_forest, extra_trees, xgboost, lightgbm, catboost, knn, svm, naive_bayes) and `FAST_MODELS` (logistic, random_forest, xgboost, lightgbm, catboost).

**Classical model files** — each is a `BaseModel` subclass defining `name`, an Optuna `get_search_space`, and a task-branching `build_model`:
- `logistic_regression.py` — `LogisticRegressionModel` (regression branch degrades to `Ridge(alpha=1/C)`).
- `linear_models.py` — `RidgeModel`, `LassoModel`, `ElasticNetModel`, `SGDModel` (SGD switches regressor/classifier + task-dependent loss).
- `tree_models.py` — `DecisionTreeModel`, `ExtraTreesModel` (n_jobs=1; regression strips class_weight).
- `random_forest.py` — `RandomForestModel` (n_jobs=-1).
- `xgboost_model.py` — `XGBoostModel` (injects `eval_metric` logloss/mlogloss in build).
- `lightgbm_model.py` — `LightGBMModel`.
- `catboost_model.py` — `CatBoostModel` (verbose=0; regression strips auto_class_weights).
- `distance_models.py` — `KNNModel`, `SVMModel` (forces `probability=True` for soft-vote), `NaiveBayesModel` (always GaussianNB, no regression variant).

**`training/classical/ensembles.py`** — `EnsembleBuilder(task_type, random_state)`.
- `build_voting(results, X, y, top_n=3)` — soft VotingClassifier / VotingRegressor over the top-N by cv_score.
- `build_stacking(results, X, y, top_n=5)` — StackingClassifier (LogisticRegression meta) / StackingRegressor (Ridge meta), cv=3.
- `_cv_score` — f1_macro (clf) / neg_root_mean_squared_error (reg), 3-fold. ⚠️ ranks by **descending** cv_score — correct for classification, but for regression cv_score is positive RMSE so this selects the **worst** models (see §9).

**`training/deep_learning/models.py`** — `MLPModel` & `TabTransformerModel` (`family="deep_learning"`) — search spaces + `build_model` returning the sklearn-wrapped nets.

**`training/deep_learning/nn_wrappers.py`** — PyTorch nets + sklearn-compatible wrappers:
- `MLPNetwork(nn.Module)` — Linear→[BN]→GELU→Dropout ×n_layers + head.
- `TabularMLP(BaseEstimator)` — `fit/predict/predict_proba/_forward`; binary uses a **single logit + sigmoid** (predict_proba rebuilds `[1-p, p]`); subclasses `TabularMLPClassifier/Regressor`.
- `TabTransformerNetwork` — tokenizes each scalar feature via shared `Linear(1, d_model)`, prepends a learnable CLS token, `TransformerEncoder`, reads CLS → head (`n_heads` hardcoded 4).
- `TabularTransformer(TabularMLP)` — overrides `fit` to build the transformer.

**`training/deep_learning/trainer.py`** — `DLConfig` + `DLTrainer(config, task_type)`.
- `train(model, X_train, y_train, ...)` — 85/15 internal val split, AdamW + OneCycleLR (per-batch step), grad-clip 1.0, loss by task, early stopping, restores best state.
- ⚠️ `_evaluate` returns a **loss** (lower=better) but the classification branch tests `improved = metric > best_metric` (with `best_metric=-inf`), i.e. it keeps the epoch with the **highest** val loss — inverted early-stopping for classification (see §9).

**`training/hpo/optuna_optimizer.py`** — `ModelResult` + `OptunaOptimizer(task_type, metric_name, n_trials=30, cv_folds=5)`.
- `optimize_model(model_cls, X, y)` — TPE study (maximize unless RMSE), objective runs `cross_val_score`; rebuilds `best_params` through `_FixedTrial` into `get_search_space` (re-derives dependent params); trains final model; **measures latency** (mean of 10 `predict` calls on ≤100 rows); returns `ModelResult`.
- `_FixedTrial` replays stored params. ⚠️ `MedianPruner` is configured but never prunes (no intermediate values reported).

**`training/hpo/pareto.py`** — `ParetoPoint` + `ParetoAnalyzer.compute(model_results, maximize_score=True)` (O(n²) domination: no worse in score AND latency, strictly better in one) + `to_dict`. Missing latency defaults to 0.0.

### 7.5 Evaluation

**`evaluation/metrics_calculator.py`** — `MetricsCalculator.compute(y_true, y_pred, y_proba, task_type, is_binary)` → dict.
- Classification always: accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, mcc, cohen_kappa, confusion_matrix; binary+proba adds roc_auc, pr_auc, brier_score, log_loss, ece, f1.
- Regression: rmse, mae, r2, adjusted_r2 (uses fixed p=1), mape, max_error.
- `_expected_calibration_error` — 10 equal-width bins, Σ weight·|acc−confidence|.

**`evaluation/explainability.py`** — `ExplainabilityEngine.explain(model, X, feature_names, model_name, output_dir, y, task_type, max_samples=200, enable_lime, enable_pdp)`.
- `_get_shap_values` chooses TreeExplainer (tree models) / DeepExplainer (mlp, tab_transformer) / KernelExplainer (else). Writes `shap_summary.json` + `shap_beeswarm.png`.
- LIME only for classification; PDP + permutation importance only if `y` given. Returns a combined summary (top-20 SHAP features + optional lime/pdp/permutation).

**`evaluation/error_analysis.py`** — `ErrorAnalyzer.analyze(...)`.
- `_worst_predictions` (top-20 by log-loss or 0/1), `_confusion_analysis` (top-10 confused pairs), `_slice_analysis` (per-value/quartile accuracy; underperforming if < overall−0.1) for classification; `_residual_analysis` for regression. Writes `error_analysis.json`.

**`evaluation/fairness_auditor.py`** — `FairnessAuditor.audit(df, y_true, y_pred, y_proba, semantic, output_dir)`.
- Sensitive columns = those flagged `sensitive` in the semantic profile. Per subgroup (≥5 rows): accuracy, positive_rate, mean proba; `disparate_impact_ratio = min/max positive_rate`. Flags DI outside (0.8, 1.25) or accuracy gap > 0.1. Writes `fairness_report.json`.

**`evaluation/llm_report_generator.py`** — `LLMReportGenerator.generate(...)` — builds a prompt from all results and calls the LLM (try/except → `_heuristic_report` template fallback); always writes markdown and returns it.

### 7.6 Deployment

**`deployment/model_exporter.py`** — `ModelExporter.export(artifact_dir, deployment_dir) -> ExportResult` — copies a whitelist (best_model.joblib, profile.json, test_metrics.json, feature_selection.json, semantic_profile.json) + writes `manifest.json`.

**`deployment/api_generator.py`** — `APIGenerator.generate(deployment_dir)` — writes a standalone `serve.py` (loads `ModelServer`, dynamic Pydantic `PredictRequest` from `input_columns`, routes `/health`, `/model-info`, `/predict`, `/predict/batch` on :8080), `Dockerfile`, `requirements.txt`, `docker-compose.yml`.

**`deployment/batch_pipeline.py`** — `BatchPipelineGenerator.generate(deployment_dir, ...)` — renders an Airflow DAG (`forge_batch_scoring`) that batch-scores a CSV via `ModelServer`.

**`deployment/model_card_generator.py`** — `ModelCardGenerator.generate(artifact_dir, task_description, target_column, output_path=None)` — reads the artifact JSONs + bundle, writes a markdown model card (details, data, metrics, fairness, limitations, intended use, ethics).

**`deployment/deploy_manager.py`** — `DeployManager(deployments_root="deployments")` + `DeploymentResult`.
- `deploy(experiment_id, artifact_dir, task_description, target_column, reference_data=None)` — `deployment_id = <exp>-<UTC ts>`; runs export → `copytree` full artifacts → APIGenerator → BatchPipelineGenerator → ModelCardGenerator → optional DriftMonitor baseline; writes `deployment.json`; `api_url` hardcoded to `http://localhost:8080`.
- `get_deployment`, `list_deployments`.

**`deployment/inference.py`** — `ModelServer(artifact_dir)`.
- `__init__` loads the bundle, merges profile.json, loads selected_features + test_metrics, builds a `FeaturePipeline`.
- `predict(features: dict | list[dict])` — normalizes to a DataFrame, `transform_raw`, `model.predict`; adds `probability`/`confidence` (binary) or `probabilities` (multiclass); `_decode_prediction` maps back via label_encoder.
- `model_info()` — model_name, feature_names, input_columns, metrics, task_type.

### 7.7 Monitoring

**`monitoring/drift_monitor.py`** — `DriftMonitor(monitoring_dir)`, `PSI_THRESHOLD=0.25`.
- `set_baseline(df, target_column)` — writes feature baseline parquet + meta.
- `check_drift(current_df, target_column="")` — Evidently if importable, else `_statistical_drift`: per-column `_compute_psi` (10 buckets, clip to 0.001 floor), feature `drifted` if PSI>0.25, `overall_drift_score` = mean PSI; writes a timestamped report.

**`monitoring/performance_tracker.py`** — `PredictionRecord` + `PerformanceTracker(monitoring_dir, max_records=10000)`.
- `set_training_metrics` (numeric only), `record(latency, prediction, confidence, error)` (bounded deque), `summary()` → total_requests, error_rate, latency p50/p95/p99, prediction_distribution, avg_confidence, training_metrics. Nested `Timer` context manager auto-records latency.

### 7.8 API layer

**`api/app.py`** — FastAPI app (title "FORGE API", v0.2.0). `lifespan` calls `load_demo_seed()` on startup. **CORS is fully open** (`allow_origins=["*"]` + credentials). Routes (all `/api/v1`): `health`, `POST experiments` (upload+run_async), `GET experiments`, `GET experiments/{id}`, `.../status`, `.../profile`, `.../models`, `.../evaluation`, `.../report` (eda/analysis FileResponse), `.../errors`, `POST .../deploy`, `POST .../predict`, `.../monitoring`, `.../model-info`, `.../model-card`. `_to_response` maps `Experiment`→`ExperimentResponse`.

**`api/schemas.py`** — `ExperimentCreate` (target_column required; trials 1–100), `ExperimentResponse` (adds stage, progress_log), `ExperimentStatusResponse`.

**`api/demo_loader.py`** — `load_demo_seed() -> bool` — if `demo/{result.json, meta.json, artifacts/}` exist, rewrites path fields to the committed location, registers a COMPLETED `Experiment` via `store.add`, and `deployment_service.deploy(...)` so the Playground works instantly. Idempotent; no-op without the seed.

**`api/services/experiment_service.py`** — `ExperimentStatus` enum, `Experiment` dataclass (see §6), `ExperimentStore` (in-memory dict + lock), module singleton `store`.
- `create`, `add`, `get`, `list_all`, `run_async` (daemon thread), `_run_pipeline` (builds config, defines `on_progress`, runs the pipeline, populates `result` or sets `error`). *Note:* `get`/`list_all` and in-thread field mutations are unsynchronized.

**`api/services/deployment_service.py`** — `DeploymentState` + `DeploymentService` (singleton `deployment_service`).
- `deploy` (registers ModelServer+DriftMonitor+PerformanceTracker), `predict`/`predict_batch` (times + records), `check_drift`, `get_monitoring` (safe when undeployed), `get_model_info`, `is_deployed`, `_require`.

### 7.9 Frontend (`frontend/src/`)

- **`main.tsx`** — mounts `<App/>` inside `<StrictMode>` + `<BrowserRouter>`.
- **`App.tsx`** — `Nav` + `Footer` + route table (see §5.5); `Contained` wraps inner pages; landing is full-bleed; atmospheric `atmos`/`grain` chrome.
- **`services/api.ts`** — typed client. `Experiment` interface (incl. stage, progress_log). Functions: `createExperiment` (multipart, hardcodes `fast_mode='true'`), `listExperiments`, `getExperiment`, `deployExperiment`, `predict`, `reportUrl` (string only), `getModelInfo`, `getMonitoring`. `BASE = VITE_API_URL || ''` → `${BASE}/api/v1` (relative in dev, absolute on Netlify).
- **`hooks/useReveal.ts`** — one-shot IntersectionObserver adding `.in` to `.reveal` descendants.
- **`pages/LandingPage.tsx`** — marketing page; `EmberField`/`ForgeCore` decorative; CTAs to `/new` and `/experiments/demo`.
- **`pages/HomePage.tsx`** — lists experiments, polls `listExperiments` every 3s; `StatusBadge`; `demo` chip.
- **`pages/NewExperimentPage.tsx`** — upload form → `createExperiment` → navigate to the experiment.
- **`pages/ExperimentPage.tsx`** — polls `getExperiment` every 2s (stops on terminal); `RunningView` renders the stage-stepper from `exp.stage`+`progress_log` with an elapsed clock; completed view shows metrics, model table, SHAP bars, Pareto table, report links; `notFound` fallback.
- **`pages/DeployPage.tsx`** — one-click `deployExperiment` then shows deployment info + basic perf.
- **`pages/PlaygroundPage.tsx`** — `getModelInfo` → dynamic form → `predict` (numeric coercion).
- **`pages/MonitoringPage.tsx`** — polls `getMonitoring` every 5s; latency/error/distribution cards.

### 7.10 Tests & benchmarks
- **`tests/conftest.py`** — `sample_classification_df`, `sample_regression_df` fixtures (10 rows).
- **`tests/unit/`** — `test_profiler` (type detection, target analysis, metric), `test_cleaner` (fit_transform, label encoder presence), `test_sandbox` (valid exec, blocks target ref + forbidden import), `test_pareto` (frontier membership), `test_mlp` (trains, search space), `test_semantic_profiler` (heuristic path).
- **`tests/integration/test_deployment.py`** — builds a mock bundle; `test_deploy_manager` (serve.py + model_card produced), `test_model_server_predict`, `test_drift_monitor` (identical data → low drift).
- **`benchmarks/run_all_benchmarks.py`** — `run_benchmarks(output_dir, trials=5)` runs the pipeline over `BENCHMARKS` (currently Titanic), writes `benchmark_summary.json`.

---

## 8. Cross-cutting patterns (explain these and you sound fluent)
1. **The bundle contract** — the whole train↔serve boundary is the `best_model.joblib` dict (§6). Training writes it; `ModelServer`/`transform_raw` read it. Understanding this dict = understanding how a trained model becomes a live API.
2. **LLM-with-heuristic-fallback** — `LLMClient.complete_json` returns `{}` when no key; semantic profiler, feature engineer, and report generator all try the LLM in a try/except and degrade to deterministic heuristics. So FORGE runs fully offline.
3. **Progress callback** — the pipeline is UI-agnostic: it calls `_progress(stage, msg)`; the API supplies a callback that fills `exp.stage/progress/progress_log`; the frontend polls and renders the stepper. Clean separation of engine vs transport.
4. **Metric-direction handling** — `"rmse"` is the sole minimize case; `_pick_best`, the Optuna study direction, and the Pareto `maximize_score` all special-case it. (The ensemble builder does **not** — that's the bug in §9.)
5. **Feature-name reconstruction at inference** — `transform_raw` rebuilds the full preprocessor output names rather than trusting the (possibly selection-subset) `feature_names`; this is what makes prediction robust after feature selection.
6. **Artifacts written per run** (in `outputs/<stem>/`): `eda_report.html`, `profile.json`, `semantic_profile.json`, `test_metrics.json`, `pareto_frontier.json`, `feature_selection.json`, `llm_features.json`, `best_model.joblib`, `reference_data.parquet`, `model_card.md`, `analysis_report.md`, and `explainability/`, `errors/`, `fairness/` subfolders.
7. **Lazy heavy imports** — torch (DL models), openai/anthropic (LLM), uvicorn (serve), evidently (drift), shap/lime are all imported lazily so the base install and CLI stay light.

---

## 9. Known bugs, gotchas & design trade-offs (senior-dev talking points)

**Genuine correctness bugs (verified in code):**
- **Inverted DL early-stopping for classification** (`training/deep_learning/trainer.py:85`): `_evaluate` returns a *loss* (lower is better), but the classification branch keeps the epoch where `metric > best_metric` (best init `-inf`), i.e. it selects/early-stops on the **highest** validation loss. Only the regression branch minimizes correctly. Real impact: DL classifiers keep a worse checkpoint. (DL only runs in non-fast mode, so the API never hits it.)
- **Ensemble base-model selection wrong for regression** (`training/classical/ensembles.py:31,49`): sorts by *descending* `cv_score` and takes the top-N. For classification (f1/auc) that's correct; for regression `cv_score` is positive RMSE, so it picks the **highest-error** models to ensemble.
- **`adjusted_r2` uses a fixed predictor count `p=1`** (`metrics_calculator.py`) regardless of real feature count — so it's not a true adjusted R².
- **MedianPruner never prunes** (`optuna_optimizer.py`) — the objective reports no intermediate values, so the configured pruner is a no-op (minor; just misleading).
- **Missing-indicator columns are dropped** (`cleaner.py`): `_handle_missing_indicators` creates `{col}_is_missing` columns but never adds them to the transformer's column lists, so `remainder="drop"` discards them.

**Security / production gaps (own these before a senior asks):**
- **The "sandbox" isn't isolated** — in-process `exec` behind a regex/AST denylist, builtins intact, post-hoc timeout. Bypassable; fine for trusted self-hosted use, not for untrusted input.
- **CORS is wide open** (`allow_origins=["*"]` + credentials) — lock to the frontend origin before public exposure.
- **In-memory experiment store** — experiments vanish on restart (only the seeded `demo` reloads); no persistence, no auth, no rate limiting; training runs in an unbounded daemon thread.
- **Hardcoded frontend URLs** — `ExperimentPage`'s analysis-report link and `DeployPage`'s model-card link hit `/api/v1/...` directly, bypassing `VITE_API_URL`; they only work same-origin. `DeployPage` imports `predict` unused.
- **`api_url` hardcoded** to `http://localhost:8080` in `DeployManager`.

**Deliberate trade-offs (defensible choices):**
- torch is an optional extra + lazy-loaded → lean API image; DL skipped gracefully if absent.
- LLM optional with heuristic fallback → no hard dependency on a paid API.
- Pareto selection (accuracy *vs* latency) instead of blind best-score → production-minded.
- One `BaseModel` interface across 16 heterogeneous algorithms → adding a model is one small class.

---

## 10. "Explain it in 5 minutes" script + likely Q&A

**The script:**
> "FORGE is an LLM-assisted AutoML platform. A user uploads a CSV and describes their goal in plain
> English. A single orchestrator, `ForgePipeline`, runs five stages: it **profiles** the data
> (statistics + an LLM semantic read), **engineers** features (the LLM proposes pandas transforms
> that run in a validation sandbox, then a 4-stage selector prunes them), **trains** about sixteen
> model families — linear, trees, boosting, and PyTorch nets — each Bayesian-tuned with Optuna under
> cross-validation, plus voting/stacking ensembles, and picks the winner on an accuracy-vs-latency
> Pareto frontier. Then it **evaluates** with SHAP/LIME, calibration, error analysis and a fairness
> audit, and **deploys** the best model — it serializes a bundle and a FastAPI `ModelServer` serves
> live predictions with drift monitoring. Everything's wrapped in a FastAPI backend that runs the
> pipeline in a background thread and streams stage-by-stage progress to a React frontend. The whole
> thing degrades gracefully with no LLM key by falling back to heuristics."

**Likely questions → where to point:**
- *"How does a trained model become a live endpoint?"* → the bundle dict + `ModelServer.predict` + `transform_raw` (§6, §8.1, §8.5).
- *"How do you pick the best model?"* → `_pick_best` + Pareto frontier; metric-direction handling (§7.1, §8.4).
- *"You execute LLM-generated code?"* → `CodeSandbox` denylist + output validation, and be honest about its limits (§7.3, §9).
- *"How does the UI show progress during a 10-min run?"* → progress callback → `exp.stage/progress_log` → polling + `RunningView` (§5.2, §8.3).
- *"What breaks under load / in prod?"* → in-memory store, open CORS, threaded training, no auth (§9).
- *"What's the hardest bug you found?"* → the inference feature-name mismatch after selection (fixed), or the inverted DL early-stopping (§9).

---

## 11. Glossary
- **HPO** — hyperparameter optimization (here, Optuna TPE + CV).
- **Pareto frontier** — the set of models not dominated on both accuracy and latency.
- **SHAP / LIME / PDP** — model explainability: global feature attribution / local per-prediction reasons / partial-dependence.
- **Calibration (Brier, ECE)** — how well predicted probabilities match observed frequencies.
- **PSI** — Population Stability Index; drift score comparing live vs baseline distributions.
- **Bundle** — the serialized `best_model.joblib` dict (model + preprocessor + metadata) that couples training to serving.
- **The Foundry / the yard / the anvil** — UI names for the landing, experiment list, and upload pages.
