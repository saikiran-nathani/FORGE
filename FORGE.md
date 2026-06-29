# FORGE — LLM-Powered Automated ML Pipeline Builder
### Complete Project Specification Document
### Version 1.0

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Profiling Engine](#3-data-profiling-engine)
4. [LLM Feature Engineering Agent](#4-llm-feature-engineering-agent)
5. [Model Training Engine](#5-model-training-engine)
6. [Hyperparameter Optimization](#6-hyperparameter-optimization)
7. [Evaluation & Explanation Engine](#7-evaluation--explanation-engine)
8. [LLM Report Generator](#8-llm-report-generator)
9. [Deployment Engine](#9-deployment-engine)
10. [Monitoring Setup](#10-monitoring-setup)
11. [Backend API](#11-backend-api)
12. [Frontend Dashboard](#12-frontend-dashboard)
13. [Database Schemas](#13-database-schemas)
14. [DevOps & Infrastructure](#14-devops--infrastructure)
15. [Testing Strategy](#15-testing-strategy)
16. [Benchmark Datasets](#16-benchmark-datasets)
17. [Performance Targets](#17-performance-targets)
18. [Build Phases](#18-build-phases)
19. [Tech Stack Summary](#19-tech-stack-summary)
20. [Directory Structure](#20-directory-structure)

---

## 1. PROJECT OVERVIEW

### 1.1 What is Forge?

Forge is an LLM-powered automated ML platform where you upload a dataset, describe your prediction goal in natural language, and the system automatically profiles the data, engineers features using an LLM agent, trains and compares 26 ML/DL model architectures with Bayesian hyperparameter optimization, generates comprehensive evaluation reports with explainability, and deploys the best model as a production API — all without writing a single line of code.

### 1.2 What Makes Forge Different from AutoML

| Feature | Traditional AutoML (H2O, Auto-sklearn) | Forge |
|---------|---------------------------------------|-------|
| Feature engineering | Basic (encoding, scaling) | LLM analyzes column semantics, generates domain-aware features |
| Model explanation | Minimal | Full SHAP/LIME/PDP suite + LLM-narrated report |
| Error analysis | None | LLM identifies patterns in misclassifications |
| Feature engineering rationale | None | LLM explains WHY each feature was created |
| Deep learning | Limited | TabTransformer, FT-Transformer, TabNet, LSTM, TCN, TFT, N-BEATS |
| Improvement recommendations | None | LLM suggests specific actions to improve performance |
| Fairness auditing | Rare | Built-in across sensitive attributes |
| Deployment | None | One-click FastAPI + Docker + monitoring |

### 1.3 Core Capabilities

1. **Automated data profiling** — type detection, distributions, correlations, quality scoring, LLM semantic analysis
2. **LLM-powered feature engineering** — generates Pandas code from column semantics, executes in sandbox, iteratively refines
3. **26 model architectures** — from Logistic Regression to FT-Transformer, all trained and compared
4. **Bayesian HPO** — Optuna with pruning, multi-objective (accuracy vs latency), 500+ trials
5. **Comprehensive evaluation** — every metric, calibration, fairness, error analysis
6. **Full explainability** — SHAP (3 explainer types), LIME, PDP/ICE, permutation importance, attention visualization
7. **LLM-generated reports** — natural language interpretation of all results
8. **One-click deployment** — auto-generated FastAPI + Docker + monitoring
9. **Batch prediction pipeline** — Airflow DAG for scheduled batch scoring
10. **Drift monitoring** — automatic setup with Evidently AI

---

## 2. SYSTEM ARCHITECTURE

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                    │
│  ┌────────────────┐  ┌────────────────────────────────────────────┐ │
│  │ Dataset Upload  │  │ Task Description (Natural Language)        │ │
│  │ CSV / Parquet / │  │ "Predict customer churn based on usage    │ │
│  │ JSON / SQL      │  │  patterns and account information"        │ │
│  └───────┬────────┘  └───────────────────┬────────────────────────┘ │
└──────────┼───────────────────────────────┼──────────────────────────┘
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   STAGE 1: DATA PROFILING ENGINE                     │
│                                                                      │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐     │
│  │ Statistical Profiler │  │ LLM Semantic Profiler            │     │
│  │ ├─ Type detection    │  │ ├─ Column meaning inference      │     │
│  │ ├─ Distributions     │  │ ├─ Sensitive column detection    │     │
│  │ ├─ Correlations      │  │ ├─ Task type classification     │     │
│  │ ├─ Missing patterns  │  │ └─ Data quality recommendations │     │
│  │ ├─ Outlier detection │  │                                  │     │
│  │ └─ Quality scoring   │  │                                  │     │
│  └─────────────────────┘  └──────────────────────────────────┘     │
│                                                                      │
│  Output: EDA Report (interactive visualizations + LLM narrative)     │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│               STAGE 2: LLM FEATURE ENGINEERING AGENT                 │
│                                                                      │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐    │
│  │ Phase 1:      │  │ Phase 2:          │  │ Phase 3:            │    │
│  │ Data Cleaning │  │ LLM-Generated     │  │ Feature Selection   │    │
│  │ ├─ Imputation │  │ Transformations   │  │ ├─ Mutual info      │    │
│  │ ├─ Outliers   │  │ ├─ Datetime feats │  │ ├─ Correlation      │    │
│  │ ├─ Dedup      │  │ ├─ Numerical feats│  │ ├─ SHAP-based       │    │
│  │ └─ Type fix   │  │ ├─ Categorical    │  │ └─ RFE              │    │
│  │               │  │ ├─ Text features  │  │                      │    │
│  │               │  │ ├─ Interactions   │  │                      │    │
│  │               │  │ └─ Domain-aware   │  │                      │    │
│  └──────────────┘  └──────────────────┘  └────────────────────┘    │
│                                                                      │
│  Code Execution Sandbox: LLM generates Pandas code → execute →       │
│  validate → if error → LLM debugs → retry (max 3)                   │
│                                                                      │
│  Output: Clean, engineered feature matrix + serializable pipeline    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  STAGE 3: MODEL TRAINING ENGINE                      │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Task Router → Binary / Multi-class / Regression / TimeSeries│    │
│  └──────────┬──────────────────────────────────────────────────┘    │
│             │                                                        │
│  ┌──────────▼──────────────────────────────────────────────────┐    │
│  │  CLASSICAL ML (Scikit-learn + XGBoost + LightGBM + CatBoost) │    │
│  │                                                               │    │
│  │  Linear:    LogReg, Ridge, Lasso, ElasticNet, SGD            │    │
│  │  Trees:     DecisionTree, RandomForest, ExtraTrees,          │    │
│  │             XGBoost, LightGBM, CatBoost                      │    │
│  │  Distance:  KNN, SVM (RBF, Poly)                             │    │
│  │  Prob:      GaussianNB, MultinomialNB                         │    │
│  │  Ensemble:  VotingClassifier, StackingClassifier              │    │
│  └──────────┬────────────────────────────────────────────────────┘    │
│             │                                                        │
│  ┌──────────▼──────────────────────────────────────────────────┐    │
│  │  DEEP LEARNING (PyTorch — custom training loops)              │    │
│  │                                                               │    │
│  │  Tabular:   MLP, TabTransformer, FT-Transformer, TabNet      │    │
│  │  TimeSeries: LSTM+Attn, TCN, TFT, N-BEATS                    │    │
│  └──────────┬────────────────────────────────────────────────────┘    │
│             │                                                        │
│  ┌──────────▼──────────────────────────────────────────────────┐    │
│  │  HYPERPARAMETER OPTIMIZATION (Optuna)                         │    │
│  │  ├─ Bayesian (TPE) per model                                  │    │
│  │  ├─ Pruning (MedianPruner — stop bad trials early)            │    │
│  │  ├─ Multi-objective (accuracy vs latency)                     │    │
│  │  ├─ 500+ total trials                                         │    │
│  │  └─ All logged in MLflow                                      │    │
│  └──────────┬────────────────────────────────────────────────────┘    │
│             │                                                        │
│  ┌──────────▼──────────────────────────────────────────────────┐    │
│  │  MODEL SELECTION                                              │    │
│  │  ├─ Compare all models: metric, latency, model size           │    │
│  │  ├─ Statistical significance (paired t-test across folds)     │    │
│  │  ├─ Pareto frontier (accuracy vs latency)                     │    │
│  │  └─ Auto-select or user choice from top 3                     │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Output: Best model + all candidate models + comparison metrics      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│            STAGE 4: EVALUATION & EXPLANATION ENGINE                   │
│                                                                      │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐  │
│  │ Metrics Suite   │ │ Explainability │ │ Error Analysis          │  │
│  │ ├─ Classification│ │ ├─ SHAP (3 types)│ ├─ Worst predictions    │  │
│  │ ├─ Regression   │ │ ├─ LIME        │ │ ├─ Confusion deep dive  │  │
│  │ ├─ Calibration  │ │ ├─ PDP / ICE   │ │ ├─ Slice analysis       │  │
│  │ └─ Fairness     │ │ ├─ Perm. Imp.  │ │ └─ LLM error patterns  │  │
│  │                 │ │ └─ Attention viz│ │                         │  │
│  └────────────────┘ └────────────────┘ └────────────────────────┘  │
│                                                                      │
│  Output: Comprehensive evaluation report + all plots + LLM narrative │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                STAGE 5: DEPLOYMENT ENGINE                             │
│                                                                      │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐  │
│  │ Model Export    │ │ API Generation │ │ Monitoring Setup        │  │
│  │ ├─ joblib       │ │ ├─ FastAPI     │ │ ├─ Evidently drift     │  │
│  │ ├─ TorchScript  │ │ ├─ Pydantic   │ │ ├─ Performance track   │  │
│  │ └─ ONNX         │ │ ├─ Dockerfile  │ │ └─ Alert rules         │  │
│  │                 │ │ └─ Swagger docs│ │                         │  │
│  └────────────────┘ └────────────────┘ └────────────────────────┘  │
│                                                                      │
│  ┌────────────────┐ ┌────────────────┐                              │
│  │ Batch Pipeline  │ │ Model Card     │                              │
│  │ ├─ Airflow DAG  │ │ ├─ Auto-filled │                              │
│  │ └─ Scheduled    │ │ └─ Limitations │                              │
│  └────────────────┘ └────────────────┘                              │
│                                                                      │
│  Output: Running API + Docker container + monitoring + model card     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. DATA PROFILING ENGINE

### 3.1 Statistical Profiler

```
File: src/profiling/statistical_profiler.py

Input: pandas DataFrame + target column name + task description

Processing:

1. Column Type Detection:
   ├── Automatic inference with override logic:
   │   ├── If unique_values < 20 AND dtype is int → CATEGORICAL
   │   ├── If dtype is float or int with high cardinality → NUMERICAL
   │   ├── If parseable as date → DATETIME
   │   ├── If average string length > 50 → TEXT
   │   ├── If column name contains "id" or all unique → ID (exclude from features)
   │   └── If dtype is bool → BINARY
   ├── Store: {column_name: detected_type} mapping
   └── LLM validates: "Given column name '{name}' with sample values {samples}, confirm type"

2. Per-Column Statistics:
   
   Numerical Columns:
   ├── count, mean, std, min, max, median
   ├── percentiles: 1st, 5th, 25th, 75th, 95th, 99th
   ├── skewness, kurtosis
   ├── number of zeros, number of negatives
   ├── number of outliers (IQR method: < Q1-1.5*IQR or > Q3+1.5*IQR)
   └── distribution fit test (Shapiro-Wilk for normality)

   Categorical Columns:
   ├── unique values count, top 10 values with frequencies
   ├── cardinality ratio (unique / total)
   ├── entropy (Shannon entropy of value distribution)
   ├── mode, mode frequency
   └── rare categories (frequency < 1%)

   Datetime Columns:
   ├── min date, max date, range
   ├── most common day of week, month
   ├── gaps in time series (missing dates)
   └── frequency detection (daily, hourly, monthly)

   Text Columns:
   ├── avg length, min/max length
   ├── vocabulary size
   ├── language detection
   └── sample values

3. Target Variable Analysis:
   ├── Classification: class distribution, class imbalance ratio
   ├── Regression: distribution, skewness, potential log-transform benefit
   └── Recommended evaluation metric based on class balance + task

4. Correlation Analysis:
   ├── Numerical × Numerical: Pearson + Spearman correlation matrix
   ├── Categorical × Target: Cramér's V, mutual information
   ├── Numerical × Target: point-biserial correlation (for binary target)
   ├── Multicollinearity: VIF (Variance Inflation Factor) for top features
   └── Flag: highly correlated feature pairs (|r| > 0.9)

5. Missing Data Analysis:
   ├── Missing count and percentage per column
   ├── Missing pattern: MCAR / MAR / MNAR estimation
   ├── Missing correlation matrix (do columns co-miss?)
   └── Recommendation: imputation strategy per column

6. Outlier Analysis:
   ├── IQR method per numerical column
   ├── Z-score method per numerical column
   ├── Visualization: box plots with outlier points
   └── Recommendation: treatment per column (clip, remove, keep)

7. Data Quality Score (0–100):
   ├── Completeness (weight: 0.3): (1 - avg_missing_rate) × 100
   ├── Uniqueness (weight: 0.2): ID columns properly unique
   ├── Consistency (weight: 0.2): types consistent, no mixed types
   ├── Validity (weight: 0.2): values within expected ranges
   └── Timeliness (weight: 0.1): data recency for datetime columns

Output: ProfileReport object with all statistics + quality score
```

### 3.2 LLM Semantic Profiler

```
File: src/profiling/semantic_profiler.py

Purpose: Use LLM to understand WHAT the data means, not just its statistics

LLM Prompt:
"You are a data scientist analyzing a dataset. Given:
- Task description: {task_description}
- Column profiles: {column_name, type, sample_values, statistics}

For each column, provide:
1. Semantic meaning (what does this column represent in the real world?)
2. Importance estimate for the target (HIGH / MEDIUM / LOW / IRRELEVANT)
3. Is this a sensitive/protected attribute? (age, gender, race, etc.)
4. Suggested feature engineering transformations
5. Potential data quality issues

Also provide:
- Overall data quality assessment
- Key feature interactions to explore
- Potential data leakage risks

Output as JSON."

Example Output:
{
  "columns": {
    "tenure_months": {
      "meaning": "Number of months the customer has been with the company",
      "importance": "HIGH",
      "sensitive": false,
      "suggested_transforms": [
        "Bin into loyalty tiers (0-6: new, 6-24: growing, 24-48: established, 48+: loyal)",
        "Create 'is_new_customer' flag (tenure < 6 months)",
        "Interaction with monthly_charges (total_lifetime_value)"
      ],
      "quality_issues": "None detected"
    },
    "gender": {
      "meaning": "Customer gender",
      "importance": "LOW",
      "sensitive": true,
      "suggested_transforms": ["Binary encode"],
      "quality_issues": "None"
    },
    "monthly_charges": {
      "meaning": "Monthly subscription cost",
      "importance": "HIGH",
      "sensitive": false,
      "suggested_transforms": [
        "Percentile rank across all customers",
        "Ratio to tenure (price_per_month_of_loyalty)",
        "Interaction with contract_type"
      ],
      "quality_issues": "Slight right skew — consider log transform"
    }
  },
  "data_quality_summary": "Good quality dataset with 0.2% missing values...",
  "key_interactions": ["tenure × monthly_charges", "contract_type × payment_method"],
  "leakage_risks": ["'total_charges' is derived from 'tenure × monthly_charges' — high correlation expected, may leak"]
}
```

### 3.3 Auto-Generated EDA Report

```
File: src/profiling/eda_report.py

Output: Interactive HTML report (served in frontend)

Report Sections:
1. Dataset Overview
   ├── Shape (rows × columns)
   ├── Data quality score
   ├── Column type breakdown (bar chart)
   └── Memory usage

2. Target Variable
   ├── Distribution (histogram for regression, bar chart for classification)
   ├── Class balance analysis
   └── Recommended metric

3. Feature Distributions
   ├── Histogram + KDE for each numerical column
   ├── Bar chart for each categorical column
   └── Highlighting outliers and skewness

4. Correlation Analysis
   ├── Heatmap (Pearson correlations)
   ├── Mutual information with target
   └── Top 10 correlated feature pairs

5. Missing Data
   ├── Missing percentage bar chart
   ├── Missing pattern heatmap (missingno library)
   └── Recommended imputation strategies

6. LLM Insights
   ├── Semantic column descriptions
   ├── Key feature interactions to explore
   ├── Data quality recommendations
   └── Potential leakage warnings
```

---

## 4. LLM FEATURE ENGINEERING AGENT

### 4.1 Cleaning Pipeline

```
File: src/feature_engineering/cleaner.py

Step 1: Missing Value Handling
├── Strategy per column (determined by profiler + LLM):
│   ├── Numerical with < 5% missing → median imputation
│   ├── Numerical with 5–30% missing → KNN imputation or iterative imputer
│   ├── Numerical with > 30% missing → drop column (warn user)
│   ├── Categorical with < 10% missing → mode imputation
│   ├── Categorical with 10–30% missing → "MISSING" category
│   └── All: create binary indicator column: {col}_is_missing
├── Imputation fitted on training data only (prevent leakage)
└── Serialized: imputation parameters saved for serving consistency

Step 2: Outlier Handling
├── Strategy per column (configurable):
│   ├── Clip: cap at 1st/99th percentile (default for most)
│   ├── Winsorize: replace outliers with boundary values
│   ├── Flag: create binary {col}_is_outlier feature
│   └── Keep: don't touch (for columns where extremes are meaningful)
├── Thresholds: IQR × 3 or Z-score > 5 (conservative)
└── Applied on training data, same thresholds applied to serving data

Step 3: Deduplication
├── Exact duplicate rows: remove
├── Near-duplicate detection: flag for review
└── Log: number of duplicates removed

Step 4: Type Correction
├── Fix mixed types (e.g., "123" and 123 in same column)
├── Parse dates from strings
├── Convert boolean strings ("True"/"False", "Yes"/"No") to binary
└── Encode target variable (LabelEncoder for classification)
```

### 4.2 LLM Feature Engineering

```
File: src/feature_engineering/llm_engineer.py

This is the CORE INNOVATION of Forge.

Process:
1. For each column (or group of related columns), send to LLM:

   Prompt:
   "You are an expert feature engineer. Given this column:
   
   Column: {column_name}
   Type: {detected_type}
   Sample values: {10 sample values}
   Statistics: {mean, std, min, max, unique_count, ...}
   Task: {task_description}
   Target: {target_column_name}
   
   Generate Python Pandas code that creates new features from this column.
   Requirements:
   - Each transformation must be a single line or small function
   - Use only pandas, numpy, and sklearn.preprocessing
   - Return a DataFrame with new columns
   - Column names must be descriptive (snake_case)
   - Do NOT modify the original column
   - Do NOT use the target variable in any transformation
   
   Output ONLY the Python code, no explanations."

2. Example LLM-Generated Transformations:

   For column 'signup_date' (datetime):
   ```python
   df['signup_day_of_week'] = pd.to_datetime(df['signup_date']).dt.dayofweek
   df['signup_month'] = pd.to_datetime(df['signup_date']).dt.month
   df['signup_quarter'] = pd.to_datetime(df['signup_date']).dt.quarter
   df['signup_is_weekend'] = df['signup_day_of_week'].isin([5, 6]).astype(int)
   df['signup_day_of_week_sin'] = np.sin(2 * np.pi * df['signup_day_of_week'] / 7)
   df['signup_day_of_week_cos'] = np.cos(2 * np.pi * df['signup_day_of_week'] / 7)
   df['days_since_signup'] = (pd.Timestamp.now() - pd.to_datetime(df['signup_date'])).dt.days
   ```

   For columns 'tenure_months' + 'monthly_charges' (numerical pair):
   ```python
   df['total_lifetime_value'] = df['tenure_months'] * df['monthly_charges']
   df['monthly_price_sensitivity'] = df['monthly_charges'] / (df['tenure_months'] + 1)
   df['tenure_bucket'] = pd.cut(df['tenure_months'], bins=[0, 6, 12, 24, 48, 999],
                                 labels=['very_new', 'new', 'mid', 'established', 'loyal'])
   df['is_new_customer'] = (df['tenure_months'] <= 6).astype(int)
   df['charges_percentile'] = df['monthly_charges'].rank(pct=True)
   df['charges_zscore'] = (df['monthly_charges'] - df['monthly_charges'].mean()) / df['monthly_charges'].std()
   ```

   For column 'contract_type' (categorical):
   ```python
   df['is_month_to_month'] = (df['contract_type'] == 'Month-to-month').astype(int)
   contract_freq = df['contract_type'].value_counts(normalize=True).to_dict()
   df['contract_type_frequency'] = df['contract_type'].map(contract_freq)
   ```

3. Code Execution Sandbox:

   File: src/feature_engineering/code_sandbox.py

   Implementation:
   ├── Restricted execution environment:
   │   ├── Allowed imports: pandas, numpy, sklearn.preprocessing, math, datetime, re
   │   ├── Blocked: os, sys, subprocess, socket, requests, open()
   │   ├── Timeout: 30 seconds per transformation
   │   └── Memory limit: 2GB
   │
   ├── Execution flow:
   │   1. Parse LLM-generated code
   │   2. Validate: check for forbidden imports/functions
   │   3. Execute on a COPY of the dataframe
   │   4. Validate output:
   │      a. No NaN explosion (new columns should have < 50% NaN)
   │      b. Correct number of rows (same as input)
   │      c. No constant columns (all same value)
   │      d. No duplicate of existing columns
   │      e. Data types are valid (no objects that should be numeric)
   │   5. If validation fails: send error back to LLM → regenerate → retry (max 3)
   │   6. If all retries fail: skip this transformation, log warning
   │
   └── Logging:
       ├── All generated code stored (reproducibility)
       ├── Execution time per transformation
       └── Error messages for failed transformations

4. Iterative Refinement:

   After initial model training (quick XGBoost run):
   ├── Compute SHAP feature importances
   ├── Send to LLM:
   │   "Current feature importances: {top_20_features_with_shap_values}
   │    Weakest areas: {features_with_low_importance}
   │    Task: {task_description}
   │    
   │    Generate ADDITIONAL features that might improve the model.
   │    Focus on:
   │    - Interaction features between the most important features
   │    - Non-linear transforms of important features
   │    - Features that capture the weaknesses identified"
   ├── Execute new features
   ├── Re-train and compare metrics
   └── Keep new features only if they improve the evaluation metric
```

### 4.3 Feature Selection

```
File: src/feature_engineering/feature_selector.py

Methods (applied in sequence):

1. Correlation Filtering
   ├── Remove features with |correlation| > 0.95 with another feature
   ├── Keep the one with higher correlation to target
   └── Log removed features and reason

2. Mutual Information Scoring
   ├── sklearn.feature_selection.mutual_info_classif / mutual_info_regression
   ├── Rank all features by MI with target
   └── Remove features with MI < 0.01 (near-zero information)

3. SHAP-Based Selection
   ├── Train quick XGBoost/LightGBM model
   ├── Compute SHAP values for all features
   ├── Remove features with mean(|SHAP|) < threshold
   └── Threshold: adaptive based on feature count

4. Recursive Feature Elimination (RFE)
   ├── sklearn.feature_selection.RFECV
   ├── With cross-validation to find optimal number of features
   ├── Base estimator: LightGBM (fast)
   └── Step: remove 10% of features per iteration

5. LLM Sanity Check
   ├── Send final feature list to LLM:
   │   "Review these features for a {task_description} task:
   │    {feature_list_with_importance_scores}
   │    Flag any features that:
   │    - Might cause data leakage
   │    - Are redundant
   │    - Don't make domain sense"
   └── Remove any LLM-flagged features after human review

Output:
├── Selected feature list with importance ranking
├── Feature selection report (which features removed and why)
└── Serializable feature pipeline (same transforms applied at serving time)
```

---

## 5. MODEL TRAINING ENGINE

### 5.1 Task Router

```
File: src/training/task_router.py

Detection Logic:
├── Binary Classification:
│   └── Target has exactly 2 unique values
├── Multi-Class Classification:
│   └── Target has 3–50 unique values AND (target is categorical OR integer with low cardinality)
├── Regression:
│   └── Target is numerical with high cardinality (> 50 unique values)
├── Multi-Label Classification:
│   └── Target is multi-hot encoded (multiple columns, all binary)
├── Time Series Forecasting:
│   └── Dataset has datetime index + target is numerical + user description mentions "forecast"
└── Override: user can specify task type explicitly
```

### 5.2 Classical ML Models

```
File: src/training/classical/

Each model is implemented as a class with:
- get_search_space() → Optuna search space
- train(X_train, y_train, params) → fitted model
- predict(X_test) → predictions
- predict_proba(X_test) → probabilities (classification)
- get_feature_importance() → importance scores

=== LINEAR MODELS ===

LogisticRegression:
  File: src/training/classical/logistic_regression.py
  Library: sklearn.linear_model.LogisticRegression
  Search Space:
    C: loguniform(1e-4, 1e2)
    penalty: categorical(['l1', 'l2', 'elasticnet'])
    solver: conditional on penalty
    class_weight: categorical([None, 'balanced'])
    max_iter: 1000
  
  Why implement: Interpretable baseline, coefficient-based feature importance

RidgeRegression:
  File: src/training/classical/ridge.py
  Library: sklearn.linear_model.Ridge
  Search Space:
    alpha: loguniform(1e-4, 1e3)
    fit_intercept: categorical([True, False])

LassoRegression:
  File: src/training/classical/lasso.py
  Library: sklearn.linear_model.Lasso
  Search Space:
    alpha: loguniform(1e-4, 1e2)
  Why: L1 produces sparse solutions — automatic feature selection

ElasticNet:
  File: src/training/classical/elastic_net.py
  Search Space:
    alpha: loguniform(1e-4, 1e2)
    l1_ratio: uniform(0.0, 1.0)

SGDClassifier:
  File: src/training/classical/sgd.py
  Library: sklearn.linear_model.SGDClassifier
  Search Space:
    loss: categorical(['hinge', 'log_loss', 'modified_huber'])
    alpha: loguniform(1e-6, 1e-1)
    penalty: categorical(['l1', 'l2', 'elasticnet'])
  Why: Scales to very large datasets (online learning)

=== TREE-BASED MODELS ===

DecisionTree:
  File: src/training/classical/decision_tree.py
  Library: sklearn.tree.DecisionTreeClassifier / Regressor
  Search Space:
    max_depth: int(2, 20)
    min_samples_split: int(2, 50)
    min_samples_leaf: int(1, 20)
    criterion: categorical(['gini', 'entropy']) / (['mse', 'mae'])
    max_features: categorical(['sqrt', 'log2', None])
  Why: Interpretable, good baseline, visualizable

RandomForest:
  File: src/training/classical/random_forest.py
  Library: sklearn.ensemble.RandomForestClassifier / Regressor
  Search Space:
    n_estimators: int(100, 1000)
    max_depth: int(3, 20) or None
    min_samples_split: int(2, 30)
    min_samples_leaf: int(1, 15)
    max_features: categorical(['sqrt', 'log2', 0.5, 0.8])
    class_weight: categorical([None, 'balanced', 'balanced_subsample'])
    n_jobs: -1

ExtraTrees:
  File: src/training/classical/extra_trees.py
  Library: sklearn.ensemble.ExtraTreesClassifier / Regressor
  Search Space: same as RandomForest
  Why: More randomized than RF, sometimes better generalization

XGBoost:
  File: src/training/classical/xgboost_model.py
  Library: xgboost.XGBClassifier / XGBRegressor
  Search Space:
    n_estimators: int(100, 2000)
    max_depth: int(3, 10)
    learning_rate: loguniform(0.005, 0.3)
    min_child_weight: int(1, 20)
    subsample: uniform(0.6, 1.0)
    colsample_bytree: uniform(0.6, 1.0)
    gamma: loguniform(1e-8, 5.0)
    reg_alpha: loguniform(1e-8, 10.0)
    reg_lambda: loguniform(1e-8, 10.0)
    tree_method: 'hist'
    eval_metric: task-dependent
    early_stopping_rounds: 50
    scale_pos_weight: auto (for imbalanced)

LightGBM:
  File: src/training/classical/lightgbm_model.py
  Library: lightgbm.LGBMClassifier / LGBMRegressor
  Search Space:
    n_estimators: int(100, 2000)
    max_depth: int(3, 12)
    learning_rate: loguniform(0.005, 0.3)
    num_leaves: int(15, 127)
    min_child_samples: int(5, 100)
    subsample: uniform(0.5, 1.0)
    colsample_bytree: uniform(0.5, 1.0)
    reg_alpha: loguniform(1e-8, 10.0)
    reg_lambda: loguniform(1e-8, 10.0)
    min_split_gain: loguniform(1e-8, 1.0)
    verbose: -1

CatBoost:
  File: src/training/classical/catboost_model.py
  Library: catboost.CatBoostClassifier / CatBoostRegressor
  Search Space:
    iterations: int(200, 2000)
    depth: int(3, 10)
    learning_rate: loguniform(0.01, 0.3)
    l2_leaf_reg: loguniform(1e-3, 10.0)
    bagging_temperature: uniform(0.0, 1.0)
    random_strength: uniform(0.0, 1.0)
    border_count: int(32, 255)
    auto_class_weights: categorical([None, 'Balanced', 'SqrtBalanced'])
    cat_features: auto-detected categorical columns
  Why: Native categorical handling — no encoding needed

=== DISTANCE/KERNEL MODELS ===

KNN:
  File: src/training/classical/knn.py
  Library: sklearn.neighbors.KNeighborsClassifier / Regressor
  Search Space:
    n_neighbors: int(3, 50)
    weights: categorical(['uniform', 'distance'])
    metric: categorical(['euclidean', 'manhattan', 'minkowski'])
    p: int(1, 5) (for minkowski)

SVM:
  File: src/training/classical/svm.py
  Library: sklearn.svm.SVC / SVR
  Search Space:
    C: loguniform(1e-3, 1e3)
    kernel: categorical(['rbf', 'poly'])
    gamma: categorical(['scale', 'auto']) or loguniform(1e-5, 1e1)
    degree: int(2, 5) (for poly kernel)
    class_weight: categorical([None, 'balanced'])
  Note: Use with PCA-reduced features if n_features > 50 (SVM doesn't scale)

=== PROBABILISTIC MODELS ===

GaussianNB:
  File: src/training/classical/naive_bayes.py
  Library: sklearn.naive_bayes.GaussianNB
  Search Space:
    var_smoothing: loguniform(1e-12, 1e-3)
  Why: Very fast, good baseline for text-like features

=== ENSEMBLE MODELS ===

VotingClassifier:
  File: src/training/classical/voting.py
  Library: sklearn.ensemble.VotingClassifier
  Implementation: After individual models are trained, create:
    - Hard voting (majority vote) from top 3 models
    - Soft voting (average probabilities) from top 3 models
  No HPO needed — uses already-tuned base models

StackingClassifier:
  File: src/training/classical/stacking.py
  Library: sklearn.ensemble.StackingClassifier
  Implementation:
    - Base estimators: top 5 models (after individual HPO)
    - Meta-learner: LogisticRegression (simple, avoids overfitting)
    - 5-fold cross-validation for base model predictions
    - Final meta-learner trained on OOF predictions
  Why: Often the best overall model — combines diverse model strengths
```

### 5.3 Deep Learning Models

```
File: src/training/deep_learning/

All DL models use CUSTOM PyTorch training loops (not just Trainer API).

=== COMMON TRAINING INFRASTRUCTURE ===

File: src/training/deep_learning/trainer.py

class DLTrainer:
    """Generic PyTorch training loop with all best practices"""
    
    def train(self, model, train_loader, val_loader, config):
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=config.lr, epochs=config.epochs,
                               steps_per_epoch=len(train_loader))
        scaler = GradScaler()  # Mixed precision
        
        best_val_metric = float('-inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            # Training
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                with autocast():  # fp16
                    outputs = model(batch)
                    loss = self.compute_loss(outputs, batch.targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Log to W&B
                wandb.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
            
            # Validation
            val_metric = self.evaluate(model, val_loader)
            wandb.log({"val_metric": val_metric, "epoch": epoch})
            
            # Early stopping
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
                self.save_checkpoint(model, "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break
        
        model.load_state_dict(torch.load("best_model.pt"))
        return model

Features:
├── Mixed precision training (fp16/bf16 via torch.cuda.amp)
├── Gradient accumulation (configurable effective batch size)
├── Gradient clipping (max_norm=1.0)
├── Learning rate scheduling (OneCycleLR default, configurable)
├── Early stopping with patience
├── Weights & Biases integration (all metrics + loss curves)
├── Best model checkpoint saving
├── Reproducibility (seed all random, deterministic algorithms)
└── GPU memory optimization (pin_memory, non_blocking transfers)

=== TABULAR DEEP LEARNING MODELS ===

MLP (Multi-Layer Perceptron):
  File: src/training/deep_learning/mlp.py
  
  Architecture:
  ├── Input: (batch, n_features)
  ├── Embedding layer for categorical features (learned embeddings)
  ├── Concatenate: [numerical_features || cat_embeddings]
  ├── Hidden layers (configurable depth):
  │   ├── Linear(in, hidden) → BatchNorm → GELU → Dropout
  │   ├── Residual connections every 2 layers (if depth > 3)
  │   └── Default: [256, 128, 64]
  ├── Output: Linear(64, n_classes) → Softmax (classification)
  │          Linear(64, 1) (regression)
  
  Search Space:
    n_layers: int(2, 5)
    hidden_dim: categorical([64, 128, 256, 512])
    dropout: uniform(0.1, 0.5)
    learning_rate: loguniform(1e-4, 1e-2)
    batch_size: categorical([64, 128, 256, 512])
    weight_decay: loguniform(1e-6, 1e-2)
    embedding_dim: int(4, 32)
    use_batch_norm: categorical([True, False])

TabTransformer:
  File: src/training/deep_learning/tab_transformer.py
  
  Architecture:
  ├── Categorical features → Learned Embeddings → Transformer Encoder
  │   ├── Column embedding (per categorical column)
  │   ├── Multi-head self-attention (4 heads, 2 layers)
  │   ├── Feed-forward network per layer
  │   └── Output: contextualized categorical embeddings
  ├── Numerical features → MLP (Linear → ReLU → Linear)
  ├── Concatenate: [transformed_categoricals || numerical_mlp_output]
  ├── MLP head: Linear(combined, 128) → ReLU → Dropout → Linear(128, output)
  
  Search Space:
    d_model: categorical([32, 64, 128])
    n_heads: categorical([2, 4, 8])
    n_layers: int(1, 4)
    dropout: uniform(0.1, 0.4)
    learning_rate: loguniform(1e-4, 5e-3)
    embedding_dim: int(8, 32)

FT-Transformer (Feature Tokenizer + Transformer):
  File: src/training/deep_learning/ft_transformer.py
  
  Architecture:
  ├── ALL features (numerical + categorical) tokenized:
  │   ├── Numerical: Linear(1, d_model) per feature
  │   ├── Categorical: Embedding(n_categories, d_model) per feature
  │   ├── [CLS] token prepended
  │   └── Result: (batch, n_features + 1, d_model)
  ├── Transformer Encoder (multi-head self-attention)
  │   ├── Same as standard Transformer encoder
  │   ├── Learns inter-feature interactions
  │   └── n_layers: 3, n_heads: 8
  ├── CLS token → MLP head → output
  
  Search Space:
    d_model: categorical([64, 128, 192, 256])
    n_heads: categorical([4, 8])
    n_layers: int(2, 6)
    d_ff_factor: uniform(1.33, 4.0)
    dropout: uniform(0.1, 0.3)
    attention_dropout: uniform(0.0, 0.2)
    learning_rate: loguniform(1e-5, 1e-3)

TabNet:
  File: src/training/deep_learning/tabnet.py
  Library: pytorch_tabnet
  
  Architecture:
  ├── Sequential attention mechanism (step-by-step feature selection)
  ├── Sparse feature selection (built-in interpretability)
  ├── Shared + decision-specific layers
  ├── Entropy-based regularization for sparsity
  
  Search Space:
    n_d: int(8, 64)              (width of decision layers)
    n_a: int(8, 64)              (width of attention layers)
    n_steps: int(3, 10)          (number of sequential steps)
    gamma: uniform(1.0, 2.0)     (coefficient for feature reuse)
    lambda_sparse: loguniform(1e-6, 1e-2)
    learning_rate: loguniform(1e-4, 1e-2)
    batch_size: categorical([256, 512, 1024])

=== TIME SERIES DEEP LEARNING ===

LSTM with Attention:
  File: src/training/deep_learning/lstm_attention.py
  
  Architecture:
  ├── Input: (batch, seq_len, n_features)
  ├── LSTM(n_features, hidden=128, num_layers=2, bidirectional=True)
  ├── Bahdanau Attention over LSTM outputs
  ├── Context vector → Linear(256, 64) → ReLU → Linear(64, 1)
  
  Search Space:
    hidden_size: categorical([64, 128, 256])
    num_layers: int(1, 3)
    bidirectional: categorical([True, False])
    dropout: uniform(0.1, 0.5)
    seq_len: int(16, 128)

TCN (Temporal Convolutional Network):
  File: src/training/deep_learning/tcn.py
  (Same architecture as defined in Sentinel — reusable module)

Temporal Fusion Transformer (TFT):
  File: src/training/deep_learning/tft.py
  
  Architecture:
  ├── Variable Selection Networks (per input type: static, past, future)
  ├── Static Covariate Encoders (context vectors from static features)
  ├── LSTM Encoder + LSTM Decoder (seq2seq backbone)
  ├── Temporal Self-Attention (interpretable multi-head attention)
  ├── Gated Residual Networks (throughout)
  ├── Multi-horizon output (predict multiple future timesteps)
  
  Search Space:
    hidden_size: categorical([32, 64, 128, 160])
    attention_heads: categorical([1, 4])
    dropout: uniform(0.1, 0.4)
    learning_rate: loguniform(1e-4, 1e-2)
  
  Why: State-of-the-art for time series with mixed inputs (static + temporal)

N-BEATS (Neural Basis Expansion):
  File: src/training/deep_learning/nbeats.py
  
  Architecture:
  ├── Stack of fully-connected blocks
  ├── Each block: MLP → fork into backcast (explain past) and forecast (predict future)
  ├── Residual: next block operates on what previous block couldn't explain
  ├── Interpretable version: trend + seasonality stacks
  
  Search Space:
    n_stacks: int(2, 5)
    n_blocks: int(1, 3)
    hidden_dim: categorical([128, 256, 512])
    share_weights: categorical([True, False])
```

---

## 6. HYPERPARAMETER OPTIMIZATION

```
File: src/training/hpo/optuna_optimizer.py

Framework: Optuna

Configuration:
├── Sampler: TPESampler (Tree-structured Parzen Estimator) — Bayesian
├── Pruner: MedianPruner (stop trial if intermediate value < median of completed trials)
│   ├── n_startup_trials: 10 (don't prune first 10 trials)
│   ├── n_warmup_steps: 5 (don't prune first 5 folds)
│   └── interval_steps: 1
├── Trials per model:
│   ├── Linear models: 30 trials
│   ├── Tree models: 50 trials
│   ├── XGBoost/LightGBM/CatBoost: 100 trials
│   ├── DL models: 30 trials (more expensive)
│   └── Total: ~500–800 trials across all models
├── Cross-validation: Stratified 5-fold (classification) or 5-fold (regression)
│   Time series: TimeSeriesSplit with 5 folds
├── Metric: configurable (default: ROC-AUC for binary, F1-macro for multi-class, RMSE for regression)
└── Timeout: 30 minutes per model family (safety limit)

Multi-Objective Optimization:
├── Objectives: (1) primary metric, (2) inference latency
├── Pareto frontier: set of models where you can't improve one without hurting the other
├── Visualization: Pareto plot in dashboard
└── User can click on any point in Pareto frontier to select that tradeoff

MLflow Integration:
├── Every trial logged as MLflow run:
│   ├── Parameters: all hyperparameters
│   ├── Metrics: CV score (mean + std), training time, inference latency
│   ├── Artifacts: trained model (for best trial only)
│   └── Tags: model_family, trial_number, pruned
├── Best model per family registered in MLflow Model Registry
└── Parent run groups all trials for a model family
```

---

## 7. EVALUATION & EXPLANATION ENGINE

### 7.1 Metrics Suite

```
File: src/evaluation/metrics_calculator.py

=== CLASSIFICATION METRICS ===
├── Accuracy
├── Precision (macro, micro, weighted, per-class)
├── Recall (macro, micro, weighted, per-class)
├── F1 Score (macro, micro, weighted, per-class)
├── ROC-AUC (binary: single value, multi-class: OVR macro)
├── PR-AUC (Average Precision)
├── MCC (Matthews Correlation Coefficient)
├── Log Loss
├── Cohen's Kappa
├── Confusion Matrix (with percentages)
├── Classification Report (per-class breakdown)

=== REGRESSION METRICS ===
├── MSE, RMSE
├── MAE
├── MAPE (Mean Absolute Percentage Error)
├── R² (coefficient of determination)
├── Adjusted R²
├── Max Error
├── Explained Variance

=== CALIBRATION METRICS ===
├── Brier Score (reliability of probabilities)
├── Calibration Curve (reliability diagram)
├── Expected Calibration Error (ECE)
├── Platt Scaling (post-hoc calibration if needed)
├── Isotonic Regression (non-parametric calibration)

=== FAIRNESS METRICS (if sensitive columns detected) ===
├── Demographic Parity: P(ŷ=1 | group=A) ≈ P(ŷ=1 | group=B)
├── Equalized Odds: TPR and FPR equal across groups
├── Calibration across groups: P(y=1 | ŷ=p, group) ≈ p for all groups
├── Disparate Impact Ratio: P(ŷ=1 | group=A) / P(ŷ=1 | group=B) — should be 0.8–1.25
├── Performance per group: accuracy, precision, recall per sensitive attribute value
└── Flagging: alert if any fairness metric deviates beyond threshold
```

### 7.2 Explainability Suite

```
File: src/evaluation/explainability.py

1. SHAP Values:
   ├── TreeExplainer: for tree-based models (XGBoost, LightGBM, CatBoost, RF)
   │   └── Fast, exact SHAP values for tree models
   ├── DeepExplainer: for PyTorch DL models (MLP, TabTransformer, etc.)
   │   └── Approximation based on DeepLIFT
   ├── KernelExplainer: for any model (SVM, KNN, etc.)
   │   └── Model-agnostic, slower but universal
   │
   ├── Visualizations generated:
   │   ├── Summary plot (beeswarm): global feature importance + value impact
   │   ├── Bar plot: mean |SHAP| per feature (simple importance ranking)
   │   ├── Waterfall plot: per-prediction explanation (top N features)
   │   ├── Dependence plot: SHAP vs feature value (for top 5 features)
   │   └── Interaction plot: SHAP interaction values for top feature pairs

2. LIME (Local Interpretable Model-agnostic Explanations):
   ├── Library: lime
   ├── For individual predictions: generate local linear approximation
   ├── Show which features pushed prediction in each direction
   └── Complement to SHAP — different perspective

3. Partial Dependence Plots (PDP) + ICE:
   ├── Library: sklearn.inspection.PartialDependenceDisplay
   ├── PDP: average effect of a feature on prediction
   ├── ICE: individual conditional expectation (per-sample PDP)
   ├── Generate for top 5 most important features
   └── 2D PDP for top 2 interaction pairs

4. Permutation Importance:
   ├── Library: sklearn.inspection.permutation_importance
   ├── Model-agnostic feature importance
   ├── Shuffle each feature → measure performance drop
   └── More robust than built-in importance for some models

5. Attention Visualization (for Transformer models):
   ├── Extract attention weights from TabTransformer / FT-Transformer
   ├── Heatmap: which features attend to which other features
   ├── Per-prediction: attention pattern for specific instances
   └── Useful for understanding learned feature interactions
```

### 7.3 Error Analysis

```
File: src/evaluation/error_analysis.py

1. Worst Predictions Analysis:
   ├── Select top 20 predictions with highest loss
   ├── For each: show features, true label, predicted label, probability
   ├── Cluster worst predictions: do they share patterns?
   └── LLM analyzes: "Why might the model be struggling with these cases?"

2. Confusion Matrix Deep Dive (classification):
   ├── Per confused class pair: show example misclassifications
   ├── Feature distribution comparison between confused classes
   └── LLM insight: "Class A and Class B have overlapping tenure distributions..."

3. Slice Analysis:
   ├── Performance stratified by each feature:
   │   For each categorical feature: accuracy per category
   │   For each numerical feature: accuracy per quartile
   ├── Identify underperforming slices
   └── Flag: any slice with performance > 10% below average

4. Residual Analysis (regression):
   ├── Residual distribution (should be normal, centered at 0)
   ├── Residuals vs predicted (should be random, no pattern)
   ├── Residuals vs each feature (identify heteroscedasticity)
   └── QQ plot for normality check
```

---

## 8. LLM REPORT GENERATOR

```
File: src/evaluation/llm_report_generator.py

Purpose: Generate natural language analysis of all ML results

LLM Prompt:
"You are a senior data scientist writing an analysis report. Given:

Dataset: {dataset_description}
Task: {task_description}
Features: {n_original} original → {n_engineered} after feature engineering
Best Model: {model_name} with {metric_name} = {metric_value}
Runner-up: {runner_up_model} with {metric_value}

Feature Importance (top 10):
{feature_importance_list}

Key findings from error analysis:
{error_analysis_summary}

Fairness metrics (if applicable):
{fairness_summary}

Write a comprehensive but concise report covering:
1. Executive Summary (2-3 sentences)
2. Feature Engineering Insights (which LLM-generated features were most useful and why)
3. Model Selection Rationale (why the best model won, what it does well)
4. Key Feature Insights (interpret the top features in business terms)
5. Error Patterns (what the model gets wrong and why)
6. Fairness Assessment (if applicable)
7. Recommendations (3-5 specific actions to improve further)
8. Deployment Considerations (latency, monitoring, when to retrain)

Use concrete numbers and be specific. No vague platitudes."

Output Format:
├── Structured markdown document
├── 500–1,000 words
├── Inline references to specific metrics and features
└── Actionable recommendations

Example Output Section:
"## Feature Engineering Insights
The LLM-generated feature 'monthly_price_sensitivity' (monthly_charges / (tenure + 1))
became the 3rd most important feature with SHAP value 0.18. This captures the intuition
that customers paying high rates relative to their loyalty are more likely to churn.
The feature 'total_lifetime_value' (tenure × charges) was the 5th most important,
representing the company's financial investment in retaining each customer."
```

---

## 9. DEPLOYMENT ENGINE

```
File: src/deployment/

9.1 Model Export:
├── Classical ML: joblib serialization (full pipeline: preprocessor + model)
├── PyTorch DL: TorchScript (traced) for production inference
├── ONNX: convert PyTorch models for cross-platform deployment
├── Feature pipeline: entire preprocessing + feature engineering serialized alongside
└── All saved to MLflow artifacts

9.2 Auto-Generated FastAPI:
File: src/deployment/api_generator.py

Generates:
├── main.py — FastAPI application
│   ├── POST /predict — single prediction
│   │   Input: JSON with raw feature values
│   │   Output: {prediction, probability, confidence, top_features}
│   ├── POST /predict/batch — batch predictions
│   ├── POST /explain — prediction + SHAP explanation
│   ├── GET /health — model loaded, dependencies available
│   └── GET /model-info — model name, version, metrics, features
├── schemas.py — Auto-generated Pydantic models from feature definitions
│   ├── Input validation: correct types, ranges, required fields
│   ├── Output schema: prediction + metadata
│   └── Error responses
├── model_loader.py — Load model + preprocessor on startup
├── Dockerfile — Multi-stage build, optimized
├── docker-compose.yml — API + Redis (prediction caching)
└── requirements.txt — Pinned dependencies

Performance:
├── Classical ML inference: < 50ms
├── DL model inference: < 200ms
├── Batch (1000 rows): < 5 seconds
└── Caching: Redis LRU cache for repeated inputs

9.3 Batch Prediction Pipeline:
├── Airflow DAG: scheduled batch scoring
│   ├── Load new data from source (CSV/SQL)
│   ├── Apply feature pipeline
│   ├── Score with production model
│   ├── Store predictions (PostgreSQL/S3)
│   └── Generate summary report
└── Configurable schedule (daily, hourly, on-demand)

9.4 Model Card:
File: src/deployment/model_card_generator.py
├── Auto-generated documentation:
│   ├── Model name, version, training date
│   ├── Task description, target variable
│   ├── Training data: size, date range, features
│   ├── Performance metrics (all computed metrics)
│   ├── Fairness assessment (if applicable)
│   ├── Known limitations (from error analysis)
│   ├── Intended use and out-of-scope use
│   └── Feature descriptions and expected ranges
└── Format: Markdown → rendered in dashboard
```

---

## 10. MONITORING SETUP

```
File: src/monitoring/

Auto-configured when model is deployed:

1. Data Drift (Evidently):
   ├── Compare incoming feature distributions vs training data
   ├── PSI per feature, overall drift score
   ├── Report generated daily
   └── Alert if PSI > 0.25 for any feature

2. Prediction Drift:
   ├── Monitor prediction distribution over time
   ├── Alert if distribution shifts significantly
   └── Track confidence score distribution

3. Performance Monitoring:
   ├── If ground truth labels available (delayed):
   │   ├── Compute live metrics vs training metrics
   │   ├── Alert if any metric drops > 5%
   │   └── Track performance over time
   └── If no labels: rely on drift monitoring as proxy

4. System Monitoring:
   ├── Inference latency (p50, p95, p99)
   ├── Request volume
   ├── Error rate
   └── Memory/CPU usage
```

---

## 11–14. BACKEND, FRONTEND, DB, DEVOPS

### Backend API Endpoints

```
POST /api/v1/experiments — Create new experiment (upload dataset + task description)
GET  /api/v1/experiments/{id}/status — Get experiment status
GET  /api/v1/experiments/{id}/profile — Get EDA report
GET  /api/v1/experiments/{id}/features — Get feature engineering results
POST /api/v1/experiments/{id}/features/approve — Approve/reject LLM features
GET  /api/v1/experiments/{id}/training — Get training progress
GET  /api/v1/experiments/{id}/models — Get all model results
GET  /api/v1/experiments/{id}/evaluation — Get evaluation report
GET  /api/v1/experiments/{id}/explanation/{model_id} — Get SHAP/LIME for a model
POST /api/v1/experiments/{id}/deploy — Deploy best model
POST /api/v1/experiments/{id}/predict — Prediction playground
GET  /api/v1/experiments/{id}/monitoring — Get drift/performance reports
GET  /api/v1/experiments/{id}/report — Get LLM-generated analysis report
```

### Frontend Pages

```
1. NEW EXPERIMENT: Upload dataset + describe task → start pipeline
2. EDA DASHBOARD: Interactive visualizations, LLM insights, quality score
3. FEATURE REVIEW: See LLM-proposed features, approve/edit/reject each
4. TRAINING PROGRESS: Real-time loss curves, trial progress, ETA
5. MODEL COMPARISON: Side-by-side metrics, Pareto frontier, rank table
6. EXPLANATION EXPLORER: SHAP plots, PDP, LIME, attention — interactive
7. ERROR ANALYSIS: Worst predictions, slice performance, confusion deep dive
8. DEPLOYMENT: One-click deploy, prediction playground, API docs
9. MONITORING: Drift detection, performance tracking, alerts
10. REPORT: LLM-generated analysis document
```

### Database Schema

```
experiments: id, name, task_description, dataset_path, status, config, created_at
datasets: id, experiment_id, original_path, profile_report, n_rows, n_cols
features: id, experiment_id, name, source (original/llm_generated), importance, selected
models: id, experiment_id, model_family, model_name, hyperparameters, metrics, training_time, inference_latency, stage
evaluations: id, model_id, metrics, shap_summary, error_analysis, fairness_report
deployments: id, model_id, api_url, docker_image, status, deployed_at
predictions: id, deployment_id, input_data, prediction, confidence, created_at
drift_reports: id, deployment_id, report_type, drift_score, created_at
```

---

## 15. TESTING STRATEGY

```
Unit Tests:
├── test_profiler.py — type detection, statistics, quality scoring
├── test_cleaner.py — imputation, outlier handling, deduplication
├── test_feature_engineer.py — each transformation type, sandbox validation
├── test_all_models.py — each model trains on toy data, correct output shape
├── test_optuna.py — search space validity, pruning works
├── test_metrics.py — all metrics computed correctly on known data
├── test_shap.py — SHAP values sum to prediction, correct shapes
├── test_api_generator.py — generated API passes validation
├── test_model_export.py — save/load round-trip, predictions match
└── Coverage: 85%+

Integration Tests:
├── test_full_pipeline.py — upload CSV → profile → features → train → evaluate → deploy
├── test_llm_feature_engineering.py — LLM generates valid code, sandbox catches errors
├── test_deployment.py — deployed API serves correct predictions
└── test_monitoring.py — drift detection triggers correctly

Benchmark Tests:
├── test_kaggle_titanic.py — ROC-AUC ≥ 0.88
├── test_uci_adult.py — F1 ≥ 0.87
├── test_house_prices.py — top 15% Kaggle RMSE
├── test_credit_fraud.py — PR-AUC ≥ 0.85
├── test_telco_churn.py — Recall ≥ 0.85
└── test_walmart_sales.py — MAPE ≤ 5%
```

---

## 16. BENCHMARK DATASETS

```
Required benchmarks (run all, report results in README):

1. Kaggle Titanic — Binary classification, small (891 rows), mixed types
2. UCI Adult Income — Binary classification, medium (48K rows), fairness testing
3. Kaggle House Prices — Regression, medium (1460 rows), heavy feature engineering
4. Credit Card Fraud — Imbalanced binary classification (284K rows, 0.17% positive)
5. Telco Customer Churn — Binary classification (7K rows), domain features
6. Walmart Sales Forecasting — Time series regression (421K rows)

For each: report Forge results vs (a) manual baseline, (b) AutoML baseline (H2O/auto-sklearn)
```

---

## 17. PERFORMANCE TARGETS

```
┌───────────────────────────────────┬──────────────────┐
│ Metric                            │ Target           │
├───────────────────────────────────┼──────────────────┤
│ Kaggle Titanic ROC-AUC            │ ≥ 0.88           │
│ UCI Adult F1                      │ ≥ 0.87           │
│ House Prices: Top 15% Kaggle RMSE │ Yes              │
│ Credit Fraud PR-AUC              │ ≥ 0.85           │
│ Telco Churn Recall               │ ≥ 0.85           │
│ Walmart Sales MAPE               │ ≤ 5%             │
│ LLM features: useful rate        │ ≥ 60%            │
│ LLM features: metric lift        │ +2–5%            │
│ Code generation success          │ ≥ 85% first pass │
│ Self-correction success          │ ≥ 90%            │
│ Models implemented               │ 26               │
│ HPO trials total                 │ 500+             │
│ End-to-end time (upload→deploy)  │ < 30 minutes     │
│ API inference latency (classical)│ < 50ms           │
│ API inference latency (DL)       │ < 200ms          │
│ Test coverage                    │ ≥ 85%            │
└───────────────────────────────────┴──────────────────┘
```

---

## 18. BUILD PHASES

```
Phase 1 — Core Pipeline MVP (Week 1–2):
├── Data profiler (statistical only)
├── Basic feature engineering (encoding, scaling, imputation — no LLM yet)
├── 5 classical models: LogReg, RF, XGBoost, LightGBM, CatBoost
├── Basic Optuna HPO (30 trials per model)
├── Evaluation: accuracy, F1, ROC-AUC, confusion matrix
├── MLflow tracking
├── Test on Titanic + Adult datasets
└── Deliverable: CLI tool that takes CSV → outputs best model + metrics

Phase 2 — LLM Feature Engineering + More Models (Week 3–4):
├── LLM semantic profiler
├── LLM feature engineering agent with code sandbox
├── All remaining classical models (KNN, SVM, NB, DT, ExtraTrees, SGD, ElasticNet)
├── Ensemble models (Voting, Stacking)
├── SHAP explainability (TreeExplainer + KernelExplainer)
├── Feature selection pipeline
├── Auto-generated EDA report
├── FastAPI backend + basic React frontend
└── Deliverable: Web app with LLM-powered feature engineering

Phase 3 — Deep Learning + Full Explainability (Week 5–6):
├── MLP, TabTransformer, FT-Transformer, TabNet (PyTorch)
├── LSTM, TCN, TFT, N-BEATS (time series)
├── Custom training loops (mixed precision, gradient accumulation, W&B)
├── DeepExplainer (SHAP for DL), LIME, PDP/ICE, attention visualization
├── Error analysis engine
├── LLM report generator
├── Fairness metrics
├── Full multi-objective Optuna (accuracy vs latency Pareto)
└── Deliverable: Full model suite with comprehensive evaluation

Phase 4 — Deployment + Monitoring + Polish (Week 7–8):
├── Auto-generated FastAPI + Dockerfile deployment
├── Batch prediction pipeline (Airflow)
├── Evidently drift monitoring (auto-configured)
├── Model card generation
├── Full React dashboard (all 10 pages)
├── Prediction playground
├── Benchmark all 6 datasets, document results
├── Full test suite (unit + integration + benchmark)
├── Docker Compose for full stack
├── Clean GitHub README with architecture diagram + benchmark results
└── Deliverable: Production-grade AutoML platform
```

---

## 19. TECH STACK SUMMARY

```
Language          │ Python 3.11+, TypeScript
Classical ML      │ Scikit-learn, XGBoost, LightGBM, CatBoost
Deep Learning     │ PyTorch, pytorch-tabnet
HPO               │ Optuna
LLM               │ Anthropic Claude / OpenAI GPT-4o (feature eng + reports)
Explainability    │ SHAP, LIME, sklearn.inspection (PDP/ICE/permutation)
Experiment Track  │ MLflow, Weights & Biases
Data Profiling    │ pandas-profiling concepts (custom), missingno
Drift Monitoring  │ Evidently AI
API Framework     │ FastAPI, Uvicorn
Orchestration     │ Airflow (batch predictions)
Database          │ PostgreSQL
Cache             │ Redis
Frontend          │ React 18, TypeScript, Tailwind, Recharts
Containerization  │ Docker, Docker Compose
CI/CD             │ GitHub Actions
Testing           │ pytest, pytest-cov
Data Processing   │ Pandas, NumPy
```

---

## 20. DIRECTORY STRUCTURE

```
forge/
├── README.md
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.frontend
├── Makefile
├── pyproject.toml
├── requirements/
│
├── src/
│   ├── config.py
│   ├── profiling/
│   │   ├── statistical_profiler.py
│   │   ├── semantic_profiler.py
│   │   └── eda_report.py
│   │
│   ├── feature_engineering/
│   │   ├── cleaner.py
│   │   ├── llm_engineer.py
│   │   ├── code_sandbox.py
│   │   ├── feature_selector.py
│   │   └── pipeline_serializer.py
│   │
│   ├── training/
│   │   ├── task_router.py
│   │   ├── classical/
│   │   │   ├── logistic_regression.py
│   │   │   ├── ridge.py
│   │   │   ├── lasso.py
│   │   │   ├── elastic_net.py
│   │   │   ├── sgd.py
│   │   │   ├── decision_tree.py
│   │   │   ├── random_forest.py
│   │   │   ├── extra_trees.py
│   │   │   ├── xgboost_model.py
│   │   │   ├── lightgbm_model.py
│   │   │   ├── catboost_model.py
│   │   │   ├── knn.py
│   │   │   ├── svm.py
│   │   │   ├── naive_bayes.py
│   │   │   ├── voting.py
│   │   │   └── stacking.py
│   │   ├── deep_learning/
│   │   │   ├── trainer.py
│   │   │   ├── mlp.py
│   │   │   ├── tab_transformer.py
│   │   │   ├── ft_transformer.py
│   │   │   ├── tabnet.py
│   │   │   ├── lstm_attention.py
│   │   │   ├── tcn.py
│   │   │   ├── tft.py
│   │   │   └── nbeats.py
│   │   └── hpo/
│   │       └── optuna_optimizer.py
│   │
│   ├── evaluation/
│   │   ├── metrics_calculator.py
│   │   ├── explainability.py
│   │   ├── error_analysis.py
│   │   ├── fairness_auditor.py
│   │   └── llm_report_generator.py
│   │
│   ├── deployment/
│   │   ├── model_exporter.py
│   │   ├── api_generator.py
│   │   ├── batch_pipeline.py
│   │   ├── model_card_generator.py
│   │   └── templates/
│   │       ├── fastapi_template.py
│   │       ├── dockerfile_template
│   │       └── airflow_dag_template.py
│   │
│   ├── monitoring/
│   │   ├── drift_monitor.py
│   │   └── performance_tracker.py
│   │
│   ├── api/
│   │   ├── app.py
│   │   ├── routers/
│   │   │   ├── experiments.py
│   │   │   ├── training.py
│   │   │   ├── evaluation.py
│   │   │   ├── deployment.py
│   │   │   └── health.py
│   │   ├── schemas/
│   │   └── services/
│   │
│   └── utils/
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── NewExperimentPage.tsx
│   │   │   ├── EDAPage.tsx
│   │   │   ├── FeatureReviewPage.tsx
│   │   │   ├── TrainingProgressPage.tsx
│   │   │   ├── ModelComparisonPage.tsx
│   │   │   ├── ExplanationExplorerPage.tsx
│   │   │   ├── ErrorAnalysisPage.tsx
│   │   │   ├── DeploymentPage.tsx
│   │   │   ├── MonitoringPage.tsx
│   │   │   └── ReportPage.tsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── store/
│   └── Dockerfile
│
├── benchmarks/
│   ├── datasets/
│   ├── run_all_benchmarks.py
│   └── results/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── benchmark/
│   └── conftest.py
│
├── notebooks/
└── docs/
```

---

*This is the complete specification for Forge. Every algorithm, every feature, every pipeline, every endpoint is defined. Build from this document.*
