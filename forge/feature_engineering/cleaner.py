"""Basic feature engineering: cleaning, encoding, scaling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from forge.profiling.models import ColumnType, ProfileReport


@dataclass
class FeaturePipelineResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    preprocessor: ColumnTransformer
    label_encoder: LabelEncoder | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FeaturePipeline:
    """Phase 1 feature pipeline: imputation, encoding, scaling."""

    def __init__(self, target_column: str, random_state: int = 42):
        self.target_column = target_column
        self.random_state = random_state
        self.label_encoder: LabelEncoder | None = None

    def fit_transform(
        self,
        df: pd.DataFrame,
        profile: ProfileReport,
        test_size: float = 0.2,
    ) -> FeaturePipelineResult:
        df = df.dropna(subset=[self.target_column]).copy()
        df, engineered = self._apply_stateless_engineering(df, profile)
        engineered_names = [c for r in engineered for c in r["new_columns"]]
        feature_cols = [c for c in df.columns if c != self.target_column]
        id_cols = [
            c for c in feature_cols
            if profile.column_types.get(c) == ColumnType.ID
        ]
        feature_cols = [c for c in feature_cols if c not in id_cols]

        X = df[feature_cols]
        y = df[self.target_column]

        if profile.target_analysis["task_type"] == "classification":
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y.astype(str)), index=y.index)

        from sklearn.model_selection import train_test_split

        stratify = None
        if profile.target_analysis["task_type"] == "classification":
            class_counts = y.value_counts()
            if class_counts.min() >= 2:
                stratify = y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        numerical_cols = [
            c for c in feature_cols
            if profile.column_types.get(c) in (ColumnType.NUMERICAL, ColumnType.DATETIME)
        ]
        # Stateless engineered features are all numeric — route them into the
        # preprocessor explicitly (they aren't in profile.column_types, so
        # remainder="drop" would otherwise silently discard them: bug #28).
        numerical_cols += [
            c for c in engineered_names if c in feature_cols and c not in numerical_cols
        ]
        categorical_cols = [
            c for c in feature_cols
            if profile.column_types.get(c) in (ColumnType.CATEGORICAL, ColumnType.BINARY, ColumnType.TEXT)
        ]

        X_train = self._handle_missing_indicators(X_train, profile)
        X_test = self._handle_missing_indicators(X_test, profile)
        numerical_cols = [c for c in numerical_cols if c in X_train.columns]
        categorical_cols = [c for c in categorical_cols if c in X_train.columns]

        X_train, clip_bounds = self._clip_outliers(
            X_train, profile, numerical_cols, return_bounds=True
        )
        X_test = self._apply_clip_bounds(
            X_test, clip_bounds
        )

        for col in categorical_cols:
            X_train[col] = X_train[col].astype(str).replace("nan", "MISSING")
            X_test[col] = X_test[col].astype(str).replace("nan", "MISSING")

        transformers = []
        if numerical_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, numerical_cols))

        if categorical_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipe, categorical_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        X_train_arr = preprocessor.fit_transform(X_train)
        X_test_arr = preprocessor.transform(X_test)

        feature_names = self._get_feature_names(preprocessor, numerical_cols, categorical_cols)
        X_train_df = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)

        return FeaturePipelineResult(
            X_train=X_train_df,
            X_test=X_test_df,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            preprocessor=preprocessor,
            label_encoder=self.label_encoder,
            metadata={
                "numerical_cols": numerical_cols,
                "categorical_cols": categorical_cols,
                "dropped_id_cols": id_cols,
                # Original input schema only — engineered cols are recreated at
                # inference from these, so they must NOT be listed as required inputs.
                "input_columns": [c for c in feature_cols if c not in engineered_names],
                "engineered_features": engineered,
                "clip_bounds": clip_bounds,
            },
        )

    def transform_raw(self, df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
        """Apply training-time transforms for inference."""
        profile_dict = bundle["profile"]
        meta = bundle["metadata"]
        preprocessor = bundle["preprocessor"]

        from forge.profiling.models import ProfileReport
        from forge.profiling.statistical_profiler import StatisticalProfiler

        profile = ProfileReport(
            n_rows=profile_dict["n_rows"],
            n_cols=profile_dict["n_cols"],
            column_types={k: ColumnType(v) for k, v in profile_dict["column_types"].items()},
            column_profiles={},
            target_analysis=profile_dict["target_analysis"],
            correlations=profile_dict.get("correlations", {}),
            missing_analysis=profile_dict["missing_analysis"],
            outlier_analysis=profile_dict.get("outlier_analysis", {}),
            quality_score=profile_dict["quality_score"],
            recommended_metric=profile_dict["recommended_metric"],
            memory_usage_mb=profile_dict.get("memory_usage_mb", 0),
        )

        cols = [c for c in meta["input_columns"] if c in df.columns]
        X = df[cols].copy()
        # Recreate the SAME stateless engineered features the model was trained on
        # (deterministic from the raw row + persisted profile — no fitted state).
        X, _ = self._apply_stateless_engineering(X, profile)
        X = self._handle_missing_indicators(X, profile)
        X = self._apply_clip_bounds(X, meta.get("clip_bounds", {}))
        for col in meta.get("categorical_cols", []):
            if col in X.columns:
                X[col] = X[col].astype(str).replace("nan", "MISSING")
        arr = preprocessor.transform(X)
        # Reconstruct the FULL preprocessor output names. bundle["feature_names"]
        # may be a post-feature-selection subset, which wouldn't match the column
        # count of the transformed array; rebuild from the fitted preprocessor.
        names = self._get_feature_names(
            preprocessor, meta.get("numerical_cols", []), meta.get("categorical_cols", [])
        )
        result = pd.DataFrame(arr, columns=names)
        selected = bundle.get("selected_features")
        if selected:
            result = result[[c for c in selected if c in result.columns]]
        return result

    def _apply_stateless_engineering(
        self, df: pd.DataFrame, profile: ProfileReport
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """Add STATELESS engineered features (each row depends only on itself).

        Statelessness is the whole point: because no cross-row statistic is used,
        (a) computing pre- vs post-split gives identical values → no train/test
        leakage, and (b) a single-row prediction reproduces the exact same value →
        no train/serve skew, so these can be replayed verbatim at inference.
        Only log / datetime-part / interaction transforms qualify. z-score
        (redundant with the StandardScaler) and rank / frequency-encode (need
        fitted train statistics) are intentionally excluded.
        """
        df = df.copy()
        records: list[dict[str, Any]] = []
        feature_cols = [c for c in df.columns if c != self.target_column]
        numeric = [
            c for c in feature_cols if profile.column_types.get(c) == ColumnType.NUMERICAL
        ]
        datetime_cols = [
            c for c in feature_cols if profile.column_types.get(c) == ColumnType.DATETIME
        ]
        for col in numeric:
            name = f"{col}_log"
            df[name] = np.log1p(df[col].clip(lower=0))
            records.append({"source_column": col, "new_columns": [name]})
        for col in datetime_cols:
            dt = pd.to_datetime(df[col], errors="coerce")
            cols = [f"{col}_dayofweek", f"{col}_month", f"{col}_is_weekend"]
            df[cols[0]] = dt.dt.dayofweek
            df[cols[1]] = dt.dt.month
            df[cols[2]] = dt.dt.dayofweek.isin([5, 6]).astype(int)
            records.append({"source_column": col, "new_columns": cols})
        if len(numeric) >= 2:
            c1, c2 = numeric[0], numeric[1]
            cols = [f"{c1}_x_{c2}", f"{c1}_ratio_{c2}"]
            df[cols[0]] = df[c1] * df[c2]
            df[cols[1]] = df[c1] / (df[c2].replace(0, np.nan) + 1e-8)
            records.append({"source_column": f"{c1},{c2}", "new_columns": cols})
        return df, records

    def _apply_clip_bounds(self, df: pd.DataFrame, bounds: dict[str, dict[str, float]]) -> pd.DataFrame:
        df = df.copy()
        for col, b in bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(b["low"], b["high"])
        return df

    def _handle_missing_indicators(
        self, df: pd.DataFrame, profile: ProfileReport
    ) -> pd.DataFrame:
        df = df.copy()
        for col, pct in profile.missing_analysis.get("missing_pct_by_column", {}).items():
            if col in df.columns and pct > 0:
                df[f"{col}_is_missing"] = df[col].isna().astype(int)
        return df

    def _clip_outliers(
        self,
        df: pd.DataFrame,
        profile: ProfileReport,
        numerical_cols: list[str],
        ref_df: pd.DataFrame | None = None,
        return_bounds: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        df = df.copy()
        ref = ref_df if ref_df is not None else df
        bounds: dict[str, dict[str, float]] = {}
        for col in numerical_cols:
            rec = profile.outlier_analysis.get("per_column", {}).get(col, {})
            # Only record bounds for columns we ACTUALLY clip at train time.
            # _apply_clip_bounds (test split + every inference request) clips every
            # column present in `bounds`, so storing bounds for un-clipped columns
            # caused train/serve skew: trained unclipped, served clipped.
            if rec.get("recommendation") != "clip":
                continue
            p01, p99 = ref[col].quantile([0.01, 0.99])
            bounds[col] = {"low": float(p01), "high": float(p99)}
            df[col] = df[col].clip(p01, p99)
        if return_bounds:
            return df, bounds
        return df

    def _get_feature_names(
        self,
        preprocessor: ColumnTransformer,
        numerical_cols: list[str],
        categorical_cols: list[str],
    ) -> list[str]:
        names: list[str] = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                encoder = transformer.named_steps["encoder"]
                cat_names = encoder.get_feature_names_out(cols)
                names.extend(cat_names.tolist())
        return self._sanitize_names(names)

    @staticmethod
    def _sanitize_names(names: list[str]) -> list[str]:
        """Make feature names safe for all estimators (XGBoost rejects [ ] < >).

        One-hot categories like 'income_<=50K' otherwise crash XGBoost. Applied
        at both fit and inference so train/serve names stay identical. De-dupes
        any collisions introduced by the substitution.
        """
        import re

        clean: list[str] = []
        seen: dict[str, int] = {}
        for n in names:
            s = re.sub(r"[\[\]<>]", "_", str(n)).strip()
            if s in seen:
                seen[s] += 1
                s = f"{s}__{seen[s]}"
            else:
                seen[s] = 0
            clean.append(s)
        return clean
