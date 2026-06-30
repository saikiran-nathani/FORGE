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
                "input_columns": feature_cols,
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
            p01, p99 = ref[col].quantile([0.01, 0.99])
            bounds[col] = {"low": float(p01), "high": float(p99)}
            if rec.get("recommendation") == "clip":
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
        return names
