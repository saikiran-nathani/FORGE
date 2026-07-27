"""Statistical data profiling engine."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from forge.profiling.models import ColumnProfile, ColumnType, ProfileReport


class StatisticalProfiler:
    """Profiles a dataset with type detection, statistics, and quality scoring."""

    def __init__(self, target_column: str, task_description: str = ""):
        self.target_column = target_column
        self.task_description = task_description

    def profile(self, df: pd.DataFrame) -> ProfileReport:
        df = df.copy()
        column_types = self._detect_column_types(df)
        column_profiles = {
            col: self._profile_column(df, col, column_types[col])
            for col in df.columns
            if col != self.target_column
        }
        target_analysis = self._analyze_target(df)
        correlations = self._correlation_analysis(df, column_types)
        missing_analysis = self._missing_analysis(df)
        outlier_analysis = self._outlier_analysis(df, column_types)
        quality_score = self._compute_quality_score(
            df, column_types, missing_analysis, outlier_analysis
        )
        recommended_metric = self._recommend_metric(target_analysis)
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        return ProfileReport(
            n_rows=len(df),
            n_cols=len(df.columns),
            column_types=column_types,
            column_profiles=column_profiles,
            target_analysis=target_analysis,
            correlations=correlations,
            missing_analysis=missing_analysis,
            outlier_analysis=outlier_analysis,
            quality_score=quality_score,
            recommended_metric=recommended_metric,
            memory_usage_mb=round(memory_usage_mb, 2),
        )

    def _detect_column_types(self, df: pd.DataFrame) -> dict[str, ColumnType]:
        types: dict[str, ColumnType] = {}
        for col in df.columns:
            series = df[col]
            if col == self.target_column:
                types[col] = self._infer_target_type(series)
                continue
            if series.dtype == bool or set(series.dropna().unique()) <= {0, 1, True, False}:
                types[col] = ColumnType.BINARY
                continue
            if self._is_id_column(col, series):
                types[col] = ColumnType.ID
                continue
            if self._is_datetime(series):
                types[col] = ColumnType.DATETIME
                continue
            if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
                avg_len = series.dropna().astype(str).str.len().mean()
                if avg_len and avg_len > 50:
                    types[col] = ColumnType.TEXT
                else:
                    types[col] = ColumnType.CATEGORICAL
                continue
            if pd.api.types.is_numeric_dtype(series):
                n_unique = series.nunique(dropna=True)
                if n_unique < 20 and pd.api.types.is_integer_dtype(series):
                    types[col] = ColumnType.CATEGORICAL
                else:
                    types[col] = ColumnType.NUMERICAL
                continue
            types[col] = ColumnType.CATEGORICAL
        return types

    def _infer_target_type(self, series: pd.Series) -> ColumnType:
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique(dropna=True)
            n_rows = len(series.dropna())
            if n_unique > 50:
                return ColumnType.NUMERICAL
            if pd.api.types.is_float_dtype(series) and n_unique > 5 and n_unique / max(n_rows, 1) > 0.5:
                return ColumnType.NUMERICAL
            if pd.api.types.is_integer_dtype(series) and n_unique > 5 and n_unique / max(n_rows, 1) > 0.8:
                return ColumnType.NUMERICAL
            if n_unique <= 2:
                return ColumnType.BINARY
            return ColumnType.CATEGORICAL
        return ColumnType.CATEGORICAL

    def _is_id_column(self, name: str, series: pd.Series) -> bool:
        # Name-based: split on separators AND camelCase boundaries, then look for
        # an explicit identifier token. Catches id, user_id, patient_id, userId,
        # uuid — without false-matching substrings inside real words
        # (grid, video, candidate, paid, width all tokenize without an "id" token).
        tokens = set(
            re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).lower())
        )
        if tokens & {"id", "uid", "uuid", "guid"}:
            return True
        n = len(series)
        if n == 0 or series.nunique(dropna=True) != n:
            return False
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            return True
        if pd.api.types.is_integer_dtype(series) and series.min() == 0 and series.max() == n - 1:
            return True
        return False

    def _is_datetime(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if pd.api.types.is_numeric_dtype(series):
            return False
        if series.dtype != object:
            return False
        sample = series.dropna().head(20)
        if sample.empty:
            return False
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            return parsed.notna().mean() >= 0.8
        except (ValueError, TypeError):
            return False

    def _profile_column(
        self, df: pd.DataFrame, col: str, col_type: ColumnType
    ) -> ColumnProfile:
        series = df[col]
        missing_pct = float(series.isna().mean() * 100)
        n_unique = int(series.nunique(dropna=True))
        stats_dict: dict[str, Any] = {"sample_values": series.dropna().head(5).tolist()}

        if col_type == ColumnType.NUMERICAL:
            stats_dict.update(self._numerical_stats(series))
        elif col_type in (ColumnType.CATEGORICAL, ColumnType.BINARY):
            stats_dict.update(self._categorical_stats(series))
        elif col_type == ColumnType.DATETIME:
            stats_dict.update(self._datetime_stats(series))
        elif col_type == ColumnType.TEXT:
            stats_dict.update(self._text_stats(series))

        return ColumnProfile(
            name=col,
            detected_type=col_type,
            statistics=stats_dict,
            missing_pct=missing_pct,
            n_unique=n_unique,
        )

    def _numerical_stats(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna()
        if clean.empty:
            return {}
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(((clean < lower) | (clean > upper)).sum())
        result: dict[str, Any] = {
            "count": int(clean.count()),
            "mean": float(clean.mean()),
            "std": float(clean.std()),
            "min": float(clean.min()),
            "max": float(clean.max()),
            "median": float(clean.median()),
            "skewness": float(clean.skew()),
            "kurtosis": float(clean.kurtosis()),
            "n_zeros": int((clean == 0).sum()),
            "n_negatives": int((clean < 0).sum()),
            "n_outliers_iqr": outliers,
            "percentiles": {
                p: float(clean.quantile(p / 100))
                for p in [1, 5, 25, 75, 95, 99]
            },
        }
        if len(clean) >= 3 and len(clean) <= 5000:
            _, p_value = stats.shapiro(clean.sample(min(len(clean), 5000), random_state=42))
            result["shapiro_p_value"] = float(p_value)
        return result

    def _categorical_stats(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna().astype(str)
        if clean.empty:
            return {}
        counts = clean.value_counts()
        total = len(clean)
        freqs = (counts / total).to_dict()
        top_10 = {str(k): int(v) for k, v in counts.head(10).items()}
        probs = counts / total
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        rare = [str(k) for k, v in freqs.items() if v < 0.01]
        return {
            "unique_count": int(clean.nunique()),
            "cardinality_ratio": float(clean.nunique() / total),
            "entropy": entropy,
            "mode": str(counts.index[0]),
            "mode_frequency": int(counts.iloc[0]),
            "top_10_values": top_10,
            "rare_categories": rare[:20],
        }

    def _datetime_stats(self, series: pd.Series) -> dict[str, Any]:
        dt = pd.to_datetime(series, errors="coerce").dropna()
        if dt.empty:
            return {}
        return {
            "min_date": str(dt.min()),
            "max_date": str(dt.max()),
            "range_days": int((dt.max() - dt.min()).days),
            "most_common_dow": int(dt.dt.dayofweek.mode().iloc[0]),
            "most_common_month": int(dt.dt.month.mode().iloc[0]),
        }

    def _text_stats(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna().astype(str)
        if clean.empty:
            return {}
        lengths = clean.str.len()
        return {
            "avg_length": float(lengths.mean()),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "vocabulary_size": int(clean.str.split().explode().nunique()),
        }

    def _analyze_target(self, df: pd.DataFrame) -> dict[str, Any]:
        series = df[self.target_column]
        n_unique = series.nunique(dropna=True)
        n_rows = len(series.dropna())
        target_type = self._infer_target_type(series)
        task_type = "regression" if target_type == ColumnType.NUMERICAL else "classification"

        result: dict[str, Any] = {
            "task_type": task_type,
            "n_unique": int(n_unique),
            "missing_pct": float(series.isna().mean() * 100),
        }

        if task_type == "classification":
            counts = series.value_counts()
            result["class_distribution"] = {str(k): int(v) for k, v in counts.items()}
            result["class_imbalance_ratio"] = float(counts.min() / counts.max()) if len(counts) > 1 else 1.0
            result["is_binary"] = n_unique == 2
        else:
            clean = series.dropna()
            result["mean"] = float(clean.mean())
            result["std"] = float(clean.std())
            result["skewness"] = float(clean.skew())
            result["log_transform_benefit"] = float(clean.skew()) > 1.0

        return result

    def _correlation_analysis(
        self, df: pd.DataFrame, column_types: dict[str, ColumnType]
    ) -> dict[str, Any]:
        numerical_cols = [
            c for c, t in column_types.items()
            if t == ColumnType.NUMERICAL and c in df.columns
        ]
        result: dict[str, Any] = {"high_correlation_pairs": []}

        if len(numerical_cols) >= 2:
            corr = df[numerical_cols].corr(method="pearson")
            pairs = []
            for i, c1 in enumerate(numerical_cols):
                for c2 in numerical_cols[i + 1:]:
                    r = corr.loc[c1, c2]
                    if abs(r) > 0.9:
                        pairs.append({"col1": c1, "col2": c2, "pearson": float(r)})
            result["pearson_matrix"] = corr.round(4).to_dict()
            result["high_correlation_pairs"] = pairs

        target = self.target_column
        if target in df.columns and column_types.get(target) != ColumnType.NUMERICAL:
            cat_cols = [
                c for c, t in column_types.items()
                if t in (ColumnType.CATEGORICAL, ColumnType.BINARY) and c != target
            ]
            mi_scores = {}
            for col in cat_cols[:50]:
                try:
                    contingency = pd.crosstab(df[col].fillna("MISSING"), df[target])
                    chi2 = stats.chi2_contingency(contingency)[0]
                    n = contingency.sum().sum()
                    mi_scores[col] = float(chi2 / n) if n else 0.0
                except (ValueError, TypeError):
                    continue
            result["categorical_target_association"] = mi_scores

        return result

    def _missing_analysis(self, df: pd.DataFrame) -> dict[str, Any]:
        missing_pct = (df.isna().mean() * 100).round(2)
        cols_with_missing = missing_pct[missing_pct > 0].to_dict()
        co_missing = df.isna().astype(int).corr()
        high_co_missing = []
        cols = list(df.columns)
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                r = co_missing.loc[c1, c2]
                if r > 0.5:
                    high_co_missing.append({"col1": c1, "col2": c2, "correlation": float(r)})

        recommendations = {}
        for col, pct in cols_with_missing.items():
            if pct > 30:
                recommendations[col] = "drop"
            elif pct > 5:
                recommendations[col] = "advanced_imputation"
            elif pct > 0:
                recommendations[col] = "simple_imputation"

        return {
            "missing_pct_by_column": {k: float(v) for k, v in cols_with_missing.items()},
            "avg_missing_rate": float(df.isna().mean().mean()),
            "high_co_missing_pairs": high_co_missing,
            "imputation_recommendations": recommendations,
        }

    def _outlier_analysis(
        self, df: pd.DataFrame, column_types: dict[str, ColumnType]
    ) -> dict[str, Any]:
        per_column = {}
        for col, col_type in column_types.items():
            if col_type != ColumnType.NUMERICAL or col == self.target_column:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_outliers = int(((series < lower) | (series > upper)).sum())
            z_outliers = int((np.abs(stats.zscore(series)) > 3).sum())
            per_column[col] = {
                "iqr_outliers": iqr_outliers,
                "zscore_outliers": z_outliers,
                "recommendation": "clip" if iqr_outliers / len(series) > 0.01 else "keep",
            }
        return {"per_column": per_column}

    def _compute_quality_score(
        self,
        df: pd.DataFrame,
        column_types: dict[str, ColumnType],
        missing_analysis: dict[str, Any],
        outlier_analysis: dict[str, Any],
    ) -> float:
        completeness = (1 - missing_analysis["avg_missing_rate"]) * 100

        id_cols = [c for c, t in column_types.items() if t == ColumnType.ID]
        if id_cols:
            uniqueness = np.mean([
                df[c].nunique() / len(df) for c in id_cols if c in df.columns
            ]) * 100
        else:
            uniqueness = 90.0

        type_consistency = 95.0
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(100).astype(str)
            if sample.empty:
                continue
            has_numeric = sample.str.match(r"^-?\d+\.?\d*$").any()
            has_text = sample.str.match(r"^[A-Za-z]").any()
            if has_numeric and has_text:
                type_consistency -= 5

        validity = 90.0
        for col, info in outlier_analysis.get("per_column", {}).items():
            if info["iqr_outliers"] > len(df) * 0.1:
                validity -= 2

        timeliness = 85.0
        datetime_cols = [c for c, t in column_types.items() if t == ColumnType.DATETIME]
        if datetime_cols:
            for col in datetime_cols:
                dt = pd.to_datetime(df[col], errors="coerce").dropna()
                if not dt.empty:
                    days_old = (pd.Timestamp.now(tz=None) - dt.max()).days
                    timeliness = max(50, 100 - days_old / 365 * 10)

        score = (
            completeness * 0.3
            + uniqueness * 0.2
            + type_consistency * 0.2
            + validity * 0.2
            + timeliness * 0.1
        )
        return round(min(100, max(0, score)), 1)

    def _recommend_metric(self, target_analysis: dict[str, Any]) -> str:
        if target_analysis["task_type"] == "regression":
            return "rmse"
        if target_analysis.get("is_binary"):
            imbalance = target_analysis.get("class_imbalance_ratio", 1.0)
            return "roc_auc" if imbalance < 0.3 else "f1"
        return "f1_macro"
