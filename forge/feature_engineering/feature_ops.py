"""Declarative, whitelisted feature-transformation registry.

Why this module exists
----------------------
The LLM feature-engineering path used to emit free-form pandas snippets. Two
properties made those snippets unusable for anything but advice:

1. They could not be replayed on a single prediction row. A z-score needs the
   TRAIN mean/std; a one-row batch has neither, so ``(x - x.mean()) / x.std()``
   quietly evaluates to 0 (or NaN) at serving time while looking correct during
   training. Train/serve skew, no error raised.
2. Anything that computes a statistic over the rows it is transforming leaks if
   it runs before the train/test split.

This module replaces free-form code with a closed set of ops, each of which is a
proper *fitted transformer*: ``fit`` learns its parameters from the training
rows and stores them; ``transform`` only ever combines those stored parameters
with the row in front of it. That is what makes stateful features (target
encoding, winsorizing, quantile binning, group aggregates) legal: their
parameters are learned on TRAIN ONLY and persisted alongside the model.

There is no code generation and no code execution anywhere in this module. A
spec is data: an op name plus a small dict of parameters.

The two invariants every op must satisfy
----------------------------------------
1. **No leakage.** Every learned quantity comes from the data handed to ``fit``.
   ``transform`` never computes a statistic across the rows it is given.
2. **Train/serve parity.** Transforming a single row in isolation produces
   exactly the values that row gets inside a large batch.

Both are covered by property tests in ``tests/unit/test_feature_ops.py``.

Conventions that hold for every op
----------------------------------
* Every engineered column is ``float64``, so missingness is always
  representable and the output dtype never depends on the batch. (For example
  ``Series.dt.dayofweek`` is ``int32`` or ``float64`` depending on whether the
  batch happens to contain a NaT -- exactly the kind of batch dependence that
  breaks parity.)
* **Missing input -> missing output.** We never substitute a fabricated value
  for absent data; the downstream imputer owns that decision. The one
  deliberate exception is an *unseen* category/group -- a value that is present
  but was never seen during ``fit``. "Never seen in train" is itself learned
  information, so those get the documented fallback (0.0 / global mean / global
  aggregate).
* Non-finite results (+/-inf from overflow or infinite inputs) are emitted as
  NaN. inf breaks every downstream estimator, and NaN is the honest
  "not computable" marker that the imputer already handles.
* Nothing is silently swallowed. A malformed spec is REPORTED by
  :func:`validate_specs`; an op that genuinely cannot fit (a column with no
  usable values, a target encoder with no target) raises.
* Deterministic, no randomness, no global or hidden state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pandas.tseries.api import guess_datetime_format
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "AGGREGATIONS",
    "OPS",
    "DatetimePartsOp",
    "FeatureOp",
    "FeatureSpec",
    "FeatureSpecTransformer",
    "FrequencyEncodeOp",
    "GroupAggregateOp",
    "Log1pOp",
    "ProductOp",
    "QuantileBinOp",
    "RatioOp",
    "TargetEncodeOp",
    "WinsorizeOp",
    "validate_specs",
]

#: Aggregations `group_aggregate` accepts. `std` is the pandas default sample
#: standard deviation (ddof=1), so a group with a single training row has an
#: undefined std -- see :class:`GroupAggregateOp` for how that is handled.
AGGREGATIONS: tuple[str, ...] = ("mean", "median", "std", "max", "min")

_MAX_COLUMNS_IN_ERROR = 12


@dataclass
class FeatureSpec:
    """One declarative feature request: an op name plus its parameters.

    A spec is inert data -- it is not executable and carries no code. It is
    JSON-round-trippable so it can be persisted next to the model and replayed
    at inference time.
    """

    op: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict) -> FeatureSpec:
        """Rebuild a spec from its dict form.

        This validates *structure* only and raises on structurally broken input
        (not a mapping, no ``op``, non-mapping ``params``). It deliberately does
        NOT check whether the op exists or the params make sense -- that is
        :func:`validate_specs`' job, which reports instead of raising so a whole
        batch of LLM-proposed specs can be triaged at once.
        """
        if not isinstance(d, Mapping):
            raise TypeError(
                f"FeatureSpec.from_dict expects a mapping, got {type(d).__name__}"
            )
        if "op" not in d:
            raise ValueError(f"FeatureSpec.from_dict: missing 'op' key in {d!r}")
        op = d["op"]
        if not isinstance(op, str) or not op:
            raise ValueError(
                f"FeatureSpec.from_dict: 'op' must be a non-empty string, got {op!r}"
            )
        params = d.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            raise TypeError(
                f"FeatureSpec.from_dict: 'params' must be a mapping, "
                f"got {type(params).__name__}"
            )
        return cls(op=op, params=dict(params))


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _describe_columns(columns: Sequence[Any]) -> str:
    names = [str(c) for c in columns]
    if len(names) > _MAX_COLUMNS_IN_ERROR:
        shown = ", ".join(names[:_MAX_COLUMNS_IN_ERROR])
        return f"{shown}, ... (+{len(names) - _MAX_COLUMNS_IN_ERROR} more)"
    return ", ".join(names) if names else "<none>"


def _require_column(X: pd.DataFrame, col: str, label: str) -> pd.Series:
    """Fetch one column, failing loudly rather than degrading to a default."""
    if col not in X.columns:
        raise ValueError(
            f"{label}: column '{col}' is not in the frame "
            f"(columns: {_describe_columns(X.columns)})"
        )
    series = X[col]
    if isinstance(series, pd.DataFrame):
        raise ValueError(
            f"{label}: column '{col}' appears more than once in the frame; "
            "de-duplicate the columns before engineering features"
        )
    return series


def _numeric_values(X: pd.DataFrame, col: str, label: str) -> np.ndarray:
    """Return ``X[col]`` as a float64 array.

    Conversion is element-wise, so a single row and a large batch yield
    identical values (train/serve parity). A non-numeric column fails loudly
    instead of being coerced to NaN.
    """
    series = _require_column(X, col, label)
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label}: column '{col}' (dtype {series.dtype}) is not numeric, so it "
            f"cannot be used by this op: {exc}"
        ) from exc
    return np.asarray(numeric.astype("float64").to_numpy(), dtype="float64")


def _finite(values: np.ndarray) -> np.ndarray:
    """Drop NaN/+-inf before learning a statistic.

    Infinities are not usable quantile/aggregate anchors, and silently letting
    them through would poison every learned bound. Rows containing them are
    still transformed at predict time -- clipping/digitising handles them.
    """
    return values[np.isfinite(values)]


def _as_python(value: Any) -> Any:
    """numpy scalar -> python scalar, so learned tables stay inspectable.

    Hash and equality are preserved, so dict lookups behave identically.
    datetime64/timedelta64 are left alone: ``.item()`` on nanosecond precision
    returns an int, which would no longer match the original key.
    """
    if isinstance(value, np.generic) and not isinstance(
        value, (np.datetime64, np.timedelta64)
    ):
        return value.item()
    return value


def _sanitise(values: np.ndarray) -> np.ndarray:
    """Map +-inf to NaN. See the module docstring for the rationale."""
    return np.where(np.isfinite(values), values, np.nan)


def _lookup_with_fallback(
    raw: pd.Series, table: Mapping[Any, float], unseen_value: float
) -> np.ndarray:
    """Row-independent dict lookup -> float64 array.

    * key present in ``table``  -> the stored value (which may itself be NaN,
      e.g. the sample std of a one-row group)
    * key absent but not null   -> ``unseen_value`` (the documented fallback)
    * key null                  -> NaN (missing in, missing out)

    Both ``Series.map`` and ``Series.isin`` are element-wise, so this is
    parity-safe by construction: nothing here looks at other rows.
    """
    mapped = raw.map(dict(table))
    out = np.asarray(mapped.astype("float64").to_numpy(), dtype="float64").copy()
    known = np.asarray(raw.isin(list(table)).to_numpy(), dtype=bool)
    null = np.asarray(raw.isna().to_numpy(), dtype=bool)
    out[~known & ~null] = unseen_value
    out[null] = np.nan
    return out


# --------------------------------------------------------------------------- #
# op base class
# --------------------------------------------------------------------------- #


class FeatureOp:
    """Base class for a whitelisted op.

    Contract for subclasses:

    * ``fit(X, y)`` may look at every row of ``X`` (training data only) and must
      store everything it needs on ``self``.
    * ``transform(X)`` may use ONLY ``self`` plus the row it is transforming. No
      statistic may be computed across the rows being transformed -- that is
      what breaks train/serve parity and what this module exists to prevent.
    * ``transform`` returns ``{output column name: float64 ndarray}``, aligned
      positionally with ``X``.
    """

    #: registry key
    name: ClassVar[str] = ""
    #: params that must be supplied
    required_params: ClassVar[tuple[str, ...]] = ()
    #: params that may be supplied, with their defaults
    optional_params: ClassVar[Mapping[str, Any]] = {}
    #: subset of params whose value is a column name in the input frame
    column_params: ClassVar[tuple[str, ...]] = ()
    #: True when ``fit`` learns parameters from the training rows
    stateful: ClassVar[bool] = False

    def __init__(self, **params: Any) -> None:
        errors = self.param_errors(params)
        if errors:
            raise ValueError(
                f"op '{self.name}': invalid params {params!r} -> " + "; ".join(errors)
            )
        self.params: dict[str, Any] = self.resolve_params(params)
        self.names: list[str] = self.output_names(self.params)
        self._fitted = False

    # -- params ------------------------------------------------------------- #

    @classmethod
    def resolve_params(cls, params: Mapping[str, Any]) -> dict[str, Any]:
        """Merge supplied params over the declared defaults."""
        resolved = dict(cls.optional_params)
        resolved.update(params)
        return resolved

    @classmethod
    def param_errors(cls, params: Mapping[str, Any]) -> list[str]:
        """Report (never raise) everything wrong with ``params``.

        Single source of truth shared by :func:`validate_specs` (which reports)
        and ``__init__`` (which raises) so the two can never disagree.
        """
        if not isinstance(params, Mapping):
            return [f"params must be a mapping, got {type(params).__name__}"]
        errors: list[str] = []
        allowed = set(cls.required_params) | set(cls.optional_params)
        for name in cls.required_params:
            if name not in params:
                errors.append(f"missing required param '{name}'")
        for name in sorted(set(params) - allowed):
            errors.append(
                f"unexpected param '{name}' (allowed: {', '.join(sorted(allowed))})"
            )
        for name in cls.column_params:
            if name in params and not isinstance(params[name], str):
                errors.append(
                    f"param '{name}' must be a column name (str), "
                    f"got {type(params[name]).__name__}"
                )
        if errors:
            # Value-level checks on an incomplete/misspelled spec would only add
            # noise on top of the real problem.
            return errors
        return cls._value_errors(cls.resolve_params(params))

    @classmethod
    def _value_errors(cls, params: Mapping[str, Any]) -> list[str]:
        """Op-specific checks on resolved params (all present, right types)."""
        return []

    @classmethod
    def output_names(cls, params: Mapping[str, Any]) -> list[str]:
        raise NotImplementedError

    # -- fit / transform ---------------------------------------------------- #

    @property
    def label(self) -> str:
        return f"op '{self.name}' {self.params!r}"

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureOp:
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _mark_fitted(self) -> FeatureOp:
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.label}: transform() called before fit()")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.params!r})"


# --------------------------------------------------------------------------- #
# stateless ops
# --------------------------------------------------------------------------- #


class Log1pOp(FeatureOp):
    """``log1p(max(col, 0))`` -- stateless.

    Negative values are clipped to 0 first so the result is always defined
    (log1p is undefined below -1 and infinite at -1). NaN stays NaN.
    """

    name = "log1p"
    required_params = ("col",)
    column_params = ("col",)

    @classmethod
    def output_names(cls, params: Mapping[str, Any]) -> list[str]:
        return [f"{params['col']}_log1p"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> Log1pOp:
        # Nothing to learn, but validate the column now so a bad spec fails at
        # training time rather than at serving time.
        _numeric_values(X, self.params["col"], self.label)
        return self._mark_fitted()

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        self._check_fitted()
        values = _numeric_values(X, self.params["col"], self.label)
        return {self.names[0]: _sanitise(np.log1p(np.clip(values, 0.0, None)))}


class RatioOp(FeatureOp):
    """``a / b`` -- stateless, with a safe denominator.

    A zero denominator yields NaN, never inf and never a fabricated
    epsilon-padded number: ``a / (b + 1e-8)`` silently reports 1e8 * a for b=0,
    which is a made-up value masquerading as a measurement. NaN is the honest
    "not computable" marker and the downstream imputer already handles it.
    Non-finite results (overflow, infinite inputs) collapse to NaN too.
    """

    name = "ratio"
    required_params = ("a", "b")
    column_params = ("a", "b")

    @classmethod
    def output_names(cls, params: Mapping[str, Any]) -> list[str]:
        return [f"{params['a']}_ratio_{params['b']}"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> RatioOp:
        _numeric_values(X, self.params["a"], self.label)
        _numeric_values(X, self.params["b"], self.label)
        return self._mark_fitted()

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        self._check_fitted()
        numerator = _numeric_values(X, self.params["a"], self.label)
        denominator = _numeric_values(X, self.params["b"], self.label)
        safe = np.where(denominator == 0.0, np.nan, denominator)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = numerator / safe
        return {self.names[0]: _sanitise(ratio)}


class ProductOp(FeatureOp):
    """``a * b`` -- stateless. Overflow to +-inf is reported as NaN."""

    name = "product"
    required_params = ("a", "b")
    column_params = ("a", "b")

    @classmethod
    def output_names(cls, params: Mapping[str, Any]) -> list[str]:
        return [f"{params['a']}_x_{params['b']}"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> ProductOp:
        _numeric_values(X, self.params["a"], self.label)
        _numeric_values(X, self.params["b"], self.label)
        return self._mark_fitted()

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        self._check_fitted()
        left = _numeric_values(X, self.params["a"], self.label)
        right = _numeric_values(X, self.params["b"], self.label)
        with np.errstate(over="ignore", invalid="ignore"):
            product = left * right
        return {self.names[0]: _sanitise(product)}


class DatetimePartsOp(FeatureOp):
    """Calendar parts of a datetime column: dayofweek, month, is_weekend.

    The arithmetic is row-local, but the *parsing* is not free: pandas infers a
    datetime format from the first non-null element of whatever it is handed, so
    ``pd.to_datetime`` can genuinely disagree between a batch and a single row
    (a column starting ``2020-01-02`` makes ``03/04/2020`` a NaT in the batch,
    while that same value parsed alone becomes 2020-03-04). That is a parity
    break, so this op is stateful in exactly one respect: ``fit`` pins the parse
    strategy and ``transform`` reuses it.

    * ``native`` -- the training column was already datetime64; no parsing.
    * ``format`` -- one strptime format explains the whole training column.
    * ``mixed``  -- formats are heterogeneous, so each element is parsed
      independently (``format="mixed"``), which is also row-independent.

    Unparseable values become NaT and therefore NaN in all three outputs; we do
    not invent a date. ``is_weekend`` is 1.0 for Saturday/Sunday.
    """

    name = "datetime_parts"
    required_params = ("col",)
    column_params = ("col",)

    _NATIVE = "native"
    _FORMAT = "format"
    _MIXED = "mixed"

    @classmethod
    def output_names(cls, params: Mapping[str, Any]) -> list[str]:
        col = params["col"]
        return [f"{col}_dayofweek", f"{col}_month", f"{col}_is_weekend"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> DatetimePartsOp:
        col = self.params["col"]
        series = _require_column(X, col, self.label)
        self.strategy_, self.format_ = self._learn_parse_strategy(series, col)
        parsed = self._parse(series)
        if series.notna().any() and not parsed.notna().any():
            raise ValueError(
                f"{self.label}: no value in column '{col}' could be parsed as a "
                "datetime, so no datetime parts can be derived from it"
            )
        return self._mark_fitted()

    def _learn_parse_strategy(
        self, series: pd.Series, col: str
    ) -> tuple[str, str | None]:
        if pd.api.types.is_datetime64_any_dtype(series):
            return self._NATIVE, None
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise ValueError(
                f"{self.label}: column '{col}' has dtype {series.dtype}; a numeric "
                "column would be silently reinterpreted as nanoseconds since the "
                "epoch. Convert it to datetimes (or date strings) first."
            )
        non_null = series.dropna()
        if non_null.empty:
            raise ValueError(
                f"{self.label}: column '{col}' has no non-null values, so no parse "
                "strategy can be learned from the training data"
            )
        first = non_null.iloc[0]
        guessed = guess_datetime_format(first) if isinstance(first, str) else None
        if guessed is None:
            return self._MIXED, None
        # Only pin the format if it explains the training column at least as
        # well as per-element parsing does; otherwise fall back to per-element
        # parsing, which is slower but equally row-independent.
        with_format = pd.to_datetime(series, format=guessed, errors="coerce")
        per_element = pd.to_datetime(series, format="mixed", errors="coerce")
        if int(with_format.notna().sum()) < int(per_element.notna().sum()):
            return self._MIXED, None
        return self._FORMAT, guessed

    def _parse(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        if bool(series.isna().all()):
            # An all-null column has nothing to reinterpret, so the numeric guard
            # below does not apply. This matters for parity: a one-row frame
            # built from a JSON payload whose timestamp is null arrives as
            # float64, while the same null inside a batch of date strings
            # arrives as object. Both must yield NaT.
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise ValueError(
                f"{self.label}: column '{self.params['col']}' has dtype "
                f"{series.dtype}; a numeric column would be silently reinterpreted "
                "as nanoseconds since the epoch. Convert it to datetimes first."
            )
        if self.strategy_ == self._FORMAT:
            return pd.to_datetime(series, format=self.format_, errors="coerce")
        return pd.to_datetime(series, format="mixed", errors="coerce")

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        self._check_fitted()
        parsed = self._parse(_require_column(X, self.params["col"], self.label))
        # Force float64: .dt.dayofweek is int32 when the batch has no NaT and
        # float64 when it does, which would make the dtype batch-dependent.
        dayofweek = np.asarray(
            parsed.dt.dayofweek.astype("float64").to_numpy(), dtype="float64"
        )
        month = np.asarray(parsed.dt.month.astype("float64").to_numpy(), dtype="float64")
        is_weekend = np.where(np.isnan(dayofweek), np.nan, (dayofweek >= 5).astype(float))
        return {
            self.names[0]: dayofweek,
            self.names[1]: month,
            self.names[2]: is_weekend,
        }


# --------------------------------------------------------------------------- #
# stateful ops -- every parameter below is learned from fit() data only
# --------------------------------------------------------------------------- #


class WinsorizeOp(FeatureOp):
    """Clip a column to the TRAIN 1st/99th percentiles.

    ``fit`` learns ``lower_``/``upper_`` from the training rows only; a test row
    beyond either bound is clipped to the train bound rather than shifting it.
    NaN stays NaN. The source column is left untouched -- the clipped values are
    appended as a new column, so the caller decides what to keep.
    """

    name = "winsorize"
    required_params = ("col",)
    column_params = ("col",)
    stateful = True

    LOWER_PERCENTILE: ClassVar[float] = 1.0
    UPPER_PERCENTILE: ClassVar[float] = 99.0

    @classmethod
    def output_names(cls, params: Mapping[str, Any]) -> list[str]:
        return [f"{params['col']}_winsor"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> WinsorizeOp:
        col = self.params["col"]
        usable = _finite(_numeric_values(X, col, self.label))
        if usable.size == 0:
            raise ValueError(
                f"{self.label}: column '{col}' has no finite values in the training "
                "data, so no percentile bounds can be learned"
            )
        lower, upper = np.percentile(usable, [self.LOWER_PERCENTILE, self.UPPER_PERCENTILE])
        self.lower_ = float(lower)
        self.upper_ = float(upper)
        return self._mark_fitted()

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        self._check_fitted()
        values = _numeric_values(X, self.params["col"], self.label)
        return {self.names[0]: np.clip(values, self.lower_, self.upper_)}


class QuantileBinOp(FeatureOp):
    """Digitise a column against TRAIN quantile edges.

    ``fit`` learns the interior bin edges from the training rows;
    ``transform`` runs ``np.digitize`` against those fixed edges, so values
    below the first edge land in bin 0 and values above the last edge land in
    the top bin (unseen extremes clamp instead of creating new bins).

    ``n_bins`` is the *requested* number of bins. Ties collapse duplicate
    quantiles, so the realised count (``n_bins_``) can be lower; if the column
    is so tied that not even one interior edge survives, ``fit`` raises rather
    than emitting a constant column dressed up as a computed feature.

    The output is float64 so that a NaN input can stay NaN (``np.digitize``
    otherwise silently files NaN in the top bin).
    """

    name = "quantile_bin"
    required_params = ("col",)
    optional_params = {"n_bins": 4}
    column_params = ("col",)
    stateful = True

    @classmethod
    def output_names(cls, params: Mapping[str, Any]) -> list[str]:
        return [f"{params['col']}_bin{params['n_bins']}"]

    @classmethod
    def _value_errors(cls, params: Mapping[str, Any]) -> list[str]:
        n_bins = params["n_bins"]
        if isinstance(n_bins, bool) or not isinstance(n_bins, int):
            return [
                f"param 'n_bins' must be an int, got {type(n_bins).__name__} "
                f"({n_bins!r})"
            ]
        if n_bins < 2:
            return [f"param 'n_bins' must be >= 2, got {n_bins}"]
        return []

    def fit(self, X: pd.DataFrame, y: Any = None) -> QuantileBinOp:
        col = self.params["col"]
        usable = _finite(_numeric_values(X, col, self.label))
        if usable.size == 0:
            raise ValueError(
                f"{self.label}: column '{col}' has no finite values in the training "
                "data, so no bin edges can be learned"
            )
        quantiles = np.linspace(0.0, 1.0, int(self.params["n_bins"]) + 1)[1:-1]
        edges = np.unique(np.quantile(usable, quantiles))
        # Degeneracy test must be on the RESULTING BINS, not on edge count: a
        # constant column yields edges=[c] (size 1, not 0), and a heavily tied
        # column can yield edges that still place every training row in one bin.
        # Either way the feature would be constant — refuse rather than emit it.
        if edges.size == 0 or np.unique(np.digitize(usable, edges, right=False)).size < 2:
            raise ValueError(
                f"{self.label}: column '{col}' has too few distinct training values "
                f"({np.unique(usable).size}) to form {self.params['n_bins']} "
                "quantile bins; every row would land in the same bin"
            )
        self.edges_ = edges.astype("float64")
        self.n_bins_ = int(edges.size + 1)
        return self._mark_fitted()

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        self._check_fitted()
        values = _numeric_values(X, self.params["col"], self.label)
        bins = np.digitize(values, self.edges_, right=False).astype("float64")
        bins[np.isnan(values)] = np.nan
        return {self.names[0]: bins}


class FrequencyEncodeOp(FeatureOp):
    """Replace a category with its TRAIN frequency.

    ``fit`` learns ``frequencies_[c] = (# training rows with value c) / (#
    training rows)``. An unseen-but-present category encodes to 0.0, which is
    the literal truth ("this value occurred in 0% of training rows") and is the
    documented fallback. A missing input stays NaN.
    """

    name = "frequency_encode"
    required_params = ("col",)
    column_params = ("col",)
    stateful = True

    @classmethod
    def output_names(cls, params: Mapping[str, Any]) -> list[str]:
        return [f"{params['col']}_freq"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> FrequencyEncodeOp:
        col = self.params["col"]
        series = _require_column(X, col, self.label)
        n_rows = int(len(series))
        if n_rows == 0:
            raise ValueError(
                f"{self.label}: cannot learn frequencies from an empty training frame"
            )
        counts = series.value_counts(dropna=True)
        if counts.empty:
            raise ValueError(
                f"{self.label}: column '{col}' has no non-null values in the "
                "training data, so no category frequencies can be learned"
            )
        self.n_fit_rows_ = n_rows
        self.frequencies_ = {
            _as_python(key): float(count) / n_rows for key, count in counts.items()
        }
        return self._mark_fitted()

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        self._check_fitted()
        raw = _require_column(X, self.params["col"], self.label)
        return {self.names[0]: _lookup_with_fallback(raw, self.frequencies_, 0.0)}


class TargetEncodeOp(FeatureOp):
    """Smoothed per-category mean of the target.

    ``fit`` learns, from the training rows and the ``y`` passed to ``fit`` only::

        encoding[c] = (n_c * mean_c + smoothing * global_mean) / (n_c + smoothing)

    so a category with few rows is pulled hard toward the global mean while a
    well-populated category keeps its own mean. An unseen category encodes to
    the global mean; a missing input stays NaN.

    ``y`` must be numeric (label-encode classification targets first) and is
    read positionally, per the scikit-learn convention -- never re-aligned by
    index. Rows with a non-finite target are excluded from the statistics.
    ``transform`` never sees ``y``, which is what makes this safe to replay at
    inference time.
    """

    name = "target_encode"
    required_params = ("col",)
    optional_params = {"smoothing": 10.0}
    column_params = ("col",)
    stateful = True

    @classmethod
    def output_names(cls, params: Mapping[str, Any]) -> list[str]:
        return [f"{params['col']}_target_enc"]

    @classmethod
    def _value_errors(cls, params: Mapping[str, Any]) -> list[str]:
        smoothing = params["smoothing"]
        if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
            return [
                f"param 'smoothing' must be a number, got "
                f"{type(smoothing).__name__} ({smoothing!r})"
            ]
        if not math.isfinite(float(smoothing)):
            return [f"param 'smoothing' must be finite, got {smoothing!r}"]
        if float(smoothing) < 0:
            return [f"param 'smoothing' must be >= 0, got {smoothing!r}"]
        return []

    def fit(self, X: pd.DataFrame, y: Any = None) -> TargetEncodeOp:
        col = self.params["col"]
        raw = _require_column(X, col, self.label)
        targets = self._target_values(y, len(X))
        usable = np.isfinite(targets)
        if not usable.any():
            raise ValueError(
                f"{self.label}: the target has no finite values, so no category "
                "means can be learned"
            )
        self.global_mean_ = float(targets[usable].mean())
        frame = pd.DataFrame(
            {"key": raw.reset_index(drop=True), "target": targets}
        )[usable]
        stats = frame.groupby("key", dropna=True, observed=True)["target"].agg(
            ["count", "mean"]
        )
        if stats.empty:
            raise ValueError(
                f"{self.label}: column '{col}' has no non-null values paired with a "
                "finite target, so no category means can be learned"
            )
        smoothing = float(self.params["smoothing"])
        counts = stats["count"].to_numpy(dtype="float64")
        means = stats["mean"].to_numpy(dtype="float64")
        smoothed = (counts * means + smoothing * self.global_mean_) / (
            counts + smoothing
        )
        self.encodings_ = {
            _as_python(key): float(value) for key, value in zip(stats.index, smoothed)
        }
        return self._mark_fitted()

    def _target_values(self, y: Any, n_rows: int) -> np.ndarray:
        if y is None:
            raise ValueError(
                f"{self.label}: target encoding needs y at fit() time. Pass the "
                "TRAINING target; transform() never sees it."
            )
        values = y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
        if values.ndim > 1:
            if values.shape[1] != 1:
                raise ValueError(
                    f"{self.label}: y must be 1-dimensional, got shape {values.shape}"
                )
            values = values.reshape(-1)
        try:
            targets = np.asarray(values, dtype="float64")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.label}: y must be numeric for target encoding "
                f"(label-encode classification targets first): {exc}"
            ) from exc
        if targets.shape[0] != n_rows:
            raise ValueError(
                f"{self.label}: y has {targets.shape[0]} rows but X has {n_rows}; "
                "y is matched to X positionally"
            )
        return targets

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        self._check_fitted()
        raw = _require_column(X, self.params["col"], self.label)
        return {
            self.names[0]: _lookup_with_fallback(
                raw, self.encodings_, self.global_mean_
            )
        }


class GroupAggregateOp(FeatureOp):
    """Per-group aggregate of a numeric column, learned on TRAIN.

    ``fit`` learns one aggregate per group plus the same aggregate over all
    training rows; ``transform`` is a pure lookup, so an unseen group falls back
    to the global aggregate and a null group key stays NaN.

    ``agg="std"`` is the pandas sample standard deviation (ddof=1), so a group
    with a single training row has an *undefined* std: the learned value is NaN
    and it propagates. We deliberately do not substitute the global std there --
    that would present a fabricated number as this group's statistic. The
    global fallback exists for groups we have never seen, not for groups whose
    statistic does not exist.
    """

    name = "group_aggregate"
    required_params = ("by", "col")
    optional_params = {"agg": "mean"}
    column_params = ("by", "col")
    stateful = True

    @classmethod
    def output_names(cls, params: Mapping[str, Any]) -> list[str]:
        return [f"{params['col']}_{params['agg']}_by_{params['by']}"]

    @classmethod
    def _value_errors(cls, params: Mapping[str, Any]) -> list[str]:
        agg = params["agg"]
        if not isinstance(agg, str):
            return [f"param 'agg' must be a string, got {type(agg).__name__}"]
        if agg not in AGGREGATIONS:
            return [
                f"param 'agg' must be one of {', '.join(AGGREGATIONS)}, got '{agg}'"
            ]
        if params["by"] == params["col"]:
            return ["params 'by' and 'col' must be different columns"]
        return []

    def fit(self, X: pd.DataFrame, y: Any = None) -> GroupAggregateOp:
        by, col = self.params["by"], self.params["col"]
        agg = str(self.params["agg"])
        keys = _require_column(X, by, self.label).reset_index(drop=True)
        values = pd.Series(_numeric_values(X, col, self.label))
        if int(values.notna().sum()) == 0:
            raise ValueError(
                f"{self.label}: column '{col}' has no non-null values in the "
                f"training data, so no '{agg}' can be learned"
            )
        global_value = float(values.agg(agg))
        if not math.isfinite(global_value):
            raise ValueError(
                f"{self.label}: the global '{agg}' of column '{col}' is not finite "
                f"({global_value}), so unseen groups would have no usable fallback"
            )
        grouped = (
            pd.DataFrame({"key": keys, "value": values})
            .groupby("key", dropna=True, observed=True)["value"]
            .agg(agg)
        )
        if grouped.empty:
            raise ValueError(
                f"{self.label}: column '{by}' has no non-null values in the training "
                "data, so no groups can be learned"
            )
        self.global_value_ = global_value
        self.aggregates_ = {
            _as_python(key): float(value) for key, value in grouped.items()
        }
        return self._mark_fitted()

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        self._check_fitted()
        # Only the group key is read: the value column of the rows being
        # transformed must NOT influence the aggregate, or the feature would be
        # computed from the batch instead of from the training data.
        keys = _require_column(X, self.params["by"], self.label)
        return {
            self.names[0]: _lookup_with_fallback(
                keys, self.aggregates_, self.global_value_
            )
        }


#: The whitelist. Anything not in here is rejected by :func:`validate_specs`.
OPS: dict[str, type] = {
    op_cls.name: op_cls
    for op_cls in (
        Log1pOp,
        RatioOp,
        ProductOp,
        WinsorizeOp,
        QuantileBinOp,
        FrequencyEncodeOp,
        TargetEncodeOp,
        GroupAggregateOp,
        DatetimePartsOp,
    )
}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def validate_specs(specs: list[FeatureSpec], columns: list[str]) -> list[str]:
    """Report everything wrong with ``specs``; an empty list means all valid.

    Checks: unknown op, missing/unexpected params, bad param values, referenced
    column not present in ``columns``, output name colliding with an existing
    column, and duplicate output names across specs.

    This function NEVER raises -- it reports. That is the whole point: a batch
    of machine-proposed specs can be triaged (report the bad ones, keep the good
    ones) instead of one typo taking the run down. Plain dicts are accepted
    alongside :class:`FeatureSpec` so freshly-deserialised specs can be checked
    before construction; structurally broken entries are reported too.

    ``columns`` should be the columns the transformer will actually see. Do NOT
    include the target column: an op that read the target would leak it.
    """
    errors: list[str] = []
    if specs is None:
        return errors
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        return [
            f"specs must be a list of FeatureSpec, got {type(specs).__name__}"
        ]
    known_columns = list(columns) if columns is not None else []
    produced_by: dict[str, int] = {}

    for index, item in enumerate(specs):
        prefix = f"spec[{index}]"
        if isinstance(item, FeatureSpec):
            spec = item
        elif isinstance(item, Mapping):
            op_name = item.get("op")
            if not isinstance(op_name, str) or not op_name:
                errors.append(
                    f"{prefix}: 'op' must be a non-empty string, got {op_name!r}"
                )
                continue
            raw_params = item.get("params", {})
            if raw_params is None:
                raw_params = {}
            if not isinstance(raw_params, Mapping):
                errors.append(
                    f"{prefix} op '{op_name}': 'params' must be a mapping, got "
                    f"{type(raw_params).__name__}"
                )
                continue
            unexpected_keys = sorted(set(item) - {"op", "params"})
            if unexpected_keys:
                errors.append(
                    f"{prefix} op '{op_name}': unexpected key(s) "
                    f"{', '.join(unexpected_keys)} (expected 'op' and 'params')"
                )
            spec = FeatureSpec(op=op_name, params=dict(raw_params))
        else:
            errors.append(
                f"{prefix}: expected a FeatureSpec or dict, got {type(item).__name__}"
            )
            continue

        prefix = f"{prefix} op '{spec.op}'"
        op_cls = OPS.get(spec.op)
        if op_cls is None:
            errors.append(
                f"{prefix}: unknown op (whitelisted ops: {', '.join(sorted(OPS))})"
            )
            continue

        param_errors = op_cls.param_errors(spec.params)
        if param_errors:
            errors.extend(f"{prefix}: {message}" for message in param_errors)
            # Column and output-name checks on a spec with bad params would be
            # guesswork; the caller has to fix the params first anyway.
            continue

        resolved = op_cls.resolve_params(spec.params)
        missing_columns = [
            resolved[name]
            for name in op_cls.column_params
            if resolved[name] not in known_columns
        ]
        if missing_columns:
            errors.append(
                f"{prefix}: column(s) {', '.join(repr(c) for c in missing_columns)} "
                f"not in the input data (available: {_describe_columns(known_columns)})"
            )
            continue

        for output in op_cls.output_names(resolved):
            if output in known_columns:
                errors.append(
                    f"{prefix}: output column '{output}' already exists in the input "
                    "data; it would overwrite a real column"
                )
            elif output in produced_by:
                errors.append(
                    f"{prefix}: output column '{output}' is already produced by "
                    f"spec[{produced_by[output]}]"
                )
            else:
                produced_by[output] = index

    return errors


# --------------------------------------------------------------------------- #
# the transformer
# --------------------------------------------------------------------------- #


class FeatureSpecTransformer(BaseEstimator, TransformerMixin):
    """Compile a list of :class:`FeatureSpec` into one sklearn transformer.

    ``fit(X, y)`` learns every parameter from ``X`` (and ``y``, for target
    encoding) -- training data only. ``transform(X)`` appends the engineered
    columns to ``X`` and touches nothing else; because every op is a pure
    lookup/arithmetic step over stored parameters, one row transforms exactly as
    it would inside a batch, which is what makes these features replayable at
    inference time.

    Parameters
    ----------
    specs:
        List of :class:`FeatureSpec` (plain dicts are accepted and normalised at
        fit time, so a JSON round-trip needs no extra step). ``None`` or ``[]``
        means "engineer nothing" and transform becomes a copy.

    Notes
    -----
    * ``fit`` runs :func:`validate_specs` and raises with the full list of
      problems if any spec is invalid. Call :func:`validate_specs` first if you
      would rather report and drop the bad specs than fail the run.
    * ``transform`` requires the source columns the fitted ops read; other
      columns may come and go and are passed through untouched.
    * Serialisable with joblib/pickle: all learned state is plain dicts, floats
      and numpy arrays.
    """

    def __init__(self, specs: list[FeatureSpec] | None = None):
        # sklearn contract: store constructor args unmodified (clone() checks
        # object identity), do all work in fit().
        self.specs = specs

    # -- fit ---------------------------------------------------------------- #

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureSpecTransformer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"FeatureSpecTransformer.fit expects a pandas DataFrame, got "
                f"{type(X).__name__}"
            )
        columns = list(X.columns)
        errors = validate_specs(self.specs, columns)
        if errors:
            raise ValueError(
                "FeatureSpecTransformer.fit: invalid feature specs:\n  - "
                + "\n  - ".join(errors)
            )

        specs = [
            spec if isinstance(spec, FeatureSpec) else FeatureSpec.from_dict(spec)
            for spec in (self.specs or [])
        ]
        fitted_ops: list[FeatureOp] = []
        for spec in specs:
            op = OPS[spec.op](**spec.params)
            op.fit(X, y)
            fitted_ops.append(op)

        self.specs_ = specs
        self.fitted_ops_ = fitted_ops
        self.engineered_names_ = [name for op in fitted_ops for name in op.names]
        self.input_columns_ = columns
        self.n_features_in_ = X.shape[1]
        if all(isinstance(c, str) for c in columns):
            # Same convention as sklearn: only set when every name is a string.
            self.feature_names_in_ = np.asarray(columns, dtype=object)
        return self

    # -- transform ---------------------------------------------------------- #

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "fitted_ops_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"FeatureSpecTransformer.transform expects a pandas DataFrame, got "
                f"{type(X).__name__}"
            )
        missing = [c for c in self.required_columns_ if c not in X.columns]
        if missing:
            raise ValueError(
                "FeatureSpecTransformer.transform: missing source column(s) "
                f"{', '.join(repr(c) for c in missing)} required by the fitted specs "
                f"(got: {_describe_columns(X.columns)})"
            )
        clashing = [name for name in self.engineered_names_ if name in X.columns]
        if clashing:
            raise ValueError(
                "FeatureSpecTransformer.transform: engineered column(s) "
                f"{', '.join(repr(c) for c in clashing)} already exist in the input; "
                "refusing to overwrite them"
            )

        blocks: dict[str, np.ndarray] = {}
        for op in self.fitted_ops_:
            produced = op.transform(X)
            unexpected = sorted(set(produced) - set(op.names))
            if unexpected or len(produced) != len(op.names):
                raise RuntimeError(
                    f"{op.label}: produced {sorted(produced)} but declared "
                    f"{op.names}; this is a bug in the op"
                )
            for name, values in produced.items():
                if len(values) != len(X):
                    raise RuntimeError(
                        f"{op.label}: produced {len(values)} values for {len(X)} rows; "
                        "this is a bug in the op"
                    )
                blocks[name] = np.asarray(values, dtype="float64")

        if not blocks:
            return X.copy()
        engineered = pd.DataFrame(
            {name: blocks[name] for name in self.engineered_names_},
            index=X.index,
            columns=self.engineered_names_,
        )
        return pd.concat([X, engineered], axis=1)

    # -- introspection ------------------------------------------------------ #

    @property
    def required_columns_(self) -> list[str]:
        """Source columns the fitted ops read (order-preserving, de-duplicated)."""
        check_is_fitted(self, "fitted_ops_")
        needed: list[str] = []
        for op in self.fitted_ops_:
            for param in type(op).column_params:
                column = op.params[param]
                if column not in needed:
                    needed.append(column)
        return needed

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "fitted_ops_")
        passthrough = [str(c) for c in self.input_columns_]
        if input_features is not None:
            given = [str(c) for c in input_features]
            if given != passthrough:
                raise ValueError(
                    "input_features is not equal to the columns seen during fit: "
                    f"expected {passthrough}, got {given}"
                )
        return np.asarray(passthrough + list(self.engineered_names_), dtype=object)

    def learned_parameters(self) -> list[dict[str, Any]]:
        """The fitted state, for audit trails and model cards.

        Everything listed here was learned from the fit() data only -- that is
        the property that makes these features safe to persist and replay.
        """
        check_is_fitted(self, "fitted_ops_")
        report: list[dict[str, Any]] = []
        for spec, op in zip(self.specs_, self.fitted_ops_):
            learned = {
                key: value
                for key, value in vars(op).items()
                if key.endswith("_") and not key.startswith("_")
            }
            report.append(
                {
                    "op": spec.op,
                    "params": dict(op.params),
                    "outputs": list(op.names),
                    "stateful": type(op).stateful,
                    "learned": learned,
                }
            )
        return report
