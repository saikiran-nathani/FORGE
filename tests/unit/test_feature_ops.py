"""Property tests for the declarative feature-op registry.

The registry exists to make two invariants true, so they are tested as
properties over *every* whitelisted op rather than spot-checked:

1. **No leakage** -- every learned parameter comes from the frame passed to
   ``fit``. Tests hand-compute the expected parameters from the TRAIN rows and
   then transform a test frame drawn from a deliberately different
   distribution: the parameters must not budge and the output must be explained
   entirely by the train statistics.
2. **Train/serve parity** -- a single row transforms to exactly the values it
   gets inside a batch. This is checked row by row, for every op, both by
   slicing the batch (``iloc[[i]]``) and by rebuilding the row from a dict (the
   real serving path, where dtypes are re-inferred from a JSON payload).

``test_registry_is_fully_covered_by_parity_tests`` makes the coverage
non-optional: adding an op to ``OPS`` without adding it to ``ALL_SPECS`` fails.
"""

from __future__ import annotations

import copy
import json
import math
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from forge.feature_engineering.feature_ops import (
    AGGREGATIONS,
    OPS,
    FeatureSpec,
    FeatureSpecTransformer,
    validate_specs,
)

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

#: Covers every op in the registry, every aggregation, and both stateful params.
ALL_SPECS: list[FeatureSpec] = [
    FeatureSpec("log1p", {"col": "amount"}),
    FeatureSpec("ratio", {"a": "amount", "b": "qty"}),
    FeatureSpec("product", {"a": "amount", "b": "qty"}),
    FeatureSpec("winsorize", {"col": "amount"}),
    FeatureSpec("quantile_bin", {"col": "amount", "n_bins": 4}),
    FeatureSpec("quantile_bin", {"col": "qty", "n_bins": 3}),
    FeatureSpec("frequency_encode", {"col": "city"}),
    FeatureSpec("target_encode", {"col": "city", "smoothing": 5.0}),
    FeatureSpec("target_encode", {"col": "dept", "smoothing": 0.0}),
    *[
        FeatureSpec("group_aggregate", {"by": "dept", "col": "amount", "agg": agg})
        for agg in AGGREGATIONS
    ],
    FeatureSpec("datetime_parts", {"col": "signup_ts"}),
]

SPEC_IDS = [
    f"{s.op}-{'-'.join(str(v) for v in s.params.values())}" for s in ALL_SPECS
]


def _train_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Training data: 240 rows, imbalanced categories, a singleton group.

    Deliberately includes the awkward cases -- NaN, a zero denominator, a
    negative value, one huge outlier -- so the properties are tested where naive
    code breaks, not only on clean data.
    """
    rng = np.random.default_rng(20240813)
    n = 240
    amount = rng.gamma(2.0, 40.0, n).round(3)
    amount[3] = -12.5  # negative: log1p must clip, winsorize must bound
    amount[7] = 25_000.0  # outlier the p99 must exclude
    amount[11] = np.nan
    qty = rng.integers(0, 6, n).astype(float)
    qty[2] = 0.0  # zero denominator for `ratio`
    qty[13] = np.nan
    city = rng.choice(["NYC", "LA", "CHI", "SF"], n, p=[0.5, 0.25, 0.15, 0.10])
    city = np.asarray(city, dtype=object)
    city[5] = None  # missing category
    dept = rng.choice(["eng", "ops", "sales"], n).astype(object)
    dept[0] = "solo"  # singleton group -> sample std is undefined
    signup = pd.date_range("2021-01-01", periods=n, freq="19h")
    frame = pd.DataFrame(
        {
            "amount": amount,
            "qty": qty,
            "city": city,
            "dept": dept,
            "signup_ts": signup.strftime("%Y-%m-%d"),
        }
    )
    # Target depends on city, so target encoding has real signal to learn.
    y = pd.Series(
        rng.normal(0.0, 1.0, n) + (frame["city"] == "NYC").to_numpy() * 3.0,
        name="target",
    )
    return frame, y


def _serving_frame() -> pd.DataFrame:
    """Serving data: different distribution, unseen category and group, nulls.

    Nothing here may influence a transform result -- that is the leakage
    property -- and every row must transform identically alone or in the batch.
    """
    rng = np.random.default_rng(99)
    n = 60
    amount = (rng.gamma(2.0, 400.0, n) + 500.0).round(3)  # ~10x the train scale
    amount[0] = -9_999.0  # below the train p01
    amount[1] = 10_000_000.0  # above the train p99
    amount[2] = np.nan
    qty = rng.integers(0, 12, n).astype(float)
    qty[0] = 0.0
    qty[3] = np.nan
    city = np.asarray(rng.choice(["NYC", "LA", "MARS"], n), dtype=object)
    city[0] = "MARS"  # never seen during fit
    city[4] = None
    dept = np.asarray(rng.choice(["eng", "atlantis"], n), dtype=object)
    dept[0] = "atlantis"  # never seen during fit
    dept[5] = None
    dept[6] = "solo"  # seen, but only once -> undefined sample std
    signup = np.asarray(
        pd.date_range("2024-06-01", periods=n, freq="31h").strftime("%Y-%m-%d"),
        dtype=object,
    )
    signup[7] = None
    signup[8] = "not a date"
    return pd.DataFrame(
        {
            "amount": amount,
            "qty": qty,
            "city": city,
            "dept": dept,
            "signup_ts": signup,
        }
    )


@pytest.fixture
def train():
    return _train_frame()


@pytest.fixture
def serving():
    return _serving_frame()


@pytest.fixture
def fitted_all(train):
    X, y = train
    return FeatureSpecTransformer(ALL_SPECS).fit(X, y)


def _engineered(transformer: FeatureSpecTransformer, frame: pd.DataFrame) -> np.ndarray:
    out = transformer.transform(frame)
    return out[transformer.engineered_names_].to_numpy(dtype="float64")


# --------------------------------------------------------------------------- #
# invariant 2: train/serve parity
# --------------------------------------------------------------------------- #


def test_registry_is_fully_covered_by_parity_tests():
    assert {spec.op for spec in ALL_SPECS} == set(OPS), (
        "every whitelisted op must appear in ALL_SPECS so the parity and "
        "determinism properties actually cover it"
    )


@pytest.mark.parametrize("spec", ALL_SPECS, ids=SPEC_IDS)
def test_single_row_equals_its_row_in_the_batch(spec, train, serving):
    """Invariant 2, per op: transform(X.iloc[[i]]) == row i of transform(X)."""
    X, y = train
    transformer = FeatureSpecTransformer([spec]).fit(X, y)
    batch = _engineered(transformer, serving)
    assert batch.shape == (len(serving), len(transformer.engineered_names_))

    for i in range(len(serving)):
        single = _engineered(transformer, serving.iloc[[i]])
        np.testing.assert_array_equal(
            single[0],
            batch[i],
            err_msg=(
                f"{spec.op}{spec.params}: row {i} transformed alone does not match "
                "the same row inside the batch"
            ),
        )


def test_row_rebuilt_from_a_dict_matches_the_batch(fitted_all, serving):
    """Parity through the real serving path, where dtypes are re-inferred.

    A prediction request arrives as JSON, not as a slice of the training frame,
    so the one-row DataFrame can legitimately have different dtypes (an int
    where the batch had float64, float64-NaN where the batch had object). The
    engineered values must still be identical.
    """
    batch = _engineered(fitted_all, serving)
    for i in range(len(serving)):
        payload = serving.iloc[i].to_dict()
        one_row = pd.DataFrame([payload])
        single = _engineered(fitted_all, one_row)
        np.testing.assert_array_equal(
            single[0],
            batch[i],
            err_msg=f"row {i} rebuilt from a dict does not match the batch",
        )


def test_transform_is_invariant_to_batch_composition(fitted_all, serving):
    """No transform may depend on which other rows happen to be present."""
    full = _engineered(fitted_all, serving)

    subset_positions = [0, 5, 17, 42]
    subset = _engineered(fitted_all, serving.iloc[subset_positions])
    np.testing.assert_array_equal(subset, full[subset_positions])

    shuffled_positions = list(reversed(range(len(serving))))
    shuffled = _engineered(fitted_all, serving.iloc[shuffled_positions])
    np.testing.assert_array_equal(shuffled, full[shuffled_positions])

    duplicated = _engineered(fitted_all, pd.concat([serving, serving], axis=0))
    np.testing.assert_array_equal(duplicated, np.vstack([full, full]))


def test_engineered_columns_are_always_float64(fitted_all, serving):
    """A batch-dependent dtype is a parity break waiting to happen."""
    out = fitted_all.transform(serving)
    for name in fitted_all.engineered_names_:
        assert out[name].dtype == np.dtype("float64"), name
    one_row = fitted_all.transform(serving.iloc[[0]])
    for name in fitted_all.engineered_names_:
        assert one_row[name].dtype == np.dtype("float64"), name


def test_datetime_parsing_is_pinned_at_fit_time():
    """The hazard this op is defensive about, demonstrated then excluded.

    ``pd.to_datetime`` infers a format from the first non-null element of
    whatever it is handed, so naive parsing genuinely disagrees between a batch
    and a single row. The op pins the strategy during fit, so it does not.
    """
    raw = pd.Series(["2021-01-04", "03/05/2021", "2021-01-06"])

    naive_batch = pd.to_datetime(raw, errors="coerce")
    naive_single = pd.to_datetime(raw.iloc[[1]], errors="coerce")
    assert pd.isna(naive_batch.iloc[1]), "expected the naive batch parse to lose row 1"
    assert pd.notna(naive_single.iloc[0]), "expected the naive single-row parse to keep it"

    frame = pd.DataFrame({"ts": raw})
    transformer = FeatureSpecTransformer(
        [FeatureSpec("datetime_parts", {"col": "ts"})]
    ).fit(frame)
    batch = _engineered(transformer, frame)
    for i in range(len(frame)):
        np.testing.assert_array_equal(_engineered(transformer, frame.iloc[[i]])[0], batch[i])
    assert not np.isnan(batch).any(), "all three dates are parseable"


# --------------------------------------------------------------------------- #
# invariant 1: no leakage -- hand-computed TRAIN parameters
# --------------------------------------------------------------------------- #


def test_winsorize_bounds_are_the_train_percentiles(train, serving):
    X, y = train
    transformer = FeatureSpecTransformer(
        [FeatureSpec("winsorize", {"col": "amount"})]
    ).fit(X, y)
    op = transformer.fitted_ops_[0]

    train_values = X["amount"].to_numpy(dtype="float64")
    finite = train_values[np.isfinite(train_values)]
    expected_low, expected_high = np.percentile(finite, [1.0, 99.0])
    assert op.lower_ == pytest.approx(float(expected_low))
    assert op.upper_ == pytest.approx(float(expected_high))
    # The serving frame is ~10x larger in scale; the bound must not follow it.
    assert op.upper_ < serving["amount"].median()

    out = transformer.transform(serving)["amount_winsor"]
    assert out.max() == pytest.approx(op.upper_)
    assert out.min() == pytest.approx(op.lower_)
    assert op.lower_ == pytest.approx(float(expected_low)), "transform refitted a bound"


def test_quantile_bin_edges_are_the_train_quantiles(train, serving):
    X, y = train
    spec = FeatureSpec("quantile_bin", {"col": "amount", "n_bins": 4})
    transformer = FeatureSpecTransformer([spec]).fit(X, y)
    op = transformer.fitted_ops_[0]

    train_values = X["amount"].to_numpy(dtype="float64")
    finite = train_values[np.isfinite(train_values)]
    expected = np.unique(np.quantile(finite, [0.25, 0.5, 0.75]))
    np.testing.assert_allclose(op.edges_, expected)
    assert op.n_bins_ == len(expected) + 1

    binned = transformer.transform(serving)["amount_bin4"]
    # Unseen extremes clamp into the end bins instead of inventing new ones.
    assert binned.iloc[0] == 0.0, "a value below every train edge belongs in bin 0"
    assert binned.iloc[1] == float(op.n_bins_ - 1), "a value above every edge clamps"
    assert math.isnan(binned.iloc[2]), "a NaN input must stay NaN, not land in a bin"
    observed = binned.dropna().unique()
    assert set(observed) <= set(float(b) for b in range(op.n_bins_))
    np.testing.assert_allclose(op.edges_, expected), "transform refitted the edges"


def test_frequency_encoding_uses_train_counts_only(serving):
    X = pd.DataFrame({"city": ["a", "a", "a", "b", None]})
    transformer = FeatureSpecTransformer(
        [FeatureSpec("frequency_encode", {"col": "city"})]
    ).fit(X)
    op = transformer.fitted_ops_[0]
    # 5 train rows: 'a' in 3 of them, 'b' in 1, one row missing.
    assert op.frequencies_ == {"a": 0.6, "b": 0.2}

    # A serving batch where 'b' is now the common value must not shift anything.
    later = pd.DataFrame({"city": ["b", "b", "b", "b", "a", "zzz", None]})
    out = transformer.transform(later)["city_freq"].tolist()
    assert out[:5] == [0.2, 0.2, 0.2, 0.2, 0.6]
    assert out[5] == 0.0, "an unseen category is 0.0 of the train rows"
    assert math.isnan(out[6]), "a missing input stays missing"
    assert op.frequencies_ == {"a": 0.6, "b": 0.2}, "transform refitted the counts"


def test_target_encoding_uses_train_target_only():
    X = pd.DataFrame({"city": ["a", "a", "b", "b", "b", "c"]})
    y = pd.Series([10.0, 20.0, 0.0, 0.0, 30.0, 100.0])
    smoothing = 2.0
    transformer = FeatureSpecTransformer(
        [FeatureSpec("target_encode", {"col": "city", "smoothing": smoothing})]
    ).fit(X, y)
    op = transformer.fitted_ops_[0]

    global_mean = (10 + 20 + 0 + 0 + 30 + 100) / 6
    assert op.global_mean_ == pytest.approx(global_mean)

    def smoothed(count: int, mean: float) -> float:
        return (count * mean + smoothing * global_mean) / (count + smoothing)

    assert op.encodings_["a"] == pytest.approx(smoothed(2, 15.0))
    assert op.encodings_["b"] == pytest.approx(smoothed(3, 10.0))
    assert op.encodings_["c"] == pytest.approx(smoothed(1, 100.0))

    # A serving batch with a wildly different target distribution available in
    # another column must not (cannot) change the encoding: transform never
    # sees y at all.
    later = pd.DataFrame({"city": ["a", "b", "c", "unseen", None]})
    out = transformer.transform(later)["city_target_enc"].tolist()
    assert out[0] == pytest.approx(op.encodings_["a"])
    assert out[3] == pytest.approx(global_mean), "unseen category -> global mean"
    assert math.isnan(out[4])
    assert transformer.transform(later)["city_target_enc"].tolist()[0] == pytest.approx(
        out[0]
    )


def test_target_encoding_never_reads_y_at_transform_time(train, serving):
    X, y = train
    transformer = FeatureSpecTransformer(
        [FeatureSpec("target_encode", {"col": "city", "smoothing": 5.0})]
    ).fit(X, y)
    before = copy.deepcopy(transformer.fitted_ops_[0].encodings_)
    # Same rows, opposite target: if any of the target leaked into transform the
    # two outputs would differ.
    with_target = serving.copy()
    with_target["target"] = np.linspace(-1000, 1000, len(serving))
    first = transformer.transform(serving)["city_target_enc"].to_numpy()
    second = transformer.transform(with_target)["city_target_enc"].to_numpy()
    np.testing.assert_array_equal(first, second)
    assert transformer.fitted_ops_[0].encodings_ == before


def test_target_encode_requires_y_and_a_numeric_target():
    X = pd.DataFrame({"city": ["a", "b"]})
    spec = FeatureSpec("target_encode", {"col": "city"})
    with pytest.raises(ValueError, match="needs y at fit"):
        FeatureSpecTransformer([spec]).fit(X)
    with pytest.raises(ValueError, match="must be numeric"):
        FeatureSpecTransformer([spec]).fit(X, pd.Series(["yes", "no"]))
    with pytest.raises(ValueError, match="positionally"):
        FeatureSpecTransformer([spec]).fit(X, pd.Series([1.0]))


def test_target_encode_reads_y_positionally_not_by_index():
    """A misaligned index must not silently reshuffle the target."""
    X = pd.DataFrame({"city": ["a", "a", "b", "b"]}, index=[10, 11, 12, 13])
    y_positional = pd.Series([1.0, 1.0, 0.0, 0.0], index=[0, 1, 2, 3])
    y_same_values = pd.Series([1.0, 1.0, 0.0, 0.0], index=[10, 11, 12, 13])
    spec = FeatureSpec("target_encode", {"col": "city", "smoothing": 0.0})
    first = FeatureSpecTransformer([spec]).fit(X, y_positional).fitted_ops_[0]
    second = FeatureSpecTransformer([spec]).fit(X, y_same_values).fitted_ops_[0]
    assert first.encodings_ == second.encodings_ == {"a": 1.0, "b": 0.0}


def test_target_encode_smoothing_pulls_small_categories_to_the_global_mean():
    """Requirement 4: a 1-row category is pulled far harder than a 50-row one."""
    rows = ["rare"] + ["common"] * 50 + ["filler"] * 49
    targets = [100.0] + [100.0] * 50 + [0.0] * 49
    X = pd.DataFrame({"g": rows})
    y = pd.Series(targets)
    smoothing = 10.0
    op = (
        FeatureSpecTransformer(
            [FeatureSpec("target_encode", {"col": "g", "smoothing": smoothing})]
        )
        .fit(X, y)
        .fitted_ops_[0]
    )
    global_mean = op.global_mean_
    rare_gap = abs(op.encodings_["rare"] - global_mean)
    common_gap = abs(op.encodings_["common"] - global_mean)

    # Both categories have the same raw mean (100.0), so the only thing that
    # separates them is the shrinkage.
    assert rare_gap < common_gap
    assert rare_gap < 0.2 * common_gap, (
        f"1-row category kept {rare_gap:.3f} of its signal vs {common_gap:.3f} for "
        "the 50-row category; shrinkage is not doing its job"
    )
    assert op.encodings_["rare"] == pytest.approx(
        (1 * 100.0 + smoothing * global_mean) / (1 + smoothing)
    )
    assert op.encodings_["common"] == pytest.approx(
        (50 * 100.0 + smoothing * global_mean) / (50 + smoothing)
    )

    # More smoothing -> everything closer to the global mean; zero smoothing ->
    # the raw category mean.
    gaps = []
    for value in (0.0, 1.0, 10.0, 100.0):
        encodings = (
            FeatureSpecTransformer(
                [FeatureSpec("target_encode", {"col": "g", "smoothing": value})]
            )
            .fit(X, y)
            .fitted_ops_[0]
            .encodings_
        )
        gaps.append(abs(encodings["common"] - global_mean))
    assert gaps == sorted(gaps, reverse=True)
    unsmoothed = (
        FeatureSpecTransformer(
            [FeatureSpec("target_encode", {"col": "g", "smoothing": 0.0})]
        )
        .fit(X, y)
        .fitted_ops_[0]
        .encodings_
    )
    assert unsmoothed["rare"] == pytest.approx(100.0)
    assert unsmoothed["filler"] == pytest.approx(0.0)


@pytest.mark.parametrize("agg", AGGREGATIONS)
def test_group_aggregate_learns_train_group_statistics(agg):
    X = pd.DataFrame(
        {
            "dept": ["a", "a", "a", "b", "b", "c", None],
            "amount": [1.0, 3.0, 5.0, 10.0, 20.0, 7.0, 999.0],
        }
    )
    transformer = FeatureSpecTransformer(
        [FeatureSpec("group_aggregate", {"by": "dept", "col": "amount", "agg": agg})]
    ).fit(X)
    op = transformer.fitted_ops_[0]

    expected = X.groupby("dept")["amount"].agg(agg).to_dict()
    expected_global = float(X["amount"].agg(agg))
    assert set(op.aggregates_) == {"a", "b", "c"}
    for key, value in expected.items():
        if math.isnan(value):
            assert math.isnan(op.aggregates_[key])
        else:
            assert op.aggregates_[key] == pytest.approx(value)
    assert op.global_value_ == pytest.approx(expected_global)

    # A serving batch whose amounts are 1000x larger cannot move the aggregate:
    # transform only reads the group key.
    later = pd.DataFrame(
        {
            "dept": ["a", "b", "c", "unseen", None],
            "amount": [1e6, 1e6, 1e6, 1e6, 1e6],
        }
    )
    name = f"amount_{agg}_by_dept"
    out = transformer.transform(later)[name].tolist()
    for i, key in enumerate(["a", "b", "c"]):
        if math.isnan(op.aggregates_[key]):
            assert math.isnan(out[i])
        else:
            assert out[i] == pytest.approx(op.aggregates_[key])
    assert out[3] == pytest.approx(expected_global), "unseen group -> global aggregate"
    assert math.isnan(out[4]), "a null group key stays missing"


def test_group_aggregate_std_of_a_singleton_group_is_nan_not_the_global_std():
    """Documented decision: an undefined statistic is reported, not faked."""
    X = pd.DataFrame({"g": ["a", "a", "solo"], "v": [1.0, 3.0, 50.0]})
    transformer = FeatureSpecTransformer(
        [FeatureSpec("group_aggregate", {"by": "g", "col": "v", "agg": "std"})]
    ).fit(X)
    op = transformer.fitted_ops_[0]
    assert math.isnan(op.aggregates_["solo"])
    out = transformer.transform(pd.DataFrame({"g": ["solo", "unseen"], "v": [0.0, 0.0]}))
    values = out["v_std_by_g"].tolist()
    assert math.isnan(values[0]), "sample std of one row is undefined"
    assert values[1] == pytest.approx(op.global_value_), "unseen group -> global std"


def test_learned_state_is_untouched_by_transform(fitted_all, serving):
    """Generic no-leakage guard across every op: transform never refits."""
    before = pickle.dumps(fitted_all.learned_parameters())
    fitted_all.transform(serving)
    fitted_all.transform(serving.iloc[[0]])
    fitted_all.transform(pd.concat([serving, serving]))
    assert pickle.dumps(fitted_all.learned_parameters()) == before


def test_parameters_actually_depend_on_the_fit_data(train, serving):
    """Guards against a vacuous no-leakage test.

    If the learned parameters were constants, every leakage assertion above
    would pass trivially. Fitting on train+serving must move them.
    """
    X, y = train
    clean = FeatureSpecTransformer(ALL_SPECS).fit(X, y)
    polluted_X = pd.concat([X, serving], ignore_index=True)
    polluted_y = pd.concat([y, pd.Series(np.zeros(len(serving)))], ignore_index=True)
    polluted = FeatureSpecTransformer(ALL_SPECS).fit(polluted_X, polluted_y)
    assert pickle.dumps(clean.learned_parameters()) != pickle.dumps(
        polluted.learned_parameters()
    )
    # ... and the difference is visible in the output for the same rows.
    assert not np.array_equal(
        _engineered(clean, serving), _engineered(polluted, serving), equal_nan=True
    )


# --------------------------------------------------------------------------- #
# unseen values, missing values, and the stateless ops
# --------------------------------------------------------------------------- #


def test_unseen_values_get_the_documented_fallback_not_nan(train, serving):
    X, y = train
    specs = [
        FeatureSpec("frequency_encode", {"col": "city"}),
        FeatureSpec("target_encode", {"col": "city", "smoothing": 5.0}),
        FeatureSpec("group_aggregate", {"by": "dept", "col": "amount", "agg": "mean"}),
    ]
    transformer = FeatureSpecTransformer(specs).fit(X, y)
    freq_op, target_op, group_op = transformer.fitted_ops_

    assert "MARS" not in freq_op.frequencies_
    assert "MARS" not in target_op.encodings_
    assert "atlantis" not in group_op.aggregates_

    out = transformer.transform(serving)
    unseen_city = out["city"] == "MARS"
    unseen_dept = out["dept"] == "atlantis"
    assert unseen_city.any() and unseen_dept.any()

    assert (out.loc[unseen_city, "city_freq"] == 0.0).all()
    assert out.loc[unseen_city, "city_freq"].notna().all()
    # np.allclose, not `Series == pytest.approx(scalar)`: approx does not
    # broadcast elementwise across a Series, so that idiom compares a Series to
    # an approx object and silently yields False.
    assert np.allclose(
        out.loc[unseen_city, "city_target_enc"].to_numpy(), target_op.global_mean_
    )
    assert np.allclose(
        out.loc[unseen_dept, "amount_mean_by_dept"].to_numpy(), group_op.global_value_
    )

    # Present-and-seen rows must not be NaN either: the only NaNs in these three
    # columns come from null inputs.
    for column, source in [
        ("city_freq", "city"),
        ("city_target_enc", "city"),
        ("amount_mean_by_dept", "dept"),
    ]:
        assert out[column].isna().tolist() == out[source].isna().tolist(), column


def test_missing_inputs_produce_missing_outputs_not_fabricated_values(fitted_all, serving):
    row = serving.iloc[[2]]  # amount is NaN here
    out = fitted_all.transform(row)
    for name in ["amount_log1p", "amount_winsor", "amount_bin4", "amount_ratio_qty"]:
        assert math.isnan(out[name].iloc[0]), name
    # A NaN group key or category is missing, not "unseen".
    null_city = fitted_all.transform(serving.iloc[[4]])
    assert math.isnan(null_city["city_freq"].iloc[0])
    assert math.isnan(null_city["city_target_enc"].iloc[0])


def test_ratio_reports_a_zero_denominator_as_nan_not_infinity():
    X = pd.DataFrame({"a": [1.0, 1.0, -1.0, 1.0], "b": [2.0, 0.0, 0.0, np.nan]})
    transformer = FeatureSpecTransformer(
        [FeatureSpec("ratio", {"a": "a", "b": "b"})]
    ).fit(X)
    values = transformer.transform(X)["a_ratio_b"].tolist()
    assert values[0] == pytest.approx(0.5)
    assert math.isnan(values[1]), "1/0 must be NaN, not inf and not a padded epsilon"
    assert math.isnan(values[2])
    assert math.isnan(values[3])
    assert np.isfinite(np.asarray(values, dtype=float)).sum() == 1


def test_stateless_ops_match_their_definition():
    X = pd.DataFrame({"a": [-5.0, 0.0, 3.0, np.nan], "b": [2.0, 4.0, 0.5, 2.0]})
    transformer = FeatureSpecTransformer(
        [
            FeatureSpec("log1p", {"col": "a"}),
            FeatureSpec("product", {"a": "a", "b": "b"}),
        ]
    ).fit(X)
    out = transformer.transform(X)
    np.testing.assert_allclose(
        out["a_log1p"].to_numpy(),
        [0.0, 0.0, math.log1p(3.0), np.nan],
    )
    np.testing.assert_allclose(out["a_x_b"].to_numpy(), [-10.0, 0.0, 1.5, np.nan])


def test_datetime_parts_are_calendar_correct():
    X = pd.DataFrame({"d": ["2021-01-02", "2021-03-15", None, "not a date"]})
    transformer = FeatureSpecTransformer(
        [FeatureSpec("datetime_parts", {"col": "d"})]
    ).fit(X)
    out = transformer.transform(X)
    np.testing.assert_allclose(out["d_dayofweek"].to_numpy(), [5.0, 0.0, np.nan, np.nan])
    np.testing.assert_allclose(out["d_month"].to_numpy(), [1.0, 3.0, np.nan, np.nan])
    np.testing.assert_allclose(out["d_is_weekend"].to_numpy(), [1.0, 0.0, np.nan, np.nan])


def test_datetime_parts_accepts_native_datetime64_columns():
    X = pd.DataFrame({"d": pd.to_datetime(["2021-01-02", "2021-03-15"])})
    transformer = FeatureSpecTransformer(
        [FeatureSpec("datetime_parts", {"col": "d"})]
    ).fit(X)
    assert transformer.fitted_ops_[0].strategy_ == "native"
    out = transformer.transform(X)
    np.testing.assert_allclose(out["d_is_weekend"].to_numpy(), [1.0, 0.0])
    # Same values when the column arrives as strings at serving time.
    as_strings = pd.DataFrame({"d": ["2021-01-02", "2021-03-15"]})
    np.testing.assert_array_equal(
        _engineered(transformer, as_strings), _engineered(transformer, X)
    )


# --------------------------------------------------------------------------- #
# validate_specs: reports, never raises
# --------------------------------------------------------------------------- #


def test_validate_specs_accepts_a_valid_batch(train):
    X, _ = train
    assert validate_specs(ALL_SPECS, list(X.columns)) == []
    assert validate_specs([spec.to_dict() for spec in ALL_SPECS], list(X.columns)) == []
    assert validate_specs([], ["a"]) == []


def test_validate_specs_reports_unknown_op():
    errors = validate_specs([FeatureSpec("rolling_zscore", {"col": "a"})], ["a"])
    assert len(errors) == 1
    assert "unknown op" in errors[0]
    assert "rolling_zscore" in errors[0]
    assert "log1p" in errors[0], "the message should list the whitelist"


def test_validate_specs_reports_missing_param():
    errors = validate_specs([FeatureSpec("ratio", {"a": "x"})], ["x", "y"])
    assert len(errors) == 1
    assert "missing required param 'b'" in errors[0]

    errors = validate_specs([FeatureSpec("group_aggregate", {"col": "x"})], ["x"])
    assert any("missing required param 'by'" in e for e in errors)


def test_validate_specs_reports_nonexistent_column():
    errors = validate_specs(
        [FeatureSpec("log1p", {"col": "does_not_exist"})], ["a", "b"]
    )
    assert len(errors) == 1
    assert "does_not_exist" in errors[0]
    assert "not in the input data" in errors[0]


def test_validate_specs_reports_extra_param_and_bad_values():
    cases = {
        "unexpected param": FeatureSpec("log1p", {"col": "a", "n_bins": 3}),
        "'n_bins' must be an int": FeatureSpec("quantile_bin", {"col": "a", "n_bins": "4"}),
        "'n_bins' must be >= 2": FeatureSpec("quantile_bin", {"col": "a", "n_bins": 1}),
        "'smoothing' must be >= 0": FeatureSpec(
            "target_encode", {"col": "a", "smoothing": -1.0}
        ),
        "'smoothing' must be finite": FeatureSpec(
            "target_encode", {"col": "a", "smoothing": float("inf")}
        ),
        "'agg' must be one of": FeatureSpec(
            "group_aggregate", {"by": "a", "col": "b", "agg": "sum"}
        ),
        "must be a column name (str)": FeatureSpec("log1p", {"col": 7}),
        "must be different columns": FeatureSpec(
            "group_aggregate", {"by": "a", "col": "a"}
        ),
    }
    for fragment, spec in cases.items():
        errors = validate_specs([spec], ["a", "b"])
        assert any(fragment in e for e in errors), (fragment, errors)


def test_validate_specs_reports_duplicate_and_colliding_output_names():
    duplicated = validate_specs(
        [
            FeatureSpec("log1p", {"col": "a"}),
            FeatureSpec("log1p", {"col": "a"}),
        ],
        ["a"],
    )
    assert len(duplicated) == 1
    assert "already produced by spec[0]" in duplicated[0]

    collision = validate_specs([FeatureSpec("log1p", {"col": "a"})], ["a", "a_log1p"])
    assert len(collision) == 1
    assert "already exists in the input data" in collision[0]

    # Different params -> different output names -> no collision.
    assert (
        validate_specs(
            [
                FeatureSpec("quantile_bin", {"col": "a", "n_bins": 3}),
                FeatureSpec("quantile_bin", {"col": "a", "n_bins": 5}),
            ],
            ["a"],
        )
        == []
    )


def test_validate_specs_reports_every_problem_in_one_pass():
    errors = validate_specs(
        [
            FeatureSpec("log1p", {"col": "a"}),  # fine
            FeatureSpec("nope", {}),
            FeatureSpec("ratio", {"a": "a"}),
            FeatureSpec("winsorize", {"col": "ghost"}),
        ],
        ["a"],
    )
    assert len(errors) == 3
    assert all(err.startswith("spec[") for err in errors)
    assert [err.split("]")[0] for err in errors] == ["spec[1", "spec[2", "spec[3"]


def test_validate_specs_never_raises_on_hostile_input():
    hostile = [
        None,
        42,
        "log1p",
        {},
        {"op": ""},
        {"op": 5, "params": {}},
        {"op": "log1p", "params": "col=a"},
        {"op": "log1p", "params": {"col": "a"}, "code": "import os"},
        {"op": "log1p"},
        FeatureSpec("log1p", {}),
        FeatureSpec("log1p", "not-a-dict"),
    ]
    for index, item in enumerate(hostile):
        errors = validate_specs([item], ["a"])
        assert isinstance(errors, list), index
        assert all(isinstance(e, str) for e in errors), index
        if index != 8:  # {"op": "log1p"} defaults params to {} -> only 'col' missing
            assert errors, f"item {index} ({item!r}) should have been reported"
    # The whole hostile batch at once, still no exception.
    assert len(validate_specs(hostile, ["a"])) >= len(hostile) - 1
    for bad_container in [None, "log1p", 42, {"op": "log1p"}]:
        assert isinstance(validate_specs(bad_container, ["a"]), list)


def test_fit_refuses_invalid_specs_and_reports_all_of_them(train):
    X, y = train
    specs = [FeatureSpec("nope", {}), FeatureSpec("log1p", {"col": "ghost"})]
    with pytest.raises(ValueError) as excinfo:
        FeatureSpecTransformer(specs).fit(X, y)
    message = str(excinfo.value)
    assert "unknown op" in message
    assert "ghost" in message


# --------------------------------------------------------------------------- #
# loud failures instead of silent defaults
# --------------------------------------------------------------------------- #


def test_ops_that_cannot_fit_raise(train):
    X, _ = train
    with pytest.raises(ValueError, match="not numeric"):
        FeatureSpecTransformer([FeatureSpec("log1p", {"col": "city"})]).fit(X)

    constant = pd.DataFrame({"a": [4.0] * 10})
    with pytest.raises(ValueError, match="too few distinct training values"):
        FeatureSpecTransformer(
            [FeatureSpec("quantile_bin", {"col": "a", "n_bins": 4})]
        ).fit(constant)

    all_nan = pd.DataFrame({"a": [np.nan] * 5, "g": ["x"] * 5})
    with pytest.raises(ValueError, match="no finite values"):
        FeatureSpecTransformer([FeatureSpec("winsorize", {"col": "a"})]).fit(all_nan)
    with pytest.raises(ValueError, match="no non-null values"):
        FeatureSpecTransformer(
            [FeatureSpec("group_aggregate", {"by": "g", "col": "a"})]
        ).fit(all_nan)
    with pytest.raises(ValueError, match="no non-null values"):
        FeatureSpecTransformer(
            [FeatureSpec("frequency_encode", {"col": "empty"})]
        ).fit(pd.DataFrame({"empty": [None, None]}))

    with pytest.raises(ValueError, match="nanoseconds since the epoch"):
        FeatureSpecTransformer([FeatureSpec("datetime_parts", {"col": "amount"})]).fit(X)
    with pytest.raises(ValueError, match="could be parsed as a datetime"):
        FeatureSpecTransformer([FeatureSpec("datetime_parts", {"col": "d"})]).fit(
            pd.DataFrame({"d": ["nope", "still not a date"]})
        )


def test_transform_failures_are_loud(fitted_all, serving):
    with pytest.raises(NotFittedError):
        FeatureSpecTransformer(ALL_SPECS).transform(serving)

    with pytest.raises(ValueError, match="missing source column"):
        fitted_all.transform(serving.drop(columns=["city"]))

    with pytest.raises(ValueError, match="refusing to overwrite"):
        polluted = serving.copy()
        polluted["amount_log1p"] = 0.0
        fitted_all.transform(polluted)

    with pytest.raises(TypeError, match="pandas DataFrame"):
        fitted_all.transform(serving.to_numpy())
    with pytest.raises(TypeError, match="pandas DataFrame"):
        FeatureSpecTransformer([]).fit(serving.to_numpy())

    with pytest.raises(ValueError, match="not numeric"):
        broken = serving.copy()
        broken["amount"] = broken["amount"].astype(str) + "kg"
        fitted_all.transform(broken)


def test_duplicate_source_columns_are_rejected(train):
    X, y = train
    transformer = FeatureSpecTransformer(
        [FeatureSpec("winsorize", {"col": "amount"})]
    ).fit(X, y)
    duplicated = pd.concat([X, X[["amount"]]], axis=1)
    with pytest.raises(ValueError, match="appears more than once"):
        transformer.transform(duplicated)


def test_op_constructor_rejects_bad_params_instead_of_defaulting():
    with pytest.raises(ValueError, match="missing required param"):
        OPS["ratio"](a="x")
    with pytest.raises(ValueError, match="unexpected param"):
        OPS["log1p"](col="x", smoothing=1.0)
    with pytest.raises(ValueError, match="must be >= 2"):
        OPS["quantile_bin"](col="x", n_bins=0)


# --------------------------------------------------------------------------- #
# serialisation, determinism, sklearn contract
# --------------------------------------------------------------------------- #


def test_joblib_round_trip_reproduces_the_output(fitted_all, serving, tmp_path):
    expected = fitted_all.transform(serving)
    path = tmp_path / "feature_ops.joblib"
    joblib.dump(fitted_all, path)
    restored = joblib.load(path)

    pd.testing.assert_frame_equal(restored.transform(serving), expected, check_exact=True)
    assert list(restored.get_feature_names_out()) == list(
        fitted_all.get_feature_names_out()
    )
    assert pickle.dumps(restored.learned_parameters()) == pickle.dumps(
        fitted_all.learned_parameters()
    )
    # Parity survives the round trip too -- that is what inference actually does.
    np.testing.assert_array_equal(
        _engineered(restored, serving.iloc[[0]]),
        _engineered(fitted_all, serving.iloc[[0]]),
    )


def test_fit_and_transform_are_deterministic(train, serving):
    X, y = train
    first = FeatureSpecTransformer(ALL_SPECS).fit(X, y)
    second = FeatureSpecTransformer(ALL_SPECS).fit(X.copy(), y.copy())
    assert pickle.dumps(first.learned_parameters()) == pickle.dumps(
        second.learned_parameters()
    )
    np.testing.assert_array_equal(_engineered(first, serving), _engineered(second, serving))
    # Repeated transforms of the same frame are byte-identical.
    np.testing.assert_array_equal(_engineered(first, serving), _engineered(first, serving))
    # Column order in the input does not change the result.
    reordered = serving[list(reversed(serving.columns))]
    np.testing.assert_array_equal(_engineered(first, reordered), _engineered(first, serving))


def test_transform_appends_columns_without_touching_the_input(fitted_all, serving):
    snapshot = serving.copy(deep=True)
    out = fitted_all.transform(serving)
    pd.testing.assert_frame_equal(serving, snapshot)
    pd.testing.assert_frame_equal(out[list(serving.columns)], serving)
    assert list(out.columns) == list(serving.columns) + fitted_all.engineered_names_
    assert out.index.equals(serving.index)
    assert len(out) == len(serving)


def test_get_feature_names_out_matches_the_transformed_frame(train, fitted_all):
    X, _ = train
    names = list(fitted_all.get_feature_names_out())
    assert names == list(fitted_all.transform(X).columns)
    assert names == list(X.columns) + fitted_all.engineered_names_
    assert list(fitted_all.get_feature_names_out(list(X.columns))) == names
    with pytest.raises(ValueError, match="not equal to the columns seen during fit"):
        fitted_all.get_feature_names_out(["wrong"])
    with pytest.raises(NotFittedError):
        FeatureSpecTransformer(ALL_SPECS).get_feature_names_out()


def test_sklearn_estimator_contract(train, serving):
    X, y = train
    transformer = FeatureSpecTransformer(ALL_SPECS)
    assert transformer.get_params()["specs"] is ALL_SPECS
    fresh = clone(transformer)
    assert fresh.specs == ALL_SPECS
    assert not hasattr(fresh, "fitted_ops_"), "clone must be unfitted"

    fit_transform = transformer.fit_transform(X, y)
    step_by_step = FeatureSpecTransformer(ALL_SPECS).fit(X, y).transform(X)
    pd.testing.assert_frame_equal(fit_transform, step_by_step, check_exact=True)

    assert transformer.n_features_in_ == X.shape[1]
    assert list(transformer.feature_names_in_) == list(X.columns)
    assert set(transformer.required_columns_) == {
        "amount",
        "qty",
        "city",
        "dept",
        "signup_ts",
    }


def test_specs_may_be_plain_dicts(train, serving):
    X, y = train
    as_dicts = [spec.to_dict() for spec in ALL_SPECS]
    from_dicts = FeatureSpecTransformer(as_dicts).fit(X, y)
    from_objects = FeatureSpecTransformer(ALL_SPECS).fit(X, y)
    assert from_dicts.specs_ == from_objects.specs_
    np.testing.assert_array_equal(
        _engineered(from_dicts, serving), _engineered(from_objects, serving)
    )


def test_empty_spec_list_is_a_no_op(train):
    X, y = train
    for specs in ([], None):
        out = FeatureSpecTransformer(specs).fit(X, y).transform(X)
        pd.testing.assert_frame_equal(out, X)
        assert FeatureSpecTransformer(specs).fit(X, y).engineered_names_ == []


def test_feature_spec_round_trips_through_json():
    for spec in ALL_SPECS:
        as_dict = spec.to_dict()
        assert FeatureSpec.from_dict(as_dict) == spec
        assert FeatureSpec.from_dict(json.loads(json.dumps(as_dict))) == spec
    # to_dict must not hand out the live params dict.
    spec = FeatureSpec("log1p", {"col": "a"})
    spec.to_dict()["params"]["col"] = "hacked"
    assert spec.params == {"col": "a"}
    # Defaults are materialised at fit time, not silently at parse time.
    assert FeatureSpec.from_dict({"op": "quantile_bin", "params": {"col": "a"}}).params == {
        "col": "a"
    }


def test_feature_spec_from_dict_rejects_structural_garbage():
    with pytest.raises(TypeError, match="expects a mapping"):
        FeatureSpec.from_dict(["log1p"])
    with pytest.raises(ValueError, match="missing 'op' key"):
        FeatureSpec.from_dict({"params": {"col": "a"}})
    with pytest.raises(ValueError, match="non-empty string"):
        FeatureSpec.from_dict({"op": None})
    with pytest.raises(TypeError, match="'params' must be a mapping"):
        FeatureSpec.from_dict({"op": "log1p", "params": 3})
    # An unknown op is NOT a structural problem: validate_specs reports it.
    assert FeatureSpec.from_dict({"op": "wat", "params": {}}).op == "wat"


def test_learned_parameters_reports_the_fitted_state(fitted_all):
    report = fitted_all.learned_parameters()
    assert len(report) == len(ALL_SPECS)
    by_op = {entry["op"]: entry for entry in report}
    assert by_op["target_encode"]["stateful"] is True
    assert by_op["log1p"]["stateful"] is False
    assert by_op["log1p"]["learned"] == {}
    assert set(by_op["winsorize"]["learned"]) == {"lower_", "upper_"}
    assert set(by_op["quantile_bin"]["learned"]) == {"edges_", "n_bins_"}
    assert set(by_op["frequency_encode"]["learned"]) == {"frequencies_", "n_fit_rows_"}
    assert set(by_op["target_encode"]["learned"]) == {"encodings_", "global_mean_"}
    assert set(by_op["group_aggregate"]["learned"]) == {"aggregates_", "global_value_"}
    assert set(by_op["datetime_parts"]["learned"]) == {"strategy_", "format_"}
