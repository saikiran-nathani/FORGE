"""Unit tests for the natural-language task planning module.

These assert real behaviour, not smoke: exact capacity counts, exact cost
ratios, exact thresholds, exact metric names, provenance on every field, and
that the LLM path fails over to rules instead of silently degrading.
"""

from __future__ import annotations

import json

import pytest

from forge.planning.task_spec import (
    MODERATE_ASYMMETRY,
    REALTIME_BUDGET_MS,
    STRONG_ASYMMETRY,
    Field,
    TaskSpec,
    cost_sensitive_threshold,
    parse_task_description,
    recommend_metric,
)

CLASSIFICATION = {
    "task_type": "classification",
    "is_binary": True,
    "imbalance_ratio": 0.5,
    "n_rows": 10_000,
}
REGRESSION = {
    "task_type": "regression",
    "is_binary": False,
    "imbalance_ratio": None,
    "n_rows": 10_000,
}


class FakeLLM:
    """Stand-in for forge.llm.client.LLMClient (same `available` / `complete_json`)."""

    def __init__(self, payload=None, available=True, raises=None):
        self.payload = payload
        self.available = available
        self.raises = raises
        self.calls = 0

    def complete_json(self, system, user):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.payload


# ==========================================================================
# Capacity
# ==========================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("we can only call 100 customers a month", 100),
        ("score the top 50 leads for the sales team", 50),
        ("the fraud team can review 20 cases/day", 20),
        ("analysts review 20 cases per day", 20),
        ("we have capacity for 200 inspections", 200),
        ("give me a shortlist of 25 candidates", 25),
        ("no more than 15 accounts can be investigated", 15),
    ],
)
def test_capacity_extracted(text, expected):
    spec = parse_task_description(text)
    assert spec.capacity_k.value == expected
    assert spec.capacity_k.source == "stated"
    assert str(expected) in spec.capacity_k.rationale


def test_capacity_absent_is_explicit_default():
    spec = parse_task_description("predict which customers will churn")
    assert spec.capacity_k.value is None
    assert spec.capacity_k.source == "default"
    assert "capacity" in spec.capacity_k.rationale.lower()


@pytest.mark.parametrize(
    "text",
    [
        "target the top 10% of leads",  # a fraction, not an absolute k
        "forecast demand for the next 3 months",  # a horizon, not a capacity
        "we can only spend $5000 on this",  # money, not a count
        "must respond in under 50ms",  # latency, not a count
        "missing a fraud case costs 10x a false alarm",  # a multiplier, not a count
    ],
)
def test_numbers_that_are_not_capacity(text):
    assert parse_task_description(text).capacity_k.value is None


def test_capacity_implies_ranking_objective():
    spec = parse_task_description("we can only call 100 customers a month")
    assert spec.objective.value == "rank"
    assert spec.objective.source == "stated"
    assert "100" in spec.objective.rationale


# ==========================================================================
# Cost asymmetry
# ==========================================================================


def test_explicit_multiplier_favours_recall():
    spec = parse_task_description("missing a fraud case costs 10x a false alarm")
    assert spec.cost_ratio.value == {"fp": 1.0, "fn": 10.0}
    assert spec.cost_ratio.source == "stated"
    # The magnitude really was stated, so it must not be labelled an assumption.
    assert "assumed" not in spec.cost_ratio.rationale.lower()


def test_multiplier_direction_reverses_when_false_positive_named_first():
    spec = parse_task_description(
        "a false positive costs 5x more than a missed detection"
    )
    assert spec.cost_ratio.value == {"fp": 5.0, "fn": 1.0}


def test_false_positives_expensive_sets_direction_with_assumed_magnitude():
    spec = parse_task_description("false positives are expensive for us")
    assert spec.cost_ratio.value == {"fp": MODERATE_ASYMMETRY, "fn": 1.0}
    assert spec.cost_ratio.source == "stated"
    # Direction stated, magnitude assumed - the rationale must admit that.
    assert "assumed" in spec.cost_ratio.rationale.lower()


def test_cannot_afford_to_miss_is_strong_recall_asymmetry():
    spec = parse_task_description("we can't afford to miss any")
    assert spec.cost_ratio.value == {"fp": 1.0, "fn": STRONG_ASYMMETRY}
    assert spec.cost_ratio.source == "stated"
    assert "assumed" in spec.cost_ratio.rationale.lower()


def test_curly_apostrophe_is_normalised():
    spec = parse_task_description("we can’t afford to miss any fraud")
    assert spec.cost_ratio.value == {"fp": 1.0, "fn": STRONG_ASYMMETRY}


def test_bare_mention_of_false_positives_does_not_claim_asymmetry():
    spec = parse_task_description("we don't really care about false positives here")
    assert spec.cost_ratio.value == {"fp": 1.0, "fn": 1.0}
    assert spec.cost_ratio.source == "default"


def test_symmetric_cost_is_the_default():
    spec = parse_task_description("classify support tickets by topic")
    assert spec.cost_ratio.value == {"fp": 1.0, "fn": 1.0}
    assert spec.cost_ratio.source == "default"


# ==========================================================================
# Latency
# ==========================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("must respond in under 50ms", 50.0),
        ("the endpoint must reply within 250 milliseconds", 250.0),
        ("scoring latency must be under 2 seconds", 2000.0),
    ],
)
def test_latency_extracted(text, expected):
    spec = parse_task_description(text)
    assert spec.latency_budget_ms.value == pytest.approx(expected)
    assert spec.latency_budget_ms.source == "stated"


def test_real_time_uses_documented_placeholder():
    spec = parse_task_description("we need real-time scoring at checkout")
    assert spec.latency_budget_ms.value == REALTIME_BUDGET_MS
    assert spec.latency_budget_ms.source == "stated"
    assert "assumed" in spec.latency_budget_ms.rationale.lower()


def test_training_time_budget_is_not_a_serving_budget():
    spec = parse_task_description("retraining should finish in under 30 minutes")
    assert spec.latency_budget_ms.value is None
    assert spec.latency_budget_ms.source == "default"


def test_latency_absent_is_explicit_default():
    spec = parse_task_description("predict churn")
    assert spec.latency_budget_ms.value is None
    assert spec.latency_budget_ms.source == "default"


# ==========================================================================
# Interpretability
# ==========================================================================


@pytest.mark.parametrize(
    "text",
    [
        "we need to explain decisions to regulators",
        "the model must be interpretable",
        "give us an explainable model for the credit committee",
        "it can't be a black box",
        "we need reason codes for adverse action notices",
        "compliance requires an audit trail",
    ],
)
def test_interpretability_required(text):
    spec = parse_task_description(text)
    assert spec.interpretability_required.value is True
    assert spec.interpretability_required.source == "stated"


@pytest.mark.parametrize(
    "text",
    [
        "we can only audit 20 accounts a day",  # "audit" as work, not explainability
        "predict which customers will churn",
        "maximise revenue",
    ],
)
def test_interpretability_not_required(text):
    spec = parse_task_description(text)
    assert spec.interpretability_required.value is False
    assert spec.interpretability_required.source == "default"


# ==========================================================================
# Forecast
# ==========================================================================


@pytest.mark.parametrize(
    "text",
    [
        "forecast next quarter",
        "predict future sales",
        "this is a time series problem",
        "estimate demand 3 months ahead",
        "project revenue for the upcoming quarter",
    ],
)
def test_is_forecast(text):
    spec = parse_task_description(text)
    assert spec.is_forecast.value is True
    assert spec.is_forecast.source == "stated"
    assert "split" in spec.is_forecast.rationale.lower()


@pytest.mark.parametrize(
    "text",
    [
        "predict which customers will churn",
        "classify tickets by topic",
        "detect fraudulent transactions",
    ],
)
def test_not_forecast(text):
    spec = parse_task_description(text)
    assert spec.is_forecast.value is False
    assert spec.is_forecast.source == "default"


def test_forecast_implies_estimate_objective():
    spec = parse_task_description("forecast next quarter's revenue")
    assert spec.objective.value == "estimate"


# ==========================================================================
# Protected attributes
# ==========================================================================


def test_protected_attributes_in_text_order():
    spec = parse_task_description("don't discriminate by gender or age")
    assert spec.protected_attributes.value == ["gender", "age"]
    assert spec.protected_attributes.source == "stated"
    assert "discriminate" in spec.protected_attributes.rationale


def test_protected_attributes_multiple_canonical_names():
    spec = parse_task_description(
        "the model must be fair and must not use race, gender or zip code"
    )
    assert spec.protected_attributes.value == ["race", "gender", "zip_code"]


def test_age_does_not_match_inside_mileage():
    """`"age" in "mileage"` is the exact bug this project was burned by."""
    spec = parse_task_description("estimate the mileage of used cars from usage data")
    assert spec.protected_attributes.value == []
    assert spec.protected_attributes.source == "default"


def test_sex_does_not_double_count_sexual_orientation():
    spec = parse_task_description("do not discriminate by sexual orientation")
    assert spec.protected_attributes.value == ["sexual_orientation"]


def test_underscored_column_style_still_matches_age():
    spec = parse_task_description("don't let age_group drive the decision")
    assert "age" in spec.protected_attributes.value


def test_protected_mention_without_fairness_context_is_flagged_for_confirmation():
    spec = parse_task_description("predict income from age and education")
    assert spec.protected_attributes.value == ["age"]
    assert "confirm" in spec.protected_attributes.rationale.lower()


# ==========================================================================
# Objective
# ==========================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("rank leads by likelihood to convert", "rank"),
        ("prioritise which accounts to call", "rank"),
        ("classify support tickets by topic", "classify"),
        ("detect fraudulent transactions", "classify"),
        ("estimate how much a customer will spend", "estimate"),
        ("forecast next quarter", "estimate"),
    ],
)
def test_objective_detection(text, expected):
    spec = parse_task_description(text)
    assert spec.objective.value == expected
    assert spec.objective.source == "stated"


# ==========================================================================
# Empty / weird input
# ==========================================================================


def test_empty_string_gives_all_defaults_and_does_not_crash():
    spec = parse_task_description("")
    assert isinstance(spec, TaskSpec)
    assert spec.parser == "rules"
    assert spec.raw_text == ""

    for name in TaskSpec.FIELD_NAMES:
        field = getattr(spec, name)
        assert isinstance(field, Field), name
        assert field.source == "default", name
        assert field.rationale.strip(), name

    assert spec.objective.value == "classify"
    assert spec.cost_ratio.value == {"fp": 1.0, "fn": 1.0}
    assert spec.capacity_k.value is None
    assert spec.latency_budget_ms.value is None
    assert spec.interpretability_required.value is False
    assert spec.is_forecast.value is False
    assert spec.protected_attributes.value == []


def test_empty_text_rationales_say_nothing_was_stated():
    spec = parse_task_description("")
    for name in TaskSpec.FIELD_NAMES:
        assert "no task description was given" in getattr(spec, name).rationale.lower()


@pytest.mark.parametrize(
    "text",
    ["", "   ", "\n\t ", "?!?!", "%%%", "0", "-5", None, 123, 4.5, [], {}, "x" * 5000, "🙂🙂"],
)
def test_weird_input_never_raises(text):
    spec = parse_task_description(text)
    assert isinstance(spec, TaskSpec)
    assert spec.parser == "rules"
    for name in TaskSpec.FIELD_NAMES:
        field = getattr(spec, name)
        assert field.source in ("stated", "default"), name
        assert field.rationale.strip(), name


def test_every_field_always_carries_provenance_and_rationale():
    texts = [
        "",
        "we can only call 100 customers a month",
        "missing a fraud case costs 10x a false alarm; explain to regulators",
        "real-time fraud scoring, don't discriminate by gender or age",
        "forecast next quarter revenue, robust to outliers",
    ]
    for text in texts:
        spec = parse_task_description(text)
        for name in TaskSpec.FIELD_NAMES:
            field = getattr(spec, name)
            assert field.source in ("stated", "default"), (text, name)
            assert len(field.rationale) > 20, (text, name)


# ==========================================================================
# Determinism + serialisation
# ==========================================================================


@pytest.mark.parametrize(
    "text",
    [
        "",
        "we can only review 20 cases/day and can't afford to miss any fraud",
        "real-time interpretable scoring, don't discriminate by age or gender",
        "forecast next quarter sales within 100ms",
    ],
)
def test_deterministic(text):
    first = parse_task_description(text).to_dict()
    second = parse_task_description(text).to_dict()
    assert first == second


def test_to_dict_is_json_serialisable_and_complete():
    spec = parse_task_description(
        "top 50 leads, real-time, explain to regulators, don't use gender"
    )
    payload = spec.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["parser"] == "rules"
    assert payload["raw_text"]
    for name in TaskSpec.FIELD_NAMES:
        assert set(payload[name]) == {"value", "source", "rationale"}
    assert payload["capacity_k"]["value"] == 50


def test_combined_description_extracts_every_dimension():
    spec = parse_task_description(
        "We can only call 100 leads a month, missing a real buyer costs 5x a wasted "
        "call, we must respond in under 40ms, we need to explain decisions to "
        "regulators, and we must not discriminate by gender or age."
    )
    assert spec.capacity_k.value == 100
    assert spec.cost_ratio.value == {"fp": 1.0, "fn": 5.0}
    assert spec.latency_budget_ms.value == pytest.approx(40.0)
    assert spec.interpretability_required.value is True
    assert spec.protected_attributes.value == ["gender", "age"]
    assert spec.objective.value == "rank"


# ==========================================================================
# recommend_metric
# ==========================================================================


def test_capacity_forces_precision_at_k():
    spec = parse_task_description("we can only call 100 customers a month")
    metric, why = recommend_metric(spec, CLASSIFICATION)
    assert metric == "precision_at_k"
    assert "100" in why
    assert "10000" in why or "10,000" in why


def test_precision_at_k_wins_even_for_regression_targets():
    spec = parse_task_description("rank the top 100 properties by predicted value")
    metric, why = recommend_metric(spec, REGRESSION)
    assert metric == "precision_at_k"
    assert "numeric" in why


def test_precision_at_k_flags_k_larger_than_dataset():
    spec = parse_task_description("we can only call 500 customers a month")
    _, why = recommend_metric(spec, {**CLASSIFICATION, "n_rows": 100})
    assert "degenerates" in why


def test_precision_at_k_admits_missing_row_count():
    spec = parse_task_description("top 50 leads")
    _, why = recommend_metric(spec, {**CLASSIFICATION, "n_rows": None})
    assert "not supplied" in why


def test_cost_asymmetry_gives_fbeta_weighted_to_recall():
    spec = parse_task_description("missing a fraud case costs 10x a false alarm")
    metric, why = recommend_metric(spec, CLASSIFICATION)
    assert metric == "fbeta"
    assert "recall" in why
    assert "fn=10" in why


def test_cost_asymmetry_gives_fbeta_weighted_to_precision():
    spec = parse_task_description("a false positive costs 5x more than a missed case")
    metric, why = recommend_metric(spec, CLASSIFICATION)
    assert metric == "fbeta"
    assert "precision" in why


def test_severe_imbalance_gives_pr_auc():
    metric, why = recommend_metric(
        parse_task_description("detect fraud"),
        {**CLASSIFICATION, "imbalance_ratio": 0.01},
    )
    assert metric == "pr_auc"
    assert "0.01" in why


def test_moderate_imbalance_gives_roc_auc():
    metric, why = recommend_metric(
        parse_task_description("predict churn"),
        {**CLASSIFICATION, "imbalance_ratio": 0.1},
    )
    assert metric == "roc_auc"
    assert "imbalanc" in why


def test_ratio_given_as_majority_over_minority_is_inverted_not_misread():
    metric, why = recommend_metric(
        parse_task_description("predict churn"),
        {**CLASSIFICATION, "imbalance_ratio": 9.0},
    )
    assert metric == "roc_auc"
    assert "9:1" in why


def test_mild_imbalance_gives_f1():
    metric, _ = recommend_metric(
        parse_task_description("predict churn"),
        {**CLASSIFICATION, "imbalance_ratio": 0.5},
    )
    assert metric == "f1"


def test_balanced_binary_gives_accuracy():
    metric, why = recommend_metric(
        parse_task_description("predict churn"),
        {**CLASSIFICATION, "imbalance_ratio": 0.95},
    )
    assert metric == "accuracy"
    assert "balanced" in why


def test_unknown_imbalance_is_admitted_not_invented():
    metric, why = recommend_metric(
        parse_task_description("predict churn"),
        {**CLASSIFICATION, "imbalance_ratio": None},
    )
    assert metric == "roc_auc"
    assert "no imbalance ratio was supplied" in why


def test_multiclass_gives_f1_macro():
    metric, why = recommend_metric(
        parse_task_description("classify tickets by topic"),
        {"task_type": "classification", "is_binary": False, "imbalance_ratio": None, "n_rows": 500},
    )
    assert metric == "f1_macro"
    assert "multiclass" in why


def test_regression_gives_rmse():
    metric, why = recommend_metric(parse_task_description("predict house prices"), REGRESSION)
    assert metric == "rmse"
    assert "regression" in why


def test_regression_with_outlier_language_gives_mae():
    spec = parse_task_description(
        "predict delivery time; a few extreme outliers shouldn't dominate the model"
    )
    metric, why = recommend_metric(spec, REGRESSION)
    assert metric == "mae"
    assert "outlier" in why


def test_forecast_regression_rationale_mentions_time_split():
    spec = parse_task_description("forecast next quarter revenue")
    metric, why = recommend_metric(spec, REGRESSION)
    assert metric == "rmse"
    assert "time-ordered" in why


def test_recommend_metric_tolerates_missing_data_facts():
    metric, why = recommend_metric(parse_task_description("predict churn"), {})
    assert metric == "f1_macro"
    assert "task_type" in why


def test_recommend_metric_rationale_always_names_evidence():
    cases = [
        (parse_task_description("top 50 leads"), CLASSIFICATION),
        (parse_task_description("we can't afford to miss any fraud"), CLASSIFICATION),
        (parse_task_description("predict churn"), CLASSIFICATION),
        (parse_task_description("predict price"), REGRESSION),
        (parse_task_description(""), CLASSIFICATION),
    ]
    for spec, facts in cases:
        metric, why = recommend_metric(spec, facts)
        assert metric
        assert why.startswith(metric)
        assert len(why) > 60


# ==========================================================================
# cost_sensitive_threshold
# ==========================================================================


def test_threshold_asymmetric_recall_heavy():
    assert cost_sensitive_threshold({"fp": 1, "fn": 9}) == pytest.approx(0.1)


def test_threshold_asymmetric_precision_heavy():
    assert cost_sensitive_threshold({"fp": 9, "fn": 1}) == pytest.approx(0.9)


@pytest.mark.parametrize("costs", [{"fp": 1, "fn": 1}, {"fp": 2.5, "fn": 2.5}, {"fp": 7, "fn": 7}])
def test_threshold_symmetric_is_half(costs):
    assert cost_sensitive_threshold(costs) == 0.5


@pytest.mark.parametrize(
    "costs",
    [{}, {"fp": 1}, {"fn": 1}, {"fp": 0, "fn": 0}, {"fp": -1, "fn": 2}, {"fp": "a", "fn": 1},
     {"fp": None, "fn": 1}, {"fp": True, "fn": 1}, None, "nope", 5],
)
def test_threshold_falls_back_to_neutral_on_unusable_costs(costs):
    assert cost_sensitive_threshold(costs) == 0.5


def test_threshold_matches_parsed_spec():
    spec = parse_task_description("missing a fraud case costs 10x a false alarm")
    assert cost_sensitive_threshold(spec.cost_ratio.value) == pytest.approx(1 / 11)


def test_threshold_of_default_spec_is_half():
    spec = parse_task_description("")
    assert cost_sensitive_threshold(spec.cost_ratio.value) == 0.5


# ==========================================================================
# LLM path: refine when verifiable, fail over to rules otherwise
# ==========================================================================

_TEXT = "Rank leads for the sales team; we have limited bandwidth this quarter."


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        "not a dict",
        [],
        42,
        {"garbage": True},
        {"objective": "banana"},  # not in the enum
        {"objective": 123},  # wrong type
        {"capacity_k": "fifty"},  # wrong type
        {"capacity_k": -5},  # invalid value
        {"cost_ratio": "very high"},  # wrong type
        {"cost_ratio": {"fp": 0, "fn": 0}},  # invalid values
        {"objective": "rank"},  # valid value, but no evidence at all
        {"objective": "rank", "evidence": {}},  # empty evidence
        {"objective": "rank", "evidence": {"objective": "  "}},  # blank quote
        # Valid value + confident-looking evidence that is NOT in the text:
        {"capacity_k": 50, "evidence": {"capacity_k": "we can only call 50 people"}},
    ],
)
def test_garbage_llm_output_falls_back_to_rules(payload):
    llm = FakeLLM(payload=payload)
    spec = parse_task_description(_TEXT, llm=llm)
    assert llm.calls == 1
    assert spec.parser == "rules"
    assert spec.to_dict() == parse_task_description(_TEXT).to_dict()


@pytest.mark.parametrize("error", [RuntimeError("boom"), ValueError("bad json"), TimeoutError()])
def test_raising_llm_falls_back_to_rules(error):
    llm = FakeLLM(raises=error)
    spec = parse_task_description(_TEXT, llm=llm)
    assert spec.parser == "rules"
    assert spec.to_dict() == parse_task_description(_TEXT).to_dict()


def test_unavailable_llm_is_never_called():
    llm = FakeLLM(payload={"capacity_k": 50}, available=False)
    spec = parse_task_description(_TEXT, llm=llm)
    assert llm.calls == 0
    assert spec.parser == "rules"


def test_llm_without_available_attribute_is_ignored():
    class Bare:
        def complete_json(self, system, user):  # pragma: no cover - must not run
            raise AssertionError("should not be called")

    assert parse_task_description(_TEXT, llm=Bare()).parser == "rules"


def test_no_llm_argument_works_with_no_api_key():
    spec = parse_task_description("we can only call 100 customers a month")
    assert spec.parser == "rules"
    assert spec.capacity_k.value == 100


def test_llm_value_backed_by_a_verbatim_quote_is_accepted():
    llm = FakeLLM(
        payload={
            "capacity_k": 25,
            "evidence": {"capacity_k": "limited bandwidth"},
        }
    )
    spec = parse_task_description(_TEXT, llm=llm)
    assert spec.parser == "llm"
    assert spec.capacity_k.value == 25
    assert spec.capacity_k.source == "stated"
    assert "limited bandwidth" in spec.capacity_k.rationale
    # Untouched fields keep their rule-derived values.
    assert spec.cost_ratio.value == {"fp": 1.0, "fn": 1.0}
    assert spec.cost_ratio.source == "default"


def test_llm_accepts_only_the_evidenced_fields():
    llm = FakeLLM(
        payload={
            "capacity_k": 25,
            "interpretability_required": True,  # no evidence -> discarded
            "evidence": {"capacity_k": "limited bandwidth"},
        }
    )
    spec = parse_task_description(_TEXT, llm=llm)
    assert spec.parser == "llm"
    assert spec.capacity_k.value == 25
    assert spec.interpretability_required.value is False
    assert spec.interpretability_required.source == "default"


def test_llm_path_is_skipped_for_empty_text():
    llm = FakeLLM(payload={"capacity_k": 25, "evidence": {"capacity_k": "anything"}})
    spec = parse_task_description("", llm=llm)
    assert llm.calls == 0
    assert spec.parser == "rules"
    assert spec.capacity_k.value is None


def test_llm_refined_spec_is_deterministic_and_serialisable():
    def build():
        llm = FakeLLM(payload={"capacity_k": 25, "evidence": {"capacity_k": "limited bandwidth"}})
        return parse_task_description(_TEXT, llm=llm).to_dict()

    first, second = build(), build()
    assert first == second
    assert json.loads(json.dumps(first)) == first
