"""Unit tests for Pareto frontier analysis."""

from forge.training.hpo.pareto import ParetoAnalyzer


def test_pareto_frontier():
    results = [
        {"model_name": "fast_low", "cv_score": 0.7, "inference_latency_ms": 1.0},
        {"model_name": "slow_high", "cv_score": 0.9, "inference_latency_ms": 100.0},
        {"model_name": "mid", "cv_score": 0.8, "inference_latency_ms": 10.0},
        {"model_name": "dominated", "cv_score": 0.75, "inference_latency_ms": 50.0},
    ]
    points = ParetoAnalyzer().compute(results)
    optimal = [p.model_name for p in points if p.is_pareto_optimal]
    assert "fast_low" in optimal
    assert "slow_high" in optimal
    assert "dominated" not in optimal
