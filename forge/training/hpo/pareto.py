"""Multi-objective Pareto frontier for accuracy vs latency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParetoPoint:
    model_name: str
    cv_score: float
    latency_ms: float
    is_pareto_optimal: bool


class ParetoAnalyzer:
    """Identifies Pareto-optimal models on accuracy vs latency."""

    def compute(
        self,
        model_results: list[dict[str, Any]],
        maximize_score: bool = True,
    ) -> list[ParetoPoint]:
        points = [
            ParetoPoint(
                model_name=r["model_name"],
                cv_score=r["cv_score"],
                latency_ms=r.get("inference_latency_ms", 0.0),
                is_pareto_optimal=False,
            )
            for r in model_results
        ]

        for i, p in enumerate(points):
            dominated = False
            for j, q in enumerate(points):
                if i == j:
                    continue
                better_score = q.cv_score >= p.cv_score if maximize_score else q.cv_score <= p.cv_score
                better_latency = q.latency_ms <= p.latency_ms
                strictly_better = (better_score and q.latency_ms < p.latency_ms) or (
                    q.cv_score > p.cv_score if maximize_score else q.cv_score < p.cv_score
                ) and better_latency
                if strictly_better and better_score and better_latency:
                    dominated = True
                    break
            p.is_pareto_optimal = not dominated

        return points

    def to_dict(self, points: list[ParetoPoint]) -> list[dict[str, Any]]:
        return [
            {
                "model_name": p.model_name,
                "cv_score": p.cv_score,
                "latency_ms": p.latency_ms,
                "is_pareto_optimal": p.is_pareto_optimal,
            }
            for p in points
        ]
