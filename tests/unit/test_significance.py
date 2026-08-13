"""Known-answer tests for forge.evaluation.significance.

Every assertion here is either hand-computed (McNemar p-values, Holm reject
patterns) or a structural property that must hold by construction (a zero
difference must be a tie, a 0.4 gap must not be). Nothing is a smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge.evaluation.significance import (
    holm_bonferroni,
    indistinguishable_set,
    mcnemar_test,
    paired_bootstrap_diff_ci,
)

# --------------------------------------------------------------------------- #
# shared fixtures: 10 aligned CV folds with a shared per-fold difficulty term
# --------------------------------------------------------------------------- #
# Per-fold accuracy of the underlying data split (shared by every model).
FOLDS = np.array([0.90, 0.88, 0.91, 0.89, 0.92, 0.90, 0.88, 0.91, 0.89, 0.90])
# Two small per-model jitter patterns. mean(JIT_A) = 0.0005, mean(JIT_B) = 0.0003,
# so the two differ by 0.0002 on average with a paired SD of ~0.0045 -> a tie.
JIT_A = np.array([0.003, -0.002, 0.001, 0.004, -0.003, 0.002, -0.001, 0.000, 0.003, -0.002])
JIT_B = np.array([-0.002, 0.003, 0.004, -0.001, 0.002, -0.003, 0.001, 0.002, -0.004, 0.001])


def _unpaired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 7
) -> tuple[float, float]:
    """Reference implementation that resamples a and b INDEPENDENTLY.

    Exists only so the tests can prove the paired version is tighter.
    """
    rng = np.random.default_rng(seed)
    n = a.size
    idx_a = rng.integers(0, n, size=(n_boot, n))
    idx_b = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    lo, hi = np.percentile(diffs, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# 1. identical vectors -> exact zero difference, zero-width CI, reported tied
# --------------------------------------------------------------------------- #
def test_identical_vectors_give_zero_diff_and_ci_containing_zero():
    scores = FOLDS.copy()
    diff, lo, hi = paired_bootstrap_diff_ci(scores, scores.copy(), n_boot=500, seed=1)
    assert diff == 0.0
    # Pairing makes every resampled difference exactly zero, so the interval is
    # the degenerate point {0} -- it still contains 0, hence a tie.
    assert lo == 0.0
    assert hi == 0.0
    assert lo <= 0.0 <= hi


def test_identical_models_are_reported_as_tied():
    result = indistinguishable_set(
        {"m1": FOLDS.copy(), "m2": FOLDS.copy()}, n_boot=500, seed=1
    )
    assert set(result["tied_with_leader"]) == {"m1", "m2"}
    assert result["per_model"]["m1"]["distinguishable_from_leader"] is False
    assert result["per_model"]["m2"]["distinguishable_from_leader"] is False
    assert result["n_units"] == 10


def test_all_identical_constant_scores_are_tied_not_crashing():
    # Zero variance within AND between models: the honest answer is "tied".
    a = np.full(8, 0.75)
    b = np.full(8, 0.75)
    diff, lo, hi = paired_bootstrap_diff_ci(a, b, n_boot=200, seed=3)
    assert (diff, lo, hi) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# 2. clearly separated vectors -> CI excludes 0 -> distinguishable
# --------------------------------------------------------------------------- #
def test_clearly_separated_vectors_exclude_zero():
    a = np.full(10, 0.9)
    b = np.full(10, 0.5)
    diff, lo, hi = paired_bootstrap_diff_ci(a, b, n_boot=1000, seed=5)
    assert diff == pytest.approx(0.4)
    assert not (lo <= 0.0 <= hi)
    # Perfectly consistent gap -> the bootstrap distribution is a point mass.
    assert lo == pytest.approx(0.4)
    assert hi == pytest.approx(0.4)


def test_separated_vectors_with_noise_exclude_zero():
    rng = np.random.default_rng(11)
    a = 0.90 + rng.normal(0.0, 0.01, size=20)
    b = 0.50 + rng.normal(0.0, 0.01, size=20)
    diff, lo, hi = paired_bootstrap_diff_ci(a, b, n_boot=2000, seed=5)
    assert diff > 0.35
    assert lo > 0.0  # entire interval strictly above zero
    assert lo < diff < hi


# --------------------------------------------------------------------------- #
# 3. pairing actually matters: paired CI is much tighter than unpaired
# --------------------------------------------------------------------------- #
def test_paired_ci_is_tighter_than_unpaired_for_correlated_models():
    rng = np.random.default_rng(0)
    n = 40
    # Large shared per-unit difficulty (sd 1.0) dominates the tiny model effect.
    difficulty = rng.normal(0.0, 1.0, size=n)
    a = difficulty + 0.05 + rng.normal(0.0, 0.01, size=n)
    b = difficulty + rng.normal(0.0, 0.01, size=n)
    assert np.corrcoef(a, b)[0, 1] > 0.99  # the construction really is correlated

    diff, p_lo, p_hi = paired_bootstrap_diff_ci(a, b, n_boot=4000, seed=13)
    u_lo, u_hi = _unpaired_bootstrap_ci(a, b, n_boot=4000, seed=13)

    paired_width = p_hi - p_lo
    unpaired_width = u_hi - u_lo
    assert paired_width < unpaired_width
    # Shared difficulty cancels in the paired resample; the gain is an order of
    # magnitude here, not a rounding artefact.
    assert paired_width < unpaired_width / 10.0

    # The practical payoff: paired detects the real +0.05 effect, unpaired cannot.
    assert diff == pytest.approx(0.05, abs=0.01)
    assert p_lo > 0.0, "paired CI should exclude 0"
    assert u_lo <= 0.0 <= u_hi, "unpaired CI should be too wide to exclude 0"


# --------------------------------------------------------------------------- #
# 4. McNemar: hand-computed contingency cases and the exact/chi2 switch
# --------------------------------------------------------------------------- #
def _mcnemar_inputs(n_b: int, n_c: int, n_both_right: int = 5, n_both_wrong: int = 3):
    """Build paired predictions with exactly n_b and n_c discordant pairs."""
    total = n_b + n_c + n_both_right + n_both_wrong
    y = np.ones(total, dtype=int)
    pred_a = np.concatenate(
        [np.ones(n_b), np.zeros(n_c), np.ones(n_both_right), np.zeros(n_both_wrong)]
    ).astype(int)
    pred_b = np.concatenate(
        [np.zeros(n_b), np.ones(n_c), np.ones(n_both_right), np.zeros(n_both_wrong)]
    ).astype(int)
    return y, pred_a, pred_b


def test_mcnemar_exact_hand_computed_p_value():
    # b=10, c=2, n=12 discordant -> two-sided exact binomial:
    # p = 2 * P(X <= 2), X ~ Binom(12, 0.5) = 2 * (1 + 12 + 66) / 4096 = 158/4096.
    y, pred_a, pred_b = _mcnemar_inputs(n_b=10, n_c=2)
    out = mcnemar_test(y, pred_a, pred_b)
    assert out["b"] == 10
    assert out["c"] == 2
    assert out["method"] == "exact_binomial"
    assert out["statistic"] == 2.0  # min(b, c)
    assert out["p_value"] == pytest.approx(158.0 / 4096.0, rel=1e-12)
    assert out["p_value"] < 0.05  # direction: A significantly better than B


def test_mcnemar_b_and_c_are_direction_sensitive():
    y, pred_a, pred_b = _mcnemar_inputs(n_b=10, n_c=2)
    forward = mcnemar_test(y, pred_a, pred_b)
    reverse = mcnemar_test(y, pred_b, pred_a)
    assert (reverse["b"], reverse["c"]) == (forward["c"], forward["b"])
    # Two-sided p-value is symmetric; only b/c reveal who is ahead.
    assert reverse["p_value"] == pytest.approx(forward["p_value"], rel=1e-12)


def test_mcnemar_chi2_hand_computed_statistic():
    # b=30, c=10, n=40 >= 25 -> chi2 with continuity correction:
    # (|30-10| - 1)^2 / 40 = 361/40 = 9.025, p = chi2.sf(9.025, 1) ~= 0.00266.
    y, pred_a, pred_b = _mcnemar_inputs(n_b=30, n_c=10, n_both_right=8, n_both_wrong=0)
    out = mcnemar_test(y, pred_a, pred_b)
    assert (out["b"], out["c"]) == (30, 10)
    assert out["method"] == "chi2_continuity_corrected"
    assert out["statistic"] == pytest.approx(9.025, rel=1e-12)
    assert 0.0026 < out["p_value"] < 0.0027


def test_mcnemar_method_switches_at_25_discordant_pairs():
    # 24 discordant -> exact; 25 discordant -> chi2. Boundary is b + c < 25.
    y24, a24, b24 = _mcnemar_inputs(n_b=12, n_c=12)
    out24 = mcnemar_test(y24, a24, b24)
    assert out24["b"] + out24["c"] == 24
    assert out24["method"] == "exact_binomial"
    # 2 * P(X <= 12), X ~ Binom(24, 0.5) exceeds 1 and must be capped.
    assert out24["p_value"] == 1.0

    y25, a25, b25 = _mcnemar_inputs(n_b=13, n_c=12)
    out25 = mcnemar_test(y25, a25, b25)
    assert out25["b"] + out25["c"] == 25
    assert out25["method"] == "chi2_continuity_corrected"
    # (|13-12| - 1)^2 / 25 = 0 -> p = 1.0 exactly.
    assert out25["statistic"] == 0.0
    assert out25["p_value"] == pytest.approx(1.0)


def test_mcnemar_total_agreement_reports_no_evidence():
    y = np.array([0, 1, 1, 0, 1])
    pred = np.array([0, 1, 0, 0, 1])
    out = mcnemar_test(y, pred, pred.copy())
    assert (out["b"], out["c"]) == (0, 0)
    assert out["method"] == "exact_binomial"
    assert out["statistic"] == 0.0
    assert out["p_value"] == 1.0


def test_mcnemar_supports_string_multiclass_labels():
    y = np.array(["cat", "dog", "fox", "cat", "dog"])
    pred_a = np.array(["cat", "dog", "fox", "dog", "dog"])
    pred_b = np.array(["dog", "dog", "cat", "cat", "dog"])
    out = mcnemar_test(y, pred_a, pred_b)
    # sample 0: a right / b wrong -> b += 1 ; sample 2: a right / b wrong -> b += 1
    # sample 3: a wrong / b right -> c += 1
    assert (out["b"], out["c"]) == (2, 1)


def test_mcnemar_rejects_misaligned_lengths():
    with pytest.raises(ValueError, match="same length"):
        mcnemar_test(np.array([1, 0, 1]), np.array([1, 0]), np.array([1, 0, 1]))


# --------------------------------------------------------------------------- #
# 5. Holm-Bonferroni known-answer patterns
# --------------------------------------------------------------------------- #
def test_holm_known_reject_pattern_and_conservativeness():
    p = [0.001, 0.008, 0.039, 0.041, 0.042]
    # m=5: 0.001 <= 0.05/5=0.010 reject; 0.008 <= 0.05/4=0.0125 reject;
    #      0.039 <= 0.05/3=0.01667 FAILS -> step-down stops.
    assert holm_bonferroni(p, alpha=0.05) == [True, True, False, False, False]
    # Uncorrected testing would reject all five -> Holm is strictly conservative.
    uncorrected = [pv <= 0.05 for pv in p]
    assert uncorrected == [True] * 5
    assert sum(holm_bonferroni(p, alpha=0.05)) < sum(uncorrected)


def test_holm_preserves_input_order():
    p = [0.041, 0.001, 0.042, 0.008, 0.039]
    assert holm_bonferroni(p, alpha=0.05) == [False, True, False, True, False]


def test_holm_first_threshold_is_bonferroni():
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    # 0.01 <= 0.05/5 = 0.01 (boundary rejects); 0.02 > 0.05/4 = 0.0125 -> stop.
    assert holm_bonferroni(p, alpha=0.05) == [True, False, False, False, False]


def test_holm_edge_cases():
    assert holm_bonferroni([]) == []
    assert holm_bonferroni([0.04], alpha=0.05) == [True]  # m=1 -> plain alpha
    assert holm_bonferroni([0.06], alpha=0.05) == [False]
    assert holm_bonferroni([0.6, 0.7, 0.8]) == [False, False, False]
    # Ties must get the same verdict as each other.
    assert holm_bonferroni([0.01, 0.01], alpha=0.05) == [True, True]


def test_holm_rejects_invalid_p_values():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_bonferroni([0.1, 1.5])
    with pytest.raises(ValueError, match="non-finite"):
        holm_bonferroni([0.1, float("nan")])
    with pytest.raises(ValueError, match="alpha"):
        holm_bonferroni([0.1], alpha=0.0)


# --------------------------------------------------------------------------- #
# 6. determinism
# --------------------------------------------------------------------------- #
def test_same_seed_gives_identical_intervals():
    rng = np.random.default_rng(99)
    a = rng.normal(0.8, 0.05, size=25)
    b = rng.normal(0.79, 0.05, size=25)
    first = paired_bootstrap_diff_ci(a, b, n_boot=1500, seed=42)
    second = paired_bootstrap_diff_ci(a, b, n_boot=1500, seed=42)
    assert first == second  # exact float equality, not approx


def test_different_seed_changes_the_interval_but_not_the_observed_diff():
    rng = np.random.default_rng(99)
    a = rng.normal(0.8, 0.05, size=25)
    b = rng.normal(0.79, 0.05, size=25)
    d1, lo1, hi1 = paired_bootstrap_diff_ci(a, b, n_boot=1500, seed=42)
    d2, lo2, hi2 = paired_bootstrap_diff_ci(a, b, n_boot=1500, seed=1234)
    assert d1 == d2  # observed difference is data, not resampling
    assert (lo1, hi1) != (lo2, hi2)


def test_indistinguishable_set_is_deterministic():
    scores = {
        "a": FOLDS + JIT_A,
        "b": FOLDS + JIT_B,
        "c": FOLDS - 0.15 + JIT_B,
    }
    first = indistinguishable_set(scores, n_boot=1000, seed=7)
    second = indistinguishable_set(scores, n_boot=1000, seed=7)
    assert first == second


# --------------------------------------------------------------------------- #
# 7. indistinguishable_set on a 4-model leaderboard, both metric directions
# --------------------------------------------------------------------------- #
def test_four_model_leaderboard_higher_is_better():
    scores = {
        "lgbm": FOLDS + JIT_A,          # mean 0.8985  <- leader
        "xgboost": FOLDS + JIT_B,       # mean 0.8983  <- tied (diff 0.0002 +- 0.0028)
        "random_forest": FOLDS - 0.15 + JIT_B,
        "logistic": FOLDS - 0.30 + JIT_B,
    }
    result = indistinguishable_set(scores, higher_is_better=True, n_boot=3000, seed=42)

    assert result["leader"] == "lgbm"
    assert sorted(result["tied_with_leader"]) == ["lgbm", "xgboost"]
    assert result["n_units"] == 10
    assert result["alpha"] == 0.05
    assert result["higher_is_better"] is True

    per = result["per_model"]
    assert per["lgbm"]["mean"] == pytest.approx(0.8985)
    assert per["xgboost"]["mean"] == pytest.approx(0.8983)
    assert per["lgbm"]["diff_vs_leader"] == 0.0
    # diff is mean(model) - mean(leader), so the trailing model is negative.
    assert per["xgboost"]["diff_vs_leader"] == pytest.approx(-0.0002, abs=1e-12)
    assert per["xgboost"]["ci_lo"] < 0.0 < per["xgboost"]["ci_hi"]
    assert per["xgboost"]["distinguishable_from_leader"] is False

    for loser in ("random_forest", "logistic"):
        assert per[loser]["distinguishable_from_leader"] is True
        assert per[loser]["ci_hi"] < 0.0, "a worse model's raw diff CI sits below 0"
        assert loser not in result["tied_with_leader"]

    # Raw-unit convention: for a higher-is-better metric, non-leaders are negative.
    assert per["random_forest"]["diff_vs_leader"] == pytest.approx(-0.1502, abs=1e-12)
    assert per["logistic"]["diff_vs_leader"] == pytest.approx(-0.3002, abs=1e-12)


def test_four_model_leaderboard_lower_is_better_rmse():
    errors = np.array([0.30, 0.32, 0.29, 0.31, 0.28, 0.30, 0.32, 0.29, 0.31, 0.30])
    scores = {
        "ridge": errors + JIT_A,        # mean 0.3025
        "lgbm": errors + JIT_B,         # mean 0.3023  <- leader (lowest RMSE)
        "tree": errors + 0.15 + JIT_B,
        "mean_baseline": errors + 0.30 + JIT_B,
    }
    result = indistinguishable_set(scores, higher_is_better=False, n_boot=3000, seed=42)

    assert result["leader"] == "lgbm"
    assert sorted(result["tied_with_leader"]) == ["lgbm", "ridge"]
    assert result["higher_is_better"] is False

    per = result["per_model"]
    assert per["ridge"]["distinguishable_from_leader"] is False
    # Raw-unit convention: for an error metric, worse models are ABOVE the leader.
    assert per["tree"]["diff_vs_leader"] == pytest.approx(0.15, abs=1e-12)
    assert per["tree"]["ci_lo"] > 0.0
    assert per["tree"]["distinguishable_from_leader"] is True
    assert per["mean_baseline"]["distinguishable_from_leader"] is True


def test_metric_direction_flips_the_leader():
    scores = {"high": FOLDS + 0.05 + JIT_A, "low": FOLDS + JIT_B}
    assert indistinguishable_set(scores, higher_is_better=True, n_boot=500)["leader"] == "high"
    assert indistinguishable_set(scores, higher_is_better=False, n_boot=500)["leader"] == "low"


def test_leader_tie_break_is_name_based_and_order_independent():
    a, b = FOLDS + JIT_A, FOLDS + JIT_A  # exactly equal means
    forward = indistinguishable_set({"zeta": a, "alpha": b}, n_boot=200)
    reverse = indistinguishable_set({"alpha": b, "zeta": a}, n_boot=200)
    assert forward["leader"] == reverse["leader"] == "alpha"


def test_single_model_reports_itself_as_the_tied_set():
    result = indistinguishable_set({"only": FOLDS + JIT_A}, n_boot=200, seed=42)
    assert result["leader"] == "only"
    assert result["tied_with_leader"] == ["only"]
    entry = result["per_model"]["only"]
    assert entry["diff_vs_leader"] == 0.0
    assert (entry["ci_lo"], entry["ci_hi"]) == (0.0, 0.0)
    assert entry["distinguishable_from_leader"] is False


def test_near_tie_from_the_motivating_example_is_reported_as_a_tie():
    # The 0.9877-vs-0.9876 case this module exists for: a 1e-4 gap that moves
    # around across folds must not become a ranking.
    rng = np.random.default_rng(4)
    difficulty = rng.normal(0.0, 0.004, size=10)
    a = 0.9877 + difficulty + rng.normal(0.0, 0.002, size=10)
    b = 0.9876 + difficulty + rng.normal(0.0, 0.002, size=10)
    result = indistinguishable_set({"a": a, "b": b}, n_boot=3000, seed=42)
    assert sorted(result["tied_with_leader"]) == ["a", "b"]


# --------------------------------------------------------------------------- #
# edge cases: these must fail loudly, never degrade into a default result
# --------------------------------------------------------------------------- #
def test_unequal_length_vectors_raise():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff_ci(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2]))


def test_fewer_than_two_units_raises():
    with pytest.raises(ValueError, match="at least 2 paired units"):
        paired_bootstrap_diff_ci(np.array([0.9]), np.array([0.5]))


def test_two_units_produces_the_documented_degenerate_interval():
    # With n=2 the bootstrap has only 3 atoms {d0, (d0+d1)/2, d1}, so the
    # percentile interval collapses to [min(d), max(d)] -- it excludes 0 whenever
    # both differences share a sign. This is pinned because it is a real trap:
    # n=2 is accepted but the interval is not usable (measured FP rate ~50%).
    a = np.array([0.90, 0.80])
    b = np.array([0.88, 0.75])
    diff, lo, hi = paired_bootstrap_diff_ci(a, b, n_boot=600, seed=42)
    d = a - b  # [0.02, 0.05], both positive
    assert lo == pytest.approx(float(d.min()))
    assert hi == pytest.approx(float(d.max()))
    assert diff == pytest.approx(float(d.mean()))
    assert not (lo <= 0.0 <= hi), "same-sign differences at n=2 always exclude 0"

    # Opposite signs at n=2 always include 0, regardless of magnitude.
    a2 = np.array([0.90, 0.70])
    b2 = np.array([0.80, 0.95])
    _, lo2, hi2 = paired_bootstrap_diff_ci(a2, b2, n_boot=600, seed=42)
    assert lo2 <= 0.0 <= hi2


def test_non_finite_scores_raise():
    with pytest.raises(ValueError, match="non-finite"):
        paired_bootstrap_diff_ci(np.array([0.9, np.nan]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="non-finite"):
        paired_bootstrap_diff_ci(np.array([0.9, np.inf]), np.array([0.5, 0.5]))


def test_two_dimensional_input_raises():
    with pytest.raises(ValueError, match="1-D"):
        paired_bootstrap_diff_ci(np.zeros((5, 1)), np.zeros((5, 1)))


def test_invalid_alpha_and_n_boot_raise():
    a, b = FOLDS + JIT_A, FOLDS + JIT_B
    with pytest.raises(ValueError, match="alpha"):
        paired_bootstrap_diff_ci(a, b, alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        paired_bootstrap_diff_ci(a, b, alpha=-0.1)
    with pytest.raises(ValueError, match="n_boot"):
        paired_bootstrap_diff_ci(a, b, n_boot=1)


def test_indistinguishable_set_rejects_misaligned_models():
    with pytest.raises(ValueError, match="same number of aligned units"):
        indistinguishable_set({"a": np.zeros(10), "b": np.zeros(9)})


def test_indistinguishable_set_rejects_single_unit_instead_of_guessing():
    # One score per model carries no variance information; a "tie" verdict here
    # would be fabricated, so the function must refuse.
    with pytest.raises(ValueError, match="at least 2 aligned units"):
        indistinguishable_set({"a": np.array([0.98]), "b": np.array([0.97])})


def test_indistinguishable_set_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        indistinguishable_set({})


def test_alpha_widens_the_tied_set():
    # A borderline pair: strict alpha -> tied, permissive alpha -> distinguishable.
    rng = np.random.default_rng(21)
    difficulty = rng.normal(0.0, 0.01, size=30)
    a = 0.90 + difficulty + rng.normal(0.0, 0.01, size=30)
    b = 0.895 + difficulty + rng.normal(0.0, 0.01, size=30)
    strict = paired_bootstrap_diff_ci(a, b, n_boot=3000, alpha=0.001, seed=8)
    loose = paired_bootstrap_diff_ci(a, b, n_boot=3000, alpha=0.20, seed=8)
    assert (strict[2] - strict[1]) > (loose[2] - loose[1])


def test_bootstrap_handles_large_per_sample_vectors_in_chunks():
    # Exercises the memory-chunking path (n * n_boot exceeds one index block).
    rng = np.random.default_rng(2)
    n = 20_000
    correct_a = (rng.random(n) < 0.90).astype(np.float64)  # per-sample correctness
    correct_b = correct_a.copy()
    flip = rng.choice(n, size=600, replace=False)
    correct_b[flip] = 1.0 - correct_b[flip]
    diff, lo, hi = paired_bootstrap_diff_ci(correct_a, correct_b, n_boot=600, seed=42)
    assert diff == pytest.approx(float(correct_a.mean() - correct_b.mean()))
    assert lo < diff < hi
