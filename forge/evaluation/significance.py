"""Statistical significance testing for model comparison on a leaderboard.

Purpose: a 0.9877-vs-0.9876 leaderboard gap is noise, not a ranking. This module
decides which models are *statistically distinguishable* from the leader so the
rest can be reported as a TIE.

Design constraints (deliberate):
  * Pure functions. No I/O, no logging, no pipeline imports, no global state.
  * numpy only, plus ``scipy.stats`` for the two McNemar reference distributions.
  * Deterministic: identical ``seed`` -> bit-identical output.
  * Nothing here picks a model for the user. It reports what is tied; the
    decision (and any tie-break on latency, size, cost) belongs to the caller.
  * When a quantity cannot be computed, these functions raise ``ValueError``
    with a specific message. They never return a placeholder number and never
    swallow an error into an empty/default result.

Shared statistical assumption: every function below treats the supplied units
(CV folds, or test samples) as *exchangeable* draws -- i.e. resampling them is a
sane stand-in for repeating the experiment. Two honest caveats:

  1. k-fold CV scores are NOT independent (folds share training data), so a
     bootstrap over fold scores understates variance somewhat. Intervals from
     few folds should be read as optimistic (anti-conservative).
  2. All comparisons are made at level ``alpha`` *per comparison*. Comparing
     many models to one leader inflates the family-wise error rate. Use
     :func:`holm_bonferroni` on p-values when a family-wise guarantee matters;
     :func:`indistinguishable_set` applies no correction and says so in its
     ``multiplicity_correction`` field.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binom, chi2

__all__ = [
    "paired_bootstrap_diff_ci",
    "mcnemar_test",
    "holm_bonferroni",
    "indistinguishable_set",
]

# Cap on the number of int64 resample indices materialised at once (~64 MB).
# Bootstrapping per-sample vectors (n=100k, n_boot=2000) would otherwise
# allocate a 1.6 GB index matrix; chunking keeps memory bounded without
# changing the result for a given seed.
_MAX_INDEX_BLOCK = 8_000_000

# Below this many discordant pairs the chi-square approximation is unreliable,
# so mcnemar_test uses the exact binomial test instead.
_MCNEMAR_EXACT_MAX_DISCORDANT = 25


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _as_1d_finite_float(name: str, values: Any) -> np.ndarray:
    """Coerce ``values`` to a 1-D float64 array, or raise a specific ValueError."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; could not convert to float ({exc}).") from exc
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D vector of per-unit scores, got shape {arr.shape}. "
            "Flatten it explicitly so the unit axis is unambiguous."
        )
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s) (NaN/inf). "
            "A NaN would silently poison the mean and the confidence interval, "
            "so it must be resolved by the caller rather than imputed here."
        )
    return arr


def _validate_alpha(alpha: float) -> float:
    if not isinstance(alpha, (int, float, np.floating, np.integer)):
        raise ValueError(f"alpha must be a number, got {type(alpha).__name__}.")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}.")
    return alpha


def _validate_n_boot(n_boot: int) -> int:
    n_boot = int(n_boot)
    if n_boot < 2:
        raise ValueError(
            f"n_boot must be >= 2 to form a percentile interval, got {n_boot}. "
            "n_boot >= 1000 is recommended for a stable 95% interval."
        )
    return n_boot


# --------------------------------------------------------------------------- #
# 1. paired bootstrap confidence interval
# --------------------------------------------------------------------------- #
def paired_bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired percentile-bootstrap CI for ``mean(a) - mean(b)``.

    ``a`` and ``b`` are PAIRED per-unit values: element ``i`` of both must refer
    to the same unit (the same CV fold, or the same test sample). Typical inputs
    are per-fold CV scores, or per-sample correctness (0/1) or per-sample loss.

    Each bootstrap iteration draws one index vector and uses it to index BOTH
    ``a`` and ``b``, so whatever makes a unit easy or hard is applied to both
    models and cancels out. That is the whole point: for positively correlated
    models ``var(a - b) = var(a) + var(b) - 2 cov(a, b)`` is much smaller than
    the unpaired ``var(a) + var(b)``, giving a far tighter interval and real
    power to detect small differences.

    Assumptions: units are exchangeable draws (see module docstring); the
    percentile bootstrap needs no normality assumption but does need enough
    units for the empirical distribution to be informative.

    Degenerate-but-honest case: if the paired differences have zero variance
    (e.g. ``[0.9]*10`` vs ``[0.5]*10``), every resample yields the same
    difference and the interval collapses to a single point. That is the correct
    bootstrap answer for that data -- a perfectly consistent gap, however small,
    is reported as real. Guard against it upstream if the gap is below the
    metric's own measurement noise.

    MEASURED CALIBRATION -- read this before trusting a k-fold result. The plain
    percentile bootstrap is anti-conservative for small ``n``: it uses the
    plug-in variance and effectively normal quantiles where the mean of ``n``
    units needs Student-t quantiles. Simulated rate of a nominal 95% interval
    wrongly excluding 0 when the true difference is 0 (3000 trials each,
    Gaussian paired differences)::

        n_units :   2      3      4      5     10     20     50    100
        FP rate : 0.50   0.25   0.19   0.16   0.10   0.07   0.06   0.06

    So a "95%" interval is really ~90% at 10 folds and ~84% at 5 folds, and it
    is worthless at ``n=2`` (the bootstrap has only 3 atoms, so the interval
    degenerates to ``[min(d), max(d)]``, which excludes 0 whenever both
    differences share a sign -- hence exactly ~50%). Prefer per-sample vectors
    (thousands of units) when the comparison must be trustworthy; with few folds
    treat "distinguishable" as weak evidence and widen ``alpha`` accordingly.
    This function deliberately does not silently swap in a t-corrected or BCa
    interval -- the percentile method is what it advertises.

    Args:
        a: Per-unit values for model A.
        b: Per-unit values for model B, aligned element-wise with ``a``.
        n_boot: Number of bootstrap resamples (>= 2; >= 1000 recommended).
        alpha: Two-sided level; the interval is the ``[alpha/2, 1-alpha/2]``
            percentile range, e.g. ``alpha=0.05`` -> 95% interval.
        seed: Seed for ``np.random.default_rng``; same seed -> same interval.

    Returns:
        ``(observed_diff, ci_lo, ci_hi)`` where ``observed_diff`` is the actual
        ``mean(a) - mean(b)`` on the supplied data (not a bootstrap average).
        The interval contains 0 exactly when A and B are indistinguishable at
        this level.

    Raises:
        ValueError: if ``a`` and ``b`` differ in length, are not 1-D, contain
            non-finite values, have fewer than 2 units (sampling variability is
            not estimable from a single unit -- there is no honest interval to
            return), or if ``n_boot``/``alpha`` are out of range.
    """
    a_arr = _as_1d_finite_float("a", a)
    b_arr = _as_1d_finite_float("b", b)
    if a_arr.size != b_arr.size:
        raise ValueError(
            f"a and b must be paired and therefore the same length, got "
            f"{a_arr.size} and {b_arr.size}. Element i of each must be the same "
            "fold/sample; there is no meaningful pairing for unequal lengths."
        )
    n = a_arr.size
    if n < 2:
        raise ValueError(
            f"need at least 2 paired units to bootstrap a confidence interval, got {n}. "
            "With a single unit every resample is identical, so no interval can be "
            "estimated -- supply per-fold scores or per-sample values instead."
        )
    alpha = _validate_alpha(alpha)
    n_boot = _validate_n_boot(n_boot)

    observed_diff = float(a_arr.mean() - b_arr.mean())

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot, dtype=np.float64)
    block = max(1, min(n_boot, _MAX_INDEX_BLOCK // n))
    start = 0
    while start < n_boot:
        take = min(block, n_boot - start)
        # One index draw per iteration, applied to BOTH vectors -> paired.
        idx = rng.integers(0, n, size=(take, n))
        boot_diffs[start : start + take] = a_arr[idx].mean(axis=1) - b_arr[idx].mean(axis=1)
        start += take

    ci_lo, ci_hi = np.percentile(boot_diffs, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return observed_diff, float(ci_lo), float(ci_hi)


# --------------------------------------------------------------------------- #
# 2. McNemar test
# --------------------------------------------------------------------------- #
def mcnemar_test(y_true: Any, pred_a: Any, pred_b: Any) -> dict[str, Any]:
    """McNemar's test for two classifiers evaluated on the SAME labelled samples.

    Only the discordant pairs carry information:

      * ``b`` = count where A is right and B is wrong.
      * ``c`` = count where A is wrong and B is right.

    Samples both models get right, or both get wrong, are ignored by
    construction -- that is what makes the test paired.

    Method switch: with fewer than 25 discordant pairs (``b + c < 25``) the
    chi-square approximation is unreliable, so an exact two-sided binomial test
    against ``Binomial(b + c, 0.5)`` is used. At 25 or more, the chi-square
    statistic with Edwards' continuity correction is used.

    Assumptions: paired predictions on identical samples, samples independent of
    one another, and under the null a discordant pair is equally likely to fall
    either way. Labels may be of any comparable dtype (multiclass and string
    labels are fine); correctness is exact equality with ``y_true``.

    Returns:
        ``{"b": int, "c": int, "statistic": float, "p_value": float,
        "method": str}``. ``statistic`` is ``min(b, c)`` for the exact test (the
        binomial test statistic) and the corrected chi-square value otherwise.
        ``method`` is ``"exact_binomial"`` or ``"chi2_continuity_corrected"``.
        When the models agree on every sample (``b + c == 0``) the honest result
        is ``statistic=0.0, p_value=1.0`` via the exact branch: no evidence of a
        difference. A small p-value means the models differ; direction is read
        off ``b`` vs ``c`` (``b > c`` favours A).

    Raises:
        ValueError: if the three arrays differ in length, are not 1-D, or have
            dtypes that cannot be compared element-wise.
    """
    y = np.asarray(y_true)
    pa = np.asarray(pred_a)
    pb = np.asarray(pred_b)
    for name, arr in (("y_true", y), ("pred_a", pa), ("pred_b", pb)):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array of labels, got shape {arr.shape}.")
    if not (y.size == pa.size == pb.size):
        raise ValueError(
            f"y_true, pred_a and pred_b must be aligned per-sample and therefore the "
            f"same length, got {y.size}, {pa.size}, {pb.size}."
        )
    if y.size == 0:
        raise ValueError("y_true is empty; McNemar's test needs at least one labelled sample.")

    correct_a = y == pa
    correct_b = y == pb
    for name, mask in (("pred_a", correct_a), ("pred_b", correct_b)):
        # numpy returns a scalar (not an element-wise mask) for incomparable
        # dtypes; silently accepting that would report b == c == 0, i.e. a fake
        # "models are identical" verdict.
        if not isinstance(mask, np.ndarray) or mask.shape != y.shape:
            raise ValueError(
                f"could not compare {name} (dtype {np.asarray(mask).dtype!s} result) with "
                f"y_true element-wise; y_true dtype is {y.dtype} and {name} dtype is "
                f"{(pa if name == 'pred_a' else pb).dtype}. Cast them to a common label type."
            )

    b = int(np.count_nonzero(correct_a & ~correct_b))
    c = int(np.count_nonzero(~correct_a & correct_b))
    n_discordant = b + c

    if n_discordant < _MCNEMAR_EXACT_MAX_DISCORDANT:
        statistic = float(min(b, c))
        # Two-sided exact test: double the lower tail, capped at 1.0.
        p_value = float(min(1.0, 2.0 * binom.cdf(min(b, c), n_discordant, 0.5)))
        method = "exact_binomial"
    else:
        statistic = float((abs(b - c) - 1.0) ** 2 / n_discordant)
        p_value = float(chi2.sf(statistic, df=1))
        method = "chi2_continuity_corrected"

    return {"b": b, "c": c, "statistic": statistic, "p_value": p_value, "method": method}


# --------------------------------------------------------------------------- #
# 3. Holm-Bonferroni correction
# --------------------------------------------------------------------------- #
def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm's step-down correction for multiple comparisons.

    Sort the ``m`` p-values ascending and compare the rank-``i`` (0-based) value
    against ``alpha / (m - i)``. Reject in order until the first failure, then
    stop -- no hypothesis with a larger p-value can be rejected.

    Assumption: none beyond valid p-values. Holm controls the family-wise error
    rate at ``alpha`` under arbitrary dependence, which makes it the safe default
    for "is any model really better?" over a leaderboard. It is uniformly more
    powerful than plain Bonferroni and strictly more conservative than testing
    each hypothesis at ``alpha``.

    Args:
        p_values: One p-value per hypothesis, each in ``[0, 1]``.
        alpha: Family-wise error rate to control.

    Returns:
        Reject flags in the SAME order as ``p_values`` (``True`` = reject the
        null = the difference is significant). Empty input -> empty list.

    Raises:
        ValueError: if any p-value is non-finite or outside ``[0, 1]``, if the
            input is not 1-D, or if ``alpha`` is out of range.
    """
    alpha = _validate_alpha(alpha)
    m = len(p_values)
    if m == 0:
        return []

    p = _as_1d_finite_float("p_values", p_values)
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(
            f"p_values must all lie in [0, 1]; got min={float(p.min())}, max={float(p.max())}."
        )

    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(np.argsort(p, kind="stable")):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break  # step-down: everything with a larger p-value stays un-rejected.
    return [bool(flag) for flag in reject]


# --------------------------------------------------------------------------- #
# 4. leaderboard tie detection
# --------------------------------------------------------------------------- #
def indistinguishable_set(
    model_scores: dict[str, np.ndarray],
    higher_is_better: bool = True,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Find which leaderboard models are statistically tied with the leader.

    ``model_scores`` maps model name -> per-unit score vector. All vectors must
    be the SAME length and ALIGNED: index ``i`` must be the same CV fold (or the
    same test sample) for every model. Misaligned input would destroy the
    pairing that makes this test sensitive, so unequal lengths raise.

    The leader is the model with the best mean (largest if ``higher_is_better``,
    smallest otherwise). Every model -- the leader included -- is then compared
    to the leader with :func:`paired_bootstrap_diff_ci`, and a model counts as
    INDISTINGUISHABLE when its difference interval contains 0.

    Conventions worth knowing:
      * ``diff_vs_leader`` is ``mean(model) - mean(leader)`` in the metric's own
        raw units, NOT sign-normalised. For accuracy it is <= 0 for non-leaders;
        for an error metric like RMSE it is >= 0. ``higher_is_better`` only
        selects the leader -- the tie test itself is symmetric.
      * Every model is compared using the same bootstrap resamples (same
        ``seed``), so interval widths are directly comparable across models and
        the result does not depend on dict ordering.
      * Exact ties in mean are broken by model name (lexicographically first)
        so the output is fully deterministic.
      * No multiple-comparison correction is applied; the intervals are
        per-comparison at ``alpha`` (reported as ``multiplicity_correction``).
      * Interval calibration degrades with few units -- see the measured table
        in :func:`paired_bootstrap_diff_ci`. With 5 folds a nominal 95%
        interval behaves like ~84%, so this function will call models
        distinguishable more often than ``alpha`` suggests. ``n_units >= 10``,
        or per-sample vectors, is where the verdicts become dependable.

    This function reports; it does not choose. ``tied_with_leader`` is the set
    the caller may legitimately pick from on other grounds (latency, size,
    interpretability).

    Args:
        model_scores: name -> aligned per-unit score vector (>= 1 model).
        higher_is_better: True for accuracy/F1/R2, False for RMSE/MAE/log-loss.
        alpha: Two-sided level for each interval.
        n_boot: Bootstrap resamples per comparison.
        seed: Shared seed; same seed -> identical output.

    Returns:
        ``{"leader": str, "tied_with_leader": list[str],
        "per_model": {name: {"mean", "diff_vs_leader", "ci_lo", "ci_hi",
        "distinguishable_from_leader"}}, "alpha": float, "n_units": int,
        "higher_is_better": bool, "multiplicity_correction": "none"}``.
        ``tied_with_leader`` is in input order and always contains the leader.

    Raises:
        ValueError: if ``model_scores`` is empty, vectors differ in length, any
            vector is not 1-D or holds non-finite values, or ``n_units < 2``.
            The last case raises rather than returning an "everything is tied"
            result: with one unit per model there is no variance information, so
            any tie verdict would be fabricated.
    """
    if not isinstance(model_scores, dict):
        raise ValueError(
            f"model_scores must be a dict of name -> score vector, got {type(model_scores).__name__}."
        )
    if len(model_scores) == 0:
        raise ValueError("model_scores is empty; there is nothing to compare.")
    alpha = _validate_alpha(alpha)
    n_boot = _validate_n_boot(n_boot)

    scores: dict[str, np.ndarray] = {}
    for name, values in model_scores.items():
        scores[str(name)] = _as_1d_finite_float(f"model_scores[{name!r}]", values)

    lengths = {name: arr.size for name, arr in scores.items()}
    distinct = set(lengths.values())
    if len(distinct) > 1:
        raise ValueError(
            "all models must have the same number of aligned units (fold i must be the "
            f"same split for every model); got lengths {lengths}."
        )
    n_units = distinct.pop()
    if n_units < 2:
        raise ValueError(
            f"need at least 2 aligned units per model to test for ties, got {n_units}. "
            "A single score per model carries no variance information, so no tie or "
            "difference can be established -- pass per-fold CV scores or per-sample values."
        )

    means = {name: float(arr.mean()) for name, arr in scores.items()}
    # Best mean wins; exact ties broken by name for determinism.
    leader = min(
        means,
        key=lambda name: (-means[name] if higher_is_better else means[name], name),
    )
    leader_scores = scores[leader]

    per_model: dict[str, dict[str, Any]] = {}
    tied_with_leader: list[str] = []
    for name, arr in scores.items():
        diff, ci_lo, ci_hi = paired_bootstrap_diff_ci(
            arr, leader_scores, n_boot=n_boot, alpha=alpha, seed=seed
        )
        ci_contains_zero = bool(ci_lo <= 0.0 <= ci_hi)
        per_model[name] = {
            "mean": means[name],
            "diff_vs_leader": diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "distinguishable_from_leader": not ci_contains_zero,
        }
        if ci_contains_zero:
            tied_with_leader.append(name)

    return {
        "leader": leader,
        "tied_with_leader": tied_with_leader,
        "per_model": per_model,
        "alpha": alpha,
        "n_units": int(n_units),
        "higher_is_better": bool(higher_is_better),
        "multiplicity_correction": "none",
    }
