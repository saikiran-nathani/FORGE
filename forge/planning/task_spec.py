"""Turn the user's plain-language task description into machine-usable decision context.

Division of labour
------------------
The *data* already tells FORGE what the profiler can measure: task type
(classification vs regression), cardinality, class imbalance. Re-deriving those
from a sentence would be strictly worse than measuring them.

What the data can *never* supply is decision context:

* **cost asymmetry** - is a miss worse than a false alarm, and by how much?
* **capacity** - how many predictions can the business actually act on?
* **latency budget** - how fast must a single prediction be served?
* **interpretability** - must a human be able to justify the decision?
* **forecast** - is the target a future value (so random splits leak)?
* **protected attributes** - which columns must not drive the decision?

Only language carries those. This module extracts them.

Provenance is not optional
--------------------------
Every derived value is wrapped in a :class:`Field` that records whether it was
``"stated"`` (the text triggered it) or ``"default"`` (nothing in the text was
relevant, so a documented assumption was used), plus a short ``rationale`` the
UI shows for confirmation. FORGE must never silently decide for the user.

One nuance worth stating plainly: some phrasings state a *direction* without a
*magnitude* - "false positives are expensive" says which error hurts more but
not by how much. Those fields are marked ``source="stated"`` (the user really
did say something; calling it a "default" would be the bigger lie) and the
rationale explicitly labels the magnitude as an assumed placeholder to confirm.
The assumed magnitudes are the named constants :data:`MODERATE_ASYMMETRY` and
:data:`STRONG_ASYMMETRY`, never inline numbers.

Regex style: every pattern is token/word-boundary aware. Naive substring
matching is how ``"age" in "mileage"`` becomes a fairness constraint.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Field",
    "TaskSpec",
    "parse_task_description",
    "recommend_metric",
    "cost_sensitive_threshold",
]

# --------------------------------------------------------------------------
# Documented assumptions. These are placeholders the user is asked to confirm,
# never presented as measured values.
# --------------------------------------------------------------------------

#: Assumed cost ratio when the text names which error is worse in ordinary terms
#: ("expensive", "costly", "reduce", "too many") but gives no multiplier.
MODERATE_ASYMMETRY = 3.0

#: Assumed cost ratio for absolute phrasing ("can't afford to miss any",
#: "must catch every", "zero tolerance") - still an assumption, just a bigger one.
STRONG_ASYMMETRY = 10.0

#: Assumed serving budget when the text says "real-time" / "sub-second" but
#: quotes no number.
REALTIME_BUDGET_MS = 100.0

#: |log ratio| above which ``recommend_metric`` switches to a recall/precision
#: weighted metric instead of a symmetric one.
ASYMMETRY_METRIC_THRESHOLD = 2.0

#: Minority/majority class ratio thresholds used by ``recommend_metric``.
SEVERE_IMBALANCE = 0.05
MODERATE_IMBALANCE = 0.30
BALANCED_IMBALANCE = 0.80

VALID_OBJECTIVES = ("rank", "classify", "estimate")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Field:
    """A single derived value plus why we believe it.

    Attributes:
        value: The derived value.
        source: ``"stated"`` if the text triggered this value, ``"default"`` if
            nothing relevant was found and a documented assumption was used.
        rationale: Short human-readable justification, shown in the UI so the
            user can confirm or correct it before the pipeline runs.
    """

    value: Any
    source: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "source": self.source, "rationale": self.rationale}


@dataclass
class TaskSpec:
    """Decision context parsed out of a natural-language task description."""

    objective: Field  # "rank" | "classify" | "estimate"
    cost_ratio: Field  # {"fp": float, "fn": float}
    capacity_k: Field  # int | None
    latency_budget_ms: Field  # float | None
    interpretability_required: Field  # bool
    is_forecast: Field  # bool
    protected_attributes: Field  # list[str]
    raw_text: str
    parser: str  # "rules" | "llm"

    #: Field names in the order the UI should present them.
    FIELD_NAMES = (
        "objective",
        "cost_ratio",
        "capacity_k",
        "latency_budget_ms",
        "interpretability_required",
        "is_forecast",
        "protected_attributes",
    )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            name: getattr(self, name).to_dict() for name in self.FIELD_NAMES
        }
        payload["raw_text"] = self.raw_text
        payload["parser"] = self.parser
        return payload


# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

_QUOTE_MAP = {
    "‘": "'",
    "’": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
}


def _normalize(text: Any) -> str:
    """Collapse whitespace and fold smart quotes/dashes to ASCII.

    Accepts anything: non-string input is treated as "no description given"
    rather than raising, because this sits directly behind a web form.
    """
    if not isinstance(text, str):
        return ""
    for fancy, plain in _QUOTE_MAP.items():
        text = text.replace(fancy, plain)
    return re.sub(r"\s+", " ", text).strip()


def _quote(norm: str, low: str, start: int, end: int) -> str:
    """Return the matched snippet, preferring original casing when index-safe.

    ``str.lower()`` is length-preserving for ASCII but not for every Unicode
    codepoint, so fall back to the lowered text if the offsets could have
    drifted rather than slicing at a wrong position.
    """
    if len(low) == len(norm):
        return norm[start:end].strip()
    return low[start:end].strip()


def _mask(low: str, spans: list[tuple[int, int]]) -> str:
    """Blank out character ranges already consumed by another extractor.

    Keeps string length (and therefore all offsets) stable, so a later pattern
    cannot re-read "50ms" or "10x" as a capacity count.
    """
    if not spans:
        return low
    chars = list(low)
    for start, end in spans:
        for i in range(max(0, start), min(len(chars), end)):
            chars[i] = "\x00"
    return "".join(chars)


def _to_int(raw: str) -> int | None:
    digits = raw.replace(",", "").strip()
    if not digits.isdigit():
        return None
    return int(digits)


def _to_float(raw: str) -> float | None:
    cleaned = raw.replace(",", "").strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return None
    return float(cleaned)


# --------------------------------------------------------------------------
# Capacity  ("we can only call 100 customers a month", "top 50 leads")
# --------------------------------------------------------------------------

_TIME_UNIT = r"(?:hour|day|week|fortnight|month|quarter|year|shift|sprint)s?"

_CAPACITY_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\btop[\s-]+(?P<k>\d[\d,]*)\b",
        "an explicit top-K request",
    ),
    (
        r"\b(?P<k>\d[\d,]*)\s+(?!" + _TIME_UNIT + r"\b)[a-z_]+(?:\s+[a-z_]+)?\s*"
        r"(?:/|\bper\b|\ba\b|\beach\b)\s*" + _TIME_UNIT + r"\b",
        "a throughput limit (fixed volume per time period)",
    ),
    (
        r"\b(?:only|just|at most|no more than|not more than|up to|maximum of|max of|"
        r"limited to|room for|afford)\b[^.;!?]{0,40}?\b(?P<k>\d[\d,]*)\b",
        "an explicit upper bound on how many records can be acted on",
    ),
    (
        r"\b(?:review|reviews|call|calls|contact|inspect|audit|handle|process|"
        r"investigate|visit|screen|interview|target|chase|action|serve)\b"
        r"[^.;!?]{0,25}?\b(?P<k>\d[\d,]*)\b",
        "a hands-on work verb applied to a fixed count",
    ),
    (
        r"\b(?:capacity|bandwidth|budget|headcount|staff|resources|throughput)\b"
        r"[^.;!?]{0,30}?\b(?P<k>\d[\d,]*)\b",
        "a stated capacity/budget for a fixed count",
    ),
    (
        r"\b(?:shortlist|short list|list|queue|batch|sample)\s+of\s+(?P<k>\d[\d,]*)\b",
        "a fixed-size output list",
    ),
)

# A number followed by any of these is not a capacity count: it is a percentage,
# a multiplier, a duration or a latency figure.
_CAPACITY_BAD_SUFFIX = re.compile(
    r"^\s*(?:%|percent|pct\b|x\b|times\b|ms\b|millisec\w*|sec\w*\b|min\w*\b|"
    r"hours?\b|days?\b|weeks?\b|months?\b|quarters?\b|years?\b|bps\b)"
)
# A number preceded by a currency symbol is money, not a count.
_CAPACITY_BAD_PREFIX = re.compile(r"[$€£¥]\s*$")


def _extract_capacity(norm: str, low: str, masked: str) -> tuple[Field, list[tuple[int, int]]]:
    for pattern, why in _CAPACITY_PATTERNS:
        for match in re.finditer(pattern, masked):
            start, end = match.span("k")
            if _CAPACITY_BAD_SUFFIX.search(masked[end : end + 24]):
                continue
            if _CAPACITY_BAD_PREFIX.search(masked[max(0, start - 3) : start]):
                continue
            k = _to_int(match.group("k"))
            if k is None or k <= 0:
                continue
            snippet = _quote(norm, low, *match.span())
            return (
                Field(
                    value=k,
                    source="stated",
                    rationale=(
                        f'"{snippet}" reads as {why}, so only the top {k} predictions '
                        f"will ever be acted on; the metric should score that slice, "
                        f"not the whole test set."
                    ),
                ),
                [match.span()],
            )
    return (
        Field(
            value=None,
            source="default",
            rationale=(
                "No acting capacity mentioned, so every prediction is assumed "
                "actionable and the metric scores the full test set. If your team "
                "can only work a fixed number of cases, say so - it changes the metric."
            ),
        ),
        [],
    )


# --------------------------------------------------------------------------
# Latency  ("must respond in under 50ms", "real-time")
# --------------------------------------------------------------------------

_LATENCY_NUM = re.compile(
    r"\b(?P<v>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<u>milliseconds?|millisecs?|ms|minutes?|mins?|seconds?|secs?|s)\b(?!\s*of\b)"
)

_LATENCY_CONTEXT = re.compile(
    r"\b(?:latency|respond\w*|response|reply|replies|round[\s-]?trip|turnaround|"
    r"inference|serve|serving|served|score|scoring|predict\w*|api|endpoint|sla|"
    r"p9[0-9]|real[\s-]?time|realtime|online)\b"
)

# Time budgets that belong to *training*, not to serving a single prediction.
_TRAINING_CONTEXT = re.compile(
    r"\b(?:train\w*|fit|fitting|refit|retrain\w*|build|rebuild|refresh\w*|"
    r"backfill|nightly|overnight|batch job|etl)\b"
)

_REALTIME = re.compile(
    r"\b(?:real[\s-]?time|realtime|sub[\s-]?second|instant(?:ly|aneous)?|"
    r"low[\s-]?latency|on[\s-]the[\s-]fly|as[\s-]they[\s-]type|"
    r"at (?:the )?point of (?:sale|checkout))\b"
)


def _unit_to_ms(unit: str) -> float:
    if unit.startswith("ms") or unit.startswith("milli"):
        return 1.0
    if unit.startswith("min"):
        return 60_000.0
    return 1_000.0


def _extract_latency(norm: str, low: str) -> tuple[Field, list[tuple[int, int]]]:
    for match in _LATENCY_NUM.finditer(low):
        value = _to_float(match.group("v"))
        if value is None or value <= 0:
            continue
        unit = match.group("u")
        factor = _unit_to_ms(unit)
        # Seconds/minutes are ambiguous: they could be a training budget. Only
        # accept them with explicit serving context and no training context.
        if factor > 1.0:
            if not _LATENCY_CONTEXT.search(low) or _TRAINING_CONTEXT.search(low):
                continue
        budget = value * factor
        snippet = _quote(norm, low, *match.span())
        return (
            Field(
                value=budget,
                source="stated",
                rationale=(
                    f'"{snippet}" is a per-prediction serving budget of {budget:g}ms, '
                    f"which rules out models that cannot score a single row that fast."
                ),
            ),
            [match.span()],
        )

    realtime = _REALTIME.search(low)
    if realtime:
        snippet = _quote(norm, low, *realtime.span())
        return (
            Field(
                value=REALTIME_BUDGET_MS,
                source="stated",
                rationale=(
                    f'"{snippet}" states a real-time requirement but quotes no number, '
                    f"so {REALTIME_BUDGET_MS:g}ms is assumed as a placeholder - confirm "
                    f"your actual budget."
                ),
            ),
            [realtime.span()],
        )

    return (
        Field(
            value=None,
            source="default",
            rationale=(
                "No serving speed mentioned, so prediction latency is assumed "
                "unconstrained and no model is excluded for being slow."
            ),
        ),
        [],
    )


# --------------------------------------------------------------------------
# Cost asymmetry
# --------------------------------------------------------------------------

# An explicit magnitude: "10x", "10 times", "10:1".
_MULTIPLIER = re.compile(r"\b(?P<m>\d[\d,]*(?:\.\d+)?)\s*(?:x\b|times\b)")
_RATIO_PAIR = re.compile(r"\b(?P<a>\d[\d,]*)\s*:\s*(?P<b>\d[\d,]*)\b")

# Tier A - the text asserts which error is worse. "strong" = absolute phrasing.
_FN_COSTLY_STRONG = (
    r"can'?t afford to miss",
    r"cannot afford to miss",
    r"can'?t afford any",
    r"must not miss",
    r"mustn'?t miss",
    r"never miss",
    r"cannot miss",
    r"can'?t miss",
    r"miss(?:ing)? any",
    r"catch (?:all|every|each)",
    r"catching (?:all|every)",
    r"zero tolerance",
    r"at all costs",
    r"missed? (?:case|fraud|default|diagnos|detection)\w* (?:are|is) unacceptable",
)
_FN_COSTLY_MODERATE = (
    r"false negatives? (?:are|is) (?:worse|expensive|costly|the problem|more)",
    r"recall (?:matters|is (?:more )?important|is critical|is what matters)",
    r"maximi[sz]e (?:detection|recall|coverage)",
    r"minimi[sz]e (?:missed|misses|false negatives?)",
    r"reduce (?:missed|misses|false negatives?)",
    r"prefer(?:ring)? (?:to )?(?:over[\s-]?flag|false alarms?)",
    r"rather (?:over[\s-]?flag|have a false alarm)",
    r"don'?t want to miss",
    r"do not want to miss",
    r"avoid missing",
    r"miss(?:ing)? \w+ (?:is|are) (?:expensive|costly|bad|worse|damaging)",
)
_FP_COSTLY_STRONG = (
    r"can'?t afford (?:any )?false (?:positives?|alarms?)",
    r"cannot afford (?:any )?false (?:positives?|alarms?)",
    r"no false (?:positives?|alarms?)",
    r"zero false (?:positives?|alarms?)",
    r"must not (?:wrongly|falsely|incorrectly) \w+",
)
_FP_COSTLY_MODERATE = (
    r"false (?:positives?|alarms?|alerts?) (?:are|is|would be|can be|get) "
    r"(?:very |extremely |really |quite |too )?"
    r"(?:expensive|costly|bad|worse|painful|damaging|unacceptable|a problem|annoying)",
    r"(?:expensive|costly|painful|unacceptable|damaging)\b[^.;!?]{0,20}?"
    r"false (?:positives?|alarms?|alerts?)",
    r"too many false (?:positives?|alarms?|alerts?)",
    r"too many alerts?",
    r"alert fatigue",
    r"avoid false (?:positives?|alarms?|alerts?)",
    r"minimi[sz]e false (?:positives?|alarms?|alerts?)",
    r"reduce false (?:positives?|alarms?|alerts?)",
    r"precision (?:matters|is (?:more )?important|is critical|is what matters)",
    r"wrongly (?:flag|accus|reject|den|block)\w*",
    r"falsely (?:flag|accus|reject|den|block)\w*",
    r"annoy\w*[^.;!?]{0,20}?(?:good )?customers",
    r"upset\w*[^.;!?]{0,20}?(?:good )?customers",
)

# Tier B - a bare mention. Only used to orient an explicit multiplier, never to
# claim asymmetry on its own ("we don't care about false positives" mentions
# them too).
_FN_MENTION = (
    r"\bmiss\w*\b",
    r"\bfalse negatives?\b",
    r"\bfn\b",
    r"\bundetected\b",
    r"\bslip(?:s|ping)? through\b",
    r"\bnot caught\b",
)
_FP_MENTION = (
    r"\bfalse (?:positives?|alarms?|alerts?)\b",
    r"\bfp\b",
    r"\bfalse alerts?\b",
)

_SYMMETRIC_RATIONALE = (
    "No cost asymmetry mentioned, so a false positive and a false negative are "
    "weighted equally (1:1) and the decision threshold stays at 0.5. If one "
    "error costs more than the other, say so - it changes both the metric and "
    "the threshold."
)


def _first_span(patterns: tuple[str, ...], low: str) -> tuple[int, str] | None:
    """Earliest match across `patterns`, as (start index, matched text)."""
    best: tuple[int, str] | None = None
    for pattern in patterns:
        match = re.search(pattern, low)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), match.group(0))
    return best


def _extract_cost_ratio(norm: str, low: str) -> tuple[Field, list[tuple[int, int]]]:
    fn_strong = _first_span(_FN_COSTLY_STRONG, low)
    fp_strong = _first_span(_FP_COSTLY_STRONG, low)
    fn_moderate = _first_span(_FN_COSTLY_MODERATE, low)
    fp_moderate = _first_span(_FP_COSTLY_MODERATE, low)
    fn_claim = fn_strong or fn_moderate
    fp_claim = fp_strong or fp_moderate

    # 1. Explicit magnitude, oriented by whichever error is named first.
    mult_match = _MULTIPLIER.search(low)
    ratio_match = None if mult_match else _RATIO_PAIR.search(low)
    magnitude: float | None = None
    spans: list[tuple[int, int]] = []
    if mult_match:
        magnitude = _to_float(mult_match.group("m"))
        spans = [mult_match.span()]
    elif ratio_match:
        high = _to_float(ratio_match.group("a"))
        low_side = _to_float(ratio_match.group("b"))
        if high and low_side and low_side > 0:
            magnitude = high / low_side
            spans = [ratio_match.span()]

    if magnitude is not None and magnitude > 1.0:
        fn_pos = _first_span(_FN_MENTION, low)
        fp_pos = _first_span(_FP_MENTION, low)
        direction: str | None = None
        if fn_pos and fp_pos:
            direction = "fn" if fn_pos[0] < fp_pos[0] else "fp"
        elif fn_pos:
            direction = "fn"
        elif fp_pos:
            direction = "fp"
        elif fn_claim and not fp_claim:
            direction = "fn"
        elif fp_claim and not fn_claim:
            direction = "fp"

        if direction is not None:
            costs = (
                {"fp": 1.0, "fn": float(magnitude)}
                if direction == "fn"
                else {"fp": float(magnitude), "fn": 1.0}
            )
            worse = "a missed case" if direction == "fn" else "a false alarm"
            snippet = _quote(norm, low, *spans[0])
            return (
                Field(
                    value=costs,
                    source="stated",
                    rationale=(
                        f'"{snippet}" states an explicit magnitude and the text names '
                        f"{worse} as the costly error, giving "
                        f"fp={costs['fp']:g} / fn={costs['fn']:g}. The decision "
                        f"threshold moves to "
                        f"{cost_sensitive_threshold(costs):.3g} accordingly."
                    ),
                ),
                spans,
            )

    # 2. Direction stated, magnitude not. Assume a documented placeholder.
    if fn_claim or fp_claim:
        if fn_claim and fp_claim:
            direction = "fn" if fn_claim[0] < fp_claim[0] else "fp"
        else:
            direction = "fn" if fn_claim else "fp"
        is_strong = bool(fn_strong) if direction == "fn" else bool(fp_strong)
        assumed = STRONG_ASYMMETRY if is_strong else MODERATE_ASYMMETRY
        snippet = (fn_claim if direction == "fn" else fp_claim)[1]
        costs = (
            {"fp": 1.0, "fn": assumed} if direction == "fn" else {"fp": assumed, "fn": 1.0}
        )
        worse = "missing a positive" if direction == "fn" else "a false alarm"
        strength = "absolute" if is_strong else "ordinary"
        return (
            Field(
                value=costs,
                source="stated",
                rationale=(
                    f'"{snippet}" names {worse} as the costly error in {strength} terms '
                    f"but quotes no multiplier, so {assumed:g}:1 is assumed as a "
                    f"placeholder - confirm the real ratio. Threshold moves to "
                    f"{cost_sensitive_threshold(costs):.3g}."
                ),
            ),
            [],
        )

    # 3. Nothing stated.
    return (
        Field(value={"fp": 1.0, "fn": 1.0}, source="default", rationale=_SYMMETRIC_RATIONALE),
        [],
    )


# --------------------------------------------------------------------------
# Interpretability
# --------------------------------------------------------------------------

# Deliberately excludes the bare verb "audit": "we can only audit 20 accounts"
# is a capacity statement, not a request for an explainable model.
_INTERPRETABILITY = re.compile(
    r"\b(?:interpretab\w+|interpretable|explainab\w+|explicab\w+|"
    r"transparen\w+|auditab\w+|audit(?:or|ors|ing)? (?:trail|requirement|committee)|"
    r"auditors?\b|regulator\w*|regulatory|compliance|complian\w+|"
    r"black[\s-]?box|reason codes?|adverse action|gdpr|sr 11[\s-]?7|"
    r"fair lending|ecoa|model risk|defensib\w+|justifiab\w+|"
    r"white[\s-]?box|glass[\s-]?box|scorecard|human[\s-]readable)\b"
    r"|\b(?:explain|justify|defend|document)\b[^.;!?]{0,40}?"
    r"\b(?:decision|decisions|denial|denials|rejection|prediction|predictions|"
    r"outcome|outcomes|why|score|scores|reasoning)\b"
)


def _extract_interpretability(norm: str, low: str) -> Field:
    match = _INTERPRETABILITY.search(low)
    if match:
        snippet = _quote(norm, low, *match.span())
        return Field(
            value=True,
            source="stated",
            rationale=(
                f'"{snippet}" requires a human-defensible decision, so glass-box '
                f"models and per-prediction explanations should be preferred over a "
                f"marginally more accurate opaque model."
            ),
        )
    return Field(
        value=False,
        source="default",
        rationale=(
            "No explainability requirement mentioned, so model choice is assumed "
            "free to favour accuracy over transparency. Say so if a human has to "
            "justify each decision."
        ),
    )


# --------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------

_FORECAST = re.compile(
    r"\b(?:forecast\w*|time[\s-]?series|timeseries|seasonal\w*|seasonality|"
    r"autocorrelat\w+|backtest\w*|arima|prophet|horizon)\b"
    r"|\bnext\s+(?:\d[\d,]*\s+)?(?:day|week|month|quarter|year|fiscal|season)\w*\b"
    r"|\b(?:days?|weeks?|months?|quarters?|years?)\s+(?:ahead|out|from now)\b"
    r"|\b(?:predict|project|estimate|anticipate)\w*\b[^.;!?]{0,30}?\bfuture\b"
    r"|\bfuture\b[^.;!?]{0,25}?\b(?:sales|demand|revenue|values?|prices?|volumes?|"
    r"usage|load|traffic|churn)\b"
    r"|\b(?:upcoming|coming)\s+(?:day|week|month|quarter|year|season)\w*\b"
)


def _extract_forecast(norm: str, low: str) -> Field:
    match = _FORECAST.search(low)
    if match:
        snippet = _quote(norm, low, *match.span())
        return Field(
            value=True,
            source="stated",
            rationale=(
                f'"{snippet}" describes predicting a future period, so a random '
                f"train/test split would leak future information; validation must "
                f"split by time instead."
            ),
        )
    return Field(
        value=False,
        source="default",
        rationale=(
            "Nothing indicates a future time period, so rows are assumed "
            "exchangeable and a random split is valid. Say \"forecast\" or "
            "\"time series\" if the target is a future value."
        ),
    )


# --------------------------------------------------------------------------
# Protected attributes
# --------------------------------------------------------------------------

# Word-boundary anchored on purpose: `\bage\b` must not fire inside "mileage",
# and `\bsex\b` must not fire inside "sexual orientation".
_PROTECTED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gender", r"\bgenders?\b|\bgendered\b|\bsex\b|\bsexes\b"),
    (
        "age",
        r"\bages?\b|\bage[\s_-]?(?:group|band|bracket|range)s?\b|\bdob\b|"
        r"\bdate of birth\b|\bbirth[\s_-]?dates?\b|\bbirth[\s_-]?years?\b",
    ),
    ("race", r"\brace\b|\bracial\b|\bethnicit\w+\b|\bethnic\b"),
    ("religion", r"\breligions?\b|\breligious\b"),
    (
        "nationality",
        r"\bnationalit\w+\b|\bnational origin\b|\bcitizenships?\b|"
        r"\bimmigration status\b|\bvisa status\b",
    ),
    ("disability", r"\bdisabilit\w+\b|\bdisabled\b|\bhandicap\w*\b"),
    ("marital_status", r"\bmarital status\b|\bmarried\b|\bdivorced\b"),
    ("pregnancy", r"\bpregnan\w+\b"),
    ("sexual_orientation", r"\bsexual orientation\b|\blgbtq?\w*\b"),
    ("veteran_status", r"\bveterans?\b|\bmilitary service\b"),
    ("genetic_information", r"\bgenetic\w*\b"),
    ("zip_code", r"\bzip[\s_-]?codes?\b|\bpostal codes?\b|\bpostcodes?\b"),
)

_FAIRNESS_CONTEXT = re.compile(
    r"\b(?:discriminat\w*|bias\w*|unbiased|fair|fairly|fairness|unfair\w*|equit\w+|"
    r"disparate (?:impact|treatment)|adverse impact|demographic parity|"
    r"equal(?:i[sz]ed)? opportunity|protected (?:class|classes|attribute|attributes|"
    r"characteristic|characteristics)|proxy|proxies|blind to|"
    r"exclude|excluding|ignore|without using|regardless of)\b"
    r"|\bdo(?:n'?t| not)\s+(?:use|consider|rely)\b"
)


def _extract_protected(norm: str, low: str) -> Field:
    found: list[tuple[int, str]] = []
    for canonical, pattern in _PROTECTED_PATTERNS:
        match = re.search(pattern, low)
        if match:
            found.append((match.start(), canonical))
    # Text order, then name, so the output is fully deterministic.
    found.sort(key=lambda item: (item[0], item[1]))
    names = [name for _, name in found]

    if not names:
        return Field(
            value=[],
            source="default",
            rationale=(
                "No protected or sensitive attribute named, so no fairness "
                "constraint is applied. Name the attributes explicitly (e.g. "
                "\"don't discriminate by gender\") to enable fairness auditing."
            ),
        )

    fairness = _FAIRNESS_CONTEXT.search(low)
    if fairness:
        snippet = _quote(norm, low, *fairness.span())
        rationale = (
            f'"{snippet}" frames these as a fairness constraint, so '
            f"{', '.join(names)} should be treated as protected: audit subgroup "
            f"performance and review their use as features."
        )
    else:
        rationale = (
            f"{', '.join(names)} appear in the description as sensitive attributes, "
            f"but no explicit fairness instruction was given - confirm whether these "
            f"must be excluded from the model or only monitored."
        )
    return Field(value=names, source="stated", rationale=rationale)


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------

_RANK_WORDS = re.compile(
    r"\b(?:rank\w*|ranking|prioriti[sz]\w+|triage\w*|shortlist\w*|"
    r"lead scor\w+|order\w* by|sort\w* by|best (?:leads?|candidates?|prospects?)|"
    r"most likely|least likely|who (?:to|should we|we should) (?:call|contact|target|"
    r"chase|visit)|which \w+ (?:to|should we) (?:call|contact|target|review|"
    r"prioriti[sz]e)|worth (?:calling|contacting)|focus (?:on|our))\b"
)
_CLASSIFY_WORDS = re.compile(
    r"\b(?:classif\w+|categori[sz]\w+|label\w*|tag\w*|flag\w*|detect\w*|"
    r"identify|whether|yes[\s/]no|approve or (?:deny|reject)|"
    r"spam|fraud\w*|churn\w*|default\w*|good or bad|pass or fail|"
    r"which (?:class|category|type|segment))\b"
)
_ESTIMATE_WORDS = re.compile(
    r"\b(?:estimat\w+|regress\w+|forecast\w*|how (?:much|many|long|far)|"
    r"expected (?:value|revenue|cost|spend|demand|volume)|"
    r"predict\w* the (?:price|value|amount|number|revenue|cost|sales|demand|"
    r"volume|duration|length|time|score|total)|"
    r"(?:dollar|numeric|continuous) (?:value|amount)|"
    r"price|revenue|amount|quantity|duration|lifetime value|ltv)\b"
)


def _extract_objective(
    norm: str, low: str, capacity: Field, is_forecast: Field
) -> Field:
    if capacity.value is not None:
        return Field(
            value="rank",
            source="stated",
            rationale=(
                f"A capacity of {capacity.value} was stated, which only makes sense if "
                f"predictions are ordered and the top {capacity.value} are worked - "
                f"that is a ranking problem, not a labelling one."
            ),
        )

    rank = _RANK_WORDS.search(low)
    if rank:
        return Field(
            value="rank",
            source="stated",
            rationale=(
                f'"{_quote(norm, low, *rank.span())}" asks for an ordering of records '
                f"rather than a label, so ranking quality matters more than "
                f"threshold accuracy."
            ),
        )

    estimate = _ESTIMATE_WORDS.search(low)
    classify = _CLASSIFY_WORDS.search(low)
    if is_forecast.value and (estimate or not classify):
        return Field(
            value="estimate",
            source="stated",
            rationale=(
                "The description is a forecast of a future quantity, so the output "
                "is a numeric estimate rather than a discrete label."
            ),
        )
    if estimate and (not classify or estimate.start() < classify.start()):
        return Field(
            value="estimate",
            source="stated",
            rationale=(
                f'"{_quote(norm, low, *estimate.span())}" asks for a numeric quantity, '
                f"so the output is an estimate rather than a discrete label."
            ),
        )
    if classify:
        return Field(
            value="classify",
            source="stated",
            rationale=(
                f'"{_quote(norm, low, *classify.span())}" asks for a discrete '
                f"label/decision, so a threshold will be applied to the score."
            ),
        )
    return Field(
        value="classify",
        source="default",
        rationale=(
            "No objective wording found, so a labelling task is assumed. Data "
            "profiling still decides classification vs regression; this only "
            "records that no ranking or capacity framing was stated."
        ),
    )


# --------------------------------------------------------------------------
# Rule parser
# --------------------------------------------------------------------------

_EMPTY_TEXT_NOTE = (
    "No task description was given, so this is a documented default rather than "
    "anything you stated."
)


def _all_defaults(raw_text: str) -> TaskSpec:
    """Spec for empty/blank input: every field a default, every rationale honest."""

    def note(field: Field) -> Field:
        return Field(value=field.value, source="default", rationale=f"{_EMPTY_TEXT_NOTE} {field.rationale}")

    capacity = note(_extract_capacity("", "", "")[0])
    return TaskSpec(
        objective=note(_extract_objective("", "", capacity, Field(False, "default", ""))),
        cost_ratio=note(_extract_cost_ratio("", "")[0]),
        capacity_k=capacity,
        latency_budget_ms=note(_extract_latency("", "")[0]),
        interpretability_required=note(_extract_interpretability("", "")),
        is_forecast=note(_extract_forecast("", "")),
        protected_attributes=note(_extract_protected("", "")),
        raw_text=raw_text,
        parser="rules",
    )


def _parse_with_rules(text: Any) -> TaskSpec:
    norm = _normalize(text)
    if not norm:
        return _all_defaults(norm)

    low = norm.lower()

    # Latency and cost multipliers are extracted first, then blanked out, so the
    # capacity extractor cannot mistake "50ms" or "10x" for a headcount.
    latency, latency_spans = _extract_latency(norm, low)
    cost, cost_spans = _extract_cost_ratio(norm, low)
    masked = _mask(low, latency_spans + cost_spans)
    capacity, _ = _extract_capacity(norm, low, masked)

    is_forecast = _extract_forecast(norm, low)
    return TaskSpec(
        objective=_extract_objective(norm, low, capacity, is_forecast),
        cost_ratio=cost,
        capacity_k=capacity,
        latency_budget_ms=latency,
        interpretability_required=_extract_interpretability(norm, low),
        is_forecast=is_forecast,
        protected_attributes=_extract_protected(norm, low),
        raw_text=norm,
        parser="rules",
    )


# --------------------------------------------------------------------------
# Optional LLM refinement
# --------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You extract decision constraints from a short business description of a "
    "machine-learning task. Return valid JSON only. Never guess: if the text does "
    "not state something, return null for that key. For every non-null key you "
    "MUST supply a verbatim quote from the description in the \"evidence\" object; "
    "a key without a verbatim quote will be discarded."
)

_LLM_SCHEMA_HINT = """Return exactly this JSON shape:
{
  "objective": "rank" | "classify" | "estimate" | null,
  "cost_ratio": {"fp": <number>, "fn": <number>} | null,
  "capacity_k": <integer> | null,
  "latency_budget_ms": <number> | null,
  "interpretability_required": true | false | null,
  "is_forecast": true | false | null,
  "protected_attributes": [<string>, ...] | null,
  "evidence": {"<key you filled>": "<verbatim quote from the description>"}
}"""

_LLM_RATIONALE = (
    'LLM read "{quote}" in your description as {key} = {value}. The quote was '
    "verified to appear in your text verbatim; confirm the reading is right."
)


def _evidence_ok(evidence: Any, key: str, low: str) -> str | None:
    """Return the verbatim quote backing `key`, or None if unverifiable.

    An LLM value is only trusted when the model can point at text that actually
    exists in the description. This is the anti-hallucination gate: it stops the
    LLM path from inventing a capacity or a cost ratio the user never mentioned.
    """
    if not isinstance(evidence, dict):
        return None
    quote = evidence.get(key)
    if not isinstance(quote, str) or not quote.strip():
        return None
    needle = re.sub(r"\s+", " ", quote).strip().lower()
    if needle and needle in low:
        return quote.strip()
    return None


def _coerce_objective(value: Any) -> str | None:
    if isinstance(value, str) and value.strip().lower() in VALID_OBJECTIVES:
        return value.strip().lower()
    return None


def _coerce_cost_ratio(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    fp, fn = value.get("fp"), value.get("fn")
    if isinstance(fp, bool) or isinstance(fn, bool):
        return None
    if not isinstance(fp, (int, float)) or not isinstance(fn, (int, float)):
        return None
    if fp <= 0 or fn <= 0:
        return None
    return {"fp": float(fp), "fn": float(fn)}


def _coerce_capacity(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _coerce_latency(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _coerce_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _coerce_str_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    names = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return names or None


_LLM_COERCERS = {
    "objective": _coerce_objective,
    "cost_ratio": _coerce_cost_ratio,
    "capacity_k": _coerce_capacity,
    "latency_budget_ms": _coerce_latency,
    "interpretability_required": _coerce_bool,
    "is_forecast": _coerce_bool,
    "protected_attributes": _coerce_str_list,
}


def _refine_with_llm(norm: str, rules_spec: TaskSpec, llm: Any) -> TaskSpec | None:
    """Overlay schema-validated, quote-backed LLM values onto the rules result.

    Returns None to mean "use the rules result unchanged" - the caller then keeps
    ``parser="rules"``. Never returns a partially-built spec with empty fields.
    """
    user_prompt = f"Description:\n{norm}\n\n{_LLM_SCHEMA_HINT}"
    try:
        payload = llm.complete_json(_LLM_SYSTEM, user_prompt)
    except Exception:
        # Third-party SDK: transport, auth, rate-limit and parse errors all surface
        # as arbitrary exception types. Broad catch is deliberate, but it is LOGGED
        # and the caller falls back to the rules spec with parser="rules" - never
        # swallowed into empty output.
        logger.warning(
            "LLM task-spec refinement failed; falling back to the rule parser.",
            exc_info=True,
        )
        return None

    if not isinstance(payload, dict) or not payload:
        logger.warning(
            "LLM task-spec refinement returned %s instead of a JSON object; "
            "falling back to the rule parser.",
            type(payload).__name__,
        )
        return None

    evidence = payload.get("evidence")
    low = norm.lower()
    accepted: dict[str, Field] = {}
    for key, coerce in _LLM_COERCERS.items():
        if key not in payload:
            continue
        value = coerce(payload[key])
        if value is None:
            continue
        quote = _evidence_ok(evidence, key, low)
        if quote is None:
            logger.debug("Discarding unverified LLM value for %s.", key)
            continue
        accepted[key] = Field(
            value=value,
            source="stated",
            rationale=_LLM_RATIONALE.format(quote=quote, key=key, value=value),
        )

    if not accepted:
        logger.warning(
            "No LLM-proposed field passed schema + evidence validation; "
            "falling back to the rule parser."
        )
        return None

    return dataclasses.replace(rules_spec, parser="llm", **accepted)


def parse_task_description(text: str, llm: Any = None) -> TaskSpec:
    """Parse a plain-language task description into a :class:`TaskSpec`.

    The rule-based parser is the primary path and works with no API key. When
    `llm` is supplied and reports ``available``, its output is used only for
    fields that pass both schema validation and a verbatim-quote check against
    the text; anything else keeps the rule value. If the LLM call fails or
    nothing validates, the rules result is returned with ``parser="rules"``.

    Never raises: unusable input (empty string, ``None``, non-string) yields a
    spec of documented defaults.
    """
    rules_spec = _parse_with_rules(text)

    if llm is None or not getattr(llm, "available", False):
        return rules_spec
    if not rules_spec.raw_text:
        # Nothing to quote against; the evidence gate would reject everything.
        return rules_spec

    refined = _refine_with_llm(rules_spec.raw_text, rules_spec, llm)
    return refined if refined is not None else rules_spec


# --------------------------------------------------------------------------
# Metric selection
# --------------------------------------------------------------------------

_OUTLIER_ROBUST = re.compile(
    r"\b(?:outliers?|outlying|extreme values?|extremes?|heavy[\s-]?tail\w*|"
    r"long[\s-]?tail\w*|median|absolute error|typical (?:error|case|customer)|"
    r"robust to|spikes?|anomalous values?|skew\w*)\b"
    r"|\b(?:do(?:n'?t| not)|shouldn'?t|should not)\b[^.;!?]{0,30}?"
    r"\b(?:dominat\w+|over[\s-]?penali[sz]\w+|skew\w+)\b"
)


def _normalized_imbalance(raw: Any) -> tuple[float | None, str]:
    """Return (minority/majority ratio in (0, 1], note).

    FORGE's profiler reports ``class_imbalance_ratio`` as min/max, so a value in
    (0, 1]. A caller passing "9:1" style ratios (>1) is inverted rather than
    silently misread as balanced.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None, ""
    value = float(raw)
    if value <= 0:
        return None, ""
    if value > 1.0:
        return 1.0 / value, f" (given as {value:g}:1, read as minority share {1.0 / value:.3g})"
    return value, ""


def _cost_skew(cost_ratio: Any) -> tuple[float, float, float]:
    """Return (fp, fn, skew) where skew >= 1 is max(fp, fn) / min(fp, fn)."""
    fp, fn = 1.0, 1.0
    if isinstance(cost_ratio, dict):
        raw_fp, raw_fn = cost_ratio.get("fp"), cost_ratio.get("fn")
        if isinstance(raw_fp, (int, float)) and not isinstance(raw_fp, bool) and raw_fp > 0:
            fp = float(raw_fp)
        if isinstance(raw_fn, (int, float)) and not isinstance(raw_fn, bool) and raw_fn > 0:
            fn = float(raw_fn)
    skew = max(fp, fn) / min(fp, fn)
    return fp, fn, skew


def recommend_metric(spec: TaskSpec, data_facts: dict) -> tuple[str, str]:
    """Pick the optimisation metric from language context plus measured data facts.

    Args:
        spec: Parsed language context.
        data_facts: Plain dict from the profiler layer (this module deliberately
            does not import the profiler)::

                {"task_type": "classification" | "regression",
                 "is_binary": bool,
                 "imbalance_ratio": float | None,   # minority/majority, (0, 1]
                 "n_rows": int | None}

    Returns:
        ``(metric_name, rationale)``. The rationale always names the evidence it
        used, and says so explicitly when a fact was missing.
    """
    facts = data_facts if isinstance(data_facts, dict) else {}
    task_type = facts.get("task_type")
    is_binary = bool(facts.get("is_binary"))
    n_rows = facts.get("n_rows")
    n_rows = n_rows if isinstance(n_rows, int) and not isinstance(n_rows, bool) else None
    imbalance, imbalance_note = _normalized_imbalance(facts.get("imbalance_ratio"))
    is_regression = task_type == "regression"

    # 1. Capacity dominates everything: only the top-k slice is ever acted on.
    capacity = spec.capacity_k.value
    if isinstance(capacity, int) and not isinstance(capacity, bool) and capacity > 0:
        parts = [
            f"a capacity of {capacity} was stated in the description "
            f"({spec.capacity_k.source})"
        ]
        if n_rows:
            share = 100.0 * capacity / n_rows
            if capacity >= n_rows:
                parts.append(
                    f"note k={capacity} is >= the {n_rows} rows profiled, so "
                    f"precision@k degenerates to plain precision - check the number"
                )
            else:
                parts.append(f"k is {share:.3g}% of the {n_rows} rows profiled")
        else:
            parts.append("row count was not supplied, so k could not be sanity-checked")
        if is_regression:
            parts.append(
                "the profiled target is numeric, so this scores the top-k selected "
                "by predicted value"
            )
        return "precision_at_k", (
            "precision_at_k: " + "; ".join(parts) + ". Aggregate metrics over the full "
            "test set would score predictions nobody will ever act on."
        )

    # 2. Regression.
    if is_regression:
        if _OUTLIER_ROBUST.search(spec.raw_text.lower()):
            match = _OUTLIER_ROBUST.search(spec.raw_text.lower())
            quote = match.group(0) if match else ""
            return "mae", (
                f'mae: the profiled task is regression and the description says '
                f'"{quote}", which asks for robustness to extreme values. RMSE '
                f"squares errors and would let a few outliers dominate model choice."
            )
        forecast_note = (
            " The description is a forecast, so validate with a time-ordered split "
            "as well." if spec.is_forecast.value else ""
        )
        return "rmse", (
            "rmse: the profiled task is regression and nothing in the description "
            "asks for outlier robustness, so squared error is the default and large "
            "misses are penalised heavily." + forecast_note
        )

    # 3. Strong cost asymmetry on a classification task.
    fp, fn, skew = _cost_skew(spec.cost_ratio.value)
    if skew >= ASYMMETRY_METRIC_THRESHOLD:
        beta = (fn / fp) ** 0.5
        favoured = "recall" if fn > fp else "precision"
        threshold = cost_sensitive_threshold({"fp": fp, "fn": fn})
        return "fbeta", (
            f"fbeta (beta={beta:.3g}, weighting {favoured}): the description gives "
            f"fp={fp:g} / fn={fn:g} ({spec.cost_ratio.source}), a {skew:g}x asymmetry, "
            f"so a symmetric metric would optimise the wrong error. Decision "
            f"threshold {threshold:.3g}. Rationale for the ratio: "
            f"{spec.cost_ratio.rationale}"
        )

    # 4. Imbalance-driven choice for binary classification.
    if is_binary:
        if imbalance is None:
            return "roc_auc", (
                "roc_auc: the profiled target is binary but no imbalance ratio was "
                "supplied, and the description states no cost asymmetry or capacity. "
                "ROC-AUC is threshold-free, so it is the safe choice until the class "
                "balance is known."
            )
        if imbalance < SEVERE_IMBALANCE:
            return "pr_auc", (
                f"pr_auc: the profiled target is binary with a minority share of "
                f"{imbalance:.3g}{imbalance_note}, below {SEVERE_IMBALANCE:g}. Under "
                f"severe imbalance ROC-AUC looks good even for a useless model, while "
                f"PR-AUC tracks the minority class."
            )
        if imbalance < MODERATE_IMBALANCE:
            return "roc_auc", (
                f"roc_auc: the profiled target is binary and imbalanced (minority "
                f"share {imbalance:.3g}{imbalance_note}, below {MODERATE_IMBALANCE:g}), "
                f"so accuracy would reward always predicting the majority class. "
                f"ROC-AUC is threshold-free."
            )
        if imbalance < BALANCED_IMBALANCE:
            return "f1", (
                f"f1: the profiled target is binary with mild imbalance (minority "
                f"share {imbalance:.3g}{imbalance_note}), and the description states "
                f"no cost asymmetry, so precision and recall are balanced equally."
            )
        return "accuracy", (
            f"accuracy: the profiled target is binary and close to balanced (minority "
            f"share {imbalance:.3g}{imbalance_note}, at or above "
            f"{BALANCED_IMBALANCE:g}), and the description states no cost asymmetry, "
            f"capacity limit or ranking need, so plain accuracy is honest here."
        )

    # 5. Multiclass, or task type not supplied.
    if task_type == "classification":
        return "f1_macro", (
            "f1_macro: the profiled target is multiclass and the description states "
            "no cost asymmetry or capacity limit, so macro-F1 weights every class "
            "equally instead of letting the largest class dominate."
        )
    return "f1_macro", (
        f"f1_macro: task_type was {task_type!r} rather than 'classification' or "
        f"'regression', so the data facts could not narrow the choice; macro-F1 is a "
        f"conservative default. Supply task_type from the profiler for a real "
        f"recommendation."
    )


# --------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------


def cost_sensitive_threshold(cost_ratio: dict) -> float:
    """Bayes-optimal decision threshold for a calibrated probability.

    Predicting positive is worth it when ``p * c_fn >= (1 - p) * c_fp``, which
    solves to ``p >= c_fp / (c_fp + c_fn)``. Symmetric costs give 0.5.

    Unusable input (not a dict, missing keys, non-numeric, non-positive, both
    zero) returns the neutral 0.5 rather than raising, since this is called from
    request handlers. Values are validated explicitly, not via a blanket catch.
    """
    if not isinstance(cost_ratio, dict):
        return 0.5

    def positive(raw: Any) -> float | None:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        return value if value > 0 else None

    fp = positive(cost_ratio.get("fp"))
    fn = positive(cost_ratio.get("fn"))
    if fp is None or fn is None:
        return 0.5
    if fp == fn:
        return 0.5
    return fp / (fp + fn)
