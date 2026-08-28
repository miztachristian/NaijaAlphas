"""
Gamble Punt Sleeve Screener
===========================
Ranks NGX micro and low-cap equities (market cap < N20B) by a composite
"punt score" tuned for asymmetric speculative bets. Designed to sit
alongside `hidden_gems` (quality screen) and the seasonal-momentum sleeve.

Spec: docs/superpowers/specs/2026-05-24-gamble-punt-sleeve-design.md

The score is NOT a quality score. Money-losing companies with accelerating
revenue can score well; profitable ones with shrinking top lines score poorly.
All scoring rules and postmortem-derived guardrails are stated explicitly in
this file so they can be audited and unit-tested.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from core.models import PuntCard

logger = logging.getLogger(__name__)

# ── Tier thresholds (Naira) ──────────────────────────────────────────
MICRO_CAP_CEILING = 5_000_000_000       # < N5B
LOW_CAP_CEILING = 20_000_000_000        # N5B .. N20B

# ── Liquidity floors (naira-turnover/day) ────────────────────────────
LIQUIDITY_FLOOR_LOW = 500_000           # low tier
LIQUIDITY_FLOOR_MICRO = 200_000         # micro tier
MICRO_VOLUME_BUFFER_MULTIPLE = 2.0      # micro penalty if volume < 2x floor

# ── Component weights (sum = 100) ────────────────────────────────────
WEIGHT_SETUP = 30
WEIGHT_CATALYST = 25
WEIGHT_INSIDER = 20
WEIGHT_BUSINESS_PULSE = 15
WEIGHT_LIQUIDITY = 10

# ── State thresholds ─────────────────────────────────────────────────
SCORE_AVOID_CEILING = 30                # score < this AND no hard-warning = AVOID
SCORE_WATCH_CEILING = 50                # WATCH = 30..49
SCORE_SETUP_CEILING = 70                # SETUP = 50..69
                                        # TRIGGER >= 70 (plus buy_zone, no fresh red flag)

MAX_ACTIONABLE_TRIGGERS = 6             # remaining are "Trigger-bench"

# ── Universe + tier classification ───────────────────────────────────

def classify_tier(market_cap: float) -> Optional[str]:
    """Return 'micro' for <N5B, 'low' for N5B-N20B, None otherwise."""
    if pd.isna(market_cap):
        return None
    if market_cap < MICRO_CAP_CEILING:
        return "micro"
    if market_cap < LOW_CAP_CEILING:
        return "low"
    return None


def build_universe(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter snapshot to punt-eligible tickers and tag each with its tier.

    Returns a DataFrame copy with the original columns plus a 'tier' column,
    sorted by ascending market cap.
    """
    out = snapshot_df.copy()
    out["tier"] = out["market_cap"].apply(classify_tier)
    out = out[out["tier"].notna()].copy()
    return out.sort_values("market_cap").reset_index(drop=True)


# ── Liquidity gate ───────────────────────────────────────────────────

def naira_turnover_1d(row: pd.Series) -> float:
    """Naira turnover for the most recent trading day. Zero on missing data."""
    vol = row.get("volume_1d")
    price = row.get("price")
    if pd.isna(vol) or pd.isna(price):
        return 0.0
    return float(vol) * float(price)


def liquidity_gate(row: pd.Series, tier: str) -> tuple[bool, list[str]]:
    """
    Apply the per-tier liquidity floor.

    Returns (passes, warnings). `passes=False` means the ticker is forced to
    AVOID regardless of score. The micro tier emits a 'micro_thin_volume'
    warning when turnover is above floor but below 2x floor.
    """
    turnover = naira_turnover_1d(row)
    floor = LIQUIDITY_FLOOR_MICRO if tier == "micro" else LIQUIDITY_FLOOR_LOW
    warnings: list[str] = []

    if turnover < floor:
        warnings.append("below_liquidity_floor")
        return False, warnings

    if tier == "micro" and turnover < floor * MICRO_VOLUME_BUFFER_MULTIPLE:
        warnings.append("micro_thin_volume")

    return True, warnings


# ── Hard-warning detection ───────────────────────────────────────────

RESULTS_OVERDUE_DAYS = 150


def detect_hard_warnings(
    row: pd.Series,
    disclosure_signals: dict,
    statement_history: list[dict],
) -> list[str]:
    """
    Return warning tags that force AVOID regardless of score.

    `statement_history` is a chronological list (oldest first) of parsed
    financial statements for the ticker, each a dict with at least
    `total_equity` and `filed_date`.

    Tags returned:
        dead_tape, late_filer, results_overdue,
        neg_equity_worsening, negative_equity, neg_equity_trend_unknown
    Of these, the first four are auto-AVOID. The last two are advisory.
    """
    warnings: list[str] = []

    rsi = row.get("rsi_14")
    if pd.notna(rsi) and (rsi >= 99.99 or rsi <= 0.01):
        warnings.append("dead_tape")

    if disclosure_signals.get("late_filer"):
        warnings.append("late_filer")

    days_since = disclosure_signals.get("days_since_results")
    if days_since is not None and days_since > RESULTS_OVERDUE_DAYS:
        warnings.append("results_overdue")

    # Negative-equity trend logic
    neg_history = [
        s for s in statement_history
        if s.get("total_equity") is not None and s["total_equity"] < 0
    ]
    if neg_history:
        if len(neg_history) >= 2:
            newest = neg_history[-1]["total_equity"]
            previous = neg_history[-2]["total_equity"]
            if newest < previous:  # more negative than before
                warnings.append("neg_equity_worsening")
            else:
                warnings.append("negative_equity")
        else:
            warnings.append("neg_equity_trend_unknown")

    return warnings


HARD_AVOID_TAGS = frozenset({
    "dead_tape",
    "late_filer",
    "results_overdue",
    "neg_equity_worsening",
})


# ── Component scorers ────────────────────────────────────────────────


def _rsi_subscore(rsi: float) -> float:
    """Triangular score peaking at RSI 45, zero at <=20 or >=80. Range [0,1]."""
    if pd.isna(rsi):
        return 0.5
    if rsi <= 20 or rsi >= 80:
        return 0.0
    if rsi <= 45:
        return (rsi - 20) / 25
    return (80 - rsi) / 35


def _ema_distance_subscore(price: float, ema: float) -> float:
    """
    Score peaks at 1.0 when price is 2-5% above ema (ideal entry zone).
    Returns 0.5 when price == ema (neutral — trend unconfirmed).
    Fades to 0 when price is >30% away in either direction.
    Range [0,1].
    """
    if pd.isna(price) or pd.isna(ema) or ema == 0:
        return 0.5
    ratio = (price - ema) / ema   # signed: positive = above ema
    if ratio >= 0.30 or ratio <= -0.30:
        return 0.0
    if 0.02 <= ratio <= 0.05:
        return 1.0
    if ratio > 0.05:
        # Fade from 1.0 at 5% to 0 at 30%
        return 1.0 - ((ratio - 0.05) / 0.25)
    if 0 <= ratio < 0.02:
        # Scale 0.5 -> 1.0 as ratio goes 0 -> 0.02
        return 0.5 + (ratio / 0.02) * 0.5
    # ratio < 0: below ema, fade from 0.5 at 0 to 0 at -30%
    return 0.5 + (ratio / 0.30) * 0.5


def _compression_subscore(vol_1m: float) -> float:
    """Lower 1-month volatility = higher score (coiled spring). Range [0,1]."""
    if pd.isna(vol_1m):
        return 0.5
    if vol_1m <= 0.5:
        return 1.0
    if vol_1m >= 5.0:
        return 0.0
    return 1.0 - ((vol_1m - 0.5) / 4.5)


def score_setup(row: pd.Series) -> float:
    """
    Setup (technical entry quality) score in [0, WEIGHT_SETUP].

    Composition (weighted average of four subscores):
        - RSI cool-down (40%)
        - Distance from EMA-50 (25%)
        - Distance from EMA-200 (15%)
        - 1-month price compression (20%)
    """
    rsi_s = _rsi_subscore(row.get("rsi_14"))
    ema50_s = _ema_distance_subscore(row.get("price"), row.get("ema_50"))
    ema200_s = _ema_distance_subscore(row.get("price"), row.get("ema_200"))
    comp_s = _compression_subscore(row.get("vol_1m"))

    weighted = (0.40 * rsi_s + 0.25 * ema50_s + 0.15 * ema200_s + 0.20 * comp_s)
    return float(np.clip(weighted, 0, 1) * WEIGHT_SETUP)


def score_catalyst(disclosure_signals: dict) -> float:
    """
    Catalyst-proximity score in [0, WEIGHT_CATALYST].

    Inputs come from `DisclosureAnalyzer.analyze(ticker)`:
        - earnings_catalyst (bool): board notice or fresh results
        - days_to_likely_earnings (int|None): from board-meeting window
        - days_since_results (int|None): freshness of last statement
        - has_fresh_forecast (bool): earnings-forecast filed recently

    Composition (weights sum to 1.0):
        - 25% imminence: peaks if days_to_likely_earnings in [0,14],
          fades to 0 by 60 days out
        - 45% freshness: peaks if days_since_results <= 14,
          fades to 0 by 130 days
        - 20% catalyst flag: 1.0 if earnings_catalyst else 0
        - 10% forecast bonus: 1.0 if has_fresh_forecast else 0
    """
    days_to = disclosure_signals.get("days_to_likely_earnings")
    days_since = disclosure_signals.get("days_since_results")
    has_catalyst = bool(disclosure_signals.get("earnings_catalyst"))
    has_forecast = bool(disclosure_signals.get("has_fresh_forecast"))

    if days_to is None and days_since is None and not has_catalyst and not has_forecast:
        return 0.0

    # Imminence: peaks at 0..14d to earnings, fades to 0 by 60d
    if days_to is None:
        imminence = 0.0
    elif days_to < 0:
        imminence = 0.0
    elif days_to <= 14:
        imminence = 1.0
    elif days_to >= 60:
        imminence = 0.0
    else:
        imminence = 1.0 - ((days_to - 14) / 46)

    # Freshness: peaks at 0..14d since results, fades to 0 by 130d
    if days_since is None:
        freshness = 0.0
    elif days_since <= 14:
        freshness = 1.0
    elif days_since >= 130:
        freshness = 0.0
    else:
        freshness = 1.0 - ((days_since - 14) / 116)

    catalyst_flag = 1.0 if has_catalyst else 0.0
    forecast_bonus = 1.0 if has_forecast else 0.0

    weighted = (0.25 * imminence + 0.45 * freshness
                + 0.20 * catalyst_flag + 0.10 * forecast_bonus)
    return float(np.clip(weighted, 0, 1) * WEIGHT_CATALYST)


def score_insider(disclosure_signals: dict) -> float:
    """
    Insider-conviction score in [-WEIGHT_INSIDER, WEIGHT_INSIDER].

    ONLY component that can return a negative value. Logic (symmetric):
        - Each buy contributes +WEIGHT_INSIDER / 2.
        - Each sale contributes -WEIGHT_INSIDER / 2.
        - Equal buys and sales cancel to zero (no signal).
        - A pattern of >=3 sales with 0 buys ("sale chain", MCNICHOLS-style)
          forces the component to -WEIGHT_INSIDER regardless of the linear math.
        - Final value clipped to [-WEIGHT_INSIDER, WEIGHT_INSIDER].

    Two buys reaches the positive cap; two sales (with no buys) reaches the
    negative cap via the linear math; three sales triggers the override.
    Mirroring keeps "buy = bullish, sale = bearish" intuitive and avoids
    the asymmetric paradox where 1 buy max-pegs the score but 2 buys + 2 sales
    cannot net to zero.

    `disclosure_signals` must contain `insider_buys_90d` and `insider_sales_90d`.
    """
    buys = int(disclosure_signals.get("insider_buys_90d", 0) or 0)
    sales = int(disclosure_signals.get("insider_sales_90d", 0) or 0)

    if buys == 0 and sales == 0:
        return 0.0

    # Sale-chain override: 3+ sales with no buys = full negative
    if buys == 0 and sales >= 3:
        return float(-WEIGHT_INSIDER)

    half = WEIGHT_INSIDER / 2
    raw = (buys * half) - (sales * half)
    return float(np.clip(raw, -WEIGHT_INSIDER, WEIGHT_INSIDER))


# Postmortem-derived clipping caps (see spec §10)
EPS_GROWTH_CAP = 150     # v2.0 lesson: one-quarter windfalls clipped
NET_MARGIN_CAP = 25      # v2.0 lesson: prevents margin spike domination


def score_business_pulse(row: pd.Series) -> tuple[float, list[str]]:
    """
    Business pulse score in [0, WEIGHT_BUSINESS_PULSE] + warnings list.

    DIRECTION not LEVEL: weights revenue acceleration most heavily so that
    money-losing companies with accelerating top lines can still score well
    (JOHNHOLT pattern), while profitable companies with shrinking revenue
    score poorly.

    Subcomponents (each clipped to [0,1] then weighted):
        - revenue_growth_ttm    (35%)  full credit at +50%, zero at <= -25%
        - revenue_growth_qoq    (25%)  full credit at +20%, zero at <= -20%
        - eps_growth_ttm (capped at +150)  (15%)  full credit at +100%, zero at <= -50%
        - fcf_growth_ttm                   (15%)  full credit at +50%, zero at <= -50%
        - net_margin_ttm (capped at +25)   (10%)  full credit at +20%, zero at <= 0%

    Warnings emitted:
        - fcf_divergence: EPS growth > 50% AND fcf_growth < 0 (MCNICHOLS pattern).
          Triggers a -5 score adjustment via the postmortem rules in Task 10.

    Weight adjustments from spec (all 7 tests pass with these exact values):
        No adjustments needed; weights match the spec 35/25/15/15/10 exactly.
    """
    def _normalise(value, full_credit_at, zero_below):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0.5
        try:
            if pd.isna(value):
                return 0.5
        except (TypeError, ValueError):
            pass
        if value <= zero_below:
            return 0.0
        if value >= full_credit_at:
            return 1.0
        return (value - zero_below) / (full_credit_at - zero_below)

    rev_ttm = row.get("revenue_growth_ttm")
    rev_qoq = row.get("revenue_growth_quarterly_qoq")
    eps = row.get("eps_growth_ttm")
    fcf = row.get("fcf_growth_ttm")
    margin = row.get("net_margin_ttm")

    eps_capped = None if (eps is None or (isinstance(eps, float) and pd.isna(eps))) else min(eps, EPS_GROWTH_CAP)
    margin_capped = None if (margin is None or (isinstance(margin, float) and pd.isna(margin))) else min(margin, NET_MARGIN_CAP)

    rev_ttm_s = _normalise(rev_ttm, full_credit_at=50, zero_below=-25)
    rev_qoq_s = _normalise(rev_qoq, full_credit_at=20, zero_below=-20)
    eps_s = _normalise(eps_capped, full_credit_at=100, zero_below=-50)
    fcf_s = _normalise(fcf, full_credit_at=50, zero_below=-50)
    margin_s = _normalise(margin_capped, full_credit_at=20, zero_below=0)

    weighted = (0.35 * rev_ttm_s + 0.25 * rev_qoq_s + 0.15 * eps_s
                + 0.15 * fcf_s + 0.10 * margin_s)
    score = float(np.clip(weighted, 0, 1) * WEIGHT_BUSINESS_PULSE)

    warnings: list[str] = []
    eps_val = row.get("eps_growth_ttm")
    fcf_val = row.get("fcf_growth_ttm")
    if (eps_val is not None and fcf_val is not None
            and not (isinstance(eps_val, float) and pd.isna(eps_val))
            and not (isinstance(fcf_val, float) and pd.isna(fcf_val))
            and eps_val > 50 and fcf_val < 0):
        warnings.append("fcf_divergence")

    return score, warnings


def score_liquidity(row: pd.Series, tier: str) -> float:
    """
    Liquidity score in [0, WEIGHT_LIQUIDITY]. Linear from floor (=0) to
    10x floor (=full weight). The hard gate is separate (`liquidity_gate`).
    """
    floor = LIQUIDITY_FLOOR_MICRO if tier == "micro" else LIQUIDITY_FLOOR_LOW
    turnover = naira_turnover_1d(row)
    if turnover < floor:
        return 0.0
    ratio = turnover / (floor * 10)  # 0..10x floor → 0..1 (below-floor gated above)
    return float(np.clip(ratio, 0, 1) * WEIGHT_LIQUIDITY)


# ── Composite + postmortem rules ─────────────────────────────────────


def composite_score(components: dict) -> float:
    """Sum the five component scores, clipping the total to >= 0."""
    total = sum(components.values())
    return float(max(0.0, total))


def apply_postmortem_rules(
    row: pd.Series,
    base_score: float,
    base_warnings: list[str],
    disclosure_signals: dict,
) -> tuple[float, list[str], Optional[str]]:
    """
    Apply v2.0/v2.1/conversation postmortem rules from spec §10.

    Returns (adjusted_score, all_warnings, state_cap):
        - adjusted_score: base_score adjusted for -5 penalties from
          fcf_divergence, value_trap, unsupported_pe (each -5, additive).
        - all_warnings: base + any newly-emitted tags.
        - state_cap: "AVOID" if a hard postmortem rule fires,
          "SETUP" if the YTD-extension guard fires, else None.
    """
    warnings = list(base_warnings)
    score = float(base_score)
    state_cap: Optional[str] = None

    # YTD-extension guard
    perf_ytd = row.get("perf_ytd")
    price = row.get("price")
    ema_200 = row.get("ema_200")
    if (pd.notna(perf_ytd) and pd.notna(price) and pd.notna(ema_200)
            and ema_200 > 0 and perf_ytd > 100 and price > 2.0 * ema_200):
        warnings.append("ytd_extension")
        state_cap = "SETUP"

    # Falling-knife block (insider buy unlocks)
    perf_3m = row.get("perf_3m")
    insider_buys = int(disclosure_signals.get("insider_buys_90d", 0) or 0)
    if pd.notna(perf_3m) and perf_3m < -30 and insider_buys == 0:
        warnings.append("falling_knife")
        state_cap = "AVOID"

    # Unsupported P/E
    pe = row.get("pe_ratio")
    eps_g = row.get("eps_growth_ttm")
    rev_g = row.get("revenue_growth_ttm")
    if (pd.notna(pe) and pe > 40 and pd.notna(eps_g) and pd.notna(rev_g)
            and eps_g > rev_g):
        warnings.append("unsupported_pe")
        score -= 5

    # -5 penalties for tags already on the warning list
    for tag in ("fcf_divergence", "value_trap"):
        if tag in base_warnings:
            score -= 5

    score = max(0.0, score)
    return score, warnings, state_cap


# ── State machine ────────────────────────────────────────────────────


def derive_state(
    score: float,
    hard_warnings: list[str],
    buy_zone,
    price: float,
    fresh_red_flag: bool,
    state_cap: Optional[str],
) -> str:
    """
    Derive the operational state per spec §5.

    Order of precedence:
        1. Any hard-AVOID tag in hard_warnings -> AVOID
        2. state_cap == "AVOID" -> AVOID
        3. score < SCORE_AVOID_CEILING -> AVOID
        4. state_cap == "SETUP" -> SETUP  (postmortem YTD-extension)
        5. score < SCORE_WATCH_CEILING -> WATCH
        6. score < SCORE_SETUP_CEILING -> SETUP
        7. TRIGGER requires: numeric buy_zone, price within, no fresh red flag
        8. Otherwise SETUP.
    """
    if any(t in HARD_AVOID_TAGS for t in hard_warnings):
        return "AVOID"
    if state_cap == "AVOID":
        return "AVOID"
    if score < SCORE_AVOID_CEILING:
        return "AVOID"
    if state_cap == "SETUP":
        return "SETUP"
    if score < SCORE_WATCH_CEILING:
        return "WATCH"
    if score < SCORE_SETUP_CEILING:
        return "SETUP"

    # TRIGGER candidates: need numeric buy_zone and price within
    is_deferred = (isinstance(buy_zone, tuple) and len(buy_zone) == 2
                   and isinstance(buy_zone[0], str))
    if is_deferred:
        return "SETUP"
    if buy_zone is None:
        return "SETUP"
    low, high = buy_zone
    if not (low <= price <= high):
        return "SETUP"
    if fresh_red_flag:
        return "SETUP"
    return "TRIGGER"


# ── Buy zone & stop ──────────────────────────────────────────────────


def calculate_buy_zone(row: pd.Series, disclosure_signals: dict):
    """
    Return a buy zone per spec §6 priority order:
      1. Insider-anchored: (insider_price, insider_price * 1.06)
      2. Extended-RSI defer: ("deferred", "wait for retrace to EMA-50 +/-3%")
      3. Default mean-reversion: (max(EMA-50, EMA-200) * 0.97, EMA-50 * 1.03)
      4. None if EMAs unavailable.
    """
    insider_price = disclosure_signals.get("latest_insider_buy_price")
    insider_buys = int(disclosure_signals.get("insider_buys_90d", 0) or 0)
    if insider_buys > 0 and insider_price and not pd.isna(insider_price):
        return (float(insider_price), float(insider_price) * 1.06)

    rsi = row.get("rsi_14")
    if pd.notna(rsi) and rsi > 75:
        return ("deferred", "wait for retrace to EMA-50 +/-3%")

    ema50 = row.get("ema_50")
    ema200 = row.get("ema_200")
    if pd.isna(ema50) or pd.isna(ema200):
        return None
    anchor = max(float(ema50), float(ema200))
    return (anchor * 0.97, float(ema50) * 1.03)


HARD_STOP_MAX_DRAWDOWN = 0.15  # never lose more than 15% from entry


def calculate_stop(buy_zone, row: pd.Series) -> Optional[float]:
    """
    Stop = highest of:
      - buy_zone_low - 1.5 * 30d_ATR
      - EMA-200 * 0.95
      - buy_zone_low * (1 - HARD_STOP_MAX_DRAWDOWN)   (hard floor)

    Highest = least painful exit; we cap the loss.
    Returns None if buy_zone is None or deferred.
    """
    if buy_zone is None:
        return None
    if isinstance(buy_zone[0], str):
        return None

    zone_low = float(buy_zone[0])
    atr = row.get("atr_30d")
    ema_200 = row.get("ema_200")

    candidates = [zone_low * (1 - HARD_STOP_MAX_DRAWDOWN)]
    if pd.notna(atr):
        candidates.append(zone_low - 1.5 * float(atr))
    if pd.notna(ema_200):
        candidates.append(float(ema_200) * 0.95)

    return float(max(candidates))


# ── Title-keyword direction inference ────────────────────────────────
#
# DisclosureAnalyzer reports an undifferentiated insider_count_90d for
# director-dealings because most NGX insider PDFs are scanned images and it
# parses metadata only. Title-keyword inference recovers buy/sale direction
# for the common patterns. It will NOT catch generic titles like
# "Notification of Insider Share Dealing" — those need PDF-body parsing.

_SALE_KEYWORDS = (
    " SALES ", " SALES,", " SALES.", " SALE ", "SALE OF", "SOLD ",
    "DISPOSAL", "DISPOSED",
)
_BUY_KEYWORDS = (
    " PURCHASE", "PURCHASED", "BOUGHT", "ACQUISITION OF SHARES",
    "ACQUIRED ", "PURCHASE OF",
)


def infer_insider_direction(title: str) -> str:
    """Classify an NGX director-dealings disclosure title as 'buy', 'sale',
    or 'unknown'. Case-insensitive; pads with spaces so substrings like
    'SALES' inside 'TOTAL SALES REVENUE' don't false-positive."""
    if not title:
        return "unknown"
    padded = f" {title.upper()} "
    if any(kw in padded for kw in _SALE_KEYWORDS):
        return "sale"
    if any(kw in padded for kw in _BUY_KEYWORDS):
        return "buy"
    return "unknown"


def _row_text_for_direction(row: dict) -> str:
    """Combined searchable text for a disclosure row: title + filename slug.

    NGX titles are usually generic ('XYZ PLC DIRECTORSDEALINGS') but the
    descriptive keyword (e.g. 'SALES', 'PURCHASE') is often in the PDF
    filename slug. We strip the URL down to the file basename and replace
    underscores with spaces so the keyword tokens match cleanly.
    """
    title = row.get("title") or ""
    url = row.get("pdf_url") or ""
    filename = ""
    if url:
        # Take everything after the last '/' and drop any '.pdf' suffix.
        tail = url.rsplit("/", 1)[-1]
        if "." in tail:
            tail = tail.rsplit(".", 1)[0]
        filename = tail.replace("_", " ").replace("-", " ")
    return f"{title} {filename}".strip()


def normalise_with_raw_rows(signals: dict, dd_rows_90d: list[dict]) -> dict:
    """Augment a normalised disclosure-signals dict with `insider_buys_90d`
    and `insider_sales_90d` derived from raw director-dealings rows. Looks
    at both the row's `title` and its `pdf_url` filename slug — NGX often
    encodes direction in the filename, not the title. Overwrites any
    existing buy/sale counts on the input."""
    buys = 0
    sales = 0
    for row in dd_rows_90d:
        if isinstance(row, dict):
            text = _row_text_for_direction(row)
        else:
            text = str(row)
        d = infer_insider_direction(text)
        if d == "buy":
            buys += 1
        elif d == "sale":
            sales += 1
    out = dict(signals)
    out["insider_buys_90d"] = buys
    out["insider_sales_90d"] = sales
    # latest_insider_buy_price needs PDF-body parsing — leave None for v1.
    out.setdefault("latest_insider_buy_price", None)
    return out


def _row_within_window(iso_str, cutoff_dt) -> bool:
    """True iff an ISO timestamp string represents a moment >= cutoff_dt."""
    if not iso_str:
        return False
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return False
    return dt >= cutoff_dt


# ── Orchestrator ─────────────────────────────────────────────────────


class GamblePuntScreener:
    """
    Coordinates universe filtering, per-ticker scoring, postmortem rule
    application, state derivation, buy zone/stop calculation, and assembly
    of PuntCard outputs.

    Two construction paths:
      - GamblePuntScreener(snapshot_date='2026-05-22')  -- production: loads
        snapshot, disclosures, statements from disk.
      - GamblePuntScreener.from_dataframes(...)         -- testing: takes
        pre-built DataFrames and dicts so tests don't touch disk.
    """

    def __init__(self, snapshot_date: Optional[str] = None):
        if snapshot_date is None:
            from analysis.snapshot_store import SnapshotStore
            snapshot_date = SnapshotStore().get_latest_snapshot_date()
        self.snapshot_date = snapshot_date

        # Load via existing analyzers (deferred imports keep tests light)
        self._snapshot = self._load_snapshot()
        self._disclosure_signals = self._load_disclosure_signals()
        self._statement_histories = self._load_statement_histories()

    @classmethod
    def from_dataframes(
        cls,
        snapshot: pd.DataFrame,
        disclosure_signals: dict[str, dict],
        statement_histories: dict[str, list[dict]],
    ) -> "GamblePuntScreener":
        """Construct without disk I/O — used by unit tests."""
        screener = cls.__new__(cls)
        screener.snapshot_date = "test"
        screener._snapshot = snapshot
        screener._disclosure_signals = disclosure_signals
        screener._statement_histories = statement_histories
        return screener

    # ── private loaders (production path) ────────────────────────────

    def _load_snapshot(self) -> pd.DataFrame:
        from pathlib import Path
        path = Path(f"data/snapshots/{self.snapshot_date}/snapshot.parquet")
        return pd.read_parquet(path)

    def _load_disclosure_signals(self) -> dict[str, dict]:
        from analysis.disclosure_analyzer import DisclosureAnalyzer
        from datetime import datetime, timezone, timedelta
        da = DisclosureAnalyzer()
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        out: dict[str, dict] = {}
        universe = build_universe(self._snapshot)
        for ticker in universe["symbol"]:
            signals = da.analyze(ticker)
            normalised = self._normalise_disclosure_signals(signals)
            # DisclosureAnalyzer's public API exposes only a total event count
            # for director-dealings; it doesn't split buys from sales (most NGX
            # insider PDFs are scanned images, so it parses metadata only). Reach
            # into the analyzer's per-ticker cache to read the raw titles and
            # infer direction from keywords. This is what makes the MCNICHOLS
            # sale-chain auto-AVOID actually fire on real data.
            raw_rows = da._by_ticker.get(ticker, [])
            dd_rows_90d = [
                r for r in raw_rows
                if r.get("type") == "DirectorsDealings"
                and _row_within_window(r.get("created"), cutoff)
            ]
            out[ticker] = normalise_with_raw_rows(normalised, dd_rows_90d)
        return out

    @staticmethod
    def _normalise_disclosure_signals(signals: dict) -> dict:
        """
        Map DisclosureAnalyzer.analyze() output to the keys this module
        consumes. Adds derived `insider_buys_90d`, `insider_sales_90d`,
        `latest_insider_buy_price`, `next_catalyst`.

        DisclosureAnalyzer v1 returns a flat dict with `insider_count_90d`
        (all directors-dealings events, no buy/sell split) and `insider_score`
        (0..1 float). We map `insider_count_90d` -> `insider_buys_90d` as the
        best available signal; `insider_sales_90d` stays 0 because the current
        disclosure pipeline does not parse individual transaction directions from
        the PDF text. The sale-chain AVOID rule for MCNICHOLS-style patterns
        therefore requires a future PDF-parsing upgrade to take effect against
        real data. Unit tests still exercise it via synthetic `from_dataframes`
        construction, which injects the split counts directly.
        """
        if not isinstance(signals, dict):
            signals = {}
        # Support both the legacy nested-insider shape (unit-test synthetic dicts)
        # and the flat DisclosureAnalyzer output.
        nested = signals.get("insider")
        if isinstance(nested, dict):
            # Synthetic / future enhanced format: explicit buy/sell split
            buys_count = nested.get("buys_count", 0) or 0
            sales_count = nested.get("sales_count", 0) or 0
            latest_buy_price = nested.get("latest_buy_price")
        else:
            # Real DisclosureAnalyzer output: only total count available
            buys_count = int(signals.get("insider_count_90d", 0) or 0)
            sales_count = 0
            latest_buy_price = None
        return {
            "earnings_catalyst": signals.get("earnings_catalyst", False),
            "days_to_likely_earnings": signals.get("days_to_likely_earnings"),
            "days_since_results": signals.get("days_since_results"),
            "has_fresh_forecast": signals.get("has_fresh_forecast", False),
            "late_filer": signals.get("late_filer", False),
            "insider_buys_90d": buys_count,
            "insider_sales_90d": sales_count,
            "latest_insider_buy_price": latest_buy_price,
            "next_catalyst": signals.get("next_catalyst", ""),
        }

    def _load_statement_histories(self) -> dict[str, list[dict]]:
        # Lazy import to avoid pulling pypdf in unit tests
        try:
            from ingest.parse_statements import parse_statements
        except ImportError:
            return {}
        universe = build_universe(self._snapshot)
        tickers = list(universe["symbol"])
        try:
            latest = parse_statements(tickers=tickers, use_cache=True)
        except Exception as exc:
            logger.warning("statement parse failed: %s", exc)
            return {}
        # parse_statements returns one record per ticker; for trend
        # detection we'd need history. v1: wrap latest as single-element list.
        return {t: [r] for t, r in latest.items() if r}

    # ── evaluation ───────────────────────────────────────────────────

    def evaluate(self, ticker: str) -> PuntCard:
        """Score and stage a single ticker."""
        universe = build_universe(self._snapshot)
        rows = universe[universe["symbol"] == ticker]
        if rows.empty:
            raise ValueError(f"{ticker} not in punt universe")
        row = rows.iloc[0]
        return self._evaluate_row(row)

    def evaluate_universe(self) -> list[PuntCard]:
        """Score every punt-eligible ticker and return ordered list."""
        universe = build_universe(self._snapshot)
        cards = [self._evaluate_row(row) for _, row in universe.iterrows()]
        cards.sort(key=lambda c: c.score, reverse=True)
        return cards

    def _evaluate_row(self, row: pd.Series) -> PuntCard:
        ticker = row["symbol"]
        tier = row["tier"]
        disclosure_signals = self._disclosure_signals.get(ticker, {})
        statement_history = self._statement_histories.get(ticker, [])

        # 1. Liquidity gate (also produces warnings)
        passes_liq, liq_warnings = liquidity_gate(row, tier=tier)

        # 2. Hard warnings (dead tape, late filer, results overdue, neg equity)
        hard_warnings = detect_hard_warnings(row, disclosure_signals, statement_history)

        # 3. Component scoring
        setup = score_setup(row)
        catalyst = score_catalyst(disclosure_signals)
        insider = score_insider(disclosure_signals)
        business_pulse, bp_warnings = score_business_pulse(row)
        liquidity = score_liquidity(row, tier=tier)
        components = {"setup": setup, "catalyst": catalyst, "insider": insider,
                      "business_pulse": business_pulse, "liquidity": liquidity}

        base_score = composite_score(components)

        # 4. Advisory: hidden_gems helpers (warnings only, scored in postmortem)
        advisory_warnings: list[str] = []
        try:
            from analysis.hidden_gems import _detect_value_trap
            if _detect_value_trap(row):
                advisory_warnings.append("value_trap")
        except Exception:
            pass

        base_warnings = sorted(set(
            liq_warnings + hard_warnings + bp_warnings + advisory_warnings
        ))

        # 5. Apply postmortem rules
        adjusted_score, all_warnings, state_cap = apply_postmortem_rules(
            row, base_score=base_score,
            base_warnings=base_warnings,
            disclosure_signals=disclosure_signals,
        )

        # 6. Force AVOID-equivalent score if liquidity gate failed
        if not passes_liq:
            adjusted_score = 0.0
            state_cap = "AVOID"

        # 7. Buy zone + stop
        buy_zone = calculate_buy_zone(row, disclosure_signals)
        stop = calculate_stop(buy_zone, row)

        # 8. Derive state
        fresh_red_flag = "fcf_divergence" in all_warnings or "unsupported_pe" in all_warnings
        state = derive_state(
            score=adjusted_score, hard_warnings=all_warnings,
            buy_zone=buy_zone, price=float(row["price"]),
            fresh_red_flag=fresh_red_flag, state_cap=state_cap,
        )

        # 9. Build human-readable strings
        catalyst_str = disclosure_signals.get("next_catalyst") or "no scheduled catalyst"
        next_change = _build_next_rating_change(state, all_warnings, buy_zone, row)

        return PuntCard(
            ticker=ticker, tier=tier,
            score=round(adjusted_score, 1),
            state=state,
            component_scores={k: round(v, 1) for k, v in components.items()},
            buy_zone=buy_zone, stop=stop,
            catalyst=catalyst_str,
            next_rating_change=next_change,
            warnings=all_warnings,
            evidence={
                "price": float(row.get("price", np.nan)),
                "rsi_14": float(row.get("rsi_14", np.nan)),
                "ema_50": float(row.get("ema_50", np.nan)),
                "ema_200": float(row.get("ema_200", np.nan)),
                "perf_ytd": float(row.get("perf_ytd", np.nan)),
                "perf_3m": float(row.get("perf_3m", np.nan)),
                "insider_buys_90d": disclosure_signals.get("insider_buys_90d", 0),
                "insider_sales_90d": disclosure_signals.get("insider_sales_90d", 0),
            },
        )


# ── Run log + diff ───────────────────────────────────────────────────

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PUNT_RUNS_DIR = Path("data/punt_runs")
BIG_SCORE_MOVE_THRESHOLD = 10


def write_run_log(cards: list[PuntCard], snapshot_date: str,
                  out_dir: Path = PUNT_RUNS_DIR) -> Path:
    """Persist this run's cards to data/punt_runs/<snapshot_date>_<ts>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%H%M%S")
    path = out_dir / f"{snapshot_date}_{ts}.json"

    def _normalise(value):
        if isinstance(value, tuple):
            return list(value)
        return value

    payload = {
        "snapshot_date": snapshot_date,
        "run_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "cards": [
            {**asdict(c), "buy_zone": _normalise(c.buy_zone)}
            for c in cards
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def load_previous_run(out_dir: Path = PUNT_RUNS_DIR) -> Optional[dict]:
    """Load the most recent prior run JSON, or None if none exists."""
    if not out_dir.exists():
        return None
    files = sorted(out_dir.glob("*.json"))
    if len(files) < 2:
        return None
    # Second-to-last is the prior run (last is the one we just wrote)
    with open(files[-2], encoding="utf-8") as f:
        return json.load(f)


def diff_runs(prev: Optional[dict], current_cards: list[PuntCard]) -> dict:
    """
    Compare a previous run payload (or None) to current cards.

    Returns dict with keys:
        state_upgrades, state_downgrades, new_triggers, dropped_triggers,
        big_score_moves, note (only present on first run).
    """
    if prev is None:
        return {
            "state_upgrades": [], "state_downgrades": [],
            "new_triggers": [], "dropped_triggers": [],
            "big_score_moves": [],
            "note": "First run — no diff to compute.",
        }

    order = ["AVOID", "WATCH", "SETUP", "TRIGGER"]
    prev_by_ticker = {c["ticker"]: c for c in prev.get("cards", [])}

    state_upgrades: list[tuple] = []
    state_downgrades: list[tuple] = []
    new_triggers: list[str] = []
    dropped_triggers: list[str] = []
    big_moves: list[dict] = []

    for card in current_cards:
        prev_card = prev_by_ticker.get(card.ticker)
        if prev_card is None:
            if card.state == "TRIGGER":
                new_triggers.append(card.ticker)
            continue
        prev_state = prev_card.get("state", "AVOID")
        if order.index(card.state) > order.index(prev_state):
            state_upgrades.append((card.ticker, prev_state, card.state))
        elif order.index(card.state) < order.index(prev_state):
            state_downgrades.append((card.ticker, prev_state, card.state))
        if card.state == "TRIGGER" and prev_state != "TRIGGER":
            new_triggers.append(card.ticker)
        if card.state != "TRIGGER" and prev_state == "TRIGGER":
            dropped_triggers.append(card.ticker)
        delta = card.score - prev_card.get("score", 0)
        if abs(delta) >= BIG_SCORE_MOVE_THRESHOLD:
            big_moves.append({"ticker": card.ticker, "delta": round(delta, 1),
                              "from": prev_card.get("score"), "to": card.score})

    # Tickers that vanished entirely (e.g., delisted) -> dropped triggers
    current_tickers = {c.ticker for c in current_cards}
    for ticker, prev_card in prev_by_ticker.items():
        if ticker not in current_tickers and prev_card.get("state") == "TRIGGER":
            dropped_triggers.append(ticker)

    return {
        "state_upgrades": state_upgrades,
        "state_downgrades": state_downgrades,
        "new_triggers": new_triggers,
        "dropped_triggers": dropped_triggers,
        "big_score_moves": big_moves,
    }


# ── Dossier rendering ────────────────────────────────────────────────

FORBIDDEN_FRAMING_WORDS = frozenset({
    "cheap", "discount", "bargain", "safe",
})


def _format_buy_zone(zone) -> str:
    if zone is None:
        return "—"
    if isinstance(zone, tuple) and isinstance(zone[0], str):
        return zone[1]
    low, high = zone
    return f"{low:.2f}–{high:.2f}"


def render_master_table_row(card: "PuntCard") -> str:
    """Single pipe-delimited row for the master table."""
    bz = _format_buy_zone(card.buy_zone)
    stop = f"{card.stop:.2f}" if card.stop is not None else "—"
    price = card.evidence.get("price", 0.0)
    return (f"| {card.ticker} | {card.tier} | {card.state} | "
            f"{card.score:.0f} | {price:.2f} | {bz} | {stop} | "
            f"{card.catalyst} | {card.next_rating_change} |")


def render_dossier_markdown(card: "PuntCard") -> str:
    """Full per-ticker dossier as a markdown string."""
    bz = _format_buy_zone(card.buy_zone)
    stop = f"₦{card.stop:.2f}" if card.stop is not None else "—"
    ev = card.evidence
    warning_str = ", ".join(card.warnings) if card.warnings else "none"

    bars = "\n".join(
        f"  {k:<16} {'█' * int(max(0, min(30, v))):<30} {v:>+5.1f}"
        for k, v in card.component_scores.items()
    )

    return f"""
### {card.ticker} — {card.state} (score {card.score:.0f})

**Tier:** {card.tier} · **Price:** ₦{ev.get('price', 0):.2f} · **RSI:** {ev.get('rsi_14', 0):.1f} · **YTD:** {ev.get('perf_ytd', 0):+.1f}% · **3M:** {ev.get('perf_3m', 0):+.1f}%

**Buy zone:** {bz} · **Stop:** {stop} · **Catalyst:** {card.catalyst}

**Next rating change:** {card.next_rating_change}

**Warnings:** {warning_str}

**Score breakdown:**
```
{bars}
```

**Technical levels:** EMA-50 ₦{ev.get('ema_50', 0):.2f} · EMA-200 ₦{ev.get('ema_200', 0):.2f}

**Insider activity (90d):** {ev.get('insider_buys_90d', 0)} buys, {ev.get('insider_sales_90d', 0)} sales
"""


def _build_next_rating_change(state: str, warnings: list[str],
                              buy_zone, row: pd.Series) -> str:
    """Plain-language description of what would move this ticker's rating."""
    if state == "AVOID":
        if "dead_tape" in warnings:
            return "Needs real trading volume to even qualify"
        if "late_filer" in warnings or "results_overdue" in warnings:
            return "Needs fresh financial statement filed"
        if "neg_equity_worsening" in warnings:
            return "Needs equity trend to stabilise or improve"
        if "falling_knife" in warnings:
            return "Needs insider purchase to confirm bottom"
        return "Needs score above 30"
    if state == "WATCH":
        return "Would advance to SETUP if score reaches 50 on next snapshot"
    if state == "SETUP":
        if isinstance(buy_zone, tuple) and isinstance(buy_zone[0], str):
            return "Would trigger after retrace to EMA-50 zone"
        if isinstance(buy_zone, tuple):
            low, high = buy_zone
            price = float(row.get("price", 0))
            if price > high:
                return f"Would trigger if price retraces to {low:.2f}-{high:.2f}"
        return "Would trigger if score reaches 70 in next snapshot"
    if state == "TRIGGER":
        return "Would degrade to SETUP if RSI > 80 or insider sale filed"
    return ""
