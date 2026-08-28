"""
Quality Scoring Module
======================
Five reusable scores computed from the expanded TV financial dataset
(ingest/tv_financials.py). Used by hidden_gems and bluechip_quality to add
a layer of intelligence beyond TTM-only fundamentals:

1. earnings_quality_score   — flags accrual-heavy earnings (NI without FCF)
2. balance_sheet_strength   — liquidity + leverage + coverage in one number
3. consistency_score        — multi-year compounder check (5y CAGR, dividend streak)
4. technical_confluence     — TV's own aggregated ratings + ADX trend strength
5. float_risk_flag          — sub-15% float warning, illiquidity guard

Each function takes a row from the snapshot DataFrame and returns either a
float (0-100), a dict, or a string. Missing data returns NaN/"" rather than
penalising the stock — the caller decides how to handle missing values.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Earnings Quality
# ---------------------------------------------------------------------------

def earnings_quality_score(row: pd.Series) -> float:
    """
    Score the quality of reported earnings 0-100.

    High score = earnings are backed by cash. Low score = accruals-heavy,
    likely to revert. Built on three signals:

    - Cash conversion ratio (CFO / Net Income). Healthy >0.7.
    - Accruals ratio: (NI - CFO) / Total Assets. Low = good. >5% = red flag.
    - FCF growth direction matches NI growth direction (no divergence).

    Banks/insurers (sector == Finance) get a partial bypass — their FCF is
    structurally noisy.
    """
    sector = row.get("sector", "")
    ni = row.get("net_income_ttm")
    cfo = row.get("cash_f_operating_activities_ttm")
    fcf_growth = row.get("free_cash_flow_yoy_growth_ttm")
    ni_growth = row.get("net_income_growth_ttm")
    total_assets = row.get("total_assets_fq", row.get("total_assets"))

    # Finance sector: cash conversion not meaningful (deposits, claims, etc.)
    # Fall back to direction-match only.
    if sector == "Finance":
        if pd.isna(fcf_growth) or pd.isna(ni_growth):
            return np.nan
        # Same sign → 75; opposite signs → 35
        return 75.0 if (fcf_growth >= 0) == (ni_growth >= 0) else 35.0

    score = 0.0
    weight = 0.0

    # Cash conversion: CFO / NI, expected ~1 for healthy company.
    if not pd.isna(ni) and not pd.isna(cfo) and ni > 0:
        ratio = cfo / ni
        # 1.0+ → full 40 points, 0.5 → 20, <=0 → 0
        cash_conv = max(0.0, min(ratio, 1.5)) / 1.5
        score += cash_conv * 40
        weight += 40

    # Accruals ratio: (NI - CFO) / total_assets. Lower abs value = higher quality.
    if not pd.isna(ni) and not pd.isna(cfo) and not pd.isna(total_assets) and total_assets > 0:
        accruals = abs(ni - cfo) / total_assets
        # <2% → full 30 points; 5% → 15; 10%+ → 0
        accrual_pts = max(0.0, 1.0 - accruals / 0.10) * 30
        score += accrual_pts
        weight += 30

    # Growth-direction consistency: NI and FCF growing/shrinking together.
    if not pd.isna(fcf_growth) and not pd.isna(ni_growth):
        same_dir = (fcf_growth >= 0) == (ni_growth >= 0)
        score += 30 if same_dir else 5
        weight += 30

    if weight == 0:
        return np.nan
    return (score / weight) * 100


def earnings_quality_flag(row: pd.Series) -> str:
    """Short label suitable for the warnings column."""
    s = earnings_quality_score(row)
    if pd.isna(s):
        return ""
    if s < 35:
        return "Low earnings quality"
    if s < 55:
        return "Mixed earnings quality"
    return ""  # 55+ is fine, no flag


# ---------------------------------------------------------------------------
# 2. Balance Sheet Strength
# ---------------------------------------------------------------------------

def balance_sheet_strength(row: pd.Series) -> float:
    """
    Score balance sheet health 0-100.

    Combines four sub-signals (each 0-25):
    - Current ratio (short-term liquidity, sweet spot 1.5-2.5)
    - Cash + ST investments / Total debt (cushion, higher better, cap at 1.0)
    - Debt/Equity (leverage, lower better, cap at 2.0)
    - Long-term debt / Assets (solvency, lower better, cap at 0.5)

    Missing components are skipped — score normalises over available signals.
    """
    score = 0.0
    weight = 0.0

    # Current ratio: 1.5-2.5 is ideal. Below 1 = stressed. Above 3 = idle capital.
    cr = row.get("current_ratio_fq", row.get("current_ratio"))
    if not pd.isna(cr) and cr > 0:
        if cr < 1.0:
            pts = cr * 10  # 0..10
        elif cr <= 2.5:
            pts = 10 + (cr - 1.0) / 1.5 * 15  # 10..25
        else:
            pts = max(15, 25 - (cr - 2.5) * 5)  # decay above 2.5
        score += pts
        weight += 25

    # Cash / Debt: higher = better; 1.0+ means net cash. Cap at 1.5x.
    c2d = row.get("cash_n_short_term_invest_to_total_debt_fq", row.get("cash_to_debt_ratio"))
    if not pd.isna(c2d) and c2d >= 0:
        pts = min(c2d / 1.5, 1.0) * 25
        score += pts
        weight += 25

    # Debt/Equity: lower = better. 0 = full 25. 2.0+ = 0.
    de = row.get("debt_to_equity_fq", row.get("debt_to_equity"))
    if not pd.isna(de) and de >= 0:
        pts = max(0.0, 1.0 - de / 2.0) * 25
        score += pts
        weight += 25

    # Long-term debt / Total assets: lower = better. 0 = full 25. 0.5+ = 0.
    lta = row.get("long_term_debt_to_assets_fq", row.get("long_term_debt_to_assets_fy"))
    if not pd.isna(lta) and lta >= 0:
        pts = max(0.0, 1.0 - lta / 0.5) * 25
        score += pts
        weight += 25

    if weight == 0:
        return np.nan
    return (score / weight) * 100


def balance_sheet_tier(score: float) -> str:
    if pd.isna(score):
        return ""
    if score >= 75:
        return "🟢 Fortress"
    if score >= 55:
        return "🟢 Healthy"
    if score >= 35:
        return "🟡 Average"
    return "🔴 Stressed"


# ---------------------------------------------------------------------------
# 3. Multi-Year Consistency
# ---------------------------------------------------------------------------

def consistency_score(row: pd.Series) -> float:
    """
    Score multi-year compounding consistency 0-100.

    Combines:
    - 5y revenue CAGR (target >10% for full points)
    - 5y EPS CAGR (target >12% for full points)
    - Dividend continuity (continuous_dividend_payout — years of unbroken payout)
    - Dividend growth streak (continuous_dividend_growth — years of growth)
    - TTM vs 5y consistency: TTM growth in same direction as 5y CAGR
    """
    score = 0.0
    weight = 0.0

    # 5y revenue CAGR — 25 points
    rev_cagr = row.get("revenue_growth_5_year_cagr")
    if not pd.isna(rev_cagr):
        pts = max(0.0, min(rev_cagr / 25.0, 1.0)) * 25  # 25%+ CAGR = full
        score += pts
        weight += 25

    # 5y EPS CAGR — 25 points
    eps_cagr = row.get("earnings_per_share_diluted_5y_growth_fy",
                       row.get("earnings_per_share_basic_cagr_5y"))
    if not pd.isna(eps_cagr):
        pts = max(0.0, min(eps_cagr / 25.0, 1.0)) * 25
        score += pts
        weight += 25

    # Continuous dividend payout — years of unbroken payout
    ddp = row.get("continuous_dividend_payout")
    if not pd.isna(ddp):
        # 5+ years = full 20 points
        pts = min(ddp / 5.0, 1.0) * 20
        score += pts
        weight += 20

    # Dividend growth streak — years of growing dividends
    ddg = row.get("continuous_dividend_growth")
    if not pd.isna(ddg):
        # 3+ years = full 15 points (rarer than payout)
        pts = min(ddg / 3.0, 1.0) * 15
        score += pts
        weight += 15

    # TTM vs 5y direction-match: penalise reversals
    rev_ttm = row.get("revenue_growth_ttm")
    if not pd.isna(rev_cagr) and not pd.isna(rev_ttm):
        same_dir = (rev_cagr >= 0) == (rev_ttm >= 0)
        score += 15 if same_dir else 0
        weight += 15

    if weight == 0:
        return np.nan
    return (score / weight) * 100


def consistency_tier(score: float) -> str:
    if pd.isna(score):
        return ""
    if score >= 70:
        return "⭐ Compounder"
    if score >= 50:
        return "✅ Steady"
    if score >= 30:
        return "🟡 Inconsistent"
    return "🔴 Volatile"


# ---------------------------------------------------------------------------
# 4. Technical Confluence
# ---------------------------------------------------------------------------

def technical_confluence(row: pd.Series) -> dict:
    """
    Combine TV's three aggregated gauges + ADX into a single read.

    Returns a dict:
        signal:       Strong Buy / Buy / Neutral / Sell / Strong Sell
        score:        -100..+100 (sum of three ratings scaled)
        trend:        Trending / Weak trend / No trend (from ADX)
        is_bullish:   True if signal in {Strong Buy, Buy} AND trending
        is_bearish:   True if signal in {Strong Sell, Sell} AND trending
        macd_cross:   "bull" / "bear" / "" (MACD vs signal line)
    """
    tech = row.get("technical_rating_1d")
    ma = row.get("moving_averages_rating_1d")
    osc = row.get("oscillators_rating_1d")
    adx = row.get("adx_14")
    macd = row.get("macd")
    macd_sig = row.get("macd_signal")

    if pd.isna(tech) and pd.isna(ma) and pd.isna(osc):
        return {"signal": "", "score": np.nan, "trend": "", "is_bullish": False,
                "is_bearish": False, "macd_cross": ""}

    # Weighted blend: technical (50%) + ma (30%) + oscillators (20%)
    parts = []
    if not pd.isna(tech): parts.append(("tech", tech, 0.5))
    if not pd.isna(ma):   parts.append(("ma",   ma,   0.3))
    if not pd.isna(osc):  parts.append(("osc",  osc,  0.2))
    total_w = sum(w for _, _, w in parts)
    raw = sum(v * w for _, v, w in parts) / total_w if total_w else np.nan
    score = raw * 100  # -100..+100

    if raw >= 0.5:    signal = "Strong Buy"
    elif raw >= 0.1:  signal = "Buy"
    elif raw > -0.1:  signal = "Neutral"
    elif raw > -0.5:  signal = "Sell"
    else:             signal = "Strong Sell"

    # ADX > 25 = trending. > 40 = strong trend.
    if pd.isna(adx):
        trend = ""
    elif adx >= 40:
        trend = "Strong trend"
    elif adx >= 25:
        trend = "Trending"
    else:
        trend = "No trend"

    is_trending = not pd.isna(adx) and adx >= 25
    is_bullish = signal in ("Strong Buy", "Buy") and is_trending
    is_bearish = signal in ("Strong Sell", "Sell") and is_trending

    macd_cross = ""
    if not pd.isna(macd) and not pd.isna(macd_sig):
        macd_cross = "bull" if macd > macd_sig else "bear"

    return {
        "signal": signal,
        "score": round(score, 1),
        "trend": trend,
        "is_bullish": is_bullish,
        "is_bearish": is_bearish,
        "macd_cross": macd_cross,
    }


# ---------------------------------------------------------------------------
# 5. Float / Liquidity Risk
# ---------------------------------------------------------------------------

def float_risk_flag(row: pd.Series) -> str:
    """
    Warning string for stocks with thin tradeable float.

    Rules:
    - float < 10%   → 🔴 Sub-10% float (manipulation risk)
    - float 10-15%  → 🟡 Thin float
    - <100 shareholders → 🟡 Concentrated holder base
    Empty string = no concerns.
    """
    flags = []
    pct = row.get("float_shares_percent_current")
    holders = row.get("number_of_shareholders_fy", row.get("number_of_shareholders"))

    if not pd.isna(pct):
        if pct < 10:
            flags.append(f"🔴 Float {pct:.1f}%")
        elif pct < 15:
            flags.append(f"🟡 Thin float {pct:.1f}%")

    if not pd.isna(holders) and holders < 100:
        flags.append(f"🟡 {int(holders)} holders")

    return " | ".join(flags)


# ---------------------------------------------------------------------------
# Aggregator: applies all five and returns a tidy frame of new columns.
# ---------------------------------------------------------------------------

NEW_COLUMNS = [
    "earnings_quality_score",
    "earnings_quality_flag",
    "balance_sheet_score",
    "balance_sheet_tier",
    "consistency_score",
    "consistency_tier",
    "technical_signal",
    "technical_score",
    "technical_trend",
    "technical_macd_cross",
    "technical_is_bullish",
    "technical_is_bearish",
    "float_risk_flag",
]


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all five scores for every row in df and attach them as new columns.
    Returns the same DataFrame with NEW_COLUMNS appended.
    """
    df = df.copy()

    df["earnings_quality_score"] = df.apply(earnings_quality_score, axis=1)
    df["earnings_quality_flag"]  = df.apply(earnings_quality_flag, axis=1)

    df["balance_sheet_score"] = df.apply(balance_sheet_strength, axis=1)
    df["balance_sheet_tier"]  = df["balance_sheet_score"].apply(balance_sheet_tier)

    df["consistency_score"] = df.apply(consistency_score, axis=1)
    df["consistency_tier"]  = df["consistency_score"].apply(consistency_tier)

    tech = df.apply(technical_confluence, axis=1)
    df["technical_signal"]     = tech.apply(lambda d: d["signal"])
    df["technical_score"]      = tech.apply(lambda d: d["score"])
    df["technical_trend"]      = tech.apply(lambda d: d["trend"])
    df["technical_macd_cross"] = tech.apply(lambda d: d["macd_cross"])
    df["technical_is_bullish"] = tech.apply(lambda d: d["is_bullish"])
    df["technical_is_bearish"] = tech.apply(lambda d: d["is_bearish"])

    df["float_risk_flag"] = df.apply(float_risk_flag, axis=1)

    return df


def merged_warnings(row: pd.Series) -> str:
    """
    Build a single ' | '-separated warning string combining earnings-quality
    and float-risk flags. Designed to be appended to existing 'warnings' columns.
    """
    parts = []
    eq = row.get("earnings_quality_flag", "") or ""
    fr = row.get("float_risk_flag", "") or ""
    if eq: parts.append(eq)
    if fr: parts.append(fr)
    return " | ".join(parts)
