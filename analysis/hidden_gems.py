"""
Hidden Gems Finder
==================
Find stocks with strong fundamentals but not yet showing momentum.
These are the "REDSTAREX-style" opportunities - beaten down but ready to run.

v2.0 — Hardened against value traps (ROYALEX/ABBEYBDS lessons, Apr 2026):
- Added _detect_value_trap() to catch accounting anomalies
- Added _generate_warnings() for transparent risk flagging
- YTD extension guard prevents buying former runners on pullback
- Tighter EPS/margin caps prevent one-time gains from inflating scores
- FCF confirmation bonus/penalty ensures earnings are backed by cash

v2.1 — Hardened against a documented multi-stock postmortem:
- max_pe filter: blocks paying >20x earnings for "hidden gems"
- min_liquidity_naira filter: requires ₦10M+/day turnover
- min_3m_perf filter: a stock down 30%+ in 3M is a falling knife, not a gem
- Renamed "Cold" heat label to "Falling — verify" to remove dangerous framing
- All three guards apply to find_value_turnarounds() too
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging
from analysis.seasonality import SeasonalAnalyzer
from analysis.quality_scores import enrich as enrich_quality_scores

logger = logging.getLogger(__name__)

def evaluate_quality_score(row):
    score = 0
    total_weight = 0
    
    roe = row.get('roe_ttm', np.nan)
    if not pd.isna(roe) and roe > 0:
        score += min(roe / 40, 1) * 15
        total_weight += 15
    
    # ROIC — better than ROE for capital-intensive companies
    roic = row.get('roic_ttm', np.nan)
    if not pd.isna(roic) and roic > 0:
        score += min(roic / 30, 1) * 15
        total_weight += 15
    
    roa = row.get('roa_ttm', np.nan)
    if not pd.isna(roa) and roa > 0:
        score += min(roa / 20, 1) * 10
        total_weight += 10
    
    nm = row.get('net_margin_ttm', np.nan)
    if not pd.isna(nm) and nm > 0:
        score += min(nm / 30, 1) * 10
        total_weight += 10
    
    om = row.get('operating_margin_ttm', np.nan)
    if not pd.isna(om) and om > 0:
        score += min(om / 30, 1) * 10
        total_weight += 10
    
    gm = row.get('gross_margin_ttm', np.nan)
    if not pd.isna(gm) and gm > 0:
        score += min(gm / 60, 1) * 5
        total_weight += 5
    
    de = row.get('debt_to_equity', np.nan)
    if not pd.isna(de) and de >= 0:
        score += max(0, 1 - de / 3) * 10
        total_weight += 10
    
    # FCF margin — continuous score instead of binary FCF check
    fcf_margin = row.get('fcf_margin_ttm', np.nan)
    if not pd.isna(fcf_margin) and fcf_margin > 0:
        score += min(fcf_margin / 20, 1) * 15
        total_weight += 15
    else:
        # Fallback to binary FCF check if margin not available
        fcf = row.get('free_cash_flow_trailing_12_months', np.nan)
        if not pd.isna(fcf):
            if fcf > 0:
                score += 8
            total_weight += 8
    
    # Earnings yield — value component, prevents overpaying for quality
    ey = row.get('earnings_yield_ttm', np.nan)
    if not pd.isna(ey) and ey > 0:
        score += min(ey / 15, 1) * 10
        total_weight += 10
    
    cr = row.get('current_ratio', np.nan)
    if not pd.isna(cr) and cr > 0:
        score += min(cr / 2, 1) * 10
        total_weight += 10
    
    if total_weight > 0:
        return (score / total_weight) * 100
    return np.nan


def _detect_value_trap(row) -> bool:
    """
    Detect stocks with accounting anomalies that make fundamentals look
    artificially strong. Returns True if the stock is a likely value trap.

    Catches cases like:
    - ROYALEX: 99% net margin, -123% FCF margin, 842% EPS growth, 0% payout
    - Stocks with massive paper gains but no real cash generation
    """
    sector = row.get('sector', '')
    net_margin = row.get('net_margin_ttm', np.nan)
    fcf_margin = row.get('fcf_margin_ttm', np.nan)
    eps_growth = row.get('eps_growth_ttm', np.nan)
    rev_growth = row.get('revenue_growth_ttm', np.nan)
    div_yield = row.get('dividend_yield_trailing_12_months', np.nan)
    payout_ratio = row.get('dividend_payout', row.get('dividend_payout_ratio_annual', np.nan))
    roe = row.get('roe_ttm', np.nan)
    employees = row.get('number_of_employees_annual', np.nan)
    market_cap = row.get('market_cap', np.nan)

    # Rule 1: FCF vs Net Margin divergence (non-Finance only)
    # Banks naturally have volatile FCF, so exempt Finance sector.
    # For non-financial companies: if net margin > 30% but FCF deeply negative,
    # the earnings are paper-only.
    if (sector != 'Finance'
            and not pd.isna(net_margin) and not pd.isna(fcf_margin)
            and net_margin > 30 and fcf_margin < -50):
        logger.info(f"Value trap: {row.get('symbol','?')} -- net margin {net_margin:.0f}% "
                    f"but FCF margin {fcf_margin:.0f}%")
        return True

    # Rule 2: Impossible margins for Finance sector
    # Insurance/banks with >70% net margin = almost always one-time accounting
    if (sector == 'Finance' and not pd.isna(net_margin) and net_margin > 70):
        logger.info(f"Value trap: {row.get('symbol','?')} -- Finance sector with "
                    f"{net_margin:.0f}% net margin")
        return True

    # Rule 3: Extreme EPS growth without revenue support
    # If EPS > 500% but revenue < 20%, it's a one-time event not business growth.
    # Threshold is 500% (not 300%) to allow legitimate operational leverage.
    if (not pd.isna(eps_growth) and not pd.isna(rev_growth)
            and eps_growth > 500 and rev_growth < 20):
        logger.info(f"Value trap: {row.get('symbol','?')} -- {eps_growth:.0f}% EPS growth "
                    f"but only {rev_growth:.0f}% revenue growth")
        return True

    # Rule 4: High profitability but zero payout
    # If margin > 50%, payout = 0%, and ROE < 20%, the company doesn't trust its own earnings
    if (not pd.isna(net_margin) and not pd.isna(payout_ratio)
            and net_margin > 50 and payout_ratio < 1
            and (pd.isna(roe) or roe < 20)):
        logger.info(f"Value trap: {row.get('symbol','?')} -- {net_margin:.0f}% margin, "
                    f"{payout_ratio:.0f}% payout, low ROE")
        return True

    # Rule 5: Suspiciously small company with large market cap
    # <50 employees with >10B market cap is a shell company red flag
    if (not pd.isna(employees) and not pd.isna(market_cap)
            and employees < 50 and market_cap > 10_000_000_000):
        logger.info(f"Value trap: {row.get('symbol','?')} -- {int(employees)} employees "
                    f"but N{market_cap/1e9:.1f}B market cap")
        return True

    return False


def _generate_warnings(row) -> str:
    """
    Generate warning flags for a stock based on known risk patterns.
    Returns a string of warning indicators to display alongside recommendations.
    """
    warnings = []
    sector = row.get('sector', '')
    net_margin = row.get('net_margin_ttm', np.nan)
    fcf_margin = row.get('fcf_margin_ttm', np.nan)
    eps_growth = row.get('eps_growth_ttm', np.nan)
    rev_growth = row.get('revenue_growth_ttm', np.nan)
    payout_ratio = row.get('dividend_payout', row.get('dividend_payout_ratio_annual', np.nan))
    perf_ytd = row.get('perf_ytd', np.nan)
    div_yield = row.get('dividend_yield_trailing_12_months', np.nan)
    employees = row.get('number_of_employees_annual', np.nan)
    eps_annual = row.get('earnings_per_share_diluted_growth_annual_yoy', np.nan)
    eps_estimate = row.get('earnings_per_share_estimate_annual', np.nan)

    # FCF negative
    if not pd.isna(fcf_margin) and fcf_margin < -20:
        warnings.append('FCF negative')

    # Extreme margin for sector
    if not pd.isna(net_margin) and net_margin > 60:
        warnings.append('Extreme margin')

    # YTD extended
    if not pd.isna(perf_ytd) and perf_ytd > 35:
        warnings.append('YTD extended')

    # No dividend despite high profits
    if (not pd.isna(net_margin) and net_margin > 30
            and (pd.isna(div_yield) or div_yield < 0.5)):
        warnings.append('No div despite profits')

    # Zero payout despite reported profits
    if not pd.isna(payout_ratio) and payout_ratio < 1:
        warnings.append('Zero payout')

    # No analyst coverage
    if pd.isna(eps_estimate):
        warnings.append('No analyst coverage')

    # EPS TTM vs Annual divergence (one-time event detection)
    if (not pd.isna(eps_growth) and not pd.isna(eps_annual)
            and eps_growth > 200 and eps_annual < 50):
        warnings.append('EPS spike may be one-time')

    # Very few employees
    if not pd.isna(employees) and employees < 100:
        warnings.append(f'Only {int(employees)} employees')

    return ' | '.join(warnings) if warnings else ''


def find_hidden_gems(
    df: pd.DataFrame,
    min_eps_growth: float = 15,
    min_revenue_growth: float = 15,
    max_1m_perf: float = 15,
    require_positive_margin: bool = True,
    top_n: int = 20,
    max_pe: float = 20.0,
    min_liquidity_naira: float = 10_000_000,
    min_3m_perf: float = -30.0,
) -> pd.DataFrame:
    """
    Find hidden gems - strong fundamentals but not yet hot.

    Strategy: Look for stocks that are:
    1. Fundamentally strong (EPS/revenue growing)
    2. Not yet showing momentum (beaten down or flat short-term)
    3. Quality indicators (positive margins, reasonable debt)
    4. Reasonably valued (P/E < cap)
    5. Liquid enough to enter and exit
    6. Not in active free-fall (3M perf above floor)

    This is how you find the next REDSTAREX BEFORE it rallies 78%.

    Args:
        df: Snapshot DataFrame with normalized column names
        min_eps_growth: Minimum EPS growth TTM (%)
        min_revenue_growth: Minimum revenue growth TTM (%)
        max_1m_perf: Maximum 1-month performance (lower = more "hidden")
        require_positive_margin: Require positive net margin
        top_n: Number of gems to return
        max_pe: Maximum P/E ratio. Stocks above this are too expensive to be
            "hidden". NaN P/E (no earnings reported, or just turned positive)
            is allowed through. Set to None to disable. Default 20x.
        min_liquidity_naira: Minimum daily turnover in naira (price × volume_1d).
            Below this you can't enter or exit a meaningful position. Default ₦10M.
        min_3m_perf: Minimum 3-month performance %. A stock down more than this
            in 3 months is a falling knife, not a gem. Default -30%.

    Returns:
        DataFrame of hidden gems sorted by score
    """
    df = df.copy()

    # Compute liquidity if not present — required for the liquidity filter.
    if 'liquidity_value_1d' not in df.columns and 'volume_1d' in df.columns:
        df['liquidity_value_1d'] = df['price'] * df['volume_1d']
    
    # === STEP 0: Value Trap Filter (NEW v2.0) ===
    # Block stocks with accounting anomalies BEFORE scoring
    if 'net_margin_ttm' in df.columns:
        value_trap_mask = df.apply(_detect_value_trap, axis=1)
        trapped_count = value_trap_mask.sum()
        if trapped_count > 0:
            trapped_symbols = df.loc[value_trap_mask, 'symbol'].tolist()
            logger.info(f"Blocked {trapped_count} value traps: {trapped_symbols}")
        df = df[~value_trap_mask]
    
    # === STEP 1: Fundamental Strength Filter ===
    # At least one of these must be strong
    fundamental_filter = (
        (df['eps_growth_ttm'] > min_eps_growth) | 
        (df['revenue_growth_ttm'] > min_revenue_growth)
    )
    
    # === STEP 2: "Hidden" Filter - Not Hot Yet ===
    # Both conditions must hold — stock can't be running hard on either timeframe
    hidden_filter = (
        (df['perf_1m'] < max_1m_perf) &  # Not rallying hard yet
        (df['perf_3m'] < 25)  # AND 3M not already extended
    )
    
    # NEW v2.0: YTD extension guard — if a stock is already up 40%+ YTD,
    # it is NOT hidden. It's a former runner pulling back.
    # This would have caught ABBEYBDS (+49% YTD when purchased).
    if 'perf_ytd' in df.columns:
        hidden_filter = hidden_filter & (df['perf_ytd'] < 40)
    
    # === STEP 3: Quality Filter ===
    quality_filter = pd.Series(True, index=df.index)
    if require_positive_margin:
        quality_filter = quality_filter & (df['net_margin_ttm'] > 0)

    # === STEP 3a (v2.1): Valuation Cap ===
    # The Apr-2026 postmortem found 3 of 4 losses had P/E 27–66. Cap at 20x.
    # NaN P/E is allowed through (early-profitable companies, no analyst coverage).
    valuation_filter = pd.Series(True, index=df.index)
    if max_pe is not None and 'pe_ratio' in df.columns:
        valuation_filter = (df['pe_ratio'] < max_pe) | df['pe_ratio'].isna()

    # === STEP 3b (v2.1): Liquidity Floor ===
    # Need to be able to actually enter and exit. INFINITY traded ₦270K/day —
    # a ₦47K position there was 17% of daily volume.
    liquidity_filter = pd.Series(True, index=df.index)
    if min_liquidity_naira is not None and 'liquidity_value_1d' in df.columns:
        liquidity_filter = df['liquidity_value_1d'] > min_liquidity_naira

    # === STEP 3c (v2.1): Falling-Knife Guard ===
    # A stock down 30%+ in 3 months has a real problem, not just sentiment.
    # MECURE was -42% 3M when this would have caught it.
    falling_knife_filter = pd.Series(True, index=df.index)
    if min_3m_perf is not None and 'perf_3m' in df.columns:
        falling_knife_filter = (df['perf_3m'] >= min_3m_perf) | df['perf_3m'].isna()

    # Combine filters
    gems = df[
        fundamental_filter
        & hidden_filter
        & quality_filter
        & valuation_filter
        & liquidity_filter
        & falling_knife_filter
    ].copy()
    
    if len(gems) == 0:
        logger.warning("No hidden gems found with current criteria")
        return pd.DataFrame()
    
    # === STEP 4: Score the Gems ===
    # Weight fundamentals heavily since momentum is intentionally low
    
    # Clip extreme values — TIGHTER caps to prevent one-time gains
    # from inflating scores (was 500, now 150 for EPS; was 50, now 35 for margin)
    eps_g = gems['eps_growth_ttm'].fillna(0).clip(-100, 150)
    rev_g = gems['revenue_growth_ttm'].fillna(0).clip(-100, 300)
    roe = gems['roe_ttm'].fillna(0).clip(-50, 100)
    margin = gems['net_margin_ttm'].fillna(0).clip(-50, 35)
    
    # Long-term momentum (shows underlying strength)
    perf_1y = gems['perf_1y'].fillna(0).clip(-50, 500)
    perf_ytd = gems['perf_ytd'].fillna(0).clip(-50, 300)
    
    # Monthly RSI timing signal — lower RSI means more "hidden"
    rsi_monthly = gems['rsi_14_1m'].fillna(50).clip(10, 90) if 'rsi_14_1m' in gems.columns else pd.Series(50, index=gems.index)
    # Invert: RSI 30 → bonus +20, RSI 50 → 0, RSI 70 → penalty -20
    rsi_bonus = (50 - rsi_monthly) * 1.0
    
    # Earnings yield — value signal
    ey = gems['earnings_yield_ttm'].fillna(0).clip(0, 30) if 'earnings_yield_ttm' in gems.columns else pd.Series(0, index=gems.index)
    
    # NEW v2.0: FCF confirmation bonus/penalty
    # Ensures reported earnings are backed by actual cash generation
    fcf_bonus = pd.Series(0.0, index=gems.index)
    if 'fcf_margin_ttm' in gems.columns:
        fcf_margin = gems['fcf_margin_ttm'].fillna(0)
        # Bonus for positive FCF (max +10)
        fcf_bonus = fcf_bonus.where(fcf_margin <= 0, fcf_margin.clip(0, 20) * 0.5)
        # Penalty for deeply negative FCF (max -20)
        fcf_bonus = fcf_bonus.where(fcf_margin >= -50, fcf_margin.clip(-50, 0) * 0.4)
    
    # NEW v2.0: Payout ratio confirmation
    # Companies that actually distribute earnings get a bonus
    payout_bonus = pd.Series(0.0, index=gems.index)
    payout_col = 'dividend_payout' if 'dividend_payout' in gems.columns else 'dividend_payout_ratio_annual'
    if payout_col in gems.columns:
        payout = gems[payout_col].fillna(0).clip(0, 100)
        payout_bonus = payout * 0.05  # Max +5 points for 100% payout
    
    # Score: Fundamentals + Quality + Long-term strength + Timing + Value + Cash confirmation
    gems['hidden_gem_score'] = (
        # Fundamentals (35% — reduced from 40%)
        eps_g * 0.175 +
        rev_g * 0.175 +
        # Quality (20%)
        roe * 0.12 +
        margin * 0.08 +
        # Long-term momentum shows underlying strength (15% — reduced from 20%)
        (perf_1y * 0.09 + perf_ytd * 0.06) +
        # Monthly RSI timing — lower = more hidden (10%)
        rsi_bonus * 0.10 +
        # Earnings yield — value component (10%)
        ey * 0.10 +
        # NEW: Cash flow confirmation (10%)
        fcf_bonus + payout_bonus
    )
    
    # Calculate standalone quality score
    gems['quality_score'] = gems.apply(evaluate_quality_score, axis=1)
    
    # Normalize to 0-100
    score_min = gems['hidden_gem_score'].min()
    score_max = gems['hidden_gem_score'].max()
    if score_max > score_min:
        gems['hidden_gem_score'] = (
            (gems['hidden_gem_score'] - score_min) /
            (score_max - score_min) * 100
        )
    else:
        # All stocks scored equally — assign 50 (neutral) to all
        gems['hidden_gem_score'] = 50.0
    
    # Sort by score
    gems = gems.sort_values('hidden_gem_score', ascending=False)
    
    # Add ranking info
    gems['gem_rank'] = range(1, len(gems) + 1)
    
    # Add "heat status" - how hot is it getting?
    # v2.1: "Cold" relabelled to "Falling — verify before buying" because the
    # original framing made the algo's biggest blind spot (good fundamentals +
    # falling price ≠ buy) look like its best feature.
    def heat_status(row):
        if row['perf_1m'] < 0:
            return "❄️ Falling — verify before buying"
        elif row['perf_1m'] < 10:
            return "🌤️ Warming up"
        elif row['perf_1m'] < 20:
            return "🔥 Getting hot"
        else:
            return "🚀 Already running"
    
    gems['heat_status'] = gems.apply(heat_status, axis=1)
    
    # NEW v2.0: Generate warning flags for each gem
    gems['warnings'] = gems.apply(_generate_warnings, axis=1)

    # NEW v2.2: Add seasonality context
    sa = SeasonalAnalyzer()
    gems['seasonality'] = gems.apply(lambda row: sa.label_current_month(sa.analyze(row['symbol'])), axis=1)

    # NEW v2.3: Layer in the five quality scores (earnings quality, balance
    # sheet strength, multi-year consistency, technical confluence, float risk).
    gems = enrich_quality_scores(gems)
    # Merge the new earnings-quality and float-risk flags into existing warnings.
    gems['warnings'] = gems.apply(
        lambda r: ' | '.join([w for w in [r.get('warnings', ''),
                                          r.get('earnings_quality_flag', ''),
                                          r.get('float_risk_flag', '')] if w]),
        axis=1,
    )

    # Select columns for output
    output_cols = [
        'gem_rank', 'symbol', 'description', 'sector',
        'price', 'hidden_gem_score', 'quality_score', 'seasonality', 'heat_status', 'warnings',
        'perf_1m', 'perf_3m', 'perf_6m', 'perf_1y', 'perf_ytd',
        'eps_growth_ttm', 'revenue_growth_ttm',
        'ebitda_growth_ttm', 'fcf_growth_ttm',
        'net_income_growth_ttm', 'gross_profit_growth_ttm',
        'roe_ttm', 'net_margin_ttm', 'fcf_margin_ttm', 'debt_to_equity',
        'earnings_yield_ttm', 'rsi_14_1m',
        'stochastic_k_1d', 'stochastic_d_1d', 'cci_20_1d',
        'relative_volume_1d',
        'dividend_payout', 'dividend_payout_ratio_annual',
        'earnings_per_share_estimate_annual', 'number_of_employees_annual',
        'volume_1d', 'liquidity_value_1d',
        # New quality columns (v2.3)
        'earnings_quality_score', 'balance_sheet_score', 'balance_sheet_tier',
        'consistency_score', 'consistency_tier',
        'technical_signal', 'technical_trend', 'technical_macd_cross',
        'float_risk_flag',
    ]
    
    available_cols = [c for c in output_cols if c in gems.columns]
    
    return gems[available_cols].head(top_n)


def find_value_turnarounds(
    df: pd.DataFrame,
    max_3m_perf: float = 0,
    min_margin: float = 0,
    top_n: int = 15,
    max_pe: float = 20.0,
    min_liquidity_naira: float = 10_000_000,
    min_3m_perf: float = -30.0,
) -> pd.DataFrame:
    """
    Find value turnaround candidates - stocks beaten down but fundamentally sound.

    These are contrarian plays:
    - Negative short-term momentum (everyone is selling)
    - But still profitable (fundamentally sound)
    - With history of strong performance (proven business)

    Higher risk, higher reward potential.

    v2.1: Adds the same valuation/liquidity/falling-knife guards as
    find_hidden_gems. Turnarounds are *especially* vulnerable to falling-knife
    risk because they're explicitly screened for negative recent performance —
    without a floor, this finder will happily recommend stocks in free fall.

    Args:
        max_pe: Maximum P/E. NaN allowed. None to disable. Default 20x.
        min_liquidity_naira: Minimum daily turnover. Default ₦10M.
        min_3m_perf: Floor on 3M performance to exclude free-falling stocks.
            Note: this acts as a floor. Stocks must satisfy
            min_3m_perf <= perf_3m < max_3m_perf. Default -30%.
    """
    df = df.copy()

    if 'liquidity_value_1d' not in df.columns and 'volume_1d' in df.columns:
        df['liquidity_value_1d'] = df['price'] * df['volume_1d']

    # NEW v2.0: Apply value trap filter to turnarounds too
    if 'net_margin_ttm' in df.columns:
        value_trap_mask = df.apply(_detect_value_trap, axis=1)
        trapped_count = value_trap_mask.sum()
        if trapped_count > 0:
            trapped_symbols = df.loc[value_trap_mask, 'symbol'].tolist()
            logger.info(f"Turnarounds: blocked {trapped_count} value traps: {trapped_symbols}")
        df = df[~value_trap_mask]

    # Beaten down recently — but with a floor (v2.1)
    beaten = df['perf_3m'] < max_3m_perf
    if min_3m_perf is not None:
        beaten = beaten & (df['perf_3m'] >= min_3m_perf)

    # But still profitable
    profitable = df['net_margin_ttm'] > min_margin

    # And has shown strength historically
    has_history = df['perf_1y'].notna()

    # v2.1 valuation cap
    valuation_filter = pd.Series(True, index=df.index)
    if max_pe is not None and 'pe_ratio' in df.columns:
        valuation_filter = (df['pe_ratio'] < max_pe) | df['pe_ratio'].isna()

    # v2.1 liquidity floor
    liquidity_filter = pd.Series(True, index=df.index)
    if min_liquidity_naira is not None and 'liquidity_value_1d' in df.columns:
        liquidity_filter = df['liquidity_value_1d'] > min_liquidity_naira

    candidates = df[
        beaten & profitable & has_history & valuation_filter & liquidity_filter
    ].copy()
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    # Score: Fundamentals strength despite price weakness
    # v2.0: Tighter caps on EPS growth (was 200, now 150) and margin (was 30, now 25)
    candidates['turnaround_score'] = (
        candidates['eps_growth_ttm'].fillna(0).clip(-50, 150) * 0.25 +
        candidates['revenue_growth_ttm'].fillna(0).clip(-50, 100) * 0.2 +
        candidates['roe_ttm'].fillna(0).clip(-20, 50) * 0.2 +
        candidates['net_margin_ttm'].fillna(0).clip(0, 25) * 0.15 +
        # Penalize very negative recent momentum (might be falling knife)
        candidates['perf_1m'].fillna(0).clip(-30, 0) * 0.10
    )
    
    # NEW v2.0: FCF confirmation bonus for turnarounds
    if 'fcf_margin_ttm' in candidates.columns:
        fcf_m = candidates['fcf_margin_ttm'].fillna(0)
        candidates['turnaround_score'] += fcf_m.clip(-30, 20) * 0.10
    
    candidates = candidates.sort_values('turnaround_score', ascending=False)
    candidates['turnaround_rank'] = range(1, len(candidates) + 1)
    
    # NEW v2.0: Add warnings to turnaround output
    candidates['warnings'] = candidates.apply(_generate_warnings, axis=1)

    # NEW v2.2: Add seasonality context
    sa = SeasonalAnalyzer()
    candidates['seasonality'] = candidates.apply(lambda row: sa.label_current_month(sa.analyze(row['symbol'])), axis=1)

    # NEW v2.3: Enrich with quality scores
    candidates = enrich_quality_scores(candidates)
    candidates['warnings'] = candidates.apply(
        lambda r: ' | '.join([w for w in [r.get('warnings', ''),
                                          r.get('earnings_quality_flag', ''),
                                          r.get('float_risk_flag', '')] if w]),
        axis=1,
    )

    output_cols = [
        'turnaround_rank', 'symbol', 'description', 'sector',
        'price', 'turnaround_score', 'seasonality', 'warnings',
        'perf_1m', 'perf_3m', 'perf_6m', 'perf_1y',
        'eps_growth_ttm', 'revenue_growth_ttm',
        'roe_ttm', 'net_margin_ttm', 'fcf_margin_ttm',
        'dividend_payout', 'number_of_employees_annual',
        'earnings_quality_score', 'balance_sheet_score', 'balance_sheet_tier',
        'consistency_score', 'consistency_tier',
        'technical_signal', 'technical_trend',
        'float_risk_flag',
    ]
    
    available_cols = [c for c in output_cols if c in candidates.columns]
    return candidates[available_cols].head(top_n)


def analyze_snapshot_for_gems(snapshot_path: Path) -> dict:
    """
    Full hidden gems analysis on a snapshot.
    
    Returns dict with:
    - hidden_gems: Main gem list
    - turnarounds: Value turnaround candidates
    - summary: Quick stats
    """
    df = pd.read_csv(snapshot_path)
    
    # Compute liquidity value if not present
    if 'liquidity_value_1d' not in df.columns:
        df['liquidity_value_1d'] = df['price'] * df['volume_1d']
    
    hidden_gems = find_hidden_gems(df)
    turnarounds = find_value_turnarounds(df)
    
    # Summary — count "falling" gems (v2.1 label, was "Cold")
    if len(hidden_gems) > 0:
        falling_gems = len(hidden_gems[hidden_gems['heat_status'].str.contains('Falling|Cold')])
    else:
        falling_gems = 0

    summary = {
        'total_stocks': len(df),
        'hidden_gems_found': len(hidden_gems),
        'cold_opportunities': falling_gems,  # legacy key kept for callers
        'falling_opportunities': falling_gems,
        'turnaround_candidates': len(turnarounds),
        'snapshot_date': snapshot_path.parent.name,
    }
    
    return {
        'hidden_gems': hidden_gems,
        'turnarounds': turnarounds,
        'summary': summary,
    }


def run_gems_report(data_dir: Path = None) -> None:
    """Run hidden gems report on latest snapshot."""
    
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data' / 'snapshots'
    
    # Find latest snapshot
    snapshots = sorted(data_dir.glob('*/snapshot_merged.csv'))
    if not snapshots:
        print("No snapshots found!")
        return
    
    latest = snapshots[-1]
    print(f"\n{'='*70}")
    print(f"HIDDEN GEMS REPORT - {latest.parent.name}")
    print(f"{'='*70}")
    
    results = analyze_snapshot_for_gems(latest)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total stocks scanned: {results['summary']['total_stocks']}")
    print(f"  Hidden gems found: {results['summary']['hidden_gems_found']}")
    print(f"  Cold opportunities (best entry): {results['summary']['cold_opportunities']}")
    print(f"  Turnaround candidates: {results['summary']['turnaround_candidates']}")
    
    # Print gems
    print(f"\n{'='*70}")
    print("TOP HIDDEN GEMS (Strong Fundamentals, Not Yet Hot)")
    print(f"{'='*70}\n")
    
    gems = results['hidden_gems']
    if len(gems) > 0:
        for _, row in gems.head(10).iterrows():
            print(f"{row['gem_rank']:2}. {row['symbol']:15} {row['heat_status']}")
            print(f"    Score: {row['hidden_gem_score']:.1f} | Price: ₦{row['price']:.2f}")
            print(f"    1M: {row['perf_1m']:+.1f}% | 3M: {row['perf_3m']:+.1f}% | 1Y: {row.get('perf_1y', 0):+.1f}%")
            print(f"    EPS Growth: {row['eps_growth_ttm']:.1f}% | Rev Growth: {row['revenue_growth_ttm']:.1f}%")
            print(f"    ROE: {row['roe_ttm']:.1f}% | Margin: {row['net_margin_ttm']:.1f}%")
            print()
    else:
        print("  No hidden gems found with current criteria.")
    
    # Print turnarounds
    print(f"\n{'='*70}")
    print("VALUE TURNAROUND CANDIDATES (Beaten Down but Profitable)")
    print(f"{'='*70}\n")
    
    turns = results['turnarounds']
    if len(turns) > 0:
        for _, row in turns.head(5).iterrows():
            print(f"{row['turnaround_rank']:2}. {row['symbol']:15}")
            print(f"    Score: {row['turnaround_score']:.1f} | Price: ₦{row['price']:.2f}")
            print(f"    3M: {row['perf_3m']:+.1f}% (beaten down)")
            print(f"    But: Margin {row['net_margin_ttm']:.1f}% | ROE {row['roe_ttm']:.1f}%")
            print()
    else:
        print("  No turnaround candidates found.")
    
    print(f"\n{'='*70}")
    print("INVESTMENT STRATEGY")
    print(f"{'='*70}")
    print("""
For hidden gems:
  - ❄️ Falling — VERIFY before buying (negative 1M is a warning, not a signal)
  - 🌤️ Warming = Good entry, building momentum
  - 🔥 Hot = May still work but less upside
  - 🚀 Running = Consider waiting for pullback

Built-in guards (v2.1):
  - P/E cap (default 20x) — won't recommend expensive stocks
  - Liquidity floor (default ₦10M/day) — won't recommend illiquid stocks
  - Falling-knife guard (default -30% 3M) — won't recommend free-falls

Before sizing up a "Falling" gem, manually check:
  - Why did it drop? Earnings miss, dilution, sector rotation, macro?
  - Cross-check with the latest annual report (analysis/report_analyzer.py)
  - Wait for 2+ weeks of price stabilization

Position sizing suggestions:
  - Hidden gems with high scores: 10-15% of portfolio
  - Turnaround candidates: 5-8% (higher risk)
  - Diversify across 3-5 gems minimum
""")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_gems_report()
