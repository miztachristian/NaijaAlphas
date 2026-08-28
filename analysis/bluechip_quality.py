"""
Blue-Chip Quality Screener
===========================
Find fundamentally strong stocks regardless of market cap.

Unlike hidden_gems which looks for small/mid-cap contrarian plays,
this screener catches quality blue-chips like DANGCEM, BUACEMENT,
BUAFOODS, and ARADEL — stocks with strong fundamentals that the
hidden gems model misses because they're never "cold" or "hidden".

Key differences from hidden_gems:
- NO negative momentum requirement (blue-chips are rarely cold)
- DYNAMIC P/E cap based on ROE (higher quality = higher allowed P/E)
- Market cap agnostic (works for N5B and N15T equally)
- Focuses on consistency: margin stability, ROE durability
- Still applies v2.1 safety gates (liquidity, falling knife, value trap)
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from analysis.quality_scores import enrich as enrich_quality_scores

logger = logging.getLogger(__name__)


def _quality_pe_cap(roe: float, eps_growth: float) -> float:
    """
    Dynamic P/E cap based on quality metrics.
    
    High-ROE businesses deserve higher P/E multiples because
    they compound shareholder value faster.
    
    Examples:
        ROE 20%, EPS +30% -> P/E cap 18x (decent quality)
        ROE 50%, EPS +80% -> P/E cap 28x (strong quality)
        ROE 80%, EPS +150% -> P/E cap 35x (exceptional)
    """
    base_pe = 15.0
    
    # ROE bonus: +1x P/E for every 5% ROE above 15%
    if not pd.isna(roe) and roe > 15:
        base_pe += min((roe - 15) / 5, 5) * 1.0  # max +5x from ROE
    
    # EPS growth bonus: +1x P/E for every 20% EPS growth above 20%
    if not pd.isna(eps_growth) and eps_growth > 20:
        base_pe += min((eps_growth - 20) / 20, 5) * 1.0  # max +5x from growth
    
    # Cap at 35x even for exceptional stocks
    return min(base_pe, 35.0)


def _bluechip_score(row) -> float:
    """
    Composite quality score for blue-chip screening.
    
    Weights:
        ROE (25%) - Return on equity, the core quality metric
        Net Margin (15%) - Pricing power and efficiency
        EPS Growth (20%) - Earnings momentum  
        Revenue Growth (10%) - Top-line growth
        FCF Margin (15%) - Cash generation (are earnings real?)
        Debt/Equity (10%) - Balance sheet health
        Dividend (5%) - Shareholder returns commitment
    """
    score = 0
    max_score = 0
    
    # ROE — 25 points, scaled to 50%
    roe = row.get('roe_ttm', np.nan)
    if not pd.isna(roe) and roe > 0:
        score += min(roe / 50, 1.0) * 25
    max_score += 25
    
    # Net Margin — 15 points, scaled to 30%
    nm = row.get('net_margin_ttm', np.nan)
    if not pd.isna(nm) and nm > 0:
        score += min(nm / 30, 1.0) * 15
    max_score += 15
    
    # EPS Growth — 20 points, scaled to 100%
    eps_g = row.get('eps_growth_ttm', np.nan)
    if not pd.isna(eps_g) and eps_g > 0:
        score += min(eps_g / 100, 1.0) * 20
    max_score += 20
    
    # Revenue Growth — 10 points, scaled to 50%
    rev_g = row.get('revenue_growth_ttm', np.nan)
    if not pd.isna(rev_g) and rev_g > 0:
        score += min(rev_g / 50, 1.0) * 10
    max_score += 10
    
    # FCF Margin — 15 points, scaled to 20%
    fcf = row.get('fcf_margin_ttm', np.nan)
    if not pd.isna(fcf) and fcf > 0:
        score += min(fcf / 20, 1.0) * 15
    max_score += 15
    
    # Debt/Equity — 10 points (lower is better)
    de = row.get('debt_to_equity', np.nan)
    if not pd.isna(de) and de >= 0:
        score += max(0, 1 - de / 2.0) * 10
    max_score += 10
    
    # Dividend yield — 5 points bonus
    div_y = row.get('dividend_yield_trailing_12_months', np.nan)
    if not pd.isna(div_y) and div_y > 0:
        score += min(div_y / 5, 1.0) * 5
    max_score += 5
    
    if max_score > 0:
        return (score / max_score) * 100
    return 0


def _generate_bluechip_warnings(row) -> str:
    """Generate warning flags specific to blue-chip stocks."""
    warnings = []
    
    pe = row.get('pe_ratio', np.nan)
    roe = row.get('roe_ttm', np.nan)
    nm = row.get('net_margin_ttm', np.nan)
    fcf = row.get('fcf_margin_ttm', np.nan)
    ytd = row.get('perf_ytd', np.nan)
    rsi = row.get('rsi_14', np.nan)
    payout = row.get('dividend_payout', np.nan)
    eps_est = row.get('earnings_per_share_estimate_annual', np.nan)
    p1m = row.get('perf_1m', np.nan)
    
    # RSI overbought
    if not pd.isna(rsi) and rsi >= 70:
        warnings.append(f'RSI {rsi:.0f} overbought')
    
    # YTD extended
    if not pd.isna(ytd) and ytd > 50:
        warnings.append(f'YTD +{ytd:.0f}% extended')
    
    # 1M hot
    if not pd.isna(p1m) and p1m > 20:
        warnings.append(f'1M +{p1m:.0f}% hot')
    
    # FCF negative
    if not pd.isna(fcf) and fcf < -10:
        warnings.append('FCF negative')
    
    # High P/E relative to quality
    if not pd.isna(pe) and pe > 25:
        warnings.append(f'P/E {pe:.0f}x premium')
    
    # Zero payout
    if not pd.isna(payout) and payout < 1:
        warnings.append('Zero payout')
    
    # No analyst coverage
    if pd.isna(eps_est):
        warnings.append('No analyst coverage')
    
    return ' | '.join(warnings) if warnings else ''


def _entry_signal(row) -> str:
    """
    Generate entry timing signal based on technicals.
    
    Returns:
        BUY NOW - RSI < 40, good entry
        WAIT - RSI > 70, overbought
        ACCUMULATE - RSI 40-60, decent entry
        CAUTIOUS - RSI 60-70, slightly hot
    """
    rsi = row.get('rsi_14', np.nan)
    p1m = row.get('perf_1m', 0) or 0
    
    if pd.isna(rsi):
        if p1m < -5:
            return '🟢 BUY DIP'
        return '⚪ NO RSI DATA'
    
    if rsi <= 30:
        return '🟢 OVERSOLD — BUY'
    elif rsi <= 45:
        return '🟢 BUY ZONE'
    elif rsi <= 60:
        return '🔵 ACCUMULATE'
    elif rsi <= 70:
        return '🟡 CAUTIOUS'
    else:
        return '🔴 WAIT — OVERBOUGHT'


def find_bluechip_quality(
    df: pd.DataFrame,
    min_roe: float = 15.0,
    min_margin: float = 3.0,
    min_eps_growth: float = 10.0,
    min_liquidity_naira: float = 10_000_000,
    min_3m_perf: float = -30.0,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Find blue-chip quality stocks with strong fundamentals.
    
    Unlike hidden_gems, this does NOT require negative momentum.
    It finds the BEST quality stocks on the exchange, period.
    
    Safety gates (v2.1 compatible):
        - Dynamic P/E cap based on ROE (not fixed 20x)
        - Liquidity floor (₦10M+/day)
        - Falling knife guard (3M > -30%)
        - Value trap detection
    
    Args:
        df: Snapshot DataFrame
        min_roe: Minimum ROE (%) 
        min_margin: Minimum net margin (%)
        min_eps_growth: Minimum EPS growth (%)
        min_liquidity_naira: Minimum daily turnover in Naira
        min_3m_perf: Minimum 3-month performance (%)
        top_n: Number of results to return
    """
    work = df.copy()
    
    # Ensure liquidity column exists
    if 'liquidity_value_1d' not in work.columns:
        if 'volume_1d' in work.columns and 'price' in work.columns:
            work['liquidity_value_1d'] = work['price'] * work['volume_1d']
        else:
            work['liquidity_value_1d'] = 0
    
    initial_count = len(work)
    
    # --- FUNDAMENTAL FILTERS ---
    
    # ROE filter
    mask = work['roe_ttm'].notna() & (work['roe_ttm'] >= min_roe)
    work = work[mask]
    logger.info(f"After ROE >= {min_roe}%: {len(work)} stocks")
    
    # Net margin filter
    mask = work['net_margin_ttm'].notna() & (work['net_margin_ttm'] >= min_margin)
    work = work[mask]
    logger.info(f"After margin >= {min_margin}%: {len(work)} stocks")
    
    # EPS growth filter (allow NaN through — some blue-chips have missing data)
    mask = (work['eps_growth_ttm'].isna()) | (work['eps_growth_ttm'] >= min_eps_growth)
    work = work[mask]
    logger.info(f"After EPS growth >= {min_eps_growth}%: {len(work)} stocks")
    
    # --- SAFETY GATES (v2.1) ---
    
    # Dynamic P/E cap
    if 'pe_ratio' in work.columns:
        def pe_passes(row):
            pe = row.get('pe_ratio', np.nan)
            if pd.isna(pe):
                return True  # No P/E = allow through
            roe = row.get('roe_ttm', np.nan)
            eps_g = row.get('eps_growth_ttm', np.nan)
            cap = _quality_pe_cap(roe, eps_g)
            return pe <= cap
        
        before = len(work)
        pe_mask = work.apply(pe_passes, axis=1)
        blocked = work[~pe_mask]
        if len(blocked) > 0:
            for _, r in blocked.iterrows():
                cap = _quality_pe_cap(r.get('roe_ttm', np.nan), r.get('eps_growth_ttm', np.nan))
                logger.info(f"P/E blocked: {r['symbol']} P/E={r['pe_ratio']:.1f}x > cap {cap:.0f}x "
                           f"(ROE={r.get('roe_ttm', 0):.0f}%)")
        work = work[pe_mask]
        logger.info(f"After dynamic P/E cap: {len(work)} stocks (blocked {before - len(work)})")
    
    # Liquidity floor
    if min_liquidity_naira > 0:
        before = len(work)
        work = work[work['liquidity_value_1d'] > min_liquidity_naira]
        logger.info(f"After liquidity > ₦{min_liquidity_naira/1e6:.0f}M: {len(work)} (blocked {before - len(work)})")
    
    # Falling knife guard
    if 'perf_3m' in work.columns:
        before = len(work)
        work = work[(work['perf_3m'].isna()) | (work['perf_3m'] > min_3m_perf)]
        logger.info(f"After 3M > {min_3m_perf}%: {len(work)} (blocked {before - len(work)})")
    
    # Value trap detection (reuse from hidden_gems)
    from analysis.hidden_gems import _detect_value_trap
    trap_mask = work.apply(lambda r: not _detect_value_trap(r), axis=1)
    traps = work[~trap_mask]['symbol'].tolist()
    if traps:
        logger.info(f"Blocked {len(traps)} value traps: {traps}")
    work = work[trap_mask]
    
    if len(work) == 0:
        logger.warning("No stocks passed all filters")
        return pd.DataFrame()
    
    # --- SCORING ---
    work['bluechip_score'] = work.apply(_bluechip_score, axis=1)
    work['warnings'] = work.apply(_generate_bluechip_warnings, axis=1)
    work['entry_signal'] = work.apply(_entry_signal, axis=1)
    
    # Dynamic P/E cap column for display
    work['pe_cap'] = work.apply(
        lambda r: _quality_pe_cap(r.get('roe_ttm', np.nan), r.get('eps_growth_ttm', np.nan)),
        axis=1
    )
    
    # Tier classification
    def classify_tier(score):
        if score >= 70:
            return '🏆 Elite'
        elif score >= 55:
            return '⭐ Premium'
        elif score >= 40:
            return '✅ Quality'
        else:
            return '📊 Watchlist'
    
    work['tier'] = work['bluechip_score'].apply(classify_tier)
    
    # Sort by score
    work = work.sort_values('bluechip_score', ascending=False).head(top_n)
    work['rank'] = range(1, len(work) + 1)

    # NEW v2.3: Layer in quality scores and merge flags into warnings.
    work = enrich_quality_scores(work)
    work['warnings'] = work.apply(
        lambda r: ' | '.join([w for w in [r.get('warnings', ''),
                                          r.get('earnings_quality_flag', ''),
                                          r.get('float_risk_flag', '')] if w]),
        axis=1,
    )

    logger.info(f"Blue-chip quality: {len(work)} stocks from {initial_count} initial")

    return work


def find_bluechip_dips(
    df: pd.DataFrame,
    min_roe: float = 20.0,
    min_margin: float = 5.0,
    max_1m_perf: float = 0.0,
    min_liquidity_naira: float = 10_000_000,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Find blue-chip stocks on a dip — quality names pulling back.
    
    These are the best entry points: proven quality at a temporary discount.
    Like catching DANGCEM at ₦680 or BUACEMENT at ₦183.
    """
    work = df.copy()
    
    if 'liquidity_value_1d' not in work.columns:
        if 'volume_1d' in work.columns and 'price' in work.columns:
            work['liquidity_value_1d'] = work['price'] * work['volume_1d']
        else:
            work['liquidity_value_1d'] = 0
    
    # Quality filters
    mask = (
        (work['roe_ttm'].notna()) & (work['roe_ttm'] >= min_roe) &
        (work['net_margin_ttm'].notna()) & (work['net_margin_ttm'] >= min_margin) &
        (work['liquidity_value_1d'] > min_liquidity_naira)
    )
    work = work[mask]
    
    # Must be pulling back (negative 1M)
    if 'perf_1m' in work.columns:
        work = work[(work['perf_1m'].notna()) & (work['perf_1m'] <= max_1m_perf)]
    
    # Not in freefall
    if 'perf_3m' in work.columns:
        work = work[(work['perf_3m'].isna()) | (work['perf_3m'] > -30)]
    
    if len(work) == 0:
        return pd.DataFrame()
    
    # Value trap filter
    from analysis.hidden_gems import _detect_value_trap
    work = work[work.apply(lambda r: not _detect_value_trap(r), axis=1)]
    
    work['bluechip_score'] = work.apply(_bluechip_score, axis=1)
    work['warnings'] = work.apply(_generate_bluechip_warnings, axis=1)
    work['entry_signal'] = work.apply(_entry_signal, axis=1)
    work['pe_cap'] = work.apply(
        lambda r: _quality_pe_cap(r.get('roe_ttm', np.nan), r.get('eps_growth_ttm', np.nan)),
        axis=1
    )
    
    work = work.sort_values('bluechip_score', ascending=False).head(top_n)
    work['rank'] = range(1, len(work) + 1)

    # NEW v2.3: Layer in quality scores and merge flags into warnings.
    work = enrich_quality_scores(work)
    work['warnings'] = work.apply(
        lambda r: ' | '.join([w for w in [r.get('warnings', ''),
                                          r.get('earnings_quality_flag', ''),
                                          r.get('float_risk_flag', '')] if w]),
        axis=1,
    )

    return work
