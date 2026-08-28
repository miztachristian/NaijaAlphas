"""
Snapshot Ranker
===============
Rank NGX stocks based on growth potential using TradingView snapshot data.
No backtests - just honest scoring based on available metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_CONFIG = {
    # Feature weights for aggressive growth ranking
    'aggressive': {
        'momentum_weight': 0.35,
        'fundamental_growth_weight': 0.30,
        'quality_weight': 0.15,
        'risk_penalty_weight': 0.10,
        'coverage_penalty_weight': 0.10,
    },
    # Feature weights for growth with guardrails
    'guardrails': {
        'momentum_weight': 0.25,
        'fundamental_growth_weight': 0.25,
        'quality_weight': 0.25,
        'risk_penalty_weight': 0.15,
        'coverage_penalty_weight': 0.10,
    },
    # Gates for filtering
    'gates': {
        'min_coverage_aggressive': 0.50,
        'min_liquidity_value_1d_aggressive': 1_000_000,  # ₦1M
        'min_coverage_guardrails': 0.60,
        'min_liquidity_value_1d_guardrails': 10_000_000,  # ₦10M
        'max_debt_to_equity_guardrails': 2.0,
        'require_positive_net_margin_guardrails': False,
    },
    # Ranking settings
    'sector_min_size_for_sector_ranks': 5,
    'winsor_percentiles': [1, 99],
}

# Features for ranking (direction: True = higher is better)
RANKING_FEATURES = {
    # Momentum (higher is better)
    'perf_1m': True,
    'perf_3m': True,
    'perf_6m': True,
    'perf_1y': True,
    'perf_ytd': True,
    
    # Fundamental growth (higher is better)
    'revenue_growth_ttm': True,
    'eps_growth_ttm': True,
    
    # Quality (higher is better)
    'roe_ttm': True,
    'net_margin_ttm': True,
    'operating_margin_ttm': True,
    'fcf_margin_ttm': True,
    
    # Risk (lower is better - will be inverted)
    'debt_to_equity': False,
    'vol_1m': False,
}

# Feature groups
MOMENTUM_FEATURES = ['perf_1m', 'perf_3m', 'perf_6m', 'perf_1y', 'perf_ytd']
GROWTH_FEATURES = ['revenue_growth_ttm', 'eps_growth_ttm']
QUALITY_FEATURES = ['roe_ttm', 'net_margin_ttm', 'operating_margin_ttm', 'fcf_margin_ttm']
RISK_FEATURES = ['debt_to_equity', 'vol_1m']

# Key features for coverage score
KEY_FEATURES = [
    'price', 'volume_1d',
    'perf_3m', 'perf_6m', 'perf_1y', 'perf_ytd',
    'revenue_growth_ttm', 'eps_growth_ttm',
    'roe_ttm',
    'net_margin_ttm', 'operating_margin_ttm', 'fcf_margin_ttm',
    'debt_to_equity',
    'vol_1m',
]


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load ranking configuration from YAML file or use defaults."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        if user_config:
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
        return config
    return DEFAULT_CONFIG.copy()


def _winsorize(series: pd.Series, lower_pct: float = 1, upper_pct: float = 99) -> pd.Series:
    """Winsorize a series to cap extreme values."""
    if series.isna().all():
        return series
    lower = np.nanpercentile(series.dropna(), lower_pct)
    upper = np.nanpercentile(series.dropna(), upper_pct)
    return series.clip(lower=lower, upper=upper)


def _compute_percentile_rank(
    series: pd.Series,
    higher_is_better: bool = True,
    ascending: bool = True,
) -> pd.Series:
    """
    Compute percentile rank (0-100) for a series.
    NaN values remain NaN.
    """
    if series.isna().all():
        return series
    
    # Rank with method='average' for ties
    if higher_is_better:
        ranks = series.rank(method='average', ascending=True, na_option='keep')
    else:
        ranks = series.rank(method='average', ascending=False, na_option='keep')
    
    # Convert to percentile (0-100)
    valid_count = series.notna().sum()
    if valid_count > 0:
        percentile_ranks = (ranks - 1) / (valid_count - 1) * 100 if valid_count > 1 else 50.0
    else:
        percentile_ranks = ranks
    
    return percentile_ranks


def _compute_sector_ranks(
    df: pd.DataFrame,
    feature: str,
    higher_is_better: bool,
    sector_col: str = 'sector',
    min_sector_size: int = 5,
) -> pd.Series:
    """
    Compute percentile ranks within sector, falling back to global if sector too small.
    """
    if feature not in df.columns:
        return pd.Series(np.nan, index=df.index)
    
    result = pd.Series(np.nan, index=df.index)
    
    # Global ranks as fallback
    global_ranks = _compute_percentile_rank(df[feature], higher_is_better)
    
    if sector_col not in df.columns or df[sector_col].isna().all():
        return global_ranks
    
    # Compute per sector
    for sector in df[sector_col].dropna().unique():
        mask = df[sector_col] == sector
        sector_size = mask.sum()
        
        if sector_size >= min_sector_size:
            sector_data = df.loc[mask, feature]
            result.loc[mask] = _compute_percentile_rank(sector_data, higher_is_better)
        else:
            # Fallback to global
            result.loc[mask] = global_ranks.loc[mask]
    
    # Fill remaining (no sector) with global
    no_sector = df[sector_col].isna()
    result.loc[no_sector] = global_ranks.loc[no_sector]
    
    return result


def compute_coverage_score(df: pd.DataFrame, features: List[str] = None) -> pd.Series:
    """
    Compute coverage score = fraction of key features present.
    """
    features = features or KEY_FEATURES
    
    # Count how many key features are present (not NaN)
    available = [f for f in features if f in df.columns]
    if not available:
        return pd.Series(0.0, index=df.index)
    
    present_counts = df[available].notna().sum(axis=1)
    coverage = present_counts / len(available)
    
    return coverage


def compute_liquidity_value(df: pd.DataFrame) -> pd.Series:
    """
    Compute liquidity value = price * volume_1d.
    """
    if 'price' not in df.columns or 'volume_1d' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    
    return df['price'] * df['volume_1d']


def rank_snapshot(
    df: pd.DataFrame,
    config: dict = None,
) -> Dict[str, pd.DataFrame]:
    """
    Rank stocks based on growth potential.
    
    Args:
        df: Merged snapshot DataFrame with normalized column names
        config: Configuration dict (uses defaults if None)
        
    Returns:
        Dict with keys:
        - 'aggressive_growth': Aggressive ranking table
        - 'growth_with_guardrails': Conservative ranking table
        - 'snapshot': Cleaned snapshot with derived columns
        - 'config': Configuration used
    """
    config = config or DEFAULT_CONFIG
    
    df = df.copy()
    
    # Ensure required columns
    if 'symbol' not in df.columns:
        raise ValueError("DataFrame must have 'symbol' column")
    
    logger.info(f"Ranking {len(df)} stocks")
    
    # === Derived columns ===
    
    # Liquidity value
    df['liquidity_value_1d'] = compute_liquidity_value(df)
    
    # Coverage score
    df['coverage_score'] = compute_coverage_score(df)
    
    # === Winsorize extreme values ===
    winsor_low, winsor_high = config.get('winsor_percentiles', [1, 99])
    
    for feature in RANKING_FEATURES.keys():
        if feature in df.columns:
            df[feature] = _winsorize(df[feature], winsor_low, winsor_high)
    
    # === Compute sector-normalized ranks ===
    min_sector_size = config.get('sector_min_size_for_sector_ranks', 5)
    
    rank_columns = {}
    for feature, higher_is_better in RANKING_FEATURES.items():
        if feature in df.columns:
            rank_col = f'{feature}_rank'
            df[rank_col] = _compute_sector_ranks(
                df, feature, higher_is_better,
                sector_col='sector',
                min_sector_size=min_sector_size,
            )
            rank_columns[feature] = rank_col
    
    # === Compute component scores ===
    
    # Momentum score (average of available momentum ranks)
    momentum_ranks = [f'{f}_rank' for f in MOMENTUM_FEATURES if f'{f}_rank' in df.columns]
    if momentum_ranks:
        df['momentum_score'] = df[momentum_ranks].mean(axis=1, skipna=True)
    else:
        df['momentum_score'] = np.nan
    
    # Fundamental growth score
    growth_ranks = [f'{f}_rank' for f in GROWTH_FEATURES if f'{f}_rank' in df.columns]
    if growth_ranks:
        df['fundamental_growth_score'] = df[growth_ranks].mean(axis=1, skipna=True)
    else:
        df['fundamental_growth_score'] = np.nan
    
    # Quality score
    quality_ranks = [f'{f}_rank' for f in QUALITY_FEATURES if f'{f}_rank' in df.columns]
    if quality_ranks:
        df['quality_score'] = df[quality_ranks].mean(axis=1, skipna=True)
    else:
        df['quality_score'] = np.nan
    
    # Risk penalty (average of risk ranks - already inverted in ranking)
    risk_ranks = [f'{f}_rank' for f in RISK_FEATURES if f'{f}_rank' in df.columns]
    if risk_ranks:
        df['risk_score'] = df[risk_ranks].mean(axis=1, skipna=True)
    else:
        df['risk_score'] = np.nan
    
    # === Compute final scores ===
    
    # Aggressive score
    agg_weights = config.get('aggressive', DEFAULT_CONFIG['aggressive'])
    df['growth_potential_score_aggressive'] = (
        df['momentum_score'].fillna(0) * agg_weights['momentum_weight'] +
        df['fundamental_growth_score'].fillna(0) * agg_weights['fundamental_growth_weight'] +
        df['quality_score'].fillna(0) * agg_weights['quality_weight'] +
        df['risk_score'].fillna(0) * agg_weights['risk_penalty_weight'] +
        df['coverage_score'] * 100 * agg_weights['coverage_penalty_weight']
    )
    
    # Guardrails score
    guard_weights = config.get('guardrails', DEFAULT_CONFIG['guardrails'])
    df['growth_potential_score_guardrails'] = (
        df['momentum_score'].fillna(0) * guard_weights['momentum_weight'] +
        df['fundamental_growth_score'].fillna(0) * guard_weights['fundamental_growth_weight'] +
        df['quality_score'].fillna(0) * guard_weights['quality_weight'] +
        df['risk_score'].fillna(0) * guard_weights['risk_penalty_weight'] +
        df['coverage_score'] * 100 * guard_weights['coverage_penalty_weight']
    )
    
    # === Apply gates and create ranking tables ===
    
    gates = config.get('gates', DEFAULT_CONFIG['gates'])
    
    # Common output columns
    output_cols = [
        'symbol', 'description', 'sector',
        'price', 'volume_1d', 'liquidity_value_1d',
        'coverage_score',
        'perf_1m', 'perf_3m', 'perf_6m', 'perf_1y', 'perf_ytd',
        'revenue_growth_ttm', 'eps_growth_ttm',
        'roe_ttm', 'net_margin_ttm', 'operating_margin_ttm', 'fcf_margin_ttm',
        'debt_to_equity', 'vol_1m',
        'momentum_score', 'fundamental_growth_score', 'quality_score', 'risk_score',
    ]
    
    # Filter to existing columns
    output_cols = [c for c in output_cols if c in df.columns]
    
    # --- Aggressive ranking ---
    agg_mask = pd.Series(True, index=df.index)
    
    # Apply gates
    if 'coverage_score' in df.columns:
        agg_mask &= df['coverage_score'] >= gates.get('min_coverage_aggressive', 0.5)
    if 'liquidity_value_1d' in df.columns:
        min_liq = gates.get('min_liquidity_value_1d_aggressive', 0)
        agg_mask &= df['liquidity_value_1d'].fillna(0) >= min_liq
    
    # NEW v2.0: Block value traps from aggressive ranking too
    try:
        from analysis.hidden_gems import _detect_value_trap
        value_trap_mask_agg = df.apply(_detect_value_trap, axis=1)
        agg_mask &= ~value_trap_mask_agg
        logger.info(f"Aggressive: blocked {value_trap_mask_agg.sum()} value traps")
    except ImportError:
        logger.warning("Could not import _detect_value_trap, skipping value trap gate")
    
    agg_df = df[agg_mask].copy()
    agg_df = agg_df.sort_values('growth_potential_score_aggressive', ascending=False)
    agg_df['rank'] = range(1, len(agg_df) + 1)
    
    agg_output_cols = output_cols + ['growth_potential_score_aggressive', 'rank']
    agg_output_cols = [c for c in agg_output_cols if c in agg_df.columns]
    aggressive_growth = agg_df[agg_output_cols].copy()
    
    # --- Guardrails ranking ---
    guard_mask = pd.Series(True, index=df.index)
    
    # Apply stricter gates
    if 'coverage_score' in df.columns:
        guard_mask &= df['coverage_score'] >= gates.get('min_coverage_guardrails', 0.6)
    if 'liquidity_value_1d' in df.columns:
        min_liq = gates.get('min_liquidity_value_1d_guardrails', 0)
        guard_mask &= df['liquidity_value_1d'].fillna(0) >= min_liq
    if 'debt_to_equity' in df.columns and gates.get('max_debt_to_equity_guardrails'):
        max_de = gates['max_debt_to_equity_guardrails']
        guard_mask &= (df['debt_to_equity'].fillna(0) <= max_de) | df['debt_to_equity'].isna()
    if gates.get('require_positive_net_margin_guardrails') and 'net_margin_ttm' in df.columns:
        guard_mask &= df['net_margin_ttm'].fillna(0) > 0
    
    # NEW v2.0: Block value traps from guardrails ranking
    try:
        from analysis.hidden_gems import _detect_value_trap
        value_trap_mask = df.apply(_detect_value_trap, axis=1)
        guard_mask &= ~value_trap_mask
        logger.info(f"Guardrails: blocked {value_trap_mask.sum()} value traps")
    except ImportError:
        logger.warning("Could not import _detect_value_trap, skipping value trap gate")
    
    guard_df = df[guard_mask].copy()
    guard_df = guard_df.sort_values('growth_potential_score_guardrails', ascending=False)
    guard_df['rank'] = range(1, len(guard_df) + 1)
    
    guard_output_cols = output_cols + ['growth_potential_score_guardrails', 'rank']
    guard_output_cols = [c for c in guard_output_cols if c in guard_df.columns]
    growth_with_guardrails = guard_df[guard_output_cols].copy()
    
    logger.info(f"Aggressive ranking: {len(aggressive_growth)} stocks passed gates")
    logger.info(f"Guardrails ranking: {len(growth_with_guardrails)} stocks passed gates")
    
    return {
        'aggressive_growth': aggressive_growth,
        'growth_with_guardrails': growth_with_guardrails,
        'snapshot': df,
        'config': config,
    }


def print_top_stocks(rankings: Dict[str, pd.DataFrame], top_k: int = 10):
    """Print top stocks from rankings to console."""
    
    print("\n" + "=" * 80)
    print("TOP GROWTH POTENTIAL STOCKS (AGGRESSIVE)")
    print("=" * 80)
    
    agg = rankings.get('aggressive_growth')
    if agg is not None and len(agg) > 0:
        display_cols = ['rank', 'symbol', 'sector', 'price', 'coverage_score', 
                       'growth_potential_score_aggressive']
        display_cols = [c for c in display_cols if c in agg.columns]
        print(agg.head(top_k)[display_cols].to_string(index=False))
    else:
        print("No stocks passed aggressive gates")
    
    print("\n" + "=" * 80)
    print("TOP GROWTH POTENTIAL STOCKS (WITH GUARDRAILS)")
    print("=" * 80)
    
    guard = rankings.get('growth_with_guardrails')
    if guard is not None and len(guard) > 0:
        display_cols = ['rank', 'symbol', 'sector', 'price', 'coverage_score',
                       'growth_potential_score_guardrails']
        display_cols = [c for c in display_cols if c in guard.columns]
        print(guard.head(top_k)[display_cols].to_string(index=False))
    else:
        print("No stocks passed guardrails gates")
    
    print()
