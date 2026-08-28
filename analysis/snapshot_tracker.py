"""
Snapshot Tracker
================
Track forward returns between snapshots to evaluate ranking performance.
No backtests - just realized returns from actual price changes between snapshots.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import logging

from .snapshot_store import SnapshotStore

logger = logging.getLogger(__name__)


def evaluate_forward_returns(
    prior_date: str,
    later_date: str,
    top_k: int = 10,
    list_name: str = 'aggressive',
    base_dir: Path = None,
) -> Optional[Dict]:
    """
    Evaluate realized forward returns from prior snapshot to later snapshot.
    
    Args:
        prior_date: Earlier snapshot date (YYYY-MM-DD)
        later_date: Later snapshot date (YYYY-MM-DD)
        top_k: Number of top stocks to evaluate
        list_name: Which ranking list to evaluate ('aggressive' or 'guardrails')
        base_dir: Base directory for snapshots
        
    Returns:
        Dict with evaluation results or None if snapshots not found
    """
    store = SnapshotStore(base_dir)
    
    # Load snapshots
    prior_snapshot = store.load_snapshot(prior_date)
    later_snapshot = store.load_snapshot(later_date)
    
    if prior_snapshot is None:
        logger.error(f"Prior snapshot not found: {prior_date}")
        return None
    
    if later_snapshot is None:
        logger.error(f"Later snapshot not found: {later_date}")
        return None
    
    # Get merged data from both snapshots
    prior_merged = prior_snapshot.get('merged')
    later_merged = later_snapshot.get('merged')
    
    if prior_merged is None or later_merged is None:
        logger.error("Merged data not found in snapshots")
        return None
    
    # Get ranking from prior date (may not exist for old-format snapshots)
    ranking_key = 'aggressive' if list_name == 'aggressive' else 'guardrails'
    prior_ranking = prior_snapshot.get(ranking_key)
    
    if prior_ranking is not None:
        # New format: use the saved ranking
        top_stocks = prior_ranking.head(top_k)['symbol'].tolist()
    else:
        # Old format: no ranking saved - rank by growth potential on the fly
        logger.info(f"No ranking found for {prior_date}, generating from merged data")
        
        # Try to rank using available columns
        rank_df = prior_merged.copy()
        
        # Build a simple composite score from available columns
        score_cols = []
        for col, weight in [('perf_1m', 0.2), ('perf_3m', 0.2), ('perf_ytd', 0.15),
                            ('eps_growth_ttm', 0.15), ('revenue_growth_ttm', 0.1),
                            ('roe_ttm', 0.1), ('net_margin_ttm', 0.1)]:
            if col in rank_df.columns:
                # Normalize to 0-100 percentile within the snapshot
                valid = rank_df[col].dropna()
                if len(valid) > 0:
                    rank_df[f'{col}_pctile'] = rank_df[col].rank(pct=True) * 100
                    score_cols.append((f'{col}_pctile', weight))
        
        if score_cols:
            rank_df['_auto_score'] = sum(rank_df[col].fillna(50) * w for col, w in score_cols)
            rank_df = rank_df.sort_values('_auto_score', ascending=False)
        
        top_stocks = rank_df.head(top_k)['symbol'].tolist()
    
    # Get prices from both snapshots
    prior_prices = prior_merged.set_index('symbol')['price'].to_dict()
    later_prices = later_merged.set_index('symbol')['price'].to_dict()
    
    # Compute returns for top picks
    returns_data = []
    for symbol in top_stocks:
        prior_price = prior_prices.get(symbol)
        later_price = later_prices.get(symbol)
        
        if prior_price and later_price and prior_price > 0:
            ret = (later_price / prior_price - 1) * 100
            returns_data.append({
                'symbol': symbol,
                'prior_price': prior_price,
                'later_price': later_price,
                'return_pct': ret,
                'found_in_both': True,
            })
        else:
            returns_data.append({
                'symbol': symbol,
                'prior_price': prior_price,
                'later_price': later_price,
                'return_pct': np.nan,
                'found_in_both': False,
            })
    
    returns_df = pd.DataFrame(returns_data)
    
    # Compute universe returns
    all_symbols = set(prior_prices.keys()) & set(later_prices.keys())
    universe_returns = []
    for symbol in all_symbols:
        prior_price = prior_prices.get(symbol)
        later_price = later_prices.get(symbol)
        if prior_price and later_price and prior_price > 0:
            ret = (later_price / prior_price - 1) * 100
            universe_returns.append(ret)
    
    universe_returns = np.array(universe_returns)
    universe_median = np.nanmedian(universe_returns) if len(universe_returns) > 0 else 0
    universe_mean = np.nanmean(universe_returns) if len(universe_returns) > 0 else 0
    
    # Compute metrics
    valid_returns = returns_df[returns_df['found_in_both']]['return_pct']
    
    top_picks_mean = valid_returns.mean() if len(valid_returns) > 0 else np.nan
    top_picks_median = valid_returns.median() if len(valid_returns) > 0 else np.nan
    hit_rate = (valid_returns > universe_median).mean() if len(valid_returns) > 0 else np.nan
    
    # Top winners and losers
    sorted_returns = returns_df[returns_df['found_in_both']].sort_values('return_pct', ascending=False)
    top_winners = sorted_returns.head(3).to_dict('records')
    top_losers = sorted_returns.tail(3).to_dict('records')
    
    # Build result
    result = {
        'prior_date': prior_date,
        'later_date': later_date,
        'list_name': list_name,
        'top_k': top_k,
        'metrics': {
            'top_picks_mean_return': round(top_picks_mean, 2) if not np.isnan(top_picks_mean) else None,
            'top_picks_median_return': round(top_picks_median, 2) if not np.isnan(top_picks_median) else None,
            'universe_mean_return': round(universe_mean, 2),
            'universe_median_return': round(universe_median, 2),
            'hit_rate_vs_universe_median': round(hit_rate * 100, 1) if not np.isnan(hit_rate) else None,
            'symbols_found_in_both': int(returns_df['found_in_both'].sum()),
            'symbols_missing': int((~returns_df['found_in_both']).sum()),
        },
        'top_winners': top_winners,
        'top_losers': top_losers,
        'returns_detail': returns_df.to_dict('records'),
        'universe_coverage': len(all_symbols),
    }
    
    return result


def save_evaluation(
    result: Dict,
    base_dir: Path = None,
) -> Path:
    """
    Save evaluation results to disk.
    
    Args:
        result: Evaluation result dict
        base_dir: Base directory for snapshots
        
    Returns:
        Path to evaluation CSV file
    """
    store = SnapshotStore(base_dir)
    
    later_date = result['later_date']
    prior_date = result['prior_date']
    list_name = result['list_name']
    
    snapshot_dir = store._get_snapshot_dir(later_date)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed returns CSV
    returns_df = pd.DataFrame(result['returns_detail'])
    csv_path = snapshot_dir / f'eval_from_{prior_date}_{list_name}.csv'
    returns_df.to_csv(csv_path, index=False)
    
    # Save summary markdown
    md_path = snapshot_dir / f'eval_from_{prior_date}_{list_name}.md'
    
    metrics = result['metrics']
    
    md_content = f"""# Forward Return Evaluation

**Prior Date:** {prior_date}  
**Later Date:** {later_date}  
**List:** {list_name}  
**Top K:** {result['top_k']}  

## Performance Metrics

| Metric | Value |
|--------|-------|
| Top Picks Mean Return | {metrics['top_picks_mean_return']}% |
| Top Picks Median Return | {metrics['top_picks_median_return']}% |
| Universe Mean Return | {metrics['universe_mean_return']}% |
| Universe Median Return | {metrics['universe_median_return']}% |
| Hit Rate vs Universe | {metrics['hit_rate_vs_universe_median']}% |
| Symbols Found in Both | {metrics['symbols_found_in_both']} / {result['top_k']} |

## Top Winners

| Symbol | Prior Price | Later Price | Return |
|--------|-------------|-------------|--------|
"""
    
    for w in result['top_winners']:
        md_content += f"| {w['symbol']} | {w['prior_price']:.2f} | {w['later_price']:.2f} | {w['return_pct']:+.1f}% |\n"
    
    md_content += """
## Top Losers

| Symbol | Prior Price | Later Price | Return |
|--------|-------------|-------------|--------|
"""
    
    for l in result['top_losers']:
        md_content += f"| {l['symbol']} | {l['prior_price']:.2f} | {l['later_price']:.2f} | {l['return_pct']:+.1f}% |\n"
    
    md_content += f"""
---
*Evaluated at: {datetime.now().isoformat()}*
"""
    
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    logger.info(f"Saved evaluation to {csv_path} and {md_path}")
    
    return csv_path


def print_evaluation(result: Dict):
    """Print evaluation results to console."""
    
    print("\n" + "=" * 70)
    print(f"FORWARD RETURN EVALUATION")
    print("=" * 70)
    print(f"Prior Snapshot: {result['prior_date']}")
    print(f"Later Snapshot: {result['later_date']}")
    print(f"List: {result['list_name'].upper()}")
    print(f"Top K: {result['top_k']}")
    
    metrics = result['metrics']
    
    print("\n" + "-" * 50)
    print("PERFORMANCE METRICS")
    print("-" * 50)
    print(f"  Top Picks Mean Return:    {metrics['top_picks_mean_return']:+.2f}%")
    print(f"  Top Picks Median Return:  {metrics['top_picks_median_return']:+.2f}%")
    print(f"  Universe Mean Return:     {metrics['universe_mean_return']:+.2f}%")
    print(f"  Universe Median Return:   {metrics['universe_median_return']:+.2f}%")
    print(f"  Hit Rate vs Universe:     {metrics['hit_rate_vs_universe_median']:.1f}%")
    print(f"  Coverage:                 {metrics['symbols_found_in_both']}/{result['top_k']} symbols")
    
    print("\n" + "-" * 50)
    print("TOP WINNERS")
    print("-" * 50)
    for w in result['top_winners']:
        print(f"  {w['symbol']:12s}  {w['prior_price']:>10.2f} -> {w['later_price']:>10.2f}  {w['return_pct']:+8.1f}%")
    
    print("\n" + "-" * 50)
    print("TOP LOSERS")
    print("-" * 50)
    for l in result['top_losers']:
        print(f"  {l['symbol']:12s}  {l['prior_price']:>10.2f} -> {l['later_price']:>10.2f}  {l['return_pct']:+8.1f}%")
    
    # Interpretation note
    print("\n" + "=" * 70)
    print("NOTE: This is NOT a backtest. These are realized returns between")
    print("      two snapshot dates. Past performance does not guarantee future results.")
    print("=" * 70 + "\n")
