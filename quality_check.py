import os
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.hidden_gems import _detect_value_trap, _generate_warnings

data_dir = Path(os.getenv('NAIJA_DATA_DIR', Path(__file__).resolve().parent / 'data')) / 'snapshots'
latest_snap = sorted(data_dir.glob('*/snapshot.parquet'))[-1]
df = pd.read_parquet(latest_snap)

tickers = ['AFRIPRUD', 'MECURE', 'SKYAVN']
picks = df[df['symbol'].isin(tickers)].copy()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 40)

# Key quality columns
quality_cols = [c for c in df.columns if any(k in c.lower() for k in [
    'symbol', 'price', 'sector', 'rsi', 'perf_', 'eps', 'revenue', 'net_margin', 
    'roe', 'roa', 'debt', 'current_ratio', 'quick_ratio', 'fcf', 'cash', 
    'bvps', 'pe_ratio', 'pb_ratio', 'dividend', 'payout', 'volume',
    'market_cap', 'operating', 'ebit', 'gross_margin', 'coverage',
    'analyst', 'technical_rating', 'moving_averages_rating'
])]

sep = "=" * 80

for t in tickers:
    row = picks[picks['symbol'] == t].iloc[0]
    desc = row.get('description', 'N/A')
    print(f"\n{sep}")
    print(f"  {t} -- {desc}")
    print(f"{sep}")
    for c in sorted(quality_cols):
        val = row.get(c)
        if pd.notna(val):
            label = c.ljust(50)
            print(f"  {label} : {val}")
    
    # Value trap check
    is_trap = _detect_value_trap(row)
    warnings = _generate_warnings(row)
    print(f"  {'---VALUE TRAP CHECK---'.ljust(50)} : {'YES TRAP' if is_trap else 'CLEAN'}")
    print(f"  {'---WARNINGS---'.ljust(50)} : {warnings if warnings else 'None'}")

# Also check: P/E, P/B, ROE, Debt/Equity for comparison
print(f"\n\n{sep}")
print("  COMPARATIVE QUALITY SCORECARD")
print(f"{sep}")

compare_cols = [
    'symbol', 'price', 'rsi_14',
    'eps_growth_ttm', 'revenue_growth_ttm', 'net_margin_ttm', 'gross_margin_ttm',
    'roe_ttm', 'roa_ttm',
    'pe_ratio_ttm', 'pb_ratio_quarterly',
    'current_ratio_quarterly', 'quick_ratio_quarterly',
    'debt_to_equity_quarterly',
    'fcf_per_share', 'eps_basic_ttm', 'bvps',
    'dividend_yield_recent', 'payout_ratio_ttm',
    'perf_1m', 'perf_3m', 'perf_6m', 'perf_ytd',
    'coverage_score'
]

available = [c for c in compare_cols if c in df.columns]
comparison = picks[available].set_index('symbol')
print(comparison.T.to_string())
