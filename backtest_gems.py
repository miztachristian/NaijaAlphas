"""Backtest the hidden gems finder on Dec 17 data."""

import pandas as pd
import sys
sys.path.insert(0, '.')
from analysis.hidden_gems import find_hidden_gems

# Run on Dec 17 snapshot
df17 = pd.read_csv('data/snapshots/2025-12-17/snapshot_merged.csv')
df17['liquidity_value_1d'] = df17['price'] * df17['volume_1d']

gems = find_hidden_gems(df17, top_n=30)

print('HIDDEN GEMS - Dec 17, 2025 (Retrospective)')
print('=' * 60)
print()

for _, row in gems.head(15).iterrows():
    status = row['heat_status']
    sym = row['symbol']
    score = row['hidden_gem_score']
    eps = row.get('eps_growth_ttm', 0)
    perf1m = row.get('perf_1m', 0)
    rank = row['gem_rank']
    print(f'{rank:2}. {sym:15} {status}')
    print(f'    Score: {score:.1f} | EPS: {eps:.0f}% | 1M: {perf1m:+.1f}%')
    print()

# Check if REDSTAREX was in top gems
redstar = gems[gems['symbol'] == 'REDSTAREX']
if len(redstar) > 0:
    print('*' * 60)
    print('REDSTAREX WAS IN THE TOP GEMS LIST!')
    print(f"   Rank: {redstar.iloc[0]['gem_rank']}")
    print('*' * 60)
else:
    print('REDSTAREX was NOT in the hidden gems list on Dec 17')

# Now check forward returns for these gems
print('\n\n')
print('=' * 60)
print('FORWARD RETURNS CHECK - What happened to Dec 17 gems?')
print('=' * 60)

# Load Jan 14 prices
df14 = pd.read_csv('data/snapshots/2026-01-14/snapshot_merged.csv')

for _, row in gems.head(15).iterrows():
    sym = row['symbol']
    price_dec = row['price']
    
    # Find Jan price
    jan_row = df14[df14['symbol'] == sym]
    if len(jan_row) > 0:
        price_jan = jan_row.iloc[0]['price']
        ret = (price_jan - price_dec) / price_dec * 100
        print(f'{sym:15} | Dec: ₦{price_dec:8.2f} → Jan: ₦{price_jan:8.2f} | Return: {ret:+6.1f}%')
    else:
        print(f'{sym:15} | Not found in Jan data')
