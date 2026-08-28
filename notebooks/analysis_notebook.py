#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Setup - Run this first
import sys
import os
from pathlib import Path

# Add project to path (parent of notebooks folder)
project_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(project_dir))

# Import new Screen + Track modules
from utils.tradingview_snapshot import load_tradingview_exports
from analysis.snapshot_ranker import rank_snapshot, print_top_stocks, load_config, DEFAULT_CONFIG
from analysis.snapshot_store import SnapshotStore, save_snapshot, load_snapshot
from analysis.snapshot_tracker import evaluate_forward_returns, print_evaluation, save_evaluation

import pandas as pd
import numpy as np
from datetime import datetime
from glob import glob

print("NGX Screen + Track System loaded!")
print(f"Project directory: {project_dir}")


# ## 1. Load TradingView Snapshot
# 
# Load and merge multiple TradingView screener CSV exports from the same date.
# Each export has different columns - the system merges them into one unified dataset.

# In[5]:


# CONFIGURE: Set the snapshot date and path to your TradingView CSVs
SNAPSHOT_DATE = "2026-05-15"  # Change this to your export date
TV_DATA_PATH = Path(os.getenv("NAIJA_TV_EXPORT_DIR", Path.home() / "Downloads"))

# Find all CSVs for this date
csv_pattern = TV_DATA_PATH / f"NG_Stocks_{SNAPSHOT_DATE}*.csv"
csv_files = list(TV_DATA_PATH.glob(f"NG_Stocks_{SNAPSHOT_DATE}*.csv"))

print(f"Snapshot Date: {SNAPSHOT_DATE}")
print(f"Found {len(csv_files)} CSV files:")
for f in csv_files[:5]:
    print(f"  - {f.name}")
if len(csv_files) > 5:
    print(f"  ... and {len(csv_files) - 5} more")


# In[6]:


# Load and merge all CSV exports
merged_df, report = load_tradingview_exports(csv_files)

print(f"Merged {report.total_symbols} stocks with {report.columns_merged} columns")
print(f"Duplicates resolved: {report.duplicate_columns_resolved}")
if report.dropped_columns:
    print(f"Dropped: {', '.join(report.dropped_columns[:5])}")

# Preview the data
print(f"\nAvailable columns ({len(merged_df.columns)}):")
print(merged_df.columns.tolist()[:20])
print("...")

merged_df.head()


# ## 2. Rank Stocks by Growth Potential
# 
# Two ranking modes:
# - **Aggressive**: Higher momentum weight, lower gates (min coverage 50%, min liquidity 1M)
# - **Guardrails**: More balanced weights, stricter gates (min coverage 60%, min liquidity 10M, max D/E 2.0)
# 
# Note: This is NOT a backtest. Scores represent current growth potential based on available metrics.

# In[ ]:


# Rank the stocks
config = load_config(project_dir / 'config' / 'snapshot_ranker.yaml')
rankings = rank_snapshot(merged_df, config=config)

print(f"Aggressive ranking: {len(rankings['aggressive_growth'])} stocks passed gates")
print(f"Guardrails ranking: {len(rankings['growth_with_guardrails'])} stocks passed gates")


# In[ ]:


# TOP 15 AGGRESSIVE GROWTH PICKS
aggressive = rankings['aggressive_growth']
display_cols = ['rank', 'symbol', 'sector', 'price', 'coverage_score', 
                'growth_potential_score_aggressive', 'momentum_score', 'fundamental_growth_score']
display_cols = [c for c in display_cols if c in aggressive.columns]

print(f"TOP 15 AGGRESSIVE GROWTH PICKS - {SNAPSHOT_DATE}")
print("="*80)
aggressive.head(15)[display_cols]


# In[ ]:


# VALUE TRAP REPORT (v2.0) — What was blocked and why
from analysis.hidden_gems import _detect_value_trap, _generate_warnings

snapshot = rankings['snapshot']

# Show what was blocked
trap_mask = snapshot.apply(_detect_value_trap, axis=1)
trapped = snapshot[trap_mask][['symbol', 'sector', 'price', 'net_margin_ttm',
                               'fcf_margin_ttm', 'eps_growth_ttm']].copy()

print(f"\n{'='*80}")
print(f"VALUE TRAP REPORT - {len(trapped)} stocks blocked from rankings")
print(f"{'='*80}")
if len(trapped) > 0:
    for _, row in trapped.iterrows():
        nm = f"{row['net_margin_ttm']:.0f}%" if pd.notna(row.get('net_margin_ttm')) else 'N/A'
        fcf = f"{row['fcf_margin_ttm']:.0f}%" if pd.notna(row.get('fcf_margin_ttm')) else 'N/A'
        eps = f"{row['eps_growth_ttm']:.0f}%" if pd.notna(row.get('eps_growth_ttm')) else 'N/A'
        print(f"  BLOCKED: {row['symbol']:<12} | Sector: {row.get('sector','?'):<10} | "
              f"Margin: {nm:>5} | FCF: {fcf:>6} | EPS Gr: {eps:>6}")
else:
    print("  No value traps detected in current snapshot")

# Show warnings on top aggressive picks
print(f"\n{'='*80}")
print(f"WARNING FLAGS ON TOP 15 AGGRESSIVE PICKS")
print(f"{'='*80}")
agg_top = aggressive.head(15).copy()
has_warnings = False
for _, row in agg_top.iterrows():
    stock_row = snapshot[snapshot['symbol'] == row['symbol']]
    if len(stock_row) > 0:
        w = _generate_warnings(stock_row.iloc[0])
        if w:
            has_warnings = True
            print(f"  {row['symbol']:<12} | {w}")
if not has_warnings:
    print("  All top 15 picks are clean - no warnings")


# In[ ]:


# TOP 15 GROWTH WITH GUARDRAILS
guardrails = rankings['growth_with_guardrails']
display_cols = ['rank', 'symbol', 'sector', 'price', 'coverage_score',
                'growth_potential_score_guardrails', 'momentum_score', 'quality_score']
display_cols = [c for c in display_cols if c in guardrails.columns]

print(f"TOP 15 GROWTH WITH GUARDRAILS - {SNAPSHOT_DATE}")
print("="*80)
guardrails.head(15)[display_cols]


# ## 3. Save Snapshot
# 
# Save the snapshot to disk for future forward-return tracking.

# In[ ]:


# Save snapshot to disk
metadata = report.to_dict()
metadata['config'] = config

snapshot_dir = save_snapshot(
    date=SNAPSHOT_DATE,
    merged_df=rankings['snapshot'],
    rankings=rankings,
    metadata=metadata,
    base_dir=project_dir / 'data' / 'snapshots'
)

print(f"Saved to: {snapshot_dir}")
print(f"\nFiles created:")
for f in Path(snapshot_dir).glob("*"):
    print(f"  - {f.name}")


# ## 4. Evaluate Forward Returns
# 
# Compare a prior snapshot to a later one to see how top picks actually performed.
# This is NOT a backtest - it's real realized returns between two dates.

# In[ ]:


# CONFIGURE: Set the evaluation period
FROM_DATE = "2025-12-17"  # Earlier snapshot
TO_DATE = "2026-05-15"    # Later snapshot
LIST_NAME = "aggressive"  # "aggressive" or "guardrails"
TOP_K = 10                # Number of top picks to evaluate

# Run evaluation
result = evaluate_forward_returns(
    prior_date=FROM_DATE,
    later_date=TO_DATE,
    top_k=TOP_K,
    list_name=LIST_NAME,
    base_dir=project_dir / 'data' / 'snapshots'
)

if result:
    print_evaluation(result)
else:
    print("Could not evaluate. Make sure both snapshots exist.")


# In[ ]:


# View detailed returns for each pick
if result:
    returns_df = pd.DataFrame(result['returns_detail'])
    returns_df = returns_df.sort_values('return_pct', ascending=False)
    print(f"DETAILED RETURNS: {FROM_DATE} -> {TO_DATE}")
    print("="*60)
    display(returns_df[['symbol', 'prior_price', 'later_price', 'return_pct', 'found_in_both']])


# ## 5. Sector Analysis

# In[ ]:


# Sector breakdown
snapshot = rankings['snapshot']

sector_summary = snapshot.groupby('sector').agg({
    'symbol': 'count',
    'price': 'median',
    'perf_1m': 'mean',
    'perf_3m': 'mean',
    'coverage_score': 'mean'
}).round(2)

sector_summary.columns = ['Count', 'Median Price', 'Avg 1M Perf', 'Avg 3M Perf', 'Avg Coverage']
sector_summary = sector_summary.sort_values('Avg 3M Perf', ascending=False)

print(f"SECTOR SUMMARY - {SNAPSHOT_DATE}")
print("="*60)
sector_summary


# In[ ]:


# Top stocks by sector
SECTOR = "Finance"  # Change to explore different sectors

# Get aggressive ranking
aggressive = rankings['aggressive_growth']
sector_stocks = aggressive[aggressive['sector'] == SECTOR].head(10)
cols = ['rank', 'symbol', 'price', 'growth_potential_score_aggressive', 'momentum_score']
cols = [c for c in cols if c in sector_stocks.columns]

print(f"TOP {SECTOR.upper()} STOCKS")
print("="*60)
sector_stocks[cols]


# ## 6. Visualizations

# In[ ]:


# Top 20 stocks bar chart
import matplotlib.pyplot as plt

# Get aggressive ranking
aggressive = rankings['aggressive_growth']
top_20 = aggressive.head(20)

fig, ax = plt.subplots(figsize=(14, 6))
colors = plt.cm.viridis(top_20['growth_potential_score_aggressive'] / 100)
bars = ax.bar(top_20['symbol'], top_20['growth_potential_score_aggressive'], color=colors)

ax.set_title(f'Top 20 Stocks by Growth Potential Score - {SNAPSHOT_DATE}')
ax.set_xlabel('Stock')
ax.set_ylabel('Growth Potential Score')
ax.axhline(y=75, color='green', linestyle='--', alpha=0.5, label='High potential')
ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Moderate potential')
plt.xticks(rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# Score components breakdown for top 10
aggressive = rankings['aggressive_growth']
top_10 = aggressive.head(10).copy()

fig, ax = plt.subplots(figsize=(12, 6))

# Get score components
score_cols = ['momentum_score', 'fundamental_growth_score', 'quality_score']
available_scores = [c for c in score_cols if c in top_10.columns]

x = np.arange(len(top_10))
width = 0.25

for i, col in enumerate(available_scores):
    offset = (i - len(available_scores)/2 + 0.5) * width
    ax.bar(x + offset, top_10[col], width, label=col.replace('_score', '').replace('_', ' ').title())

ax.set_xlabel('Stock')
ax.set_ylabel('Score (0-100)')
ax.set_title(f'Score Component Breakdown - Top 10 Stocks')
ax.set_xticks(x)
ax.set_xticklabels(top_10['symbol'], rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.show()


# ## 7. Look Up Specific Stock

# In[ ]:


# Look up a specific stock
TICKER = "TIP"  # Change this

snapshot = rankings['snapshot']
aggressive = rankings['aggressive_growth']
guardrails = rankings['growth_with_guardrails']

stock = snapshot[snapshot['symbol'] == TICKER]

if len(stock) > 0:
    stock_data = stock.iloc[0]
    print(f"STOCK DETAILS: {TICKER}")
    print("="*50)
    print(f"Sector:        {stock_data.get('sector', 'N/A')}")
    print(f"Price:         {stock_data.get('price', 'N/A')}")
    print(f"Coverage:      {stock_data.get('coverage_score', 0)*100:.0f}%")
    print()
    print("Performance:")
    print(f"  1 Month:     {stock_data.get('perf_1m', 'N/A'):.1f}%" if pd.notna(stock_data.get('perf_1m')) else "  1 Month:     N/A")
    print(f"  3 Month:     {stock_data.get('perf_3m', 'N/A'):.1f}%" if pd.notna(stock_data.get('perf_3m')) else "  3 Month:     N/A")
    print(f"  6 Month:     {stock_data.get('perf_6m', 'N/A'):.1f}%" if pd.notna(stock_data.get('perf_6m')) else "  6 Month:     N/A")
    print(f"  1 Year:      {stock_data.get('perf_1y', 'N/A'):.1f}%" if pd.notna(stock_data.get('perf_1y')) else "  1 Year:      N/A")
    print()

    # Check ranking
    agg_rank = aggressive[aggressive['symbol'] == TICKER]
    guard_rank = guardrails[guardrails['symbol'] == TICKER]

    if len(agg_rank) > 0:
        print(f"Aggressive Rank:  #{int(agg_rank.iloc[0]['rank'])} (Score: {agg_rank.iloc[0]['growth_potential_score_aggressive']:.1f})")
    else:
        print("Aggressive Rank:  Did not pass gates")

    if len(guard_rank) > 0:
        print(f"Guardrails Rank:  #{int(guard_rank.iloc[0]['rank'])} (Score: {guard_rank.iloc[0]['growth_potential_score_guardrails']:.1f})")
    else:
        print("Guardrails Rank:  Did not pass gates")
else:
    print(f"Stock {TICKER} not found in snapshot")


# ## 8. Growth Quality Heatmap
#
# Multi-dimensional growth analysis: which stocks have BROAD-BASED growth?

# In[ ]:


# Growth Quality Heatmap
snapshot = rankings['snapshot']
growth_cols = ['eps_growth_ttm', 'revenue_growth_ttm', 'ebitda_growth_ttm',
               'fcf_growth_ttm', 'net_income_growth_ttm', 'gross_profit_growth_ttm']
available_growth = [c for c in growth_cols if c in snapshot.columns]

if len(available_growth) >= 3:
    growth_df = snapshot[['symbol', 'sector'] + available_growth].copy()
    growth_df['growth_signals'] = (growth_df[available_growth] > 10).sum(axis=1)
    growth_df = growth_df.sort_values('growth_signals', ascending=False).head(20)

    if len(growth_df) > 0:
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(14, 8))
        labels = {
            'eps_growth_ttm': 'EPS', 'revenue_growth_ttm': 'Revenue',
            'ebitda_growth_ttm': 'EBITDA', 'fcf_growth_ttm': 'FCF',
            'net_income_growth_ttm': 'Net Income', 'gross_profit_growth_ttm': 'Gross Profit'
        }
        matrix = growth_df[available_growth].values
        matrix_clipped = np.clip(matrix, -50, 100)
        cmap = plt.cm.RdYlGn
        norm = mcolors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=100)

        im = ax.imshow(matrix_clipped, cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(len(available_growth)))
        ax.set_xticklabels([labels.get(c, c) for c in available_growth], rotation=45, ha='right')
        ax.set_yticks(range(len(growth_df)))
        ax.set_yticklabels([f"{row['symbol']} ({int(row['growth_signals'])})" for _, row in growth_df.iterrows()])
        ax.set_title(f'Growth Quality Heatmap (Top 20) - {SNAPSHOT_DATE}')

        for i in range(len(growth_df)):
            for j in range(len(available_growth)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 40 else 'black'
                    ax.text(j, i, f"{val:.0f}%", ha='center', va='center', fontsize=7, color=color)
                else:
                    ax.text(j, i, '---', ha='center', va='center', fontsize=7, color='gray')

        plt.colorbar(im, ax=ax, label='Growth %', shrink=0.8)
        plt.tight_layout()
        plt.show()
else:
    print('Not enough growth columns available for heatmap')


# ## 9. Technical Regime Summary
#
# Market-wide overview: how many stocks are oversold/overbought across indicators?

# In[ ]:


# Technical regime
snapshot = rankings['snapshot']

tech_indicators = {
    'rsi_14': ('RSI 14', 30, 70),
    'stochastic_k_1d': ('Stoch %K', 20, 80),
    'cci_20_1d': ('CCI 20', -100, 100),
}
available_tech = {k: v for k, v in tech_indicators.items() if k in snapshot.columns}

if available_tech:
    fig, axes = plt.subplots(1, len(available_tech), figsize=(5 * len(available_tech), 4))
    if len(available_tech) == 1:
        axes = [axes]

    for ax, (col, (name, os_thresh, ob_thresh)) in zip(axes, available_tech.items()):
        data = snapshot[col].dropna()
        n_os = (data <= os_thresh).sum()
        n_ob = (data >= ob_thresh).sum()
        n_neutral = len(data) - n_os - n_ob

        bars = ax.bar(['Oversold', 'Neutral', 'Overbought'], [n_os, n_neutral, n_ob],
                      color=['#22c55e', '#94a3b8', '#ef4444'])
        ax.set_title(f'{name} (OS<={os_thresh}, OB>={ob_thresh})')
        ax.set_ylabel('# Stocks')
        for bar, val in zip(bars, [n_os, n_neutral, n_ob]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', fontsize=10, fontweight='bold')

    plt.suptitle(f'Market Technical Regime - {SNAPSHOT_DATE}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    header = f"{'Indicator':15} | {'Oversold':>10} | {'Neutral':>10} | {'Overbought':>10}"
    print(f"\n{header}")
    print(f"{'-'*55}")
    for col, (name, os_thresh, ob_thresh) in available_tech.items():
        data = snapshot[col].dropna()
        n_os = (data <= os_thresh).sum()
        n_ob = (data >= ob_thresh).sum()
        n_neutral = len(data) - n_os - n_ob
        print(f"{name:15} | {n_os:10} | {n_neutral:10} | {n_ob:10}")
else:
    print('No technical indicator columns available')


# ## 10. Balance Sheet Strength Ranking

# In[ ]:


# Balance sheet fortress score
bs_cols = ['cash_to_debt_ratio', 'current_ratio', 'debt_to_assets_annual']
available_bs = [c for c in bs_cols if c in snapshot.columns]

if len(available_bs) >= 2:
    bs_df = snapshot[['symbol', 'sector'] + available_bs].dropna(subset=available_bs, how='all').copy()
    if 'cash_to_debt_ratio' in bs_df.columns:
        bs_df['cd_score'] = bs_df['cash_to_debt_ratio'].clip(0, 5) / 5
    else:
        bs_df['cd_score'] = 0.5
    if 'current_ratio' in bs_df.columns:
        bs_df['cr_score'] = bs_df['current_ratio'].clip(0, 4) / 4
    else:
        bs_df['cr_score'] = 0.5
    if 'debt_to_assets_annual' in bs_df.columns:
        bs_df['da_score'] = 1 - bs_df['debt_to_assets_annual'].clip(0, 1)
    else:
        bs_df['da_score'] = 0.5

    bs_df['fortress_score'] = (bs_df['cd_score'] + bs_df['cr_score'] + bs_df['da_score']) / 3 * 100
    bs_df = bs_df.sort_values('fortress_score', ascending=False)

    print(f"\n{'='*70}")
    print('BALANCE SHEET STRENGTH - Top 15 Financial Fortresses')
    print(f"{'='*70}")
    header = f"{'Symbol':12} | {'Cash/Debt':>10} | {'Current':>8} | {'D/A %':>8} | {'Score':>6}"
    print(header)
    print(f"{'-'*60}")
    for _, row in bs_df.head(15).iterrows():
        cd_val = row.get('cash_to_debt_ratio', np.nan)
        cd = f"{cd_val:.2f}" if not pd.isna(cd_val) else '  N/A'
        cr_val = row.get('current_ratio', np.nan)
        cr = f"{cr_val:.2f}" if not pd.isna(cr_val) else '  N/A'
        da_val = row.get('debt_to_assets_annual', np.nan)
        da = f"{da_val*100:.0f}%" if not pd.isna(da_val) else ' N/A'
        print(f"{row['symbol']:12} | {cd:>10} | {cr:>8} | {da:>8} | {row['fortress_score']:5.0f}")
else:
    print('Not enough balance sheet columns for fortress ranking')




# ## 11. Export Results

# In[ ]:


# Export top picks to CSV
output_dir = project_dir / 'outputs'
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Get rankings
aggressive = rankings['aggressive_growth']
guardrails = rankings['growth_with_guardrails']

# Top 20 aggressive
agg_path = output_dir / f"ngx_aggressive_top20_{SNAPSHOT_DATE}_{timestamp}.csv"
aggressive.head(20).to_csv(agg_path, index=False)
print(f"Saved: {agg_path}")

# Top 20 guardrails
guard_path = output_dir / f"ngx_guardrails_top20_{SNAPSHOT_DATE}_{timestamp}.csv"
guardrails.head(20).to_csv(guard_path, index=False)
print(f"Saved: {guard_path}")

