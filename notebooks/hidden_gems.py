import os
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Setup
import sys
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from IPython.display import display, HTML

# Add parent directory to path
sys.path.insert(0, str(Path.cwd().parent))

from analysis.hidden_gems import find_hidden_gems, find_value_turnarounds

from notebooks.nb_helpers import (
    load_report_cache,
    get_batch_report_insights,
    load_or_fetch_news,
    get_news_sentiment,
    data_freshness_banner,
    TONE_EMOJI,
    print_divergence_warnings,
    print_report_block,
    print_news_block
)

# Load caches
report_cache = load_report_cache()

# Configuration
DATA_DIR = Path(os.getenv('NAIJA_DATA_DIR', Path(__file__).resolve().parents[1] / 'data')) / 'snapshots'

print(f"📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"📁 Data Directory: {DATA_DIR}")


# ## 📊 Load Latest Snapshot

# In[2]:


# Find and load the latest snapshot
snapshots = sorted(DATA_DIR.glob('*/snapshot.parquet'))

if not snapshots:
    print("❌ No snapshots found! Run the data ingestion first.")
else:
    latest = snapshots[-1]
    snapshot_date = latest.parent.name

    df = pd.read_parquet(latest)

    # Add liquidity value if volume data exists
    if 'volume_1d' in df.columns:
        df['liquidity_value_1d'] = df['price'] * df['volume_1d']

    print(f"✅ Loaded snapshot: {snapshot_date}")
    print(f"📈 Total stocks: {len(df)}")
    print(f"📊 Total columns: {len(df.columns)}")
    print(f"\nAvailable snapshots:")
    for s in snapshots:
        marker = "👉" if s == latest else "  "
        print(f"  {marker} {s.parent.name}")


# ## 💎 Hidden Gems
# 
# Stocks with **strong fundamentals** (EPS/revenue growth) but **not yet hot** (low short-term momentum).
# 
# These are the REDSTAREX-style opportunities!

# In[3]:


# Check if required columns exist
required_cols = {
    'price': 'Last price',
    'perf_1m': 'Performance (1 Month %)',
    'perf_3m': 'Performance (3 Months %)',
    'eps_growth_ttm': 'EPS Growth TTM',
    'revenue_growth_ttm': 'Revenue Growth TTM',
    'net_margin_ttm': 'Net Margin TTM'
}

missing_cols = [f"{col} ({desc})" for col, desc in required_cols.items() if col not in df.columns]

if missing_cols:
    print(f"{'='*70}")
    print(f"⚠️  MISSING REQUIRED DATA FOR HIDDEN GEMS ANALYSIS")
    print(f"{'='*70}\n")
    print(f"Missing columns:")
    for col in missing_cols:
        print(f"  ❌ {col}")

    print(f"\n📋 TO FIX THIS:")
    print(f"\n1. Go to TradingView screener for Nigeria stocks")
    print(f"2. Add these columns:")
    print(f"   • Last (Price)")
    print(f"   • Change 1M %")
    print(f"   • Change 3M %") 
    print(f"   • Change 6M %")
    print(f"   • Change 1Y %")
    print(f"   • Change YTD %")
    print(f"   • Volume")
    print(f"\n3. Export to CSV")
    print(f"4. Re-run process_new_data.py to merge it")
    print(f"\n{'='*70}")

    print(f"\n✅ DATA YOU ALREADY HAVE:")
    have_cols = [col for col in required_cols.keys() if col in df.columns]
    for col in have_cols:
        print(f"  ✓ {col} ({required_cols[col]})")

    gems = pd.DataFrame()
else:
    # Find hidden gems
    gems = find_hidden_gems(df, top_n=25)

    # Summary stats
    cold_count = len(gems[gems['heat_status'].str.contains('Cold')]) if len(gems) > 0 else 0
    warming_count = len(gems[gems['heat_status'].str.contains('Warming')]) if len(gems) > 0 else 0

    print(f"\n{'='*70}")
    print(f"💎 HIDDEN GEMS SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Total gems found: {len(gems)}")
    print(f"  ❄️  Cold (best entry): {cold_count}")
    print(f"  🌤️  Warming up: {warming_count}")
    print(f"  🔥  Getting hot: {len(gems) - cold_count - warming_count}")


# In[4]:


# Display top gems with key metrics
if len(gems) > 0:
    display_cols = ['gem_rank', 'symbol', 'heat_status', 'price', 'hidden_gem_score',
                    'perf_1m', 'perf_3m', 'perf_1y', 'eps_growth_ttm', 'revenue_growth_ttm',
                    'roe_ttm', 'net_margin_ttm', 'warnings']
    display_df = gems[[c for c in display_cols if c in gems.columns]].copy()

    # Format for display
    display_df = display_df.rename(columns={
        'gem_rank': 'Rank',
        'symbol': 'Symbol',
        'heat_status': 'Status',
        'price': 'Price',
        'hidden_gem_score': 'Score',
        'perf_1m': '1M %',
        'perf_3m': '3M %',
        'perf_1y': '1Y %',
        'eps_growth_ttm': 'EPS Gr %',
        'revenue_growth_ttm': 'Rev Gr %',
        'roe_ttm': 'ROE %',
        'net_margin_ttm': 'Margin %',
        'warnings': 'Warnings'
    })

    # Round numeric columns
    for col in ['Score', '1M %', '3M %', '1Y %', 'EPS Gr %', 'Rev Gr %', 'ROE %', 'Margin %']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(1)

    print("\n🔝 TOP HIDDEN GEMS (v2.0 - value trap filtered):\n")
    display(display_df.head(15).style.background_gradient(subset=['Score'], cmap='Greens'))
else:
    print("No hidden gems found with current criteria.")


# In[ ]:


# 📊 Growth Confidence — multi-dimensional growth analysis
if len(gems) > 0:
    growth_cols = ['eps_growth_ttm', 'revenue_growth_ttm', 'ebitda_growth_ttm',
                   'fcf_growth_ttm', 'net_income_growth_ttm', 'gross_profit_growth_ttm']
    available_growth = [c for c in growth_cols if c in gems.columns]

    if len(available_growth) >= 3:
        print(f"\n{'='*100}")
        print(f"📊 GROWTH CONFIDENCE — How broad-based is the growth?")
        print(f"{'='*100}")
        print(f"  Stocks with 4+ growth signals >10% are HIGH conviction\n")

        # Count growth signals > 10%
        gems['growth_signals'] = (gems[available_growth] > 10).sum(axis=1)

        # Relative volume flag
        if 'relative_volume_1d' in gems.columns:
            gems['vol_flag'] = gems['relative_volume_1d'].apply(
                lambda x: '🔊' if x > 2.0 else ('📈' if x > 1.5 else ''))
        else:
            gems['vol_flag'] = ''

        col_labels = {
            'eps_growth_ttm': 'EPS',
            'revenue_growth_ttm': 'Rev',
            'ebitda_growth_ttm': 'EBITDA',
            'fcf_growth_ttm': 'FCF',
            'net_income_growth_ttm': 'NI',
            'gross_profit_growth_ttm': 'GP',
        }
        header_parts = [f"{'Symbol':12}"] + [f"{col_labels[c]:>8}" for c in available_growth] + [f"{'Signals':>8}", f"{'Vol':>4}"]
        print(f"  {'  '.join(header_parts)}")
        print(f"  {'-'*len('  '.join(header_parts))}")

        for _, row in gems.head(15).iterrows():
            parts = [f"{row['symbol']:12}"]
            for c in available_growth:
                v = row.get(c, np.nan)
                if pd.isna(v):
                    parts.append(f"{'—':>8}")
                elif v > 10:
                    parts.append(f"{v:>+7.0f}%")
                else:
                    parts.append(f"{v:>+7.0f}%")
            n_signals = int(row['growth_signals'])
            max_signals = len(available_growth)
            level = '🟢' if n_signals >= 4 else ('🟡' if n_signals >= 2 else '🔴')
            parts.append(f"{n_signals}/{max_signals} {level} ")
            parts.append(f"{row['vol_flag']:>4}")
            print(f"  {'  '.join(parts)}")


# ## ❄️ Cold Opportunities (Best Entry Points)
# 
# These stocks have **negative 1-month momentum** but strong fundamentals.
# 
# Like REDSTAREX was on Dec 17 (0% 1M, -16% 3M, but +159% EPS growth)!

# In[5]:


# Filter for cold opportunities — enhanced with technical confirmation
if len(gems) > 0:
    cold_gems = gems[gems['heat_status'].str.contains('Cold')].copy()

    if len(cold_gems) > 0:
        print(f"\n{'='*70}")
        print(f"❄️ COLD OPPORTUNITIES - Best Entry Points")
        print(f"{'='*70}")
        print(f"  Tech signals: Stoch <20 + CCI <-100 = strong oversold confirmation\n")

        for _, row in cold_gems.iterrows():
            # Technical confluence signals
            tech_signals = []
            stoch_k = row.get('stochastic_k_1d', np.nan)
            stoch_d = row.get('stochastic_d_1d', np.nan)
            cci     = row.get('cci_20_1d', np.nan)
            rvol    = row.get('relative_volume_1d', np.nan)

            if not pd.isna(stoch_k) and stoch_k < 20:
                tech_signals.append(f"Stoch K={stoch_k:.0f}")
            if not pd.isna(cci) and cci < -100:
                tech_signals.append(f"CCI={cci:.0f}")
            if not pd.isna(rvol) and rvol > 2.0:
                tech_signals.append(f"🔊 RVOL={rvol:.1f}x")

            confluence = '⚡ OVERSOLD CONFIRMED' if len(tech_signals) >= 2 else ''

            print(f"🎯 {row['symbol']} {confluence}")
            print(f"   Price: ₦{row['price']:.2f} | Score: {row['hidden_gem_score']:.1f}")
            print(f"   1M: {row['perf_1m']:+.1f}% | 3M: {row['perf_3m']:+.1f}% | 1Y: {row.get('perf_1y', 0):+.1f}%")
            print(f"   EPS Growth: {row['eps_growth_ttm']:.1f}% | Revenue Growth: {row['revenue_growth_ttm']:.1f}%")
            print(f"   ROE: {row['roe_ttm']:.1f}% | Net Margin: {row['net_margin_ttm']:.1f}%")
            if tech_signals:
                print(f"   📊 Tech: {' | '.join(tech_signals)}")
            # v2.0: Show warning flags if any
            warnings = row.get('warnings', '')
            if warnings:
                print(f"   ⚠️  WARNINGS: {warnings}")
            print()
    else:
        print("\n⚠️ No cold opportunities right now.")
        print("The market is hot - most gems are already warming up or running.")
        print("\nConsider:")
        print("  1. Wait for a market pullback")
        print("  2. Look at 🌤️ Warming stocks (still decent entry)")
        print("  3. Check turnaround candidates below")
else:
    print("No gems data available.")


# ## 🔄 Value Turnaround Candidates
# 
# Stocks that are **beaten down** (negative 3-month momentum) but still **profitable**.
# 
# Higher risk, but potential for strong recovery.

# In[6]:


# Find turnaround candidates
turnarounds = find_value_turnarounds(df, top_n=10)

if len(turnarounds) > 0:
    print(f"\n{'='*70}")
    print(f"🔄 VALUE TURNAROUND CANDIDATES")
    print(f"{'='*70}\n")

    for _, row in turnarounds.head(8).iterrows():
        warnings = row.get('warnings', '')
        warn_str = f"\n    ⚠️  {warnings}" if warnings else ""
        print(f"{row['turnaround_rank']:2}. {row['symbol']}")
        print(f"    Price: ₦{row['price']:.2f} | Score: {row['turnaround_score']:.1f}")
        print(f"    3M: {row['perf_3m']:+.1f}% (beaten down)")
        print(f"    But: Margin {row['net_margin_ttm']:.1f}% | ROE {row.get('roe_ttm', 0):.1f}%{warn_str}")
        print()
else:
    print("No turnaround candidates found.")


# ## 📈 Your PORTFOLIO
# 
# Add symbols here to track specific stocks.

# In[7]:


# Your watchlist - all current portfolio positions (Updated March 2026)
WATCHLIST = [
    'ABBEYBDS',     # 23K shares @ ₦11.00
    'ABCTRANS',
    'AIICO',
    'BETAGLAS',
    'BUACEMENT',
    'CUSTODIAN',
    'GEREGU',
    'HONYFLOUR',
    'IKEJAHOTEL',
    'INFINITY',     # 2.5K shares @ ₦18.85
    'JBERGER',
    'LEARNAFRCA',
    'MECURE',       # Broker 2 — 3K shares @ ₦77.25
    'NAHCO',
    'NASCON',
    'NGXGROUP',     # Broker 2 — 3K shares @ ₦112.12
    'OKOMUOIL',
    'PRESCO',
    'ROYALEX',
    'SKYAVN',       # 1.5K shares @ ₦140.00
    'TIP',
    'TRANSPOWER',
    'UPDC',         # Broker 2 — 5.75K shares @ ₦5.35
    'VFDGROUP',
    'VITAFOAM',
    'WAPCO',
]

def rsi_signal(rsi):
    if pd.isna(rsi):      return "  —  "
    if rsi >= 70:         return "🔴 OB "   # Overbought
    if rsi <= 30:         return "🟢 OS "   # Oversold
    if rsi >= 60:         return "🟠 Hi "
    if rsi <= 40:         return "🔵 Lo "
    return               "⚪ OK "

print(f"\n{'='*100}")
print(f"📋 MY PORTFOLIO - {len(WATCHLIST)} positions")
print(f"{'='*100}")
print(f"{'Symbol':15} | {'Price':>9} | {'1W%':>7} | {'1M%':>7} | {'RSI':>6} | {'mRSI':>6} | {'Stoch':>6} | {'RVOL':>5} | {'RSI Signal':10} | Status")
print(f"{'-'*120}")

MONTHLY_RSI_COL = 'rsi_14_1m'

for symbol in WATCHLIST:
    stock = df[df['symbol'] == symbol]
    if len(stock) > 0:
        row = stock.iloc[0]
        perf_1m = row.get('perf_1m', 0) or 0
        perf_1w = row.get('perf_1w', 0) or 0
        rsi     = row.get('rsi_14', np.nan)
        mrsi    = row.get(MONTHLY_RSI_COL, np.nan)
        stoch_k = row.get('stochastic_k_1d', np.nan)
        rvol    = row.get('relative_volume_1d', np.nan)

        if perf_1m < 0:    status = "❄️ Cold"
        elif perf_1m < 10: status = "🌤️ Warming"
        elif perf_1m < 20: status = "🔥 Hot"
        else:              status = "🚀 Running"

        rsi_str   = f"{rsi:5.1f}" if not pd.isna(rsi) else "  N/A"
        mrsi_str  = f"{mrsi:5.1f}" if not pd.isna(mrsi) else "  N/A"
        stoch_str = f"{stoch_k:5.0f}" if not pd.isna(stoch_k) else "  N/A"
        rvol_str  = f"{rvol:4.1f}x" if not pd.isna(rvol) else " N/A"
        if not pd.isna(rvol) and rvol > 2.0:
            rvol_str = f"🔊{rvol:.1f}"

        print(f"{symbol:15} | ₦{row['price']:8.2f} | {perf_1w:+6.1f}% | {perf_1m:+6.1f}% | {rsi_str} | {mrsi_str} | {stoch_str} | {rvol_str:>5} | {rsi_signal(rsi):10} | {status}")
    else:
        print(f"{symbol:15} | Not found in data")

print(f"\n  RSI key: 🔴 Overbought (≥70)  🟠 High (60-69)  ⚪ Neutral (40-59)  🔵 Low (31-40)  🟢 Oversold (≤30)")
print(f"  mRSI = Monthly RSI(14) — trend-level signal (30-50 = ‘undiscovered’)")


# In[8]:


# RSI Signals — Overbought / Oversold across full market
RSI_COL = 'rsi_14'

rsi_df = df[df[RSI_COL].notna()][['symbol', 'sector', 'price', 'perf_1m', 'perf_1w', RSI_COL]].copy()
rsi_df = rsi_df.rename(columns={RSI_COL: 'rsi'})

oversold   = rsi_df[rsi_df['rsi'] <= 30].sort_values('rsi')
overbought = rsi_df[rsi_df['rsi'] >= 70].sort_values('rsi', ascending=False)

# --- OVERSOLD (potential buy / entry zone) ---
print(f"\n{'='*70}")
print(f"🟢 OVERSOLD — RSI ≤ 30  (potential entry zone)")
print(f"{'='*70}")
if len(oversold):
    print(f"{'Symbol':12} | {'RSI':>5} | {'Price':>9} | {'1W%':>7} | {'1M%':>7} | Sector")
    print(f"{'-'*70}")
    for _, r in oversold.iterrows():
        watch = "⭐" if r['symbol'] in WATCHLIST else "  "
        print(f"{watch}{r['symbol']:10} | {r['rsi']:5.1f} | ₦{r['price']:8.2f} | {r['perf_1w']:+6.1f}% | {r['perf_1m']:+6.1f}% | {r.get('sector','')}")
else:
    print("  None currently — market not in oversold territory")

# --- OVERBOUGHT (consider taking profit) ---
print(f"\n{'='*70}")
print(f"🔴 OVERBOUGHT — RSI ≥ 70  (consider taking profit / avoid new entries)")
print(f"{'='*70}")
if len(overbought):
    print(f"{'Symbol':12} | {'RSI':>5} | {'Price':>9} | {'1W%':>7} | {'1M%':>7} | Sector")
    print(f"{'-'*70}")
    for _, r in overbought.iterrows():
        watch = "⭐" if r['symbol'] in WATCHLIST else "  "
        print(f"{watch}{r['symbol']:10} | {r['rsi']:5.1f} | ₦{r['price']:8.2f} | {r['perf_1w']:+6.1f}% | {r['perf_1m']:+6.1f}% | {r.get('sector','')}")
else:
    print("  None currently")

# --- PORTFOLIO RSI SUMMARY ---
portfolio_rsi = rsi_df[rsi_df['symbol'].isin(WATCHLIST)].sort_values('rsi', ascending=False)
print(f"\n{'='*70}")
print(f"📊 PORTFOLIO RSI DISTRIBUTION")
print(f"{'='*70}")
ob  = portfolio_rsi[portfolio_rsi['rsi'] >= 70]
hi  = portfolio_rsi[(portfolio_rsi['rsi'] >= 60) & (portfolio_rsi['rsi'] < 70)]
ok  = portfolio_rsi[(portfolio_rsi['rsi'] >= 40) & (portfolio_rsi['rsi'] < 60)]
lo  = portfolio_rsi[(portfolio_rsi['rsi'] >= 30) & (portfolio_rsi['rsi'] < 40)]
os_ = portfolio_rsi[portfolio_rsi['rsi'] < 30]
print(f"  🔴 Overbought  (≥70): {len(ob):2}  — {', '.join(ob['symbol'].tolist()) or 'none'}")
print(f"  🟠 High        (60-69): {len(hi):2}  — {', '.join(hi['symbol'].tolist()) or 'none'}")
print(f"  ⚪ Neutral     (40-59): {len(ok):2}  — {', '.join(ok['symbol'].tolist()) or 'none'}")
print(f"  🔵 Low         (31-40): {len(lo):2}  — {', '.join(lo['symbol'].tolist()) or 'none'}")
print(f"  🟢 Oversold    (≤30):   {len(os_):2}  — {', '.join(os_['symbol'].tolist()) or 'none'}")


# ## 💡 Investment Strategy
# 
# ### Position Sizing by Heat Status:
# 
# | Heat Status | Entry Quality | Suggested Position |
# |-------------|---------------|--------------------|
# | ❄️ Cold | **Best** | 10-15% of portfolio |
# | 🌤️ Warming | Good | 8-12% of portfolio |
# | 🔥 Hot | Fair | 5-8% of portfolio |
# | 🚀 Running | Wait for pullback | Hold or reduce |
# 
# ### REDSTAREX Case Study:
# - **Dec 17:** Price ₦9.20, 1M: 0%, 3M: -16%, EPS: +159% → ❄️ Cold
# - **Dec 23 (your entry):** Price ₦9.01 → Perfect timing!
# - **Jan 20:** Price ₦15.90 → **+76.5% return**
# 
# ### Key Takeaways:
# 1. **Fundamentals matter:** High EPS/revenue growth signals future price appreciation
# 2. **Short-term weakness = opportunity:** Cold stocks are overlooked by the crowd
# 3. **Size up on conviction:** If you find a cold gem with strong fundamentals, go bigger!

# In[9]:


# Summary
print(f"\n{'='*70}")
print(f"📊 WEEKLY SUMMARY - {datetime.now().strftime('%Y-%m-%d')}")
print(f"{'='*70}")
print(f"\n  Stocks analyzed: {len(df)}")
print(f"  Hidden gems found: {len(gems)}")
print(f"  ❄️ Cold (best entry): {cold_count}")
print(f"  🔄 Turnaround candidates: {len(turnarounds)}")
print(f"\n{'='*70}")
print("\n🔔 Next steps:")
print("  1. Review cold opportunities above")
print("  2. Research top gems fundamentals")
print("  3. Set position sizes based on conviction")
print("  4. Run this notebook again next week!")

