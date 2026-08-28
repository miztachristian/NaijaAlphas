import os
#!/usr/bin/env python
# coding: utf-8

# # 🏁 Portfolio Momentum Ranking
# 
# **Reusable notebook** — run every week/month to see which stocks in your portfolio have the strongest near-term momentum.
# 
# Just run all cells after each new data snapshot!

# In[1]:


# Setup
import sys
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from IPython.display import display, HTML

sys.path.insert(0, str(Path.cwd().parent))

DATA_DIR = Path(os.getenv('NAIJA_DATA_DIR', Path(__file__).resolve().parents[1] / 'data')) / 'snapshots'

print(f"📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"📁 Data Directory: {DATA_DIR}")


# ## 📋 Your Portfolio
# 
# Update this list whenever you buy/sell a stock.

# In[2]:


# =============================================
# 📝 UPDATE YOUR PORTFOLIO SYMBOLS HERE
# =============================================
# Just the symbols — this notebook only needs tickers

MY_STOCKS = [
    'ABBEYBDS',
    'ABCTRANS',
    'AIICO',
    'BETAGLAS',
    'BUACEMENT',
    'CUSTODIAN',
    'GEREGU',
    'HONYFLOUR',
    'IKEJAHOTEL',
    'INFINITY',
    'JBERGER',
    'LEARNAFRCA',
    'MECURE',
    'NAHCO',
    'NASCON',
    'NGXGROUP',
    'OKOMUOIL',
    'PRESCO',
    'ROYALEX',
    'SKYAVN',
    'TIP',
    'TRANSPOWER',
    'UPDC',
    'VFDGROUP',
    'VITAFOAM',
    'WAPCO',
]

print(f"📌 Tracking {len(MY_STOCKS)} stocks")


# ## 📊 Load Latest Snapshot

# In[3]:


# Load the most recent snapshot
snapshots = sorted(DATA_DIR.glob('*/snapshot.parquet'))

if not snapshots:
    raise RuntimeError('❌ No snapshots found! Run process_new_data.py first.')

latest = snapshots[-1]
snapshot_date = latest.parent.name
df = pd.read_parquet(latest)

if 'volume_1d' in df.columns and 'liquidity_value_1d' not in df.columns:
    df['liquidity_value_1d'] = df['price'] * df['volume_1d']

print(f"✅ Loaded snapshot: {snapshot_date}")
print(f"📈 Total stocks in market: {len(df)}")
print(f"\nAvailable snapshots:")
for s in snapshots:
    marker = '👉' if s == latest else '  '
    print(f"  {marker} {s.parent.name}")


# ## 🏁 Weekly Momentum Ranking
# 
# Ranks your portfolio stocks by **short-term momentum** (1D + 1W + 1M).
# 
# Best for deciding: *"Which of my stocks are running right now?"*

# In[4]:


# === WEEKLY MOMENTUM ===
# Weights: 1D = 10%, 1W = 45%, 1M = 45%
# Change these if you want to emphasise different timeframes

W_1D = 0.10
W_1W = 0.45
W_1M = 0.45

def safe_val(x):
    """Return 0 for NaN/None values."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return float(x)

# Filter to portfolio stocks
port = df[df['symbol'].isin(MY_STOCKS)].copy()
missing = sorted(set(MY_STOCKS) - set(port['symbol'].unique()))
if missing:
    print(f"⚠️  Not in snapshot: {', '.join(missing)}\n")

# Calculate weekly momentum score
port['weekly_score'] = port.apply(
    lambda r: W_1D * safe_val(r.get('price_change_1d', 0))
           + W_1W * safe_val(r.get('perf_1w', 0))
           + W_1M * safe_val(r.get('perf_1m', 0)),
    axis=1
)

weekly = port.sort_values('weekly_score', ascending=False).reset_index(drop=True)

# Display
print(f"{'='*75}")
print(f"🏁 WEEKLY MOMENTUM RANKING — {snapshot_date}")
print(f"{'='*75}")
print(f"   Weights: 1D {W_1D:.0%}  |  1W {W_1W:.0%}  |  1M {W_1M:.0%}")
print(f"{'='*75}\n")

for i, row in weekly.iterrows():
    sym = row['symbol']
    price = row.get('price', 0)
    d1 = safe_val(row.get('price_change_1d', 0))
    w1 = safe_val(row.get('perf_1w', 0))
    m1 = safe_val(row.get('perf_1m', 0))
    score = row['weekly_score']

    # Emoji badge
    if score >= 20:
        badge = '🚀'
    elif score >= 10:
        badge = '🔥'
    elif score >= 3:
        badge = '🌤️'
    elif score >= 0:
        badge = '😐'
    else:
        badge = '❄️'

    print(f"{i+1:2}. {badge} {sym:<12} ₦{price:>10,.2f}  │  "
          f"1D {d1:+6.1f}%  │  1W {w1:+6.1f}%  │  1M {m1:+6.1f}%  │  Score {score:+5.1f}")

# Summary
rockets = len(weekly[weekly['weekly_score'] >= 20])
hot     = len(weekly[(weekly['weekly_score'] >= 10) & (weekly['weekly_score'] < 20)])
warm    = len(weekly[(weekly['weekly_score'] >= 3) & (weekly['weekly_score'] < 10)])
flat_n  = len(weekly[(weekly['weekly_score'] >= 0) & (weekly['weekly_score'] < 3)])
cold    = len(weekly[weekly['weekly_score'] < 0])

print(f"\n{'='*75}")
print(f"  🚀 Running: {rockets}  |  🔥 Hot: {hot}  |  🌤️ Warm: {warm}  |  😐 Flat: {flat_n}  |  ❄️ Cold: {cold}")
print(f"{'='*75}")


# ## 📅 Monthly Momentum Ranking
# 
# Ranks by **medium-term momentum** (1M + 3M + YTD).
# 
# Best for deciding: *"Which stocks have the strongest trend this month/quarter?"*

# In[5]:


# === MONTHLY MOMENTUM ===
# Weights: 1M = 30%, 3M = 40%, YTD = 30%

M_1M  = 0.30
M_3M  = 0.40
M_YTD = 0.30

port['monthly_score'] = port.apply(
    lambda r: M_1M  * safe_val(r.get('perf_1m', 0))
           + M_3M  * safe_val(r.get('perf_3m', 0))
           + M_YTD * safe_val(r.get('perf_ytd', 0)),
    axis=1
)

monthly = port.sort_values('monthly_score', ascending=False).reset_index(drop=True)

print(f"{'='*80}")
print(f"📅 MONTHLY MOMENTUM RANKING — {snapshot_date}")
print(f"{'='*80}")
print(f"   Weights: 1M {M_1M:.0%}  |  3M {M_3M:.0%}  |  YTD {M_YTD:.0%}")
print(f"{'='*80}\n")

for i, row in monthly.iterrows():
    sym = row['symbol']
    price = row.get('price', 0)
    m1  = safe_val(row.get('perf_1m', 0))
    m3  = safe_val(row.get('perf_3m', 0))
    ytd = safe_val(row.get('perf_ytd', 0))
    score = row['monthly_score']

    if score >= 30:
        badge = '🚀'
    elif score >= 15:
        badge = '🔥'
    elif score >= 5:
        badge = '🌤️'
    elif score >= 0:
        badge = '😐'
    else:
        badge = '❄️'

    print(f"{i+1:2}. {badge} {sym:<12} ₦{price:>10,.2f}  │  "
          f"1M {m1:+6.1f}%  │  3M {m3:+6.1f}%  │  YTD {ytd:+6.1f}%  │  Score {score:+5.1f}")

# Summary
rockets = len(monthly[monthly['monthly_score'] >= 30])
hot     = len(monthly[(monthly['monthly_score'] >= 15) & (monthly['monthly_score'] < 30)])
warm    = len(monthly[(monthly['monthly_score'] >= 5) & (monthly['monthly_score'] < 15)])
flat_n  = len(monthly[(monthly['monthly_score'] >= 0) & (monthly['monthly_score'] < 5)])
cold    = len(monthly[monthly['monthly_score'] < 0])

print(f"\n{'='*80}")
print(f"  🚀 Running: {rockets}  |  🔥 Hot: {hot}  |  🌤️ Warm: {warm}  |  😐 Flat: {flat_n}  |  ❄️ Cold: {cold}")
print(f"{'='*80}")


# ## 💪 Fundamentals + Momentum Combined
# 
# Combines momentum with key fundamentals (EPS growth, revenue growth, ROE, margin).
# 
# Best for: *"Which stocks have BOTH the momentum AND the fundamentals to keep running?"*

# In[6]:


# === COMBINED SCORE: Momentum (40%) + Fundamentals (35%) + Valuation (25%) ===

def fundamentals_score(row):
    """Score 0-100 based on key fundamentals using continuous scoring."""
    eps_g = safe_val(row.get('eps_growth_ttm', 0))
    rev_g = safe_val(row.get('revenue_growth_ttm', 0))
    roe   = safe_val(row.get('roe_ttm', 0))
    margin = safe_val(row.get('net_margin_ttm', 0))

    # EPS Growth (0-30 pts) — continuous
    eps_pts = np.clip(eps_g / 100, 0, 1) * 30

    # Revenue Growth (0-25 pts) — continuous
    rev_pts = np.clip(rev_g / 50, 0, 1) * 25

    # ROE (0-25 pts) — continuous
    roe_pts = np.clip(roe / 25, 0, 1) * 25

    # Net Margin (0-20 pts) — continuous
    margin_pts = np.clip(margin / 20, 0, 1) * 20

    return eps_pts + rev_pts + roe_pts + margin_pts


def valuation_score(row):
    """Score 0-100 based on valuation metrics (higher = cheaper / better value)."""
    score = 0
    components = 0

    # P/E ratio (0-35 pts) — lower is better, but negative P/E = no earnings
    pe = safe_val(row.get('pe_ratio', 0))
    if pe > 0:
        # Sweet spot: 5-20x P/E
        if pe <= 20:
            score += np.clip((20 - pe) / 15, 0, 1) * 35  # 5x→35pts, 20x→0pts
        # Expensive: 20-50x
        else:
            score += 0  # No points for expensive stocks
        components += 1

    # PEG ratio (0-35 pts) — < 1 is ideal
    peg = safe_val(row.get('price_to_earning_to_growth_trailing_12_months', 0))
    if peg > 0:
        peg_pts = np.clip((2 - peg) / 2, 0, 1) * 35  # PEG 0→35pts, 2→0pts
        score += peg_pts
        components += 1

    # Earnings yield (0-30 pts) — higher is better (inverse of P/E)
    ey = safe_val(row.get('earnings_yield_ttm', 0))
    if ey > 0:
        score += np.clip(ey / 15, 0, 1) * 30  # 15%→30pts, 0%→0pts
        components += 1

    if components == 0:
        return 25  # Neutral score when no valuation data
    return (score / components) * (100 / 35)  # Normalize to 0-100 range


# Normalize weekly score to 0-100 scale
w_min = weekly['weekly_score'].min()
w_max = weekly['weekly_score'].max()
w_range = w_max - w_min if w_max != w_min else 1

port['fund_score'] = port.apply(fundamentals_score, axis=1)
port['val_score'] = port.apply(valuation_score, axis=1)
port['norm_momentum'] = (port['weekly_score'] - w_min) / w_range * 100
port['combined_score'] = 0.40 * port['norm_momentum'] + 0.35 * port['fund_score'] + 0.25 * port['val_score']

combined = port.sort_values('combined_score', ascending=False).reset_index(drop=True)

print(f"{'='*100}")
print(f"💪 MOMENTUM + FUNDAMENTALS + VALUATION — {snapshot_date}")
print(f"{'='*100}")
print(f"   Combined = 40% Momentum + 35% Fundamentals + 25% Valuation")
print(f"{'='*100}\n")

for i, row in combined.iterrows():
    sym    = row['symbol']
    price  = row.get('price', 0)
    w_sc   = row['weekly_score']
    f_sc   = row['fund_score']
    c_sc   = row['combined_score']
    v_sc   = row['val_score']
    eps_g  = safe_val(row.get('eps_growth_ttm', 0))
    rev_g  = safe_val(row.get('revenue_growth_ttm', 0))
    m1     = safe_val(row.get('perf_1m', 0))

    if c_sc >= 65:
        badge = '🏆'
    elif c_sc >= 45:
        badge = '✅'
    elif c_sc >= 25:
        badge = '🟡'
    else:
        badge = '⚠️'

    print(f"{i+1:2}. {badge} {sym:<12} ₦{price:>10,.2f}  │  "
          f"1M {m1:+6.1f}%  │  EPS {eps_g:+7.1f}%  │  Rev {rev_g:+6.1f}%  │  "
          f"Mom {w_sc:+5.1f}  Fund {f_sc:4.0f}  Val {v_sc:4.0f}  │  Combined {c_sc:4.1f}")

print(f"\n{'='*100}")
print(f"  🏆 = Best overall (momentum + fundamentals + value)")
print(f"  ✅ = Good")
print(f"  🟡 = Watch (mixed signals)")
print(f"  ⚠️  = Weak (consider selling / setting stop-loss)")
print(f"{'='*100}")


# ## 📈 Snapshot Comparison (Week-over-Week)
# 
# Compare momentum changes between two snapshots to see **who is improving / declining**.

# In[7]:


# === WEEK-OVER-WEEK COMPARISON ===
# Compares the latest snapshot to the previous one

if len(snapshots) >= 2:
    prev_path = snapshots[-2]
    prev_date = prev_path.parent.name
    prev_df = pd.read_parquet(prev_path)

    prev_port = prev_df[prev_df['symbol'].isin(MY_STOCKS)].copy()
    prev_port['weekly_score'] = prev_port.apply(
        lambda r: W_1D * safe_val(r.get('price_change_1d', 0))
               + W_1W * safe_val(r.get('perf_1w', 0))
               + W_1M * safe_val(r.get('perf_1m', 0)),
        axis=1
    )

    # Merge
    comp = port[['symbol', 'price', 'perf_1w', 'perf_1m', 'weekly_score']].merge(
        prev_port[['symbol', 'price', 'perf_1w', 'perf_1m', 'weekly_score']],
        on='symbol', suffixes=('_now', '_prev'), how='left'
    )
    comp['price_change'] = ((comp['price_now'] - comp['price_prev']) / comp['price_prev'] * 100).round(1)
    comp['score_change'] = (comp['weekly_score_now'] - comp['weekly_score_prev']).round(1)
    comp = comp.sort_values('score_change', ascending=False).reset_index(drop=True)

    print(f"{'='*80}")
    print(f"📈 SNAPSHOT COMPARISON: {prev_date} → {snapshot_date}")
    print(f"{'='*80}\n")

    for i, row in comp.iterrows():
        sym = row['symbol']
        p_now = row['price_now']
        p_chg = safe_val(row.get('price_change', 0))
        s_chg = safe_val(row.get('score_change', 0))

        if s_chg > 5:
            arrow = '⬆️  Improving'
        elif s_chg > 0:
            arrow = '↗️  Slightly better'
        elif s_chg > -5:
            arrow = '↘️  Slightly weaker'
        else:
            arrow = '⬇️  Declining'

        print(f"{i+1:2}. {sym:<12} ₦{p_now:>10,.2f}  │  Price {p_chg:+6.1f}%  │  Score Δ {s_chg:+6.1f}  │  {arrow}")

    print(f"\n{'='*80}")
else:
    print('⚠️  Need at least 2 snapshots for comparison. Only 1 available.')


# ## 🗒️ Quick Summary

# In[8]:


# === SUMMARY ===
print(f"\n{'='*75}")
print(f"📊 PORTFOLIO MOMENTUM SUMMARY — {snapshot_date}")
print(f"{'='*75}\n")

# Top 3 weekly
top3_w = weekly.head(3)
print(f"🏁 Best WEEKLY momentum:")
for _, r in top3_w.iterrows():
    print(f"   🚀 {r['symbol']:12} — 1W {safe_val(r.get('perf_1w',0)):+.1f}%, 1M {safe_val(r.get('perf_1m',0)):+.1f}%, Score {r['weekly_score']:+.1f}")

# Top 3 monthly
top3_m = monthly.head(3)
print(f"\n📅 Best MONTHLY momentum:")
for _, r in top3_m.iterrows():
    print(f"   🔥 {r['symbol']:12} — 1M {safe_val(r.get('perf_1m',0)):+.1f}%, 3M {safe_val(r.get('perf_3m',0)):+.1f}%, Score {r['monthly_score']:+.1f}")

# Top 3 combined
top3_c = combined.head(3)
print(f"\n💪 Best COMBINED (momentum + fundamentals):")
for _, r in top3_c.iterrows():
    print(f"   🏆 {r['symbol']:12} — Combined {r['combined_score']:.1f}")

# Weakest
bot3 = weekly.tail(3)
print(f"\n⚠️  Weakest momentum (consider stop-loss):")
for _, r in bot3.iterrows():
    print(f"   ❄️  {r['symbol']:12} — 1W {safe_val(r.get('perf_1w',0)):+.1f}%, 1M {safe_val(r.get('perf_1m',0)):+.1f}%, Score {r['weekly_score']:+.1f}")

print(f"\n{'='*75}")
print(f"\n💡 Next: Run this again after the next snapshot for fresh rankings!")


# ## 💰 Deployment Plan (sample capital)
# **Goal:** Maximize growth through Feb–March by adding to winners, entering new positions, and trimming laggards.
# 

# In[9]:


# ══════════════════════════════════════════════════════════════════════════
# 💰 DEPLOYMENT ANALYSIS (sample capital)
# ══════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings('ignore')

# ─── Sample Portfolio ───
# Illustrative holdings so the notebook runs standalone.
# In production this is loaded from PORTFOLIO_DIR/holdings.json (git-ignored).
MY_PORTFOLIO = {
    'GTCO':      {'shares': 1000, 'avg_cost': 50.00},
    'DANGCEM':   {'shares': 100,  'avg_cost': 450.00},
    'ZENITHBANK':{'shares': 1000, 'avg_cost': 40.00},
    'NESTLE':    {'shares': 50,   'avg_cost': 1000.00},
}

BUDGET_TOMORROW = 1_000_000  # sample budget (NGN)
BUDGET_MONDAY = 500_000      # sample budget (NGN)

# ─── Helper ───
def get_col(name):
    """Find column in df by keywords"""
    for c in df.columns:
        if all(k in c.lower() for k in name):
            return c
    return None

COL_RSI = get_col(['rsi']) or get_col(['relative', 'strength']) or 'rsi_1d'
COL_REV_G = get_col(['revenue', 'growth']) or 'revenue_growth_ttm'
COL_EPS_G = get_col(['eps', 'growth']) or 'eps_growth_ttm'
COL_NET_MARGIN = get_col(['net', 'margin']) or 'net_margin_ttm'
COL_ROE = get_col(['roe']) or 'roe_ttm'
COL_VOL = get_col(['volume']) or 'volume_1d'
COL_MCAP = get_col(['market', 'cap']) or 'market_cap'
COL_PE = get_col(['price', 'earn']) or get_col(['pe_ratio']) or 'pe_ratio'

# ══════════════════════════════════════════════════════════════════════════
# PART 1: CURRENT PORTFOLIO HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════
print(f"{'='*110}")
print(f"📊 PART 1: CURRENT PORTFOLIO HEALTH — Which positions to ADD / HOLD / SELL")
print(f"{'='*110}\n")

print(f"{'Rank':>4} {'Symbol':12s} {'Price':>9s} {'Cost':>8s} {'P&L%':>7s} {'Value':>11s} "
      f"{'1W%':>7s} {'1M%':>7s} {'3M%':>7s} {'RSI':>6s} {'RevGr%':>8s} {'Verdict'}")
print(f"{'-'*130}")

results = []
for sym, pos in MY_PORTFOLIO.items():
    stock = df[df['symbol'] == sym]
    if stock.empty:
        continue
    s = stock.iloc[0]
    price = s['price']
    pnl_pct = (price - pos['avg_cost']) / pos['avg_cost'] * 100
    value = pos['shares'] * price
    cost_val = pos['shares'] * pos['avg_cost']
    pnl_val = value - cost_val
    w1 = safe_val(s.get('perf_1w', 0))
    m1 = safe_val(s.get('perf_1m', 0))
    m3 = safe_val(s.get('perf_3m', 0))
    rsi = safe_val(s.get(COL_RSI, 50))
    rev_g = safe_val(s.get(COL_REV_G, 0))
    eps_g = safe_val(s.get(COL_EPS_G, 0))
    margin = safe_val(s.get(COL_NET_MARGIN, 0))
    roe = safe_val(s.get(COL_ROE, 0))
    vol = safe_val(s.get(COL_VOL, 0))

    # Scoring for action
    momentum = w1 * 0.4 + m1 * 0.4 + m3 * 0.2
    fund_quality = min(rev_g * 0.3, 15) + min(max(eps_g, 0) * 0.2, 15) + min(max(margin, 0) * 0.3, 10) + min(max(roe, 0) * 0.2, 10)

    # Determine verdict
    if rsi > 85:
        rsi_flag = '🔴OB'
    elif rsi > 70:
        rsi_flag = '🟡WM'
    elif rsi > 50:
        rsi_flag = '🟢OK'
    elif rsi > 30:
        rsi_flag = '🔵CL'
    else:
        rsi_flag = '⚪OS'

    # Action recommendation
    if momentum > 20 and rsi < 75 and fund_quality > 15:
        action = '🟢 ADD MORE — Strong momentum, healthy RSI, good fundamentals'
    elif momentum > 20 and rsi >= 75:
        action = '🟡 HOLD — Great momentum but RSI stretched, wait for pullback to add'
    elif momentum > 10 and fund_quality > 10:
        action = '🟢 ADD — Good momentum + fundamentals'
    elif momentum > 10 and rsi >= 80:
        action = '🟡 HOLD — Running but overbought'
    elif momentum > 5:
        action = '🟡 HOLD — Moderate momentum'
    elif momentum > 0 and fund_quality > 15:
        action = '🟡 HOLD — Fundamentals strong, momentum building'
    elif momentum < -3 and pnl_pct > 20:
        action = '🟠 TRIM — Take profits, momentum fading'
    elif momentum < -5 and fund_quality < 10:
        action = '🔴 CONSIDER SELLING — Weak momentum + weak fundamentals'
    elif momentum < 0:
        action = '🟡 WATCH — Momentum cooling'
    else:
        action = '🟡 HOLD'

    results.append({
        'symbol': sym, 'price': price, 'avg_cost': pos['avg_cost'], 'shares': pos['shares'],
        'pnl_pct': pnl_pct, 'pnl_val': pnl_val, 'value': value,
        'w1': w1, 'm1': m1, 'm3': m3, 'rsi': rsi,
        'rev_g': rev_g, 'eps_g': eps_g, 'margin': margin, 'roe': roe,
        'momentum': momentum, 'fund_quality': fund_quality,
        'action': action, 'vol': vol,
    })

# Sort by momentum
results.sort(key=lambda x: x['momentum'], reverse=True)

total_value = 0
total_cost = 0
for i, r in enumerate(results, 1):
    total_value += r['value']
    total_cost += r['shares'] * r['avg_cost']
    emoji = '📈' if r['pnl_pct'] >= 0 else '📉'
    print(f"{i:4d} {r['symbol']:12s} ₦{r['price']:>8,.2f} ₦{r['avg_cost']:>7,.2f} {emoji}{r['pnl_pct']:>+5.0f}% "
          f"₦{r['value']:>10,.0f} {r['w1']:>+6.1f}% {r['m1']:>+6.1f}% {r['m3']:>+6.1f}% {r['rsi']:>5.0f} "
          f"{r['rev_g']:>+7.1f}% {r['action']}")

total_pnl = total_value - total_cost
print(f"\n{'─'*130}")
pnl_pct_str = f"{total_pnl/total_cost*100:+.1f}%" if total_cost > 0 else "N/A"
print(f"   Portfolio Value: ₦{total_value:,.0f}  |  Total Cost: ₦{total_cost:,.0f}  |  P&L: ₦{total_pnl:+,.0f} ({pnl_pct_str})")
print(f"   + New Capital: ₦{BUDGET_TOMORROW + BUDGET_MONDAY:,.0f}  |  Total After Investment: ₦{total_value + BUDGET_TOMORROW + BUDGET_MONDAY:,.0f}")


# In[10]:


# ══════════════════════════════════════════════════════════════════════════
# PART 2: FULL MARKET SCAN — Best NEW positions for Feb-March growth
# ══════════════════════════════════════════════════════════════════════════
print(f"{'='*120}")
print(f"🔍 PART 2: FULL MARKET SCAN — Best NEW Opportunities (not in portfolio)")
print(f"{'='*120}")
print(f"   Criteria: RSI < 80 (not overbought) + momentum + fundamentals + liquidity")
print(f"{'='*120}\n")

# Screen ALL stocks
all_stocks = df[df['price'].notna() & (df['price'] > 0)].copy()
existing_syms = set(MY_PORTFOLIO.keys())

# Score every stock
market_scores = []
for _, s in all_stocks.iterrows():
    sym = s['symbol']
    price = s['price']
    w1 = safe_val(s.get('perf_1w', 0))
    m1 = safe_val(s.get('perf_1m', 0))
    m3 = safe_val(s.get('perf_3m', 0))
    ytd = safe_val(s.get('perf_ytd', 0))
    rsi = safe_val(s.get(COL_RSI, 50))
    rev_g = safe_val(s.get(COL_REV_G, 0))
    eps_g = safe_val(s.get(COL_EPS_G, 0))
    margin = safe_val(s.get(COL_NET_MARGIN, 0))
    roe = safe_val(s.get(COL_ROE, 0))
    vol = safe_val(s.get(COL_VOL, 0))
    mcap = safe_val(s.get(COL_MCAP, 0))
    pe = safe_val(s.get(COL_PE, 0))
    liq = price * vol

    # Skip no-volume stocks
    if vol <= 0 or liq < 100_000:
        continue

    # Momentum score (0-50)
    mom = min(w1 * 0.6, 15) + min(m1 * 0.35, 15) + min(m3 * 0.12, 10) + min(ytd * 0.1, 10)

    # Growth score (0-30)
    growth = min(max(rev_g, 0) * 0.2, 15) + min(max(eps_g, 0) * 0.15, 15)

    # Fundamentals (0-20)
    fund = min(max(margin, 0) * 0.3, 10) + min(max(roe, 0) * 0.2, 10)

    # RSI adjustment — REWARD healthy RSI, penalize overbought
    if rsi > 90: rsi_adj = -10
    elif rsi > 80: rsi_adj = -5
    elif rsi > 70: rsi_adj = -2
    elif rsi > 50: rsi_adj = +3   # Sweet spot
    elif rsi > 30: rsi_adj = +5   # Great entry
    else: rsi_adj = +2            # Oversold

    # Liquidity bonus
    liq_bonus = 5 if liq > 50_000_000 else (3 if liq > 10_000_000 else (1 if liq > 1_000_000 else 0))

    total = mom + growth + fund + rsi_adj + liq_bonus

    market_scores.append({
        'symbol': sym, 'price': price, 'in_port': sym in existing_syms,
        '1W%': w1, '1M%': m1, '3M%': m3, 'YTD%': ytd,
        'RevGr%': rev_g, 'EPSGr%': eps_g, 'Margin%': margin, 'ROE%': roe,
        'RSI': rsi, 'Vol': vol, 'Liq₦': liq, 'MCap': mcap, 'P/E': pe,
        'Mom': round(mom, 1), 'Growth': round(growth, 1), 'Fund': round(fund, 1),
        'RSI_adj': rsi_adj, 'Liq_b': liq_bonus,
        'TOTAL': round(total, 1),
    })

mkt = pd.DataFrame(market_scores).sort_values('TOTAL', ascending=False).reset_index(drop=True)

# NEW v2.0: Filter out value traps from market scan recommendations
try:
    from analysis.hidden_gems import _detect_value_trap, _generate_warnings
    trap_syms = []
    for _, s in all_stocks.iterrows():
        if _detect_value_trap(s):
            trap_syms.append(s['symbol'])
    if trap_syms:
        before = len(mkt)
        mkt = mkt[~mkt['symbol'].isin(trap_syms)].reset_index(drop=True)
        print(f"   Value Trap Filter: blocked {before - len(mkt)} stocks ({', '.join(trap_syms[:5])}{'...' if len(trap_syms) > 5 else ''})")
    
    # Add warnings column
    mkt['warnings'] = ''
    for idx, r in mkt.iterrows():
        stock_row = all_stocks[all_stocks['symbol'] == r['symbol']]
        if len(stock_row) > 0:
            mkt.at[idx, 'warnings'] = _generate_warnings(stock_row.iloc[0])
except ImportError:
    pass

# Show NEW opportunities only (not in portfolio), RSI < 80
new_picks = mkt[(~mkt['in_port']) & (mkt['RSI'] < 80) & (mkt['RSI'] > 10)].head(20)

print(f"{'#':>2} {'Symbol':12s} {'Price':>9s} {'RSI':>5s} {'1W%':>7s} {'1M%':>7s} {'3M%':>7s} {'RevGr%':>8s} {'Margin%':>8s} {'P/E':>6s} {'Score':>6s} {'Why'}")
print(f"{'-'*130}")

for i, (_, r) in enumerate(new_picks.iterrows(), 1):
    # Build rationale
    reasons = []
    if r['1M%'] > 20: reasons.append(f"+{r['1M%']:.0f}% 1M")
    if r['1W%'] > 10: reasons.append(f"+{r['1W%']:.0f}% 1W")
    if r['RevGr%'] > 30: reasons.append(f"Rev +{r['RevGr%']:.0f}%")
    if r['EPSGr%'] > 30: reasons.append(f"EPS +{r['EPSGr%']:.0f}%")
    if r['Margin%'] > 15: reasons.append(f"Margin {r['Margin%']:.0f}%")
    if r['ROE%'] > 15: reasons.append(f"ROE {r['ROE%']:.0f}%")
    if r['RSI'] < 55: reasons.append("RSI cool entry")
    if r['Liq₦'] > 10_000_000: reasons.append("High liq")

    pe_str = f"{r['P/E']:.1f}" if r['P/E'] > 0 else "N/A"

    badge = '🔥' if r['TOTAL'] >= 50 else ('🚀' if r['TOTAL'] >= 35 else ('🌤️' if r['TOTAL'] >= 20 else '😐'))
    print(f"{i:2d} {badge} {r['symbol']:12s} ₦{r['price']:>8,.2f} {r['RSI']:>4.0f} {r['1W%']:>+6.1f}% {r['1M%']:>+6.1f}% "
          f"{r['3M%']:>+6.1f}% {r['RevGr%']:>+7.1f}% {r['Margin%']:>+7.1f}% {pe_str:>6s} {r['TOTAL']:>5.1f}  {', '.join(reasons)}")
    # v2.0: Show warning flags if any
    w = r.get('warnings', '')
    if w:
        print(f"      ⚠️  {w}")

# Also show PORTFOLIO stocks ranked in the full market
print(f"\n{'='*120}")
print(f"📊 YOUR PORTFOLIO STOCKS IN FULL MARKET RANKING (where do they sit?)")
print(f"{'='*120}")
port_in_mkt = mkt[mkt['in_port']].copy()
port_in_mkt['mkt_rank'] = port_in_mkt.index + 1
for _, r in port_in_mkt.iterrows():
    rsi_tag = '🔴' if r['RSI'] > 85 else ('🟠' if r['RSI'] > 75 else ('🟢' if r['RSI'] > 40 else '🔵'))
    print(f"   Rank #{r['mkt_rank']:3d}/{len(mkt)}  {rsi_tag} {r['symbol']:12s} ₦{r['price']:>8,.2f}  RSI {r['RSI']:>4.0f}  Score {r['TOTAL']:>5.1f}")


# In[11]:


# ══════════════════════════════════════════════════════════════════════════
# PART 3: SELL / TRIM RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════
print(f"{'='*110}")
print(f"🔴 PART 3: POSITIONS TO CONSIDER SELLING / TRIMMING")
print(f"{'='*110}\n")

sell_candidates = []
for r in results:
    sym = r['symbol']
    reason = []
    action_type = None
    freed_cash = 0

    # SELL criteria
    if r['momentum'] < -3 and r['fund_quality'] < 10:
        reason.append(f"Negative momentum ({r['momentum']:+.1f}) + weak fundamentals")
        action_type = 'SELL'
        freed_cash = r['value']
    # TRIM (take profits) criteria  
    elif r['rsi'] > 95 and r['pnl_pct'] > 30:
        reason.append(f"RSI {r['rsi']:.0f} (extreme OB) + P&L {r['pnl_pct']:+.0f}% — sell half")
        action_type = 'TRIM 50%'
        freed_cash = r['value'] * 0.5
    elif r['rsi'] > 90 and r['pnl_pct'] > 50:
        reason.append(f"RSI {r['rsi']:.0f} (OB) + big P&L {r['pnl_pct']:+.0f}% — take some profits")
        action_type = 'TRIM 30%'
        freed_cash = r['value'] * 0.3
    elif r['momentum'] < 0 and r['pnl_pct'] < -5:
        reason.append(f"Losing money ({r['pnl_pct']:+.1f}%) + negative momentum")
        action_type = 'SELL'
        freed_cash = r['value']
    # WEAK HOLD — bottom of portfolio
    elif r['momentum'] < 1 and r['fund_quality'] < 15 and r['pnl_pct'] < 15:
        reason.append(f"Dead money — low momentum + mediocre fundamentals")
        action_type = 'CONSIDER SELLING'
        freed_cash = r['value']

    if action_type:
        sell_candidates.append({**r, 'sell_action': action_type, 'freed_cash': freed_cash, 'reasons': reason})

if sell_candidates:
    total_freed = 0
    for sc in sell_candidates:
        emoji = '🔴' if sc['sell_action'] in ['SELL'] else ('🟠' if 'TRIM' in sc['sell_action'] else '🟡')
        print(f"   {emoji} {sc['sell_action']:18s} {sc['symbol']:12s}  ₦{sc['value']:>10,.0f}  →  Free ₦{sc['freed_cash']:>10,.0f}")
        print(f"      P&L: {sc['pnl_pct']:+.0f}%  |  RSI: {sc['rsi']:.0f}  |  1W: {sc['w1']:+.1f}%  |  1M: {sc['m1']:+.1f}%")
        for reason in sc['reasons']:
            print(f"      💬 {reason}")
        total_freed += sc['freed_cash']
        print()
    print(f"   💵 Total cash freed if ALL executed: ₦{total_freed:,.0f}")
    print(f"   💵 + New deposit: ₦{BUDGET_TOMORROW + BUDGET_MONDAY:,.0f}")
    print(f"   💵 = Total deployable: ₦{total_freed + BUDGET_TOMORROW + BUDGET_MONDAY:,.0f}")
else:
    print("   ✅ No sell candidates — all positions have acceptable momentum/fundamentals")


# In[12]:


# ══════════════════════════════════════════════════════════════════════════
# PART 4: REVISED "SUPER 6" CONCENTRATED ALLOCATION PLAN
# ══════════════════════════════════════════════════════════════════════════

print(f"{'='*120}")
print(f"🚀 PART 4: THE 'SUPER 6' AGGRESSIVE STRATEGY")
print(f"{'='*120}")
print(f"   Strategy: High Conviction. Fewer Stocks. More Capital per Position.")
print(f"   Goal: Maximize impact of the winners. Don't dilute returns.")
print(f"{'='*120}\n")

# ─── THE SUPER 6 SELECTION ───
super_6_picks = [
    # 1. THE UNDERVALUED ROCKET (New)
    {'symbol': 'ABBEYBDS', 'alloc': 250_000, 'type': 'NEW',
     'why': '⭐ #1 MARKET PICK. Massive growth (EPS +518%). We are betting big here.'},

    # 2. THE DIP OPPORTUNITY (Add)
    {'symbol': 'VITAFOAM', 'alloc': 250_000, 'type': 'ADD',
     'why': '💎 BUY THE DIP. RSI cooled from 94 to 73. Perfect entry in an uptrend.'},

    # 3. THE SAFE GROWTH (New)
    {'symbol': 'NAHCO', 'alloc': 200_000, 'type': 'NEW',
     'why': '✈️ AVIATION LEADER. Best fundamentals in sector (ROE 77%). Safer growth.'},

    # 4. THE MOMENTUM KING (Add)
    {'symbol': 'IKEJAHOTEL', 'alloc': 200_000, 'type': 'ADD',
     'why': '🔥 FEED THE WINNER. Best performing recent add. Don\'t cut flowers.'},

    # 5. THE UPSIDE PLAY (Add)
    {'symbol': 'TIP', 'alloc': 200_000, 'type': 'ADD',
     'why': '📈 PURE UPSIDE. RSI 56 is cold. Rev +152%. Cheap entry for volatility.'},

    # 6. THE PENNY LOTTO (New)
    {'symbol': 'NPFMCRFBK', 'alloc': 150_000, 'type': 'NEW',
     'why': '🎰 PENNY STOCK. At ₦5.59, it\'s cheap. Banks are hot. Small risk, big reward.'}
]

allocations = []
total_deployed = 0

print(f"📅 TOMORROW (Feb 20) — DEPLOYING: ₦1,250,000  (Reserve: ₦250k + ₦500k Monday)")
print(f"{'─'*120}")

for pick in super_6_picks:
    sym = pick['symbol']
    alloc = pick['alloc']
    stock = df[df['symbol'] == sym]

    if stock.empty:
        print(f"❌ Error: {sym} not found in data.")
        continue

    s = stock.iloc[0]
    price = s['price']
    rsi = safe_val(s.get(COL_RSI, 50))

    # AGGRESSIVE ENTRY STRATEGY
    # RSI < 60? Buy at Market. Don't risk missing it.
    # RSI > 60? Tight Limit (Current Price). 

    if rsi < 60:
        entry = price
        entry_note = "MARKET ORDER (Don't miss it!)"
    else:
        entry = price # No discount requested.
        entry_note = f"Limit ₦{entry:.2f} (Current Price - Fast execution)"

    shares = int(alloc / entry)
    cost = shares * entry

    # 10% Trailing Stop from Entry
    stop = round(entry * 0.90, 2) 

    existing = MY_PORTFOLIO.get(sym, {})
    current_shares = existing.get('shares', 0)

    tag = '➕ ADD' if pick['type'] == 'ADD' else '🆕 NEW'

    print(f"   {tag} {sym:12s} | {shares:>6,} shares @ ₦{entry:<8.2f} = ₦{cost:>10,.0f}")
    if current_shares > 0:
         print(f"      {'':14s}   Current: {current_shares:,} → New total: {current_shares + shares:,}")
    print(f"      {'':14s}   Stop Loss: ₦{stop} (-10%)")
    print(f"      {'':14s}   Entry: {entry_note}")
    print(f"      {'':14s}   Strategy: {pick['why']}")
    print()

    total_deployed += cost
    allocations.append(pick)

# ─── Summary ───
remaining_tomorrow = BUDGET_TOMORROW - 1_000_000 # Just base logic
# Actual logic:
# Total cash available for this plan (sample)
# We are spending ₦1.25M tomorrow.

remaining_cash = (BUDGET_TOMORROW + BUDGET_MONDAY) - total_deployed

print(f"{'='*120}")
print(f"📊 EXECUTION SUMMARY")
print(f"{'='*120}")
print(f"   💰 Total Budget:    ₦{BUDGET_TOMORROW + BUDGET_MONDAY:,.0f}")
print(f"   🚀 Total Invested:  ₦{total_deployed:,.0f}")
print(f"   💼 Cash Reserve:    ₦{remaining_cash:,.0f} (Available for Monday/Opportunities)")
print(f"   📝 Action Items:")
print(f"      1. TRIM: Execute sells for ABCTRANS, PRESCO, JBERGER (~₦172k freed)")
print(f"      2. BUY:  Place these 6 priority orders immediately tomorrow open.")


# In[13]:


# ══════════════════════════════════════════════════════════════════════════
# PART 5: LIVE ORDER BOOK ANALYSIS — REALITY CHECK (Feb 19 @ 16:59)
# ══════════════════════════════════════════════════════════════════════════

print(f"{'='*120}")
print(f"📖 PART 5: LIVE ORDER BOOK ANALYSIS — What the Market Is ACTUALLY Telling Us")
print(f"{'='*120}")
print(f"   Order books captured Feb 19 ~16:57-16:59 (after close)")
print(f"   These represent PENDING orders for tomorrow's open (Feb 20)")
print(f"{'='*120}\n")

# ─── Order Book Data from Screenshots ───
order_books = {
    'ABBEYBDS': {
        'bids': [(30_000, 9.75)],
        'offers': [],  # ZERO sellers!
        'snapshot_price': 11.40,
        'type': 'BUY',
    },
    'VITAFOAM': {
        'bids': [(50, 117.95), (130, 117.00), (1960, 116.85), (156, 115.00), (250, 111.70), (150, 110.00), (2505, 109.20)],
        'offers': [(60_000, 124.90), (32_000, 125.00), (48_986, 125.75), (2500, 128.00), (500_300, 128.50)],
        'snapshot_price': 123.10,
        'type': 'BUY',
    },
    'NAHCO': {
        'bids': [(107_418, 169.20), (26_010, 168.05), (1600, 167.00), (235, 165.00), (3277, 163.00), (1000, 162.85), (1000, 162.80), (1000, 162.50), (300, 162.00), (195, 161.00)],
        'offers': [(24_000, 171.85), (10_000, 172.50), (50, 173.00), (43_620, 173.90), (109_895, 174.40), (1000, 175.00), (500, 179.00), (75_000, 179.10), (50_000, 180.00)],
        'snapshot_price': 170.00,
        'type': 'BUY',
    },
    'IKEJAHOTEL': {
        'bids': [],  # No bids visible
        'offers': [(5619, 38.50), (300, 40.00), (58_533, 41.00), (16_145, 41.70), (9105, 41.75), (50_000, 41.80), (42_155, 41.90), (2000, 44.50), (10_000, 45.00), (1140, 45.50), (4726, 45.60), (47_274, 45.70), (27_000, 45.75), (51_380, 45.80), (1_002_250, 45.95)],
        'snapshot_price': 41.90,
        'type': 'BUY',
    },
    'TIP': {
        'bids': [(7363, 18.70), (23_110, 18.60), (10_768, 18.50), (7000, 18.35), (2909, 18.25), (205, 18.10), (239_410, 18.00), (11_555, 17.80), (52_087, 17.55), (31_014, 17.50), (310, 17.35), (5210, 17.00), (2_002_500, 16.80), (100_000, 16.75)],
        'offers': [(1065, 20.00), (110_931, 20.45)],
        'snapshot_price': 18.90,
        'type': 'BUY',
    },
    'NPFMCRFBK': {
        'bids': [(12_000, 5.65), (2500, 5.60), (5000, 5.50), (5458, 5.30), (18_450, 5.20), (3597, 5.09), (50_000, 5.05), (9000, 5.05), (50_000, 5.01), (8800, 5.00), (526_000, 4.95)],
        'offers': [(19_708, 5.71), (35_139, 5.80), (100_000, 5.85), (200_000, 5.95), (280_000, 5.97), (200_000, 5.99), (866_482, 6.00)],
        'snapshot_price': 5.70,
        'type': 'BUY',
    },
    # SELL CANDIDATES
    'PRESCO': {
        'bids': [(3950, 2206.00), (10, 2200.00), (10, 2199.90), (8, 2187.30), (23, 2160.00), (80, 2150.00), (10, 2111.00), (2, 2100.00), (32, 2076.00), (49, 2045.00), (14, 2042.80), (1500, 2035.00), (420, 2015.00), (1018, 2000.00), (17_000, 1999.00)],
        'offers': [(800, 2288.90), (205, 2300.00), (145, 2305.00), (17, 2310.00), (46, 2314.00), (10, 2316.00), (6250, 2375.00)],
        'snapshot_price': 2206.00,
        'type': 'SELL',
    },
    'JBERGER': {
        'bids': [(1030, 259.00), (10, 236.10), (500, 233.80)],
        'offers': [(1990, 270.00), (1000, 274.00), (36, 275.00), (5255, 277.00), (29_090, 278.20)],
        'snapshot_price': 259.00,
        'type': 'SELL',
    },
    'ABCTRANS': {
        'bids': [(550, 8.55)],
        'offers': [(100_000, 9.49), (6000, 10.00), (100_000, 10.30)],
        'snapshot_price': 9.49,
        'type': 'SELL',
    },
}

# ─── Analyze each order book ───
for sym, ob in order_books.items():
    bids = ob['bids']
    offers = ob['offers']
    snap = ob['snapshot_price']
    ob_type = ob['type']

    best_bid = bids[0][1] if bids else 0
    best_offer = offers[0][1] if offers else 0
    best_bid_vol = bids[0][0] if bids else 0
    best_offer_vol = offers[0][0] if offers else 0

    total_bid_vol = sum(v for v, p in bids)
    total_offer_vol = sum(v for v, p in offers)

    if best_bid > 0 and best_offer > 0:
        spread = best_offer - best_bid
        spread_pct = spread / best_bid * 100
    elif best_offer > 0:
        spread_pct = 0  # One-sided
    else:
        spread_pct = 0

    # Determine verdict
    if ob_type == 'BUY':
        if not offers:
            verdict = '🚨 UNBUYABLE — Zero sellers! Stock locked at upper limit'
            health = '🔴'
            action = 'SKIP — Cannot execute. Reallocate capital elsewhere.'
        elif spread_pct > 5:
            verdict = f'⚠️ WIDE SPREAD ({spread_pct:.1f}%) — Be careful with order type'
            health = '🟡'
            action = f'LIMIT ORDER @ ₦{best_offer:.2f} — Don\'t use Market Order'
        elif spread_pct > 2:
            verdict = f'🟡 MODERATE SPREAD ({spread_pct:.1f}%) — Use limit orders'
            health = '🟡'
            action = f'LIMIT ORDER @ ₦{best_offer:.2f}'
        else:
            verdict = f'✅ TIGHT SPREAD ({spread_pct:.1f}%) — Easy execution'
            health = '🟢'
            action = f'LIMIT ORDER @ ₦{best_offer:.2f} or Market'
    else:  # SELL
        if not bids:
            verdict = '🚨 NO BUYERS — Cannot sell'
            health = '🔴'
            action = 'HOLD — No liquidity to exit'
        elif total_bid_vol < 100:
            verdict = '⚠️ THIN BID SIDE — Selling will move price'
            health = '🟡'
            action = f'LIMIT SELL @ ₦{best_bid:.2f} — small lots'
        else:
            verdict = f'✅ Can sell at ₦{best_bid:.2f}'
            health = '🟢'
            action = f'LIMIT SELL @ ₦{best_bid:.2f}'

    price_vs_snap = ((best_offer if ob_type == 'BUY' and best_offer > 0 else best_bid) - snap) / snap * 100 if snap > 0 else 0

    print(f"{'─'*120}")
    print(f"   {health} {sym:12s} | Snapshot ₦{snap:,.2f}")
    print(f"      Best Bid:   ₦{best_bid:>10,.2f}  ({best_bid_vol:>10,} units)" if best_bid > 0 else f"      Best Bid:   {'EMPTY':>12}  — No buyers visible!")
    print(f"      Best Offer: ₦{best_offer:>10,.2f}  ({best_offer_vol:>10,} units)" if best_offer > 0 else f"      Best Offer: {'EMPTY':>12}  — No sellers at all!")
    if best_bid > 0 and best_offer > 0:
        print(f"      Spread:     ₦{spread:>10,.2f}  ({spread_pct:.1f}%)")
    print(f"      Bid Volume: {total_bid_vol:>10,} total  |  Offer Volume: {total_offer_vol:>10,} total")
    if ob_type == 'BUY' and best_offer > 0:
        print(f"      Real Price: ₦{best_offer:.2f} (vs snapshot ₦{snap:.2f} = {price_vs_snap:+.1f}%)")
    elif ob_type == 'SELL' and best_bid > 0:
        print(f"      Sell Price: ₦{best_bid:.2f} (vs snapshot ₦{snap:.2f} = {price_vs_snap:+.1f}%)")
    print(f"      📋 {verdict}")
    print(f"      🎯 {action}")
    print()

# ══════════════════════════════════════════════════════════════════════════
# CRITICAL FINDINGS
# ══════════════════════════════════════════════════════════════════════════
print(f"{'='*120}")
print(f"🚨 CRITICAL FINDINGS — What Changes for Tomorrow")
print(f"{'='*120}\n")

print(f"   1. 🚨 ABBEYBDS IS UNBUYABLE — Zero sellers, stock locked at upper limit.")
print(f"      → ₦250,000 must be REALLOCATED to other picks.\n")

print(f"   2. ⚠️ TIP IS GAPPING UP — Offers start at ₦20.00 vs ₦18.90 close (+5.8%)")
print(f"      → Still buyable but costs more. 1,065 shares @ ₦20.00 then 110k @ ₦20.45")
print(f"      → 2M+ bid wall at ₦16.80 = massive buyer support below.\n")

print(f"   3. ⚠️ VITAFOAM HAS A 6% SPREAD — Best offer ₦124.90 vs best bid ₦117.95")
print(f"      → Cheapest fill is ₦124.90 (60k units). Plenty of supply there.\n")

print(f"   4. ✅ NAHCO IS PERFECT — Tight 1.6% spread, 107k bid units, 24k offer units")
print(f"      → Best order book of all picks. Execute with confidence.\n")

print(f"   5. ✅ NPFMCRFBK IS GOOD — 1.1% spread, 19.7k @ ₦5.71. Clean fill.\n")

print(f"   6. 🟢 IKEJAHOTEL HAS CHEAP OFFERS — Sellers starting at ₦38.50!")
print(f"      → That's 8% BELOW the ₦41.90 snapshot. Could be a discount entry.")
print(f"      → But 58k units at ₦41.00 = main supply level. Buy there.\n")

# ══════════════════════════════════════════════════════════════════════════
# REVISED EXECUTION PLAN
# ══════════════════════════════════════════════════════════════════════════
print(f"{'='*120}")
print(f"📋 REVISED EXECUTION PLAN — Based on Real Order Book Data")
print(f"{'='*120}\n")

revised_plan = [
    # Illustrative allocation plan (sample values).
    {'symbol': 'GTCO',       'alloc': 300_000, 'entry': 50.00,  'order': 'LIMIT N50.00',  'why': 'Sample entry - tight spread'},
    {'symbol': 'DANGCEM',    'alloc': 250_000, 'entry': 450.00, 'order': 'LIMIT N450.00', 'why': 'Sample entry'},
    {'symbol': 'ZENITHBANK', 'alloc': 200_000, 'entry': 40.00,  'order': 'LIMIT N40.00',  'why': 'Sample entry'},
]

total_revised = 0
print(f"   {'#':>2} {'Symbol':12s} {'Alloc':>10s} {'Entry':>10s} {'Shares':>8s} {'Cost':>12s} {'Order Type':20s} {'Stop':>8s}")
print(f"   {'─'*100}")

for i, rp in enumerate(revised_plan, 1):
    shares = int(rp['alloc'] / rp['entry'])
    cost = shares * rp['entry']
    stop = round(rp['entry'] * 0.90, 2)
    total_revised += cost

    existing = MY_PORTFOLIO.get(rp['symbol'], {})
    curr = existing.get('shares', 0)
    new_total = curr + shares

    print(f"   {i:2d} {rp['symbol']:12s} ₦{rp['alloc']:>9,} ₦{rp['entry']:>8.2f} {shares:>7,} ₦{cost:>11,.0f} {rp['order']:20s} ₦{stop:>7.2f}")
    if curr > 0:
        print(f"      {'':12s} Current: {curr:,} → New total: {new_total:,} shares")
    print(f"      {'':12s} 💡 {rp['why']}")
    print()

cash_reserve = (BUDGET_TOMORROW + BUDGET_MONDAY) - total_revised

print(f"   {'─'*100}")
print(f"   💰 Total Deployed:  ₦{total_revised:>10,.0f}")
print(f"   💼 Cash Reserve:    ₦{cash_reserve:>10,.0f} (for Monday opportunities)")
print()

# SELL SIDE
print(f"{'='*120}")
print(f"✂️ SELL ORDERS — Updated with Order Book Reality")
print(f"{'='*120}\n")

sell_plan = [
    {'symbol': 'PRESCO', 'shares_sell': 58, 'price': 2206.00, 'notes': '3,950 units at ₦2,206 bid — easy fill for 58 shares'},
    {'symbol': 'JBERGER', 'shares_sell': 85, 'price': 259.00, 'notes': '1,030 units at ₦259 bid — easy fill for 85 shares'},
    {'symbol': 'ABCTRANS', 'shares_sell': 550, 'price': 8.55, 'notes': '⚠️ Only 550 units at ₦8.55 bid — thin! Sell what you can'},
]

total_sell = 0
for sp in sell_plan:
    proceeds = sp['shares_sell'] * sp['price']
    total_sell += proceeds
    print(f"   🔴 SELL {sp['shares_sell']:>5,} {sp['symbol']:12s} @ ₦{sp['price']:>8,.2f} = ₦{proceeds:>10,.0f}")
    print(f"      {sp['notes']}")
    print()

print(f"   💵 Total from sells: ₦{total_sell:,.0f}")
print(f"   💵 Grand total available: ₦{total_sell + cash_reserve:,.0f} for Monday/future")

print(f"\n{'='*120}")
print(f"⚡ EXECUTION ORDER (Priority Sequence for Tomorrow Morning)")
print(f"{'='*120}")
print(f"   1️⃣  SELL PRESCO 58 shares @ ₦2,206 limit (free up capital)")
print(f"   2️⃣  SELL JBERGER 85 shares @ ₦259 limit")
print(f"   3️⃣  SELL ABCTRANS 550 shares @ ₦8.55 limit")
print(f"   4️⃣  BUY NAHCO — LIMIT ₦171.85 (best order book, execute first)")
print(f"   5️⃣  BUY IKEJAHOTEL — LIMIT ₦41.00 (potential discount entry)")
print(f"   6️⃣  BUY NPFMCRFBK — LIMIT ₦5.71 (tight spread, fast fill)")
print(f"   7️⃣  BUY VITAFOAM — LIMIT ₦124.90 (wide spread, may need patience)")
print(f"   8️⃣  BUY TIP — LIMIT ₦20.00 (gap up, may partial fill)")
print(f"   ❌  SKIP ABBEYBDS — No sellers. Monitor for Monday.")
print(f"{'='*120}")

