# Setup
import sys
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from IPython.display import display, HTML

sys.path.insert(0, str(Path.cwd().parent))

from analysis.bluechip_quality import (
    find_bluechip_quality,
    find_bluechip_dips,
    _quality_pe_cap,
    _entry_signal,
)

from notebooks.nb_helpers import (
    load_report_cache,
    get_batch_report_insights,
    load_or_fetch_news,
    get_news_sentiment,
    data_freshness_banner,
    TONE_EMOJI,
    print_divergence_warnings,
    print_report_block,
    print_news_block,
)

report_cache = load_report_cache()

DATA_DIR = Path(os.getenv('NAIJA_DATA_DIR', Path(__file__).resolve().parents[1] / 'data')) / 'snapshots'

print(f"📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"📁 Data Directory: {DATA_DIR}")


# Find and load the latest snapshot
snapshots = sorted(DATA_DIR.glob('*/snapshot.parquet'))

if not snapshots:
    print("❌ No snapshots found! Run create_snapshot.py first.")
    df = pd.DataFrame()
    snapshot_date = None
else:
    latest = snapshots[-1]
    snapshot_date = latest.parent.name
    df = pd.read_parquet(latest)

    if 'volume_1d' in df.columns:
        df['liquidity_value_1d'] = df['price'] * df['volume_1d']

    print(f"✅ Loaded snapshot: {snapshot_date}")
    print(f"📈 Total stocks: {len(df)}")
    print(f"📊 Total columns: {len(df.columns)}")
    print(f"\nAvailable snapshots:")
    for s in snapshots[-8:]:
        marker = "👉" if s == latest else "  "
        print(f"  {marker} {s.parent.name}")


bluechips = find_bluechip_quality(df, top_n=25)

if len(bluechips) == 0:
    print("⚠️  No blue-chip quality stocks passed the filters on this snapshot.")
else:
    elite   = (bluechips['tier'] == '🏆 Elite').sum()
    premium = (bluechips['tier'] == '⭐ Premium').sum()
    quality = (bluechips['tier'] == '✅ Quality').sum()
    watch   = (bluechips['tier'] == '📊 Watchlist').sum()

    print(f"\n{'='*70}")
    print(f"🏆 BLUE-CHIP QUALITY SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Total found        : {len(bluechips)}")
    print(f"  🏆 Elite     (≥70) : {elite}")
    print(f"  ⭐ Premium (55-69) : {premium}")
    print(f"  ✅ Quality (40-54) : {quality}")
    print(f"  📊 Watchlist (<40) : {watch}")


if len(bluechips) > 0:
    display_cols = [
        'rank', 'symbol', 'tier', 'price', 'bluechip_score',
        'pe_ratio', 'pe_cap', 'rsi_14',
        'perf_1m', 'perf_3m', 'perf_ytd',
        'eps_growth_ttm', 'roe_ttm', 'net_margin_ttm',
        'entry_signal', 'warnings'
    ]
    available = [c for c in display_cols if c in bluechips.columns]
    display_df = bluechips[available].copy()
    display_df = display_df.rename(columns={
        'rank':           'Rank',
        'symbol':         'Symbol',
        'tier':           'Tier',
        'price':          'Price',
        'bluechip_score': 'Score',
        'pe_ratio':       'P/E',
        'pe_cap':         'PE Cap',
        'rsi_14':         'RSI',
        'perf_1m':        '1M%',
        'perf_3m':        '3M%',
        'perf_ytd':       'YTD%',
        'eps_growth_ttm': 'EPS Gr',
        'roe_ttm':        'ROE',
        'net_margin_ttm': 'Margin',
        'entry_signal':   'Entry Signal',
        'warnings':       'Warnings',
    })

    numeric_cols = ['Score', 'P/E', 'PE Cap', 'RSI', '1M%', '3M%', 'YTD%',
                    'EPS Gr', 'ROE', 'Margin']
    for c in numeric_cols:
        if c in display_df.columns:
            display_df[c] = display_df[c].round(1)

    print("\n🔝 TOP BLUE-CHIP QUALITY STOCKS:\n")
    display(display_df.style.background_gradient(subset=['Score'], cmap='Greens'))
else:
    print("No blue-chip data to display.")


dips = find_bluechip_dips(df, top_n=15)

def _tier_for_score(s):
    if s >= 70: return '🏆 Elite'
    if s >= 55: return '⭐ Premium'
    if s >= 40: return '✅ Quality'
    return '📊 Watchlist'

if len(dips) > 0 and 'tier' not in dips.columns:
    dips['tier'] = dips['bluechip_score'].apply(_tier_for_score)

if len(dips) == 0:
    print("\n⚠️  No blue-chip dips on this snapshot.")
    print("    The market is hot — quality names are running.")
    print("    Strategies:")
    print("      1. Wait for a market pullback")
    print("      2. Use staggered entries on Premium/Elite tier names")
    print("      3. Take profits in any overbought blue-chips you hold")
else:
    print(f"\n{'='*70}")
    print(f"💎 BLUE-CHIP DIP OPPORTUNITIES — {len(dips)} quality stocks on pullback")
    print(f"{'='*70}\n")

    for _, row in dips.iterrows():
        # Technical confluence — same idea as hidden_gems cold opportunities
        tech_signals = []
        stoch_k = row.get('stochastic_k_1d', np.nan)
        cci     = row.get('cci_20_1d', np.nan)
        rvol    = row.get('relative_volume_1d', np.nan)

        if not pd.isna(stoch_k) and stoch_k < 20:
            tech_signals.append(f"Stoch K={stoch_k:.0f}")
        if not pd.isna(cci) and cci < -100:
            tech_signals.append(f"CCI={cci:.0f}")
        if not pd.isna(rvol) and rvol > 2.0:
            tech_signals.append(f"🔊 RVOL={rvol:.1f}x")

        confluence = '⚡ OVERSOLD CONFIRMED' if len(tech_signals) >= 2 else ''

        pe = row.get('pe_ratio', np.nan)
        pe_str = f"{pe:.1f}x" if not pd.isna(pe) else "—"
        cap_str = f"{row['pe_cap']:.0f}x"

        print(f"🎯 {row['symbol']}  {row['tier']}  {confluence}")
        print(f"   Price: ₦{row['price']:.2f} | Score: {row['bluechip_score']:.1f}")
        print(f"   1M: {row['perf_1m']:+.1f}% | 3M: {row.get('perf_3m', 0):+.1f}% | "
              f"YTD: {row.get('perf_ytd', 0):+.1f}%")
        print(f"   ROE: {row['roe_ttm']:.1f}% | Margin: {row['net_margin_ttm']:.1f}% | "
              f"EPS Gr: {row.get('eps_growth_ttm', 0):.1f}%")
        print(f"   P/E: {pe_str} (cap {cap_str}) | Entry: {row['entry_signal']}")
        if tech_signals:
            print(f"   📊 Tech: {' | '.join(tech_signals)}")
        if row.get('warnings'):
            print(f"   ⚠️  {row['warnings']}")
        print()


if len(bluechips) > 0 and 'sector' in bluechips.columns:
    def _sector_stats(group):
        top = group.sort_values('bluechip_score', ascending=False).iloc[0]
        return pd.Series({
            'n':          len(group),
            'avg_score':  group['bluechip_score'].mean(),
            'avg_roe':    group['roe_ttm'].mean(),
            'avg_margin': group['net_margin_ttm'].mean(),
            'best':       top['symbol'],
            'best_score': top['bluechip_score'],
        })

    sector_summary = (
        bluechips.groupby('sector', dropna=False)
                 .apply(_sector_stats, include_groups=False)
                 .sort_values('avg_score', ascending=False)
    )

    print(f"\n{'='*90}")
    print(f"🌍 SECTOR QUALITY HEATMAP — top sectors by average blue-chip score")
    print(f"{'='*90}")
    print(f"  {'Sector':32} | {'#':>3} | {'Avg Score':>9} | {'Avg ROE':>8} | "
          f"{'Avg Margin':>10} | {'Best Stock':12} | {'Score':>5}")
    print(f"  {'-'*90}")
    for sector, r in sector_summary.iterrows():
        sec_name = (sector or 'Unknown')[:32]
        print(f"  {sec_name:32} | {int(r['n']):>3} | {r['avg_score']:>9.1f} | "
              f"{r['avg_roe']:>7.1f}% | {r['avg_margin']:>9.1f}% | "
              f"{r['best']:12} | {r['best_score']:>5.1f}")
else:
    print("No sector data available.")


from analysis.hidden_gems import find_hidden_gems

gems_30      = find_hidden_gems(df, top_n=30)
bluechips_30 = find_bluechip_quality(df, top_n=30)

gem_set      = set(gems_30['symbol'])      if len(gems_30)      > 0 else set()
bluechip_set = set(bluechips_30['symbol']) if len(bluechips_30) > 0 else set()

both          = gem_set & bluechip_set
gems_only     = gem_set - both
bluechip_only = bluechip_set - both

print(f"\n{'='*70}")
print(f"🔁 QUALITY vs VALUE — coverage map (top 30 in each model)")
print(f"{'='*70}")
print(f"\n  💎 Hidden gems     : {len(gem_set):2}")
print(f"  🏆 Blue-chips      : {len(bluechip_set):2}")
print(f"  🟢 Appear in BOTH  : {len(both):2}  — best of both worlds")
print(f"  💎 Gems only       : {len(gems_only):2}  — small/mid-cap contrarian plays")
print(f"  🏆 Blue-chip only  : {len(bluechip_only):2}  — institutional quality not seen by gems")

def _show_set(label, syms, source_df, score_col):
    if not syms:
        print(f"\n  {label}: (none)")
        return
    print(f"\n  {label}:")
    sub = source_df[source_df['symbol'].isin(syms)].sort_values(score_col, ascending=False)
    for _, r in sub.iterrows():
        score = r[score_col]
        sec   = r.get('sector', '') or ''
        p1m   = r.get('perf_1m', 0) or 0
        print(f"    • {r['symbol']:12} score={score:5.1f}  1M={p1m:+5.1f}%  {sec}")

_show_set("🟢 BOTH (highest conviction)", both,          bluechips_30, 'bluechip_score')
_show_set("💎 Gems only",                  gems_only,     gems_30,      'hidden_gem_score')
_show_set("🏆 Blue-chip only",             bluechip_only, bluechips_30, 'bluechip_score')


WATCHLIST = [
    'ABBEYBDS', 'ABCTRANS', 'AIICO', 'BETAGLAS', 'BUACEMENT',
    'CUSTODIAN', 'GEREGU', 'HONYFLOUR', 'IKEJAHOTEL', 'INFINITY',
    'JBERGER', 'LEARNAFRCA', 'MECURE', 'NAHCO', 'NASCON',
    'NGXGROUP', 'OKOMUOIL', 'PRESCO', 'ROYALEX', 'SKYAVN',
    'TIP', 'TRANSPOWER', 'UPDC', 'VFDGROUP', 'VITAFOAM', 'WAPCO',
]

# Full blue-chip universe (no top_n cap) so we can attach tier to every match
all_bluechips = find_bluechip_quality(df, top_n=10_000)
tier_lookup   = dict(zip(all_bluechips['symbol'], all_bluechips['tier'])) if len(all_bluechips) else {}

print(f"\n{'='*120}")
print(f"📋 MY PORTFOLIO — BLUE-CHIP LENS  ({len(WATCHLIST)} positions)")
print(f"{'='*120}")
print(f"{'Symbol':12} | {'Price':>9} | {'P/E':>7} | {'PE Cap':>7} | "
      f"{'ROE':>6} | {'Margin':>7} | {'EPS Gr':>7} | {'RSI':>5} | "
      f"{'Entry Signal':22} | Tier")
print(f"{'-'*120}")

for symbol in WATCHLIST:
    stock = df[df['symbol'] == symbol]
    if len(stock) == 0:
        print(f"{symbol:12} | not found in snapshot")
        continue

    row = stock.iloc[0]
    roe   = row.get('roe_ttm', np.nan)
    eps_g = row.get('eps_growth_ttm', np.nan)
    pe    = row.get('pe_ratio', np.nan)
    nm    = row.get('net_margin_ttm', np.nan)
    rsi   = row.get('rsi_14', np.nan)
    cap   = _quality_pe_cap(roe, eps_g)
    sig   = _entry_signal(row)
    tier  = tier_lookup.get(symbol, '—')

    pe_str    = f"{pe:5.1f}x"  if not pd.isna(pe)    else "    —"
    cap_str   = f"{cap:5.1f}x"
    roe_str   = f"{roe:5.1f}%" if not pd.isna(roe)   else "    —"
    nm_str    = f"{nm:6.1f}%"  if not pd.isna(nm)    else "     —"
    epsg_str  = f"{eps_g:+6.1f}" if not pd.isna(eps_g) else "     —"
    rsi_str   = f"{rsi:5.1f}"  if not pd.isna(rsi)   else "    —"

    print(f"{symbol:12} | ₦{row['price']:8.2f} | {pe_str} | {cap_str} | "
          f"{roe_str} | {nm_str} | {epsg_str} | {rsi_str} | {sig:22} | {tier}")

print()
print("  Tier key: 🏆 Elite ≥70  ·  ⭐ Premium 55-69  ·  ✅ Quality 40-54  ·  📊 Watchlist <40")
print("  P/E is allowed up to PE Cap (cap scales with ROE & EPS growth).")


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
        print(f"{watch}{r['symbol']:10} | {r['rsi']:5.1f} | ₦{r['price']:8.2f} | "
              f"{r['perf_1w']:+6.1f}% | {r['perf_1m']:+6.1f}% | {r.get('sector','')}")
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
        print(f"{watch}{r['symbol']:10} | {r['rsi']:5.1f} | ₦{r['price']:8.2f} | "
              f"{r['perf_1w']:+6.1f}% | {r['perf_1m']:+6.1f}% | {r.get('sector','')}")
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


print(f"\n{'='*70}")
print(f"📊 BLUE-CHIP WEEKLY SUMMARY — {datetime.now().strftime('%Y-%m-%d')}")
print(f"{'='*70}")
print(f"\n  Snapshot date           : {snapshot_date}")
print(f"  Stocks analyzed         : {len(df)}")
print(f"  Blue-chips found        : {len(bluechips)}")
print(f"  Dip opportunities       : {len(dips)}")

if len(bluechips) > 0:
    by_tier = bluechips['tier'].value_counts()
    print(f"\n  Tier distribution:")
    for tier in ['🏆 Elite', '⭐ Premium', '✅ Quality', '📊 Watchlist']:
        print(f"    {tier:14}: {int(by_tier.get(tier, 0))}")

print(f"\n{'='*70}")
print("\n🔔 Next steps:")
print("  1. Prioritize 🏆 Elite + ⭐ Premium names with 🟢 BUY ZONE entry signals")
print("  2. Stage entries on Blue-Chip Dips — quality on temporary pullback")
print("  3. Cross-check 'BOTH' list (gems ∩ blue-chip) for highest conviction")
print("  4. Re-run after each new TradingView snapshot")
