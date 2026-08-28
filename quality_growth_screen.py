"""Quality + Growth Combined Stock Screen"""
import pandas as pd
import numpy as np

df = pd.read_parquet('data/snapshots/2026-08-20/snapshot.parquet')

def quality_score(row):
    score = 0
    total_weight = 0
    
    roe = row.get('roe_ttm', np.nan)
    if not pd.isna(roe) and roe > 0:
        score += min(roe / 40, 1) * 20
        total_weight += 20
    
    roa = row.get('roa_ttm', np.nan)
    if not pd.isna(roa) and roa > 0:
        score += min(roa / 20, 1) * 10
        total_weight += 10
    
    nm = row.get('net_margin_ttm', np.nan)
    if not pd.isna(nm) and nm > 0:
        score += min(nm / 30, 1) * 15
        total_weight += 15
    
    om = row.get('operating_margin_ttm', np.nan)
    if not pd.isna(om) and om > 0:
        score += min(om / 30, 1) * 10
        total_weight += 10
    
    gm = row.get('gross_margin_ttm', np.nan)
    if not pd.isna(gm) and gm > 0:
        score += min(gm / 60, 1) * 10
        total_weight += 10
    
    de = row.get('debt_to_equity', np.nan)
    if not pd.isna(de) and de >= 0:
        score += max(0, 1 - de / 3) * 15
        total_weight += 15
    
    fcf = row.get('free_cash_flow_trailing_12_months', np.nan)
    if not pd.isna(fcf):
        if fcf > 0:
            score += 10
        total_weight += 10
    
    cr = row.get('current_ratio', np.nan)
    if not pd.isna(cr) and cr > 0:
        score += min(cr / 2, 1) * 10
        total_weight += 10
    
    if total_weight > 0:
        return (score / total_weight) * 100
    return np.nan


def growth_score(row):
    score = 0
    total_weight = 0
    
    eps_g = row.get('eps_growth_ttm', np.nan)
    if not pd.isna(eps_g) and eps_g > 0:
        score += min(eps_g / 100, 1) * 30
        total_weight += 30
    elif not pd.isna(eps_g):
        total_weight += 30
    
    rev_g = row.get('revenue_growth_ttm', np.nan)
    if not pd.isna(rev_g) and rev_g > 0:
        score += min(rev_g / 60, 1) * 25
        total_weight += 25
    elif not pd.isna(rev_g):
        total_weight += 25
    
    p1y = row.get('perf_1y', np.nan)
    if not pd.isna(p1y) and p1y > 0:
        score += min(p1y / 200, 1) * 20
        total_weight += 20
    elif not pd.isna(p1y):
        total_weight += 20
    
    p3m = row.get('perf_3m', np.nan)
    if not pd.isna(p3m) and p3m > 0:
        score += min(p3m / 100, 1) * 15
        total_weight += 15
    elif not pd.isna(p3m):
        total_weight += 15
    
    roic = row.get('roic_ttm', np.nan)
    if not pd.isna(roic) and roic > 0:
        score += min(roic / 25, 1) * 10
        total_weight += 10
    
    if total_weight > 0:
        return (score / total_weight) * 100
    return np.nan


# Apply scores
df['quality'] = df.apply(quality_score, axis=1)
df['growth'] = df.apply(growth_score, axis=1)

# Filter: need both scores, both positive
valid = df[(df['quality'].notna()) & (df['growth'].notna()) & (df['quality'] > 0) & (df['growth'] > 0)].copy()

# Combined score: 50% quality + 50% growth
valid['combined'] = (valid['quality'] * 0.5) + (valid['growth'] * 0.5)

# Sort by combined
top = valid.nlargest(20, 'combined')

print('=' * 130)
print('TOP 20: QUALITY + GROWTH COMBINED RANKING (Feb 25, 2026)')
print('=' * 130)
header = f"{'Rk':>3} {'Symbol':>12} {'Price':>9} {'Quality':>8} {'Growth':>8} {'COMBINED':>9} {'RSI':>6} {'Entry':>6} | {'ROE%':>6} {'NMarg%':>7} {'D/E':>5} {'EPSGr%':>7} {'RevGr%':>7} {'1Y%':>7}"
print(header)
print('-' * 130)

for i, (_, row) in enumerate(top.iterrows(), 1):
    rsi = row.get('relative_strength_index_14_1_day', np.nan)
    if pd.isna(rsi):
        entry = '???'
    elif rsi <= 30:
        entry = 'BUY!'
    elif rsi <= 50:
        entry = 'GOOD'
    elif rsi <= 65:
        entry = 'OK'
    elif rsi <= 75:
        entry = 'WARM'
    else:
        entry = 'HOT'
    
    def fmt(val, w=6, dec=1):
        if pd.isna(val): return 'N/A'.rjust(w)
        return f"{val:.{dec}f}".rjust(w)
    
    def fmti(val, w=6):
        if pd.isna(val): return 'N/A'.rjust(w)
        return f"{val:.0f}".rjust(w)
    
    rsi_s = fmt(rsi, 5, 1)
    roe_s = fmt(row.get('roe_ttm'), 6, 1)
    nm_s = fmt(row.get('net_margin_ttm'), 7, 1)
    de_s = fmt(row.get('debt_to_equity'), 5, 2)
    eps_s = fmti(row.get('eps_growth_ttm'), 7)
    rev_s = fmti(row.get('revenue_growth_ttm'), 7)
    p1y_s = fmti(row.get('perf_1y'), 7)
    
    print(f"{i:3} {row['symbol']:>12} {row['price']:>9.2f} {row['quality']:>7.1f}  {row['growth']:>7.1f}  {row['combined']:>8.1f}  {rsi_s} {entry:>6} | {roe_s} {nm_s} {de_s} {eps_s} {rev_s} {p1y_s}")

print('=' * 130)
print()
print("LEGEND:")
print("  Quality = ROE + ROA + Margins + Low Debt + Positive FCF + Liquidity")
print("  Growth  = EPS Growth + Revenue Growth + 1Y Return + 3M Return + ROIC")
print("  Entry: BUY! (RSI<=30) | GOOD (30-50) | OK (50-65) | WARM (65-75) | HOT (75+)")
