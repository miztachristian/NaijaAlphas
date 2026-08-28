import pandas as pd
import numpy as np

df = pd.read_parquet(r'data\snapshots\2026-08-20\snapshot.parquet')

def get_pe(row):
    pe = row.get('pe_ratio', np.nan)
    if pd.isna(pe) or pe == 0:
        eps = row.get('eps_basic_ttm', np.nan)
        price = row.get('price', np.nan)
        if pd.notna(eps) and eps > 0 and pd.notna(price):
            pe = price / eps
    return pe

results = []
for _, row in df.iterrows():
    ticker = row['symbol']
    if ticker in ['TIP']: continue
    
    pe = get_pe(row)
    roe = row.get('roe_ttm', np.nan)
    rev_g = row.get('revenue_growth_ttm', np.nan)
    net_inc_g = row.get('net_income_growth_ttm_yoy', np.nan)
    margin = row.get('net_margin_ttm', np.nan)
    debt_eq = row.get('debt_to_equity', np.nan)
    perf_1y = row.get('perf_1y', np.nan)
    
    # Filter for value, growth, and profitability
    if pd.notna(pe) and 0 < pe <= 20 and pd.notna(net_inc_g) and net_inc_g > 15 and pd.notna(roe) and roe > 15:
        score = 0
        
        # Valuation (MAX +5)
        if pe < 5: score += 5
        elif pe < 8: score += 3
        elif pe < 12: score += 1
        
        # Growth (MAX +6)
        if pd.notna(rev_g):
            if rev_g > 80: score += 3
            elif rev_g > 40: score += 2
            elif rev_g > 20: score += 1
            
        if net_inc_g > 150: score += 3
        elif net_inc_g > 80: score += 2
        elif net_inc_g > 30: score += 1
        
        # Quality/Profitability (MAX +4)
        if roe > 40: score += 3
        elif roe > 25: score += 2
        
        if pd.notna(debt_eq) and debt_eq < 0.5: score += 1
        
        results.append({
            'ticker': ticker,
            'desc': row.get('description', 'N/A'),
            'price': row.get('price', np.nan),
            'pe': pe,
            'roe': roe,
            'rev_g': rev_g,
            'net_inc_g': net_inc_g,
            'margin': margin,
            'debt_eq': debt_eq,
            'perf_1y': perf_1y,
            'score': score
        })

results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df = results_df.sort_values(by=['score', 'pe'], ascending=[False, True]).head(10)
    for _, r in results_df.iterrows():
        print(f"{'='*60}")
        print(f"{r['ticker']} - {r['desc']} | GEM SCORE: {r['score']}/15")
        print(f"Price: N{r['price']:.2f} | PE: {r['pe']:.2f} | ROE: {r['roe']:.1f}%")
        print(f"Rev Growth: {r['rev_g']:.1f}% | Net Inc Growth: {r['net_inc_g']:.1f}%")
        print(f"Margin: {r['margin']:.1f}% | Debt/Eq: {r['debt_eq']} | 1Yr Perf: {r['perf_1y']:.1f}%")
else:
    print("No gems found.")
