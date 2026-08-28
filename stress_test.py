"""Stress-test a deployment plan across two consecutive daily snapshots."""
from pathlib import Path
import pandas as pd

from analysis.snapshot_ranker import rank_snapshot, load_config

proj = Path(__file__).parent
PLAN = ['OKOMUOIL','MTNN','STANBIC','ARADEL','FIDSON','CORNERST','NEIMETH']

def load(d):
    return pd.read_parquet(proj/'data'/'snapshots'/d/'snapshot.parquet')

s21, s22 = load('2026-05-21'), load('2026-05-22')
config = load_config(proj/'config'/'snapshot_ranker.yaml')
rk = rank_snapshot(s22, config=config)
guard = rk['growth_with_guardrails'].reset_index(drop=True)
guard['guard_rank'] = guard.index + 1
gmap = guard.set_index('symbol')[['guard_rank','growth_potential_score_guardrails']]

cols = ['price','perf_1w','perf_1m','perf_3m','eps_growth_ttm','roe_ttm','rsi_14']
print(f"{'TICKER':11} {'px 5/21':>9} {'px 5/22':>9} {'1d %':>7} {'1m %':>8} {'RSI':>6} "
      f"{'EPSg%':>8} {'ROE%':>7} {'guard#':>7} {'gscore':>7}")
print('-'*92)
for t in PLAN:
    a = s21[s21['symbol']==t]
    b = s22[s22['symbol']==t]
    if len(b)==0:
        print(f"{t:11} -- NOT IN 5/22 SNAPSHOT --"); continue
    a = a.iloc[0] if len(a) else None
    b = b.iloc[0]
    p21 = a['price'] if a is not None else float('nan')
    p22 = b['price']
    d1 = (p22/p21-1)*100 if p21==p21 and p21 else float('nan')
    gr = gmap.loc[t,'guard_rank'] if t in gmap.index else None
    gs = gmap.loc[t,'growth_potential_score_guardrails'] if t in gmap.index else float('nan')
    print(f"{t:11} {p21:9,.2f} {p22:9,.2f} {d1:+7.1f} {b['perf_1m']:+8.1f} "
          f"{b['rsi_14']:6.0f} {b['eps_growth_ttm']:8.0f} {b['roe_ttm']:7.0f} "
          f"{str(gr or '-'):>7} {gs:7.1f}")

print()
print("=== NEW 5/22 GUARDRAILS TOP 15 (anything new worth noting?) ===")
g = guard.head(15)[['guard_rank','symbol','sector','price','perf_1m','eps_growth_ttm',
                    'roe_ttm','growth_potential_score_guardrails']]
print(g.to_string(index=False,float_format=lambda x:f'{x:,.1f}'))
