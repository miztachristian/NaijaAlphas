import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
import os
os.chdir(r".")
sys.path.insert(0, ".")
from analysis.hidden_gems import find_hidden_gems, find_value_turnarounds
from analysis.snapshot_ranker import rank_snapshot

df = pd.read_parquet("data/snapshots/2026-02-19/snapshot.parquet")
df["liquidity_value_1d"] = df["price"] * df["volume_1d"]

MY_PORTFOLIO = {
    # Illustrative sample so the script runs standalone.
    # Production loads from PORTFOLIO_DIR/holdings.json (git-ignored).
    "GTCO":       {"shares": 1000, "avg_cost": 50.00},
    "DANGCEM":    {"shares": 100,  "avg_cost": 450.00},
    "ZENITHBANK": {"shares": 1000, "avg_cost": 40.00},
    "NESTLE":     {"shares": 50,   "avg_cost": 1000.00},
}

print("="*80)
print("PORTFOLIO P&L AS OF 2026-02-19")
print("="*80)
positions = []
for sym, pos in MY_PORTFOLIO.items():
    stock = df[df["symbol"] == sym]
    if len(stock) > 0:
        r = stock.iloc[0]
        price = r["price"]
        cost = pos["shares"] * pos["avg_cost"]
        val = pos["shares"] * price
        pnl = val - cost
        pnl_pct = (pnl/cost)*100
        def safe(row, key, default=0):
            v = row.get(key, default)
            return default if pd.isna(v) else v
        positions.append({
            "symbol": sym, "shares": pos["shares"], "avg_cost": pos["avg_cost"],
            "price": price, "cost_basis": cost, "value": val, "pnl": pnl, "pnl_pct": pnl_pct,
            "perf_1w": safe(r, "perf_1w"),
            "perf_1m": safe(r, "perf_1m"),
            "perf_3m": safe(r, "perf_3m"),
            "eps_growth": safe(r, "eps_growth_ttm"),
            "rev_growth": safe(r, "revenue_growth_ttm"),
            "roe": safe(r, "roe_ttm"),
            "margin": safe(r, "net_margin_ttm"),
            "pe": safe(r, "pe_ratio"),
            "rsi": safe(r, "relative_strength_index_14_1_day"),
            "tech_rating": safe(r, "technical_rating_1_day", ""),
            "div_yield": safe(r, "dividend_yield_trailing_12_months"),
            "volume": safe(r, "volume_1d"),
            "rel_vol": safe(r, "relative_volume_1_day"),
        })

pdf = pd.DataFrame(positions).sort_values("pnl_pct", ascending=False)
total_cost = pdf["cost_basis"].sum()
total_val = pdf["value"].sum()
total_pnl = pdf["pnl"].sum()
print(f"Total Cost: N{total_cost:,.0f}  |  Value: N{total_val:,.0f}  |  P&L: N{total_pnl:,.0f} ({total_pnl/total_cost*100:+.1f}%)")
print()
for _, p in pdf.iterrows():
    rsi_flag = " [RSI>70\!]" if p["rsi"] > 70 else ""
    sym = p["symbol"]
    line = "{:12} | N{:>10,.2f} | Cost N{:>10,.0f} | Val N{:>10,.0f} | P&L {:>+7.1f}% | 1W {:>+6.1f}% | 1M {:>+6.1f}% | RSI {:.0f}{}".format(
        sym, p["price"], p["cost_basis"], p["value"], p["pnl_pct"], p["perf_1w"], p["perf_1m"], p["rsi"], rsi_flag)
    print(line)

print()
print("="*80)
print("TOP 30 STOCKS - AGGRESSIVE GROWTH RANKING")
print("="*80)
rankings = rank_snapshot(df)
agg = rankings["aggressive_growth"]
cols = ["rank","symbol","price","perf_1w","perf_1m","perf_3m","eps_growth_ttm","revenue_growth_ttm","roe_ttm","net_margin_ttm","growth_potential_score_aggressive"]
cols = [c for c in cols if c in agg.columns]
print(agg.head(30)[cols].to_string(index=False))

print()
print("="*80)
print("TOP 30 STOCKS - GROWTH WITH GUARDRAILS")
print("="*80)
guard = rankings["growth_with_guardrails"]
cols2 = ["rank","symbol","price","perf_1w","perf_1m","perf_3m","eps_growth_ttm","revenue_growth_ttm","roe_ttm","net_margin_ttm","growth_potential_score_guardrails"]
cols2 = [c for c in cols2 if c in guard.columns]
print(guard.head(30)[cols2].to_string(index=False))

print()
print("="*80)
print("HIDDEN GEMS (Strong Fundamentals, Not Yet Hot)")
print("="*80)
gems = find_hidden_gems(df, top_n=25)
owned = set(MY_PORTFOLIO.keys())
for _, row in gems.iterrows():
    flag = " [OWNED]" if row["symbol"] in owned else ""
    eps_g = row.get("eps_growth_ttm", 0)
    if pd.isna(eps_g): eps_g = 0
    roe_v = row.get("roe_ttm", 0)
    if pd.isna(roe_v): roe_v = 0
    margin_v = row.get("net_margin_ttm", 0)
    if pd.isna(margin_v): margin_v = 0
    line = "{:2}. {:12} | N{:>8.2f} | Score: {:>5.1f} | {} | 1M: {:>+6.1f}% | EPS: {:>+7.0f}% | ROE: {:>+6.1f}% | Margin: {:>+5.1f}%{}".format(
        row["gem_rank"], row["symbol"], row["price"], row["hidden_gem_score"], row["heat_status"], row["perf_1m"], eps_g, roe_v, margin_v, flag)
    print(line)

print()
print("="*80)
print("VALUE TURNAROUND CANDIDATES")
print("="*80)
turns = find_value_turnarounds(df, top_n=15)
for _, row in turns.iterrows():
    flag = " [OWNED]" if row["symbol"] in owned else ""
    eps_g = row.get("eps_growth_ttm", 0)
    if pd.isna(eps_g): eps_g = 0
    margin_v = row.get("net_margin_ttm", 0)
    if pd.isna(margin_v): margin_v = 0
    line = "{:2}. {:12} | N{:>8.2f} | 1M: {:>+6.1f}% | 3M: {:>+6.1f}% | EPS: {:>+7.0f}% | Margin: {:>+5.1f}%{}".format(
        row["turnaround_rank"], row["symbol"], row["price"], row["perf_1m"], row["perf_3m"], eps_g, margin_v, flag)
    print(line)

print()
print("="*80)
print("BEST OPPORTUNITIES NOT IN PORTFOLIO (From Rankings)")
print("="*80)
not_owned_agg = agg[~agg["symbol"].isin(owned)].head(20)
for _, row in not_owned_agg.iterrows():
    line = "{:2}. {:12} | N{:>8.2f} | 1W: {:>+6.1f}% | 1M: {:>+6.1f}% | 3M: {:>+6.1f}% | EPS: {:>+7.0f}% | Score: {:.1f}".format(
        row["rank"], row["symbol"], row["price"], row.get("perf_1w",0), row.get("perf_1m",0), row.get("perf_3m",0), row.get("eps_growth_ttm",0), row["growth_potential_score_aggressive"])
    print(line)

print()
print("="*80)
print("YOUR HOLDINGS IN THE AGGRESSIVE RANKING")
print("="*80)
owned_in_agg = agg[agg["symbol"].isin(owned)]
for _, row in owned_in_agg.iterrows():
    line = "Rank {:3} | {:12} | Score: {:.1f} | 1M: {:>+6.1f}%".format(
        row["rank"], row["symbol"], row["growth_potential_score_aggressive"], row.get("perf_1m",0))
    print(line)

print()
print("="*80)
print("WEAKEST HOLDINGS (Consider Selling/Trimming)")
print("="*80)
for _, p in pdf.iterrows():
    sym = p["symbol"]
    in_ranking = agg[agg["symbol"] == sym]
    rank_pos = int(in_ranking.iloc[0]["rank"]) if len(in_ranking) > 0 else 999
    issues = []
    if rank_pos > 60: issues.append("Low rank #%d" % rank_pos)
    if p["perf_1m"] < 0: issues.append("Negative 1M (%+.1f%%)" % p["perf_1m"])
    if p["perf_1w"] < -3: issues.append("Dropping 1W (%+.1f%%)" % p["perf_1w"])
    if p["rsi"] > 75: issues.append("Overbought RSI=%.0f" % p["rsi"])
    if p["eps_growth"] < 0 and p["margin"] < 5: issues.append("Weak fundamentals")
    if issues:
        issue_str = ", ".join(issues)
        print("{:12} | P&L: {:>+7.1f}% | Issues: {}".format(sym, p["pnl_pct"], issue_str))
