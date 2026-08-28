"""
Compare momentum-strategy variants against an equal-weight buy-and-hold
benchmark, using real NGX historical data from data/historical/.

Variants tested:
  A: Baseline      - 12-1 momentum, quarterly, stop -15%
  B: + 200d filter - only buy names trading above their 200-day MA
  C: Monthly       - monthly rebalance instead of quarterly, +200d filter
  D: No stop-loss  - monthly, +200d filter, no -15% stop (let winners run)
  E: Loose stop    - monthly, +200d filter, -25% stop
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from analysis.backtest import BacktestEngine
from core.data_loader import NGXDataLoader

START = datetime(2024, 6, 1)
END = datetime(2026, 5, 15)
INITIAL_CAPITAL = 10_000_000
HOLDINGS = 10


def build_universe(loader: NGXDataLoader, min_rows: int) -> list[str]:
    hist_dir = Path("data/historical")
    universe = []
    for p in sorted(hist_dir.glob("*.parquet")):
        df = loader.get_price_data(p.stem)
        if df is None or len(df) < min_rows:
            continue
        universe.append(p.stem)
    return universe


def momentum_selection_basic(loader: NGXDataLoader, tickers: list[str],
                             date: datetime, n: int) -> list[str]:
    scored = []
    for t in tickers:
        prices = loader.get_price_series(t)
        if prices is None:
            continue
        prices = prices[prices.index < date]
        if len(prices) < 252:
            continue
        p_12m = prices.iloc[-252]
        p_1m = prices.iloc[-21]
        if p_12m <= 0:
            continue
        mom = (p_1m / p_12m - 1)
        scored.append((t, mom))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:n]]


def momentum_selection_with_200d(loader: NGXDataLoader, tickers: list[str],
                                 date: datetime, n: int) -> list[str]:
    """Top-N by 12-1 momentum AND currently above 200-day moving average."""
    scored = []
    for t in tickers:
        prices = loader.get_price_series(t)
        if prices is None:
            continue
        prices = prices[prices.index < date]
        if len(prices) < 252:
            continue
        p_12m = prices.iloc[-252]
        p_1m = prices.iloc[-21]
        p_now = prices.iloc[-1]
        if p_12m <= 0:
            continue
        sma_200 = prices.iloc[-200:].mean()
        if p_now <= sma_200:
            continue  # filtered out: stock not in an uptrend
        mom = (p_1m / p_12m - 1)
        scored.append((t, mom))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:n]]


def equal_weight_buyhold(loader, tickers, start, end, capital):
    per_stock = capital / len(tickers)
    final_value = 0.0
    realised = 0
    for t in tickers:
        df = loader.get_price_data(t)
        if df is None:
            continue
        df = df.set_index("Date")
        in_window = df[(df.index >= start) & (df.index <= end)]
        if len(in_window) < 2:
            continue
        start_px = float(in_window.iloc[0]["Close"])
        end_px = float(in_window.iloc[-1]["Close"])
        if start_px <= 0:
            continue
        final_value += per_stock * (end_px / start_px)
        realised += 1
    return final_value, realised


def run_variant(name: str, loader: NGXDataLoader, universe: list[str],
                rebalance: str, stop_loss: float,
                selection_builder: Callable):
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        rebalance_frequency=rebalance,
        transaction_cost=0.01,
        slippage=0.005,
        max_position_size=0.20,
        stop_loss=stop_loss,
        take_profit=0.50,
    )

    def select(tickers, date):
        return selection_builder(loader, tickers, date, HOLDINGS)

    r = engine.run_backtest(
        tickers=universe, start_date=START, end_date=END,
        selection_fn=select, n_holdings=HOLDINGS, strategy_name=name,
    )
    return r


def main():
    loader = NGXDataLoader(use_cache=False)
    universe = build_universe(loader, min_rows=300)
    print("=" * 78)
    print(f"NGX BACKTEST VARIANTS  |  {START.date()} -> {END.date()}  |  universe {len(universe)} tickers")
    print("=" * 78)

    results = []

    print("\n>>> Variant A: Baseline (12-1 momentum, quarterly, stop -15%)")
    rA = run_variant("A: Baseline", loader, universe,
                     rebalance="quarterly", stop_loss=-0.15,
                     selection_builder=momentum_selection_basic)
    results.append(("A: Baseline", rA))

    print("\n>>> Variant B: + 200-day MA filter (quarterly, stop -15%)")
    rB = run_variant("B: +200d filter", loader, universe,
                     rebalance="quarterly", stop_loss=-0.15,
                     selection_builder=momentum_selection_with_200d)
    results.append(("B: +200d filter", rB))

    print("\n>>> Variant C: Monthly + 200-day filter (stop -15%)")
    rC = run_variant("C: Monthly+200d", loader, universe,
                     rebalance="monthly", stop_loss=-0.15,
                     selection_builder=momentum_selection_with_200d)
    results.append(("C: Monthly+200d", rC))

    print("\n>>> Variant D: Monthly + 200-day, NO stop-loss")
    rD = run_variant("D: No stop", loader, universe,
                     rebalance="monthly", stop_loss=-0.99,  # effectively disabled
                     selection_builder=momentum_selection_with_200d)
    results.append(("D: No stop", rD))

    print("\n>>> Variant E: Monthly + 200-day, loose -25% stop")
    rE = run_variant("E: -25% stop", loader, universe,
                     rebalance="monthly", stop_loss=-0.25,
                     selection_builder=momentum_selection_with_200d)
    results.append(("E: -25% stop", rE))

    # Benchmark
    bench_value, n = equal_weight_buyhold(loader, universe, START, END, INITIAL_CAPITAL)
    bench_ret = (bench_value / INITIAL_CAPITAL - 1) * 100
    years = (END - START).days / 365.25
    bench_cagr = ((bench_value / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Summary table
    print()
    print("=" * 78)
    print(f"SUMMARY — equal-weight buy-and-hold benchmark: +{bench_ret:.1f}% total  "
          f"({bench_cagr:+.1f}% CAGR, {n} tickers)")
    print("=" * 78)
    print(f"{'Variant':22} {'Total%':>9} {'CAGR%':>8} {'Vol%':>7} {'MaxDD%':>8} "
          f"{'Sharpe':>7} {'WinRate':>8} {'vs B&H':>8}")
    print("-" * 78)
    for name, r in results:
        diff = r.total_return - bench_ret
        print(f"{name:22} {r.total_return:>+9.1f} {r.annualized_return:>+8.1f} "
              f"{r.volatility:>7.1f} {r.max_drawdown:>+8.1f} {r.sharpe_ratio:>7.2f} "
              f"{r.win_rate:>7.0f}% {diff:>+8.1f}")
    print("-" * 78)
    print(f"{'  Buy-and-hold (134)':22} {bench_ret:>+9.1f} {bench_cagr:>+8.1f} "
          f"{'—':>7} {'—':>8} {'—':>7} {'—':>8}      —")


if __name__ == "__main__":
    main()
