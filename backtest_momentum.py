"""
Momentum-strategy backtest on the NGX universe using real historical data
from data/historical/.

Strategy: at each rebalance date, rank tickers by 12-1 month momentum
(12-month return excluding the most recent month). Hold top N equally weighted.
Compare against an equal-weight buy-and-hold benchmark of the same universe.
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from analysis.backtest import BacktestEngine, momentum_selection
from core.data_loader import NGXDataLoader


def build_universe(loader: NGXDataLoader, min_rows_in_window: int) -> list[str]:
    """Return tickers that have enough bars in our backtest window."""
    hist_dir = Path("data/historical")
    universe: list[str] = []
    for p in sorted(hist_dir.glob("*.parquet")):
        sym = p.stem
        df = loader.get_price_data(sym)
        if df is None or len(df) < min_rows_in_window:
            continue
        universe.append(sym)
    return universe


def equal_weight_buyhold(loader, tickers, start, end, capital):
    """Benchmark: buy each ticker equally on day 1, hold until end."""
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


def main() -> None:
    loader = NGXDataLoader(use_cache=False)

    start = datetime(2024, 6, 1)
    end = datetime(2026, 5, 21)

    print("=" * 72)
    print("NGX MOMENTUM BACKTEST")
    print(f"Window: {start.date()} -> {end.date()}")
    print("=" * 72)

    universe = build_universe(loader, min_rows_in_window=300)
    print(f"Eligible universe: {len(universe)} tickers (>=300 bars cached)")

    engine = BacktestEngine(
        initial_capital=10_000_000,        # 10M NGN
        rebalance_frequency="quarterly",
        transaction_cost=0.01,
        slippage=0.005,
        max_position_size=0.20,
        stop_loss=-0.15,
        take_profit=0.50,
    )

    def select_top10(tickers, date):
        return momentum_selection(tickers, date, n=10)

    result = engine.run_backtest(
        tickers=universe,
        start_date=start,
        end_date=end,
        selection_fn=select_top10,
        n_holdings=10,
        strategy_name="12-1 Momentum, Top 10, Quarterly",
    )

    print(engine.generate_report(result))

    # Benchmark: equal-weight buy-and-hold of the same universe
    bench_value, n = equal_weight_buyhold(loader, universe, start, end, engine.initial_capital)
    bench_ret = (bench_value / engine.initial_capital - 1) * 100
    years = (end - start).days / 365.25
    bench_cagr = ((bench_value / engine.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    print()
    print("=" * 72)
    print(f"BENCHMARK — equal-weight buy-and-hold ({n} tickers)")
    print("=" * 72)
    print(f"  Final value      : N{bench_value:>15,.0f}")
    print(f"  Total return     : {bench_ret:>+8.2f}%")
    print(f"  Annualised (CAGR): {bench_cagr:>+8.2f}%")
    print(f"  Strategy vs bench: {result.total_return - bench_ret:+.2f}pp total, "
          f"{result.annualized_return - bench_cagr:+.2f}pp CAGR")


if __name__ == "__main__":
    main()
