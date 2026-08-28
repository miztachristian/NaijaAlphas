"""
pipeline.py — orchestrate the full decision run.

For a given snapshot it: validates the data, loads the macro regime, collects
every signal for every stock, fuses them into conviction scores, classifies
each into a sleeve, and constructs the portfolio order list.

Every per-stock step is wrapped — one stock (or one signal) failing degrades
that row and is recorded in the run manifest; it never aborts the run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from analysis.snapshot_store import SnapshotStore
from config.settings import BacktestConfig, get_ticker_sector
from decision_system.conviction import ConvictionEngine
from decision_system.macro_regime import load_macro_state, tilt_for
from decision_system.models import Decision, MacroState, OrderItem
from decision_system.portfolio import PortfolioConstructor, load_holdings
from decision_system.signals import SignalCollector


def _finite(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


@dataclass
class RunResult:
    """Everything one decision run produced."""

    snapshot_date: str = ""
    run_at: str = ""
    decisions: List[Decision] = field(default_factory=list)
    orders: List[OrderItem] = field(default_factory=list)
    macro_state: Optional[MacroState] = None
    capital: float = 0.0
    issues: List[str] = field(default_factory=list)   # per-stock degradations
    stats: Dict[str, int] = field(default_factory=dict)


def validate_snapshot(df: pd.DataFrame) -> List[str]:
    """Check the snapshot is usable. Raises on a fatal problem, else returns
    a list of non-fatal warnings."""
    if df is None or df.empty:
        raise ValueError("Snapshot is empty.")
    if "symbol" not in df.columns:
        raise ValueError("Snapshot has no 'symbol' column.")
    warnings: List[str] = []
    if "price" not in df.columns:
        warnings.append("Snapshot has no 'price' column — sizing will be limited.")
    n_dupes = int(df["symbol"].duplicated().sum())
    if n_dupes:
        warnings.append(f"{n_dupes} duplicate symbol rows — keeping first of each.")
    return warnings


class DecisionPipeline:
    """Runs signals -> conviction -> portfolio for a whole snapshot."""

    def __init__(self, capital: float = 0.0):
        self.capital = capital if capital > 0 else float(BacktestConfig.INITIAL_CAPITAL)
        self.engine = ConvictionEngine()

    def run(self, snapshot_date: Optional[str] = None) -> RunResult:
        store = SnapshotStore()
        date = snapshot_date or store.get_latest_snapshot_date()
        if not date:
            raise ValueError("No snapshots found under data/snapshots/.")

        loaded = store.load_snapshot(date)
        if not loaded or "merged" not in loaded:
            raise ValueError(f"Could not load snapshot for {date}.")
        df = loaded["merged"].drop_duplicates(subset="symbol", keep="first")

        result = RunResult(snapshot_date=date,
                           run_at=datetime.now().isoformat(),
                           capital=self.capital)
        result.issues.extend(validate_snapshot(df))

        macro = load_macro_state()
        result.macro_state = macro
        if macro is None:
            result.issues.append("No macro data — running without regime tilt.")

        holdings = load_holdings()
        collector = SignalCollector(snapshot_df=df)

        liquidity: Dict[str, float] = {}
        rows = df.set_index("symbol")
        for symbol, row in rows.iterrows():
            ticker = str(symbol)
            sector = str(row.get("sector") or get_ticker_sector(ticker))
            price = _to_float(row.get("price"))

            # liquidity_value_1d for the portfolio liquidity cap
            vol = _to_float(row.get("volume_1d"))
            if _finite(price) and _finite(vol):
                liquidity[ticker] = price * vol

            held = ticker in holdings
            shares = float(holdings.get(ticker, {}).get("shares", 0)) if held else 0.0

            try:
                signal_set = collector.collect(ticker, sector)
                tilt = tilt_for(macro, ticker, sector) if macro else 0.0
                score = self.engine.score(signal_set, macro_tilt=tilt, held=held)
                sleeve = self._classify_sleeve(signal_set)
            except Exception as exc:  # noqa: BLE001 - degrade this row, keep going
                result.issues.append(f"{ticker}: scoring failed ({exc}).")
                continue

            decision = Decision(
                ticker=ticker, sector=sector, price=price, score=score,
                held=held, current_shares=shares,
                current_value=(shares * price if _finite(price) else 0.0),
                sleeve=sleeve,
            )
            result.decisions.append(decision)

        result.decisions.sort(
            key=lambda d: d.score.conviction if d.score else 0.0, reverse=True)

        # Portfolio construction
        try:
            result.orders = PortfolioConstructor().build(
                result.decisions, self.capital, holdings, liquidity)
        except Exception as exc:  # noqa: BLE001
            result.issues.append(f"Portfolio construction failed ({exc}).")

        result.stats = self._stats(result)
        return result

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _classify_sleeve(signal_set) -> str:
        """Long-term (quality/fundamental led) vs short-term (momentum led)."""
        subs = signal_set.sub_scores
        lt = [subs.get(n) for n in ("fundamental", "quality")]
        st = [subs.get(n) for n in ("technical", "seasonality")]
        lt = [v for v in lt if _finite(v)]
        st = [v for v in st if _finite(v)]
        lt_avg = sum(lt) / len(lt) if lt else 0.0
        st_avg = sum(st) / len(st) if st else 0.0
        return "short_term" if st_avg > lt_avg else "long_term"

    @staticmethod
    def _stats(result: RunResult) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        for d in result.decisions:
            if d.score:
                actions[d.score.action] = actions.get(d.score.action, 0) + 1
        stats = {"stocks_scored": len(result.decisions),
                 "orders": len(result.orders),
                 "degraded_rows": len(result.issues)}
        stats.update({f"action_{k}": v for k, v in actions.items()})
        return stats


def _to_float(value) -> float:
    try:
        f = float(value)
        return f if not math.isnan(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")
