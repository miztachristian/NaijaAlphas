"""
macro_regime.py — classify the Nigeria macro environment into a regime and
per-sector conviction tilts.

Consumes what ingest/macro.py writes (data/macro/<date>.json + history.parquet)
and produces a MacroState: rate-cycle / naira / oil sub-regimes, a readable
label, and an additive sector-tilt table (conviction points). The tilts are
declarative and bounded — they nudge the conviction score, never dominate it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import MacroConfig, MACRO_DIR, get_ticker_sector
from decision_system.models import MacroState


def _clamp(value: float, cap: float) -> float:
    return max(-cap, min(cap, value))


def _window_change(series: pd.Series, window: int) -> Optional[float]:
    """Fractional change from `window` rows ago (or earliest) to the latest."""
    s = series.dropna()
    if len(s) < 2:
        return None
    past = s.iloc[-min(window, len(s))]
    latest = s.iloc[-1]
    if past == 0:
        return None
    return (latest - past) / abs(past)


class MacroRegime:
    """Classifies macro inputs into a regime + sector tilts. Pure logic —
    inject `latest`/`history` directly to unit-test without disk."""

    def __init__(self, latest: dict, history: Optional[pd.DataFrame] = None):
        self.latest = latest
        self.history = history if history is not None else pd.DataFrame()

    # --- sub-regime classifiers ------------------------------------------
    def _rate_cycle(self) -> str:
        """CUTTING / HOLDING / HIKING from the MPR history (distinct levels)."""
        if "mpr" not in self.history.columns or len(self.history) < 2:
            return "HOLDING"
        levels = self.history["mpr"].dropna()
        distinct = levels[levels != levels.shift()]
        if len(distinct) < 2:
            return "HOLDING"
        latest, prior = distinct.iloc[-1], distinct.iloc[-2]
        if latest < prior:
            return "CUTTING"
        if latest > prior:
            return "HIKING"
        return "HOLDING"

    def _naira_trend(self) -> str:
        """STRENGTHENING / STABLE / WEAKENING from USD/NGN slope."""
        if "usdngn" not in self.history.columns:
            return "STABLE"
        change = _window_change(self.history["usdngn"], MacroConfig.TREND_WINDOW_DAYS)
        if change is None:
            return "STABLE"
        if change > MacroConfig.FX_MOVE_THRESHOLD:
            return "WEAKENING"      # more naira per USD => weaker naira
        if change < -MacroConfig.FX_MOVE_THRESHOLD:
            return "STRENGTHENING"
        return "STABLE"

    def _oil_trend(self) -> str:
        """RISING / FLAT / FALLING from Brent slope."""
        if "brent" not in self.history.columns:
            return "FLAT"
        change = _window_change(self.history["brent"], MacroConfig.TREND_WINDOW_DAYS)
        if change is None:
            return "FLAT"
        if change > MacroConfig.OIL_MOVE_THRESHOLD:
            return "RISING"
        if change < -MacroConfig.OIL_MOVE_THRESHOLD:
            return "FALLING"
        return "FLAT"

    def _yield_trend(self) -> str:
        """RISING_YIELDS / STABLE_YIELDS / FALLING_YIELDS from 10Y FGN slope."""
        if "bond_10y" not in self.history.columns:
            return "STABLE_YIELDS"
        change = _window_change(self.history["bond_10y"], MacroConfig.TREND_WINDOW_DAYS)
        if change is None:
            return "STABLE_YIELDS"
        if change > MacroConfig.YIELD_MOVE_THRESHOLD:
            return "RISING_YIELDS"
        if change < -MacroConfig.YIELD_MOVE_THRESHOLD:
            return "FALLING_YIELDS"
        return "STABLE_YIELDS"

    # --- tilt table -------------------------------------------------------
    def _sector_tilts(self, rate_cycle: str, naira: str, oil: str,
                      yields: str, inflation: float) -> dict:
        """Additive conviction-point tilt per sector, bounded by the cap."""
        strong = MacroConfig.TILT_STRONG
        mild = MacroConfig.TILT_MILD
        tilts: dict = {}

        def add(sector: str, points: float) -> None:
            tilts[sector] = tilts.get(sector, 0.0) + points

        # Naira: a weak naira rewards exporters, penalises import-dependent names.
        if naira == "WEAKENING":
            add("AGRICULTURE", strong)   # PRESCO / OKOMUOIL — palm oil exporters
            add("OIL_GAS", strong)
            add("CEMENT", mild)          # domestic pricing power
            add("CONSUMER", -mild)       # imported inputs
            add("HEALTHCARE", -mild)     # imported pharma inputs
        elif naira == "STRENGTHENING":
            add("CONSUMER", mild)
            add("HEALTHCARE", mild)
            add("AGRICULTURE", -mild)
            add("OIL_GAS", -mild)

        # Oil: direct read-through to oil & gas names.
        if oil == "RISING":
            add("OIL_GAS", strong)
        elif oil == "FALLING":
            add("OIL_GAS", -strong)

        # Rate cycle: cuts help rate-sensitive sectors; hikes help bank margins.
        if rate_cycle == "CUTTING":
            add("REALESTATE", strong)
            add("INDUSTRIAL", mild)
            add("CONSUMER", mild)
            add("BANKING", -mild)
        elif rate_cycle == "HIKING":
            add("BANKING", mild)
            add("REALESTATE", -mild)
            add("INDUSTRIAL", -mild)

        # Very high inflation squeezes discretionary consumer demand.
        if inflation == inflation and inflation > 25.0:  # not-NaN check
            add("CONSUMER", -mild)

        # Bond yields: independent of policy rate (reflect market expectations).
        # Rising yields hurt rate-sensitive long-duration assets; banks benefit
        # from a steeper curve / higher reinvestment yields.
        if yields == "RISING_YIELDS":
            add("REALESTATE", -strong)
            add("INDUSTRIAL", -mild)
            add("BANKING", mild)
        elif yields == "FALLING_YIELDS":
            add("REALESTATE", strong)
            add("INDUSTRIAL", mild)
            add("BANKING", -mild)

        cap = MacroConfig.TILT_STRONG + MacroConfig.TILT_MILD + 0.0
        return {s: _clamp(p, cap) for s, p in tilts.items()}

    # --- public -----------------------------------------------------------
    def classify(self) -> MacroState:
        rate_cycle = self._rate_cycle()
        naira = self._naira_trend()
        oil = self._oil_trend()
        yields = self._yield_trend()
        inflation = float(self.latest.get("inflation", float("nan")))

        tilts = self._sector_tilts(rate_cycle, naira, oil, yields, inflation)

        # Yield label reads naturally without the "_YIELDS" suffix in the regime
        # string (RISING_YIELDS -> "Rising yields"), keeping the existing format.
        yield_label = yields.replace("_YIELDS", "").lower()
        if yields == "STABLE_YIELDS":
            yield_label = "stable"
        label = (f"{rate_cycle.title()} rates · "
                 f"{naira.title()} naira · {oil.title()} oil · "
                 f"{yield_label.title()} yields")

        return MacroState(
            asof=str(self.latest.get("date", "")),
            regime_label=label,
            rate_cycle=rate_cycle,
            naira_trend=naira,
            oil_trend=oil,
            yield_trend=yields,
            mpr=float(self.latest.get("mpr", float("nan"))),
            inflation=inflation,
            usdngn=float(self.latest.get("usdngn", float("nan"))),
            brent=float(self.latest.get("brent", float("nan"))),
            bond_10y=float(self.latest.get("bond_10y", float("nan"))),
            sector_tilts=tilts,
            sources=self.latest.get("sources", {}),
            degraded=bool(self.latest.get("degraded", False)),
        )


def tilt_for(state: MacroState, ticker: str, sector: Optional[str] = None) -> float:
    """Macro tilt (conviction points) for a ticker, via its sector."""
    sector = sector or get_ticker_sector(ticker)
    return state.sector_tilts.get(sector, 0.0)


def load_macro_state(date: Optional[str] = None,
                     macro_dir: Path = MACRO_DIR) -> Optional[MacroState]:
    """
    Build a MacroState from data/macro/. Uses the given date's JSON (or the
    most recent one) plus history.parquet for trend slopes. Returns None if no
    macro data has been ingested yet.
    """
    macro_dir = Path(macro_dir)
    if not macro_dir.exists():
        return None

    if date:
        json_path = macro_dir / f"{date}.json"
    else:
        candidates = sorted(macro_dir.glob("[0-9]*.json"))
        json_path = candidates[-1] if candidates else None

    if not json_path or not json_path.exists():
        return None

    with open(json_path, encoding="utf-8") as fh:
        latest = json.load(fh)

    history = pd.DataFrame()
    hist_path = macro_dir / "history.parquet"
    if hist_path.exists():
        history = pd.read_parquet(hist_path).sort_values("date").reset_index(drop=True)
        if date:  # only use history up to and including the chosen date
            history = history[history["date"] <= date].reset_index(drop=True)

    return MacroRegime(latest, history).classify()
