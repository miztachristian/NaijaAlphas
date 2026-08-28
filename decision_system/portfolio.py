"""
portfolio.py — turn the conviction ranking into a concrete order list.

The model is capital deployment, not full rebalancing:

  * New cash is deployed into the highest-conviction STRONG_BUY / ADD names,
    conviction-weighted, split across two sleeves (long-term ~85% /
    short-term ~15%), subject to position / sector / liquidity caps.
  * Held names flagged SELL are exited; held names flagged TRIM are shaved.
  * Held names flagged HOLD / ADD are simply kept (no churn).

The 85/15 sleeve split lives ONLY here — the conviction score is unified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import PortfolioConfig, PORTFOLIO_DIR, get_ticker_sector
from decision_system.models import Decision, OrderItem


def _finite(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def load_holdings(path: Optional[Path] = None) -> Dict[str, dict]:
    """Load the canonical holdings — {ticker: {shares, avg_cost, ...}}."""
    path = path or (PORTFOLIO_DIR / PortfolioConfig.HOLDINGS_FILE)
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("holdings", {})


class PortfolioConstructor:
    """Builds the buy/trim/sell order list from scored Decisions."""

    _BUY_ACTIONS = ("STRONG_BUY", "ADD")

    def __init__(self, config=PortfolioConfig):
        self.cfg = config

    def build(self, decisions: List[Decision], capital: float,
              holdings: Optional[Dict[str, dict]] = None,
              liquidity: Optional[Dict[str, float]] = None) -> List[OrderItem]:
        """
        Args:
            decisions: scored Decisions (each carries price, sleeve, ConvictionScore).
            capital:   new cash to deploy, in naira.
            holdings:  {ticker: {shares, avg_cost}} — canonical holdings if omitted.
            liquidity: optional {ticker: daily_traded_value_naira} for the liquidity cap.

        Returns:
            Ordered list of OrderItem (SELLs first, then BUY/TRIM by size).
        """
        holdings = load_holdings() if holdings is None else holdings
        liquidity = liquidity or {}
        by_ticker = {d.ticker: d for d in decisions}

        # --- current market values ---
        cur_val: Dict[str, float] = {}
        for tk, pos in holdings.items():
            d = by_ticker.get(tk)
            if d and _finite(d.price) and d.price > 0:
                cur_val[tk] = float(pos.get("shares", 0)) * d.price

        total_value = sum(cur_val.values()) + max(capital, 0.0)
        if total_value <= 0:
            return []

        orders: List[OrderItem] = []

        # --- 1. exits / trims on flagged holdings ---
        sold = set()
        for tk, pos in holdings.items():
            d = by_ticker.get(tk)
            if not d or not d.score:
                continue
            shares_held = int(pos.get("shares", 0))
            if shares_held <= 0 or not _finite(d.price):
                continue
            if d.score.action == "SELL":
                sold.add(tk)
                orders.append(OrderItem(
                    ticker=tk, side="SELL", shares=shares_held,
                    naira=shares_held * d.price, price=d.price, sleeve=d.sleeve,
                    reason=f"Conviction {d.score.conviction:.0f} — full exit."))
            elif d.score.action == "TRIM":
                trim = shares_held // 3
                if trim > 0:
                    orders.append(OrderItem(
                        ticker=tk, side="TRIM", shares=trim,
                        naira=trim * d.price, price=d.price, sleeve=d.sleeve,
                        reason=f"Conviction {d.score.conviction:.0f} — shave weak holding."))

        # --- 2. deploy new capital into top conviction names per sleeve ---
        deployable = max(capital, 0.0) * (1.0 - self.cfg.CASH_RESERVE)
        if deployable >= self.cfg.MIN_ORDER_NAIRA:
            buy_cands = [d for d in decisions
                         if d.score and d.score.action in self._BUY_ACTIONS
                         and d.ticker not in sold
                         and _finite(d.price) and d.price > 0]
            for sleeve, share in (("long_term", self.cfg.LONG_TERM_SLEEVE),
                                  ("short_term", self.cfg.SHORT_TERM_SLEEVE)):
                budget = deployable * share
                cands = sorted((d for d in buy_cands if d.sleeve == sleeve),
                               key=lambda d: d.score.conviction, reverse=True)
                orders.extend(self._deploy(cands, budget, total_value,
                                           cur_val, liquidity))

        # SELLs first (free up cash), then largest orders.
        orders.sort(key=lambda o: (o.side != "SELL", -o.naira))
        return orders

    # --- helpers ----------------------------------------------------------
    def _deploy(self, cands: List[Decision], budget: float, total_value: float,
                cur_val: Dict[str, float],
                liquidity: Dict[str, float]) -> List[OrderItem]:
        """Conviction-weight `budget` across the top candidates, capped."""
        if not cands or budget < self.cfg.MIN_ORDER_NAIRA:
            return []

        # How many positions fit, given the minimum position size.
        min_pos = self.cfg.MIN_POSITION * total_value
        max_k = max(1, int(budget // min_pos)) if min_pos > 0 else len(cands)
        chosen = cands[:min(max_k, len(cands))]

        convs = {d.ticker: max(d.score.conviction, 1.0) for d in chosen}
        wsum = sum(convs.values())

        # Per-sector running exposure (existing + planned buys) for the cap.
        sector_exposure: Dict[str, float] = {}
        for tk, val in cur_val.items():
            sector_exposure[get_ticker_sector(tk)] = (
                sector_exposure.get(get_ticker_sector(tk), 0.0) + val)
        max_sector = self.cfg.MAX_SECTOR * total_value
        max_pos = self.cfg.MAX_POSITION * total_value

        orders: List[OrderItem] = []
        for d in chosen:
            tk = d.ticker
            want = budget * convs[tk] / wsum

            # Cap: position size vs the whole book (existing + new).
            headroom = max(0.0, max_pos - cur_val.get(tk, 0.0))
            amount = min(want, headroom)

            # Cap: liquidity — never deploy more than a slice of daily turnover.
            liq = liquidity.get(tk)
            if liq and liq > 0:
                amount = min(amount, self.cfg.MAX_LIQUIDITY_FRACTION * liq)

            # Cap: sector concentration.
            sector = get_ticker_sector(tk)
            sector_headroom = max(0.0, max_sector - sector_exposure.get(sector, 0.0))
            amount = min(amount, sector_headroom)

            shares = int(amount / d.price)
            if shares <= 0 or shares * d.price < self.cfg.MIN_ORDER_NAIRA:
                continue
            spend = shares * d.price
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + spend
            verb = "add to" if cur_val.get(tk, 0.0) > 0 else "open"
            orders.append(OrderItem(
                ticker=tk, side="BUY", shares=shares, naira=spend,
                price=d.price, sleeve=d.sleeve,
                reason=f"Conviction {d.score.conviction:.0f} ({d.score.action}) "
                       f"— {verb}, {self._pct(spend, total_value)} of book."))
        return orders

    @staticmethod
    def _pct(value: float, total: float) -> str:
        return f"{(value / total * 100.0):.1f}%" if total > 0 else "n/a"
