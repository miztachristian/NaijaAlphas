"""concentrate.py — size FRESH capital into the 2-name Core Concentrate.

Takes the Core scores (analysis.core_strategy.compute_core_scores) and a fresh
capital amount, selects the top N=2 eligible names, and splits the ~85% core
sleeve score-weighted (single-name fallback = full sleeve). Computes per-pick
naira target, share count, and a liquidity "days-to-build" estimate.

Capital is fresh money the user is deploying now — NOT funded by selling the
existing book (see the design spec, §3).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CoreStrategyConfig  # noqa: E402
from analysis.core_strategy import eligible_ranked  # noqa: E402


@dataclass
class CorePosition:
    symbol: str
    sector: str
    core_score: float
    price: float
    weight_pct: float          # % of total fresh capital
    target_naira: float
    shares: int
    daily_turnover: float
    days_to_build: float       # at MAX_LIQUIDITY_FRACTION of daily turnover


@dataclass
class ConcentratePlan:
    capital: float
    deployable: float
    core_budget: float
    punt_budget: float
    cash_reserve: float
    positions: List[CorePosition] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([p.__dict__ for p in self.positions])


def _pick_core(elig: pd.DataFrame, cfg) -> pd.DataFrame:
    """Top N=cfg.N_CORE eligible names, honouring FORBID_SAME_SECTOR."""
    if elig.empty:
        return elig
    picks = [elig.iloc[0]]
    for _, row in elig.iloc[1:].iterrows():
        if len(picks) >= cfg.N_CORE:
            break
        if cfg.FORBID_SAME_SECTOR and any(row["sector"] == p["sector"] for p in picks):
            continue
        picks.append(row)
    return pd.DataFrame(picks).reset_index(drop=True)


def size_core(scores: pd.DataFrame, capital: float,
              cfg: Optional[type] = None) -> ConcentratePlan:
    """Build the concentrate plan for `capital` of fresh money."""
    cfg = cfg or CoreStrategyConfig
    if capital <= 0:
        raise ValueError("capital must be positive")

    deployable = capital * (1 - cfg.CASH_RESERVE)
    core_budget = deployable * cfg.CORE_SLEEVE
    punt_budget = deployable * cfg.PUNT_SLEEVE

    plan = ConcentratePlan(
        capital=capital, deployable=deployable, core_budget=core_budget,
        punt_budget=punt_budget, cash_reserve=capital - deployable,
    )

    picks = _pick_core(eligible_ranked(scores), cfg)
    if picks.empty:
        plan.flags.append("no eligible names — core stays in cash")
        return plan

    if cfg.SPLIT == "equal":
        weights = [1.0 / len(picks)] * len(picks)
    else:  # score_weighted
        ssum = picks["core_score"].sum()
        weights = [s / ssum for s in picks["core_score"]]

    for (_, row), w in zip(picks.iterrows(), weights):
        target = core_budget * w
        price = row["price"]
        turn = row["turnover"]
        cap_per_day = cfg.MAX_LIQUIDITY_FRACTION * turn if turn else 0.0
        plan.positions.append(CorePosition(
            symbol=row["symbol"], sector=row["sector"], core_score=row["core_score"],
            price=price, weight_pct=target / capital * 100.0, target_naira=target,
            shares=int(target // price) if price and price > 0 else 0,
            daily_turnover=turn,
            days_to_build=(target / cap_per_day) if cap_per_day > 0 else float("inf"),
        ))

    if len(picks) == 1:
        plan.flags.append("only one name qualified — full core sleeve in it")
    if len(picks) >= 2 and picks.iloc[0]["sector"] == picks.iloc[1]["sector"]:
        plan.flags.append(f"both picks in same sector ({picks.iloc[0]['sector']}) — doubled sector risk")
    for p in plan.positions:
        if p.days_to_build > 1.0:
            plan.flags.append(f"{p.symbol}: ~{p.days_to_build:.1f} days to build (liquidity-staged)")

    return plan
