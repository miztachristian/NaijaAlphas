"""consolidate.py — advisory view of the current book under the Core strategy.

Scores the existing holdings (data/portfolio/holdings.json) with the Core scorer
and buckets each name. ADVISORY ONLY: the core buys are funded by fresh capital
(see the design spec, §3-4), so this does not generate sells — it tells you which
current holdings are core-grade, which merely pass the gates, and which are
candidates to prune over time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PORTFOLIO_DIR, PortfolioConfig  # noqa: E402


def load_holdings(path: Optional[Path] = None) -> dict:
    path = path or (PORTFOLIO_DIR / PortfolioConfig.HOLDINGS_FILE)
    return json.loads(Path(path).read_text(encoding="utf-8")).get("holdings", {})


def consolidate(scores: pd.DataFrame, holdings: dict,
                core_symbols: Iterable[str]) -> pd.DataFrame:
    """Bucket each current holding: CORE / HOLD-ELIGIBLE / CONSIDER-TRIM.

    `scores` is the full table from compute_core_scores; `core_symbols` are
    today's top-2 picks. Returns a table sorted by core_score (desc).
    """
    core_set = set(core_symbols)
    by_symbol = scores.set_index("symbol")
    rows = []
    for sym, pos in holdings.items():
        s = by_symbol.loc[sym] if sym in by_symbol.index else None
        price = float(s["price"]) if s is not None and pd.notna(s["price"]) else float("nan")
        mv = pos.get("shares", 0) * price if price == price else float("nan")  # nan-safe
        score = float(s["core_score"]) if s is not None and pd.notna(s["core_score"]) else float("nan")
        eligible = bool(s["eligible"]) if s is not None else False
        fails = (s["gate_fails"] if s is not None else "not-in-snapshot") or ""

        if sym in core_set:
            bucket = "CORE"
        elif eligible:
            bucket = "HOLD-ELIGIBLE"
        else:
            bucket = "CONSIDER-TRIM"

        rows.append(dict(symbol=sym, bucket=bucket, core_score=score,
                         market_value=mv, shares=pos.get("shares", 0),
                         gate_fails=fails if bucket == "CONSIDER-TRIM" else ""))
    return pd.DataFrame(rows).sort_values("core_score", ascending=False,
                                          na_position="last").reset_index(drop=True)
