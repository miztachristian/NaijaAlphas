"""core_strategy.py — the Core Concentrate scorer.

Selects stocks on quality + profitability + growth, then uses volume as a
*confirmation* signal (accumulation, with a distribution veto). The output feeds
the 2-name concentrate sizer (decision_system/concentrate.py) and the advisory
consolidation report (decision_system/consolidate.py).

Pure and side-effect free: `compute_core_scores(df)` takes a snapshot DataFrame
and returns a per-stock scoring table. All thresholds live in
config.settings.CoreStrategyConfig.

Snapshot units: margins / growth / ROE-ROIC-ROA are PERCENT (25.0 == 25%);
debt_to_equity and current_ratio are ratios; relative_volume_1d ~1.0 is average.

Design: docs/superpowers/specs/2026-06-24-core-concentrate-strategy-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CoreStrategyConfig  # noqa: E402

# --- full-credit targets (scoring internals; tune here) ---
T_EPS_G, T_REV_G = 60.0, 40.0
T_NET, T_OP, T_GROSS, T_FCF = 30.0, 30.0, 60.0, 20.0
T_ROE, T_ROIC, T_ROA = 40.0, 30.0, 20.0
T_DE_MAX = 2.0          # debt/equity penalised linearly up to this
T_CR_SWEET = 2.0        # current-ratio sweet spot

OUTPUT_COLS = [
    "symbol", "sector", "price", "turnover",
    "growth", "profit", "quality", "accumulation", "base", "core_score",
    "eligible", "gate_fails",
    "roe_eff", "net_margin", "eps_growth", "rev_growth", "rvol", "perf_1m",
]


def _clip01(x: float, lo: float, hi: float) -> float:
    """Linear 0..1 ramp; NaN passes through."""
    if pd.isna(x):
        return np.nan
    return max(0.0, min(1.0, (float(x) - lo) / (hi - lo)))


def _wavg(pairs) -> float:
    """Weighted average over available (non-NaN) (score0to1, weight) pairs -> 0..100."""
    num = den = 0.0
    for s, w in pairs:
        if not pd.isna(s):
            num += s * w
            den += w
    return (num / den * 100.0) if den else np.nan


def _get(row, col) -> float:
    return row[col] if (col in row and not pd.isna(row[col])) else np.nan


def _score_row(row, cfg) -> dict:
    sym = row["symbol"]
    price, vol = _get(row, "price"), _get(row, "volume_1d")
    turnover = price * vol if not (pd.isna(price) or pd.isna(vol)) else np.nan

    roe, roic = _get(row, "roe_ttm"), _get(row, "roic_ttm")
    roe_eff = roe if not pd.isna(roe) else roic   # ROIC stands in for a missing ROE
    nmar = _get(row, "net_margin_ttm")
    ninc = _get(row, "net_income_trailing_12_months")
    epsg, revg = _get(row, "eps_growth_ttm"), _get(row, "revenue_growth_ttm")
    rvol, p1m = _get(row, "relative_volume_1d"), _get(row, "perf_1m")

    # ---- pillars (0..100) ----
    growth = _wavg([(_clip01(epsg, 0, T_EPS_G), 0.6),
                    (_clip01(revg, 0, T_REV_G), 0.4)])
    profit = _wavg([(_clip01(nmar, 0, T_NET), 0.40),
                    (_clip01(_get(row, "operating_margin_ttm"), 0, T_OP), 0.25),
                    (_clip01(_get(row, "gross_margin_ttm"), 0, T_GROSS), 0.20),
                    (_clip01(_get(row, "fcf_margin_ttm"), 0, T_FCF), 0.15)])
    d2e, cur = _get(row, "debt_to_equity"), _get(row, "current_ratio")
    q_debt = (1 - _clip01(d2e, 0, T_DE_MAX)) if not pd.isna(d2e) else np.nan
    q_cur = (1 - abs(min(max(cur, 0), 3) - T_CR_SWEET) / T_CR_SWEET) if not pd.isna(cur) else np.nan
    quality = _wavg([(_clip01(roe_eff, 0, T_ROE), 0.35),
                     (_clip01(roic, 0, T_ROIC), 0.25),
                     (_clip01(_get(row, "roa_ttm"), 0, T_ROA), 0.15),
                     (q_debt, 0.15), (q_cur, 0.10)])

    if pd.isna(growth) or pd.isna(profit) or pd.isna(quality):
        base = np.nan
    else:
        base = cfg.W_GROWTH * growth + cfg.W_PROFIT * profit + cfg.W_QUALITY * quality

    # ---- volume confirmation (accumulation) ----
    a_rvol = _clip01(rvol, 0, 2)
    a_align = _clip01(p1m, -10, 10) if not pd.isna(p1m) else 0.5
    if pd.isna(a_rvol):
        accumulation = a_align * 100.0 if not pd.isna(a_align) else np.nan
    else:
        accumulation = _wavg([(a_rvol, 0.5), (a_align, 0.5)])

    if pd.isna(base) or pd.isna(accumulation):
        core = np.nan
    else:
        core = base * (cfg.VOL_MOD_LOW + cfg.VOL_MOD_SPAN * accumulation / 100.0)

    # ---- hard gates ----
    is_etf = bool(row.get("is_etf", False)) if "is_etf" in row else False
    g_profit = (not pd.isna(nmar) and nmar > 0) and (pd.isna(ninc) or ninc > 0)
    g_growth = (not pd.isna(epsg) and epsg > 0) or (not pd.isna(revg) and revg >= cfg.REV_GROWTH_FLOOR)
    g_quality = (not pd.isna(roe_eff) and roe_eff >= cfg.ROE_FLOOR)
    g_liq = (not pd.isna(turnover) and turnover >= cfg.MIN_TURNOVER_NAIRA)
    g_data = (not pd.isna(base)) and (not pd.isna(roe_eff))
    is_distribution = (not pd.isna(p1m) and p1m < cfg.DIST_MAX_1M) and \
                      (not pd.isna(rvol) and rvol > cfg.DIST_MIN_RVOL)
    g_momo = not is_distribution

    gate_fails = [n for n, ok in [
        ("profit", g_profit), ("growth", g_growth), ("quality", g_quality),
        ("liquidity", g_liq), ("data", g_data), ("distribution", g_momo),
    ] if not ok]
    if is_etf:
        gate_fails.append("etf")
    eligible = not gate_fails

    return dict(
        symbol=sym, sector=_get(row, "sector"), price=price, turnover=turnover,
        growth=growth, profit=profit, quality=quality, accumulation=accumulation,
        base=base, core_score=core, eligible=eligible, gate_fails=",".join(gate_fails),
        roe_eff=roe_eff, net_margin=nmar, eps_growth=epsg, rev_growth=revg,
        rvol=rvol, perf_1m=p1m,
    )


def compute_core_scores(df: pd.DataFrame,
                        cfg: Optional[type] = None) -> pd.DataFrame:
    """Score every row of a snapshot. Returns a table sorted by core_score desc.

    Eligible (gate-passing) rows sort to the top; ineligible rows keep their
    score for the consolidation view but never enter the core.
    """
    cfg = cfg or CoreStrategyConfig
    if "symbol" not in df.columns:
        raise KeyError("snapshot is missing the 'symbol' column")
    rows = [_score_row(r, cfg) for _, r in df.iterrows()]
    out = pd.DataFrame(rows, columns=OUTPUT_COLS)
    return out.sort_values(["eligible", "core_score"], ascending=[False, False],
                           na_position="last").reset_index(drop=True)


def eligible_ranked(scores: pd.DataFrame) -> pd.DataFrame:
    """Just the core-eligible names, best first."""
    return scores[scores["eligible"]].sort_values("core_score", ascending=False).reset_index(drop=True)
