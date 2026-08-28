"""
outputs.py — write the artifacts of a decision run.

For each run, under outputs/decisions/<snapshot_date>/:
  decision_table.csv  one row per stock — conviction, action, every sub-score,
                      confidence and the reason list (the explainability file)
  orders.csv          the concrete buy / trim / sell list
  macro_summary.md    the macro regime and its sector tilts
  run_manifest.json   per-run stats, degraded rows, weight-set version
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import ConvictionWeights, DECISIONS_DIR
from decision_system.models import SIGNAL_NAMES
from decision_system.pipeline import RunResult


def _run_dir(result: RunResult, base: Optional[Path] = None) -> Path:
    base = base or DECISIONS_DIR
    out = Path(base) / result.snapshot_date
    out.mkdir(parents=True, exist_ok=True)
    return out


def decision_table(result: RunResult) -> pd.DataFrame:
    """Flat per-stock decision table (also the source for decision_table.csv)."""
    rows = []
    for d in result.decisions:
        s = d.score
        row = {
            "ticker": d.ticker,
            "sector": d.sector,
            "sleeve": d.sleeve,
            "price": d.price,
            "conviction": round(s.conviction, 2) if s else None,
            "action": s.action if s else None,
            "confidence": round(s.confidence, 3) if s else None,
            "held": d.held,
            "current_shares": d.current_shares,
            "macro_adj": round(s.macro_adjustment, 2) if s else None,
        }
        for name in SIGNAL_NAMES:
            val = s.sub_scores.get(name) if s else None
            row[f"sig_{name}"] = (round(val, 1)
                                  if val is not None and val == val else None)
        row["reasons"] = " ".join(s.reasons) if s else ""
        rows.append(row)
    return pd.DataFrame(rows)


def orders_table(result: RunResult) -> pd.DataFrame:
    """The order list as a DataFrame."""
    return pd.DataFrame([o.to_dict() for o in result.orders])


def macro_summary(result: RunResult) -> str:
    """Markdown summary of the macro regime."""
    m = result.macro_state
    lines = ["# Macro Regime Summary", ""]
    if m is None:
        lines.append("_No macro data ingested._")
        return "\n".join(lines)
    lines += [
        f"**As of:** {m.asof}    **Regime:** {m.regime_label}",
        "",
        f"- Rate cycle: **{m.rate_cycle}**  (MPR {m.mpr:.2f}%)",
        f"- Naira trend: **{m.naira_trend}**  (USD/NGN {m.usdngn:,.2f})",
        f"- Oil trend: **{m.oil_trend}**  (Brent ${m.brent:.2f})",
        f"- Inflation: {m.inflation:.2f}%",
        f"- Degraded inputs: {m.degraded}",
        "",
        "## Active sector tilts (conviction points)",
    ]
    if m.sector_tilts:
        for sector, tilt in sorted(m.sector_tilts.items(),
                                   key=lambda kv: kv[1], reverse=True):
            lines.append(f"- {sector}: {tilt:+.1f}")
    else:
        lines.append("- _none — neutral environment or insufficient history_")
    return "\n".join(lines)


def run_manifest(result: RunResult) -> dict:
    """Per-run metadata: stats, degraded rows, the weight set used."""
    return {
        "snapshot_date": result.snapshot_date,
        "run_at": result.run_at,
        "capital": result.capital,
        "stats": result.stats,
        "issues": result.issues,
        "conviction_weights": ConvictionWeights.as_dict(),
        "macro_regime": (result.macro_state.regime_label
                         if result.macro_state else None),
    }


def write_run(result: RunResult, base: Optional[Path] = None) -> Path:
    """Write all four artifacts; returns the run directory."""
    out = _run_dir(result, base)

    decision_table(result).to_csv(out / "decision_table.csv", index=False)

    orders = orders_table(result)
    if not orders.empty:
        orders.to_csv(out / "orders.csv", index=False)
    else:  # still write a header-only file so the artifact always exists
        pd.DataFrame(columns=["ticker", "side", "shares", "naira",
                              "price", "sleeve", "reason"]
                     ).to_csv(out / "orders.csv", index=False)

    (out / "macro_summary.md").write_text(macro_summary(result), encoding="utf-8")

    with open(out / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(run_manifest(result), fh, indent=2, default=str)

    return out
