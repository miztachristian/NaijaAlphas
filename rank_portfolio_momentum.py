"""Rank portfolio stocks by near-term momentum using the latest snapshot.

This script:
- Extracts portfolio symbols from notebooks/portfolio_tracker.ipynb (MY_PORTFOLIO)
- Loads the latest data/snapshots/*/snapshot.parquet
- Ranks symbols by a weighted momentum score emphasizing 1W and 1M

Run (PowerShell):
  $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe rank_portfolio_momentum.py
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MomentumWeights:
    one_day: float = 0.10
    one_week: float = 0.45
    one_month: float = 0.45


def _strip_python_comments(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines():
        # Remove end-of-line comments, but only when they are actual comments.
        # This portfolio dict uses comments only at end of line.
        line = re.sub(r"\s+#.*$", "", line)
        lines.append(line)
    return "\n".join(lines)


def load_portfolio_symbols_from_notebook(notebook_path: Path) -> list[str]:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "MY_PORTFOLIO" not in src:
            continue

        # Capture `MY_PORTFOLIO = { ... }` including newlines.
        m = re.search(r"MY_PORTFOLIO\s*=\s*(\{.*?\})\s*\n", src, flags=re.S)
        if not m:
            continue

        dict_literal = _strip_python_comments(m.group(1))
        portfolio = ast.literal_eval(dict_literal)
        if isinstance(portfolio, dict) and portfolio:
            return sorted(portfolio.keys())

    raise RuntimeError(f"Could not find MY_PORTFOLIO in {notebook_path}")


def load_latest_snapshot(snapshot_root: Path) -> tuple[str, pd.DataFrame]:
    snapshots = sorted(snapshot_root.glob("*/snapshot.parquet"))
    if not snapshots:
        raise RuntimeError(f"No snapshots found under {snapshot_root}")

    latest = snapshots[-1]
    snapshot_date = latest.parent.name
    df = pd.read_parquet(latest)

    if "volume_1d" in df.columns and "price" in df.columns and "liquidity_value_1d" not in df.columns:
        df["liquidity_value_1d"] = df["price"] * df["volume_1d"]

    return snapshot_date, df


def momentum_score(row: pd.Series, weights: MomentumWeights) -> float:
    def val(col: str) -> float:
        x = row.get(col)
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0.0
        return float(x)

    return (
        weights.one_day * val("price_change_1d")
        + weights.one_week * val("perf_1w")
        + weights.one_month * val("perf_1m")
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    snapshot_root = repo_root / "data" / "snapshots"
    portfolio_nb = repo_root / "notebooks" / "portfolio_tracker.ipynb"

    snapshot_date, df = load_latest_snapshot(snapshot_root)
    symbols = load_portfolio_symbols_from_notebook(portfolio_nb)

    weights = MomentumWeights()

    dfp = df.loc[df["symbol"].isin(symbols)].copy()

    missing = sorted(set(symbols) - set(dfp["symbol"].unique()))
    if missing:
        print(f"⚠️ Not found in snapshot ({len(missing)}): {', '.join(missing)}")

    dfp["momentum_score"] = dfp.apply(lambda r: momentum_score(r, weights), axis=1)

    cols = [
        "symbol",
        "price",
        "price_change_1d",
        "perf_1w",
        "perf_1m",
        "momentum_score",
        "liquidity_value_1d",
    ]
    cols = [c for c in cols if c in dfp.columns]

    ranked = dfp[cols].sort_values(by="momentum_score", ascending=False).reset_index(drop=True)

    print(f"📅 Snapshot: {snapshot_date}")
    print(f"📌 Portfolio symbols found: {len(ranked)} / {len(symbols)}")
    print(f"🧮 Momentum weights: 1D {weights.one_day:.2f}, 1W {weights.one_week:.2f}, 1M {weights.one_month:.2f}")
    print("\n🏁 February Momentum Leaderboard (higher = stronger near-term momentum)\n")

    # Pretty formatting
    def fmt_pct(x: float | None) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "   n/a"
        return f"{x:+6.1f}%"

    def fmt_num(x: float | None) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "   n/a"
        return f"{x:,.2f}"

    for i, row in ranked.head(12).iterrows():
        print(
            f"{i+1:2}. {row['symbol']:<10} "
            f"₦{fmt_num(row.get('price')):>10} | "
            f"1D {fmt_pct(row.get('price_change_1d')):>8} | "
            f"1W {fmt_pct(row.get('perf_1w')):>8} | "
            f"1M {fmt_pct(row.get('perf_1m')):>8} | "
            f"Score {row['momentum_score']:.1f}"
        )

    print("\n📉 Weak momentum (bottom 5)\n")
    for i, row in ranked.tail(5).reset_index(drop=True).iterrows():
        print(
            f"- {row['symbol']:<10} "
            f"₦{fmt_num(row.get('price')):>10} | "
            f"1W {fmt_pct(row.get('perf_1w')):>8} | "
            f"1M {fmt_pct(row.get('perf_1m')):>8} | "
            f"Score {row['momentum_score']:.1f}"
        )


if __name__ == "__main__":
    main()
