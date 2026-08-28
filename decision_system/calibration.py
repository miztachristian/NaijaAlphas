"""
calibration.py — validate the conviction signals against realized returns.

Closes the "backtests disconnected from live picks" gap. For every consecutive
pair of stored snapshots it measures how well each point-in-time factor (the
snapshot columns behind each conviction signal) predicted the forward return
to the next snapshot — a Spearman rank correlation per factor, averaged across
all pairs.

This is validation/diagnostics, not a black-box optimizer: it writes a report
(outputs/calibration/<date>_weight_report.md) and a human decides whether to
edit config.ConvictionWeights. Re-run quarterly or after ~8 new snapshots.

Usage:
    python -m decision_system.calibration
    python -m decision_system.calibration --max-pairs 6
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from analysis.snapshot_store import SnapshotStore
from config.settings import OUTPUT_DIR

# Conviction signal -> the point-in-time snapshot column(s) that proxy it.
# "higher is better" unless the factor is listed in _LOWER_IS_BETTER.
FACTOR_COLUMNS: Dict[str, List[str]] = {
    "momentum_3m": ["perf_3m"],
    "momentum_6m": ["perf_6m"],
    "eps_growth": ["eps_growth_ttm"],
    "revenue_growth": ["revenue_growth_ttm"],
    "roe": ["roe_ttm"],
    "net_margin": ["net_margin_ttm"],
    "fcf_margin": ["fcf_margin_ttm"],
    "rsi": ["rsi_14"],
    "pe_ratio": ["pe_ratio"],
}
_LOWER_IS_BETTER = {"pe_ratio", "rsi"}


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _spearman(factor: pd.Series, forward_return: pd.Series) -> Optional[float]:
    """Spearman rank correlation, NaN-safe; None if too few overlapping points.

    Computed as Pearson correlation of the ranks — avoids a scipy dependency
    (pandas' built-in method='spearman' imports scipy.stats)."""
    pair = pd.DataFrame({"f": pd.to_numeric(factor, errors="coerce"),
                         "r": pd.to_numeric(forward_return, errors="coerce")}).dropna()
    if len(pair) < 8:
        return None
    corr = pair["f"].rank().corr(pair["r"].rank())  # pearson on ranks = spearman
    return None if pd.isna(corr) else float(corr)


def evaluate_pair(prior: pd.DataFrame, later: pd.DataFrame) -> Dict[str, float]:
    """Spearman of every factor (in `prior`) vs the forward return to `later`."""
    p = prior.drop_duplicates(subset="symbol").set_index("symbol")
    l = later.drop_duplicates(subset="symbol").set_index("symbol")
    common = p.index.intersection(l.index)
    if len(common) < 8:
        return {}

    p, l = p.loc[common], l.loc[common]
    if "price" not in p.columns or "price" not in l.columns:
        return {}
    p0 = pd.to_numeric(p["price"], errors="coerce")
    p1 = pd.to_numeric(l["price"], errors="coerce")
    fwd = (p1 / p0) - 1.0
    # Invalidate non-positive base prices and infinities (mask -> NaN).
    fwd = fwd.mask(p0 <= 0).replace([float("inf"), float("-inf")], float("nan"))

    out: Dict[str, float] = {}
    for factor, candidates in FACTOR_COLUMNS.items():
        col = _first_present(p, candidates)
        if not col:
            continue
        series = pd.to_numeric(p[col], errors="coerce")
        if factor in _LOWER_IS_BETTER:
            series = -series
        corr = _spearman(series, fwd)
        if corr is not None:
            out[factor] = corr
    return out


def calibrate(max_pairs: Optional[int] = None) -> dict:
    """Run the forward-return validation across all stored snapshot pairs."""
    store = SnapshotStore()
    dates = store.list_snapshots()
    if len(dates) < 2:
        raise ValueError("Need at least 2 snapshots to calibrate.")

    pairs = list(zip(dates[:-1], dates[1:]))
    if max_pairs:
        pairs = pairs[-max_pairs:]

    per_factor: Dict[str, List[float]] = {}
    used_pairs = 0
    for d0, d1 in pairs:
        s0, s1 = store.load_snapshot(d0), store.load_snapshot(d1)
        if not s0 or not s1 or "merged" not in s0 or "merged" not in s1:
            continue
        result = evaluate_pair(s0["merged"], s1["merged"])
        if not result:
            continue
        used_pairs += 1
        for factor, corr in result.items():
            per_factor.setdefault(factor, []).append(corr)

    summary = {}
    for factor, corrs in per_factor.items():
        summary[factor] = {
            "mean_spearman": round(sum(corrs) / len(corrs), 4),
            "n_pairs": len(corrs),
            "positive_rate": round(sum(1 for c in corrs if c > 0) / len(corrs), 2),
        }
    return {"pairs_evaluated": used_pairs, "factors": summary,
            "date_range": f"{pairs[0][0]} -> {pairs[-1][1]}" if pairs else ""}


def write_report(report: dict, out_dir: Optional[Path] = None) -> Path:
    """Write the calibration report as markdown."""
    out_dir = out_dir or (OUTPUT_DIR / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    path = out_dir / f"{today}_weight_report.md"

    factors = sorted(report["factors"].items(),
                     key=lambda kv: kv[1]["mean_spearman"], reverse=True)
    lines = [
        "# Conviction Calibration Report",
        "",
        f"Generated: {today}    Snapshot pairs evaluated: {report['pairs_evaluated']}",
        f"Range: {report['date_range']}",
        "",
        "Spearman rank correlation between each point-in-time factor and the",
        "forward return to the next snapshot. Higher = more predictive.",
        "",
        "| Factor | Mean Spearman | Positive rate | Pairs |",
        "|---|---:|---:|---:|",
    ]
    for factor, stats in factors:
        lines.append(f"| {factor} | {stats['mean_spearman']:+.4f} "
                     f"| {stats['positive_rate']:.0%} | {stats['n_pairs']} |")
    lines += [
        "",
        "## How to read this",
        "",
        "- Factors with a positive mean Spearman *and* a high positive rate",
        "  predicted returns consistently — they justify their conviction weight.",
        "- Factors near zero or negative are noise over this window — consider",
        "  trimming the corresponding weight in `config.ConvictionWeights`.",
        "- This is a short history (~30 snapshots); treat it as a sanity check,",
        "  not a precise optimizer. Re-run after ~8 new snapshots.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate conviction signals.")
    ap.add_argument("--max-pairs", type=int,
                    help="Only use the most recent N snapshot pairs.")
    args = ap.parse_args()

    report = calibrate(args.max_pairs)
    path = write_report(report)

    print(f"Pairs evaluated: {report['pairs_evaluated']}  ({report['date_range']})")
    print(f"{'Factor':16s} {'Mean Spearman':>14s} {'Pos rate':>9s}")
    for factor, stats in sorted(report["factors"].items(),
                                key=lambda kv: kv[1]["mean_spearman"], reverse=True):
        print(f"  {factor:14s} {stats['mean_spearman']:+13.4f} "
              f"{stats['positive_rate']:>8.0%}")
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
