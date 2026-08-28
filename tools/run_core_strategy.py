"""run_core_strategy.py — run the Core Concentrate strategy for a snapshot.

Loads the latest (or a given) snapshot, scores every stock, sizes FRESH capital
into the top-2 core names, builds the advisory consolidation view of the current
book, writes artifacts under outputs/core_strategy/<date>/, and prints a report.

Usage:
    python -m tools.run_core_strategy --capital 2000000
    python -m tools.run_core_strategy --capital 2000000 --date 2026-06-23
    python -m tools.run_core_strategy            # no sizing, shortlist only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import OUTPUT_DIR, CoreStrategyConfig  # noqa: E402
from analysis.snapshot_store import SnapshotStore  # noqa: E402
from analysis.core_strategy import compute_core_scores, eligible_ranked  # noqa: E402
from decision_system.concentrate import size_core  # noqa: E402
from decision_system.consolidate import consolidate, load_holdings  # noqa: E402


def _naira(x) -> str:
    return f"₦{x:,.0f}" if pd.notna(x) else "n/a"


def run(capital: float = 0.0, date: str | None = None,
        cfg: type = CoreStrategyConfig) -> dict:
    store = SnapshotStore()
    date = date or store.get_latest_snapshot_date()
    if not date:
        raise SystemExit("No snapshots found under data/snapshots/.")
    loaded = store.load_snapshot(date)
    if not loaded or "merged" not in loaded:
        raise SystemExit(f"Snapshot for {date} could not be loaded.")
    df = loaded["merged"]

    scores = compute_core_scores(df, cfg)
    elig = eligible_ranked(scores)
    plan = size_core(scores, capital, cfg) if capital > 0 else None
    core_syms = [p.symbol for p in plan.positions] if plan else list(elig["symbol"].head(cfg.N_CORE))

    holdings = load_holdings()
    cons = consolidate(scores, holdings, core_syms)

    out_dir = OUTPUT_DIR / cfg.OUTPUT_SUBDIR / date
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "shortlist.csv", index=False)
    cons.to_csv(out_dir / "consolidation.csv", index=False)
    summary = {
        "date": date,
        "capital": capital,
        "eligible_count": int(len(elig)),
        "core_picks": [
            {"symbol": p.symbol, "sector": p.sector, "core_score": round(p.core_score, 1),
             "weight_pct": round(p.weight_pct, 1), "target_naira": round(p.target_naira),
             "shares": p.shares, "days_to_build": round(p.days_to_build, 2)}
            for p in (plan.positions if plan else [])
        ] or [{"symbol": s} for s in core_syms],
        "flags": plan.flags if plan else [],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print_report(date, capital, elig, plan, cons, core_syms, cfg, out_dir)
    return {"date": date, "scores": scores, "plan": plan, "consolidation": cons, "out_dir": out_dir}


def _print_report(date, capital, elig, plan, cons, core_syms, cfg, out_dir):
    print("=" * 92)
    print(f"  CORE CONCENTRATE STRATEGY — {date}   (quality+profit+growth, volume-confirmed)")
    print("=" * 92)
    print(f"  Eligible (passed all gates): {len(elig)}\n")
    print(f"  {'#':>2} {'Symbol':12} {'Core':>6} {'Grw':>4} {'Prof':>4} {'Qual':>4} {'Acc':>4} "
          f"{'1M%':>6} {'RVol':>5}  Sector")
    for i, r in elig.head(10).iterrows():
        print(f"  {i+1:>2} {r.symbol:12} {r.core_score:6.1f} {r.growth:4.0f} {r.profit:4.0f} "
              f"{r.quality:4.0f} {r.accumulation:4.0f} {r.perf_1m:6.1f} {r.rvol:5.2f}  {str(r.sector)[:26]}")

    print("\n  " + "-" * 88)
    if plan and plan.positions:
        print(f"  CORE SIZING — fresh capital {_naira(capital)} "
              f"(deployable {_naira(plan.deployable)}, core {_naira(plan.core_budget)}, "
              f"punt {_naira(plan.punt_budget)}, cash {_naira(plan.cash_reserve)}):")
        for p in plan.positions:
            print(f"     {p.symbol:12} score {p.core_score:5.1f} -> {p.weight_pct:4.1f}% "
                  f"= {_naira(p.target_naira)}  ({p.shares:,} sh @ {_naira(p.price)})  "
                  f"{'build same-day' if p.days_to_build <= 1 else f'~{p.days_to_build:.1f}d to build'}")
        for f in plan.flags:
            print(f"     ⚠  {f}")
    else:
        print(f"  CORE PICKS (no capital given — shortlist only): {', '.join(core_syms)}")

    print("\n  " + "-" * 88)
    print("  CONSOLIDATION — current holdings under the strategy (advisory):")
    for bucket in ("CORE", "HOLD-ELIGIBLE", "CONSIDER-TRIM"):
        sub = cons[cons["bucket"] == bucket]
        if sub.empty:
            continue
        print(f"    {bucket} ({len(sub)}):")
        for _, r in sub.iterrows():
            why = f"  [{r.gate_fails}]" if r.gate_fails else ""
            sc = f"{r.core_score:5.1f}" if pd.notna(r.core_score) else "  —"
            print(f"       {r.symbol:12} score {sc}  mv {_naira(r.market_value)}{why}")
    print(f"\n  Artifacts -> {out_dir}")
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser(description="Run the Core Concentrate strategy.")
    ap.add_argument("--capital", type=float, default=0.0,
                    help="Fresh capital to deploy (naira). 0 = shortlist only.")
    ap.add_argument("--date", help="Snapshot date YYYY-MM-DD (default: latest).")
    args = ap.parse_args()
    run(args.capital, args.date)


if __name__ == "__main__":
    main()
