"""
daily_ingest.py — the decision_system daily orchestrator.
=========================================================
Runs the whole pipeline end to end, one failure-isolated step at a time:

  1. Macro      — Nigeria macro series (ingest/macro.py)
  2. News       — headline scrape (ingest/news_scraper.py)        [best-effort]
  3. Disclosures— NGX corporate disclosures (ingest/fetch_disclosures.py) [best-effort]
  4. Snapshot   — use today's snapshot if present, else build one from the
                  TradingView scanner API and enrich it
  5. History    — refresh stale OHLCV parquet           [optional, --refresh-history]
  6. Decisions  — run DecisionPipeline + write outputs/decisions/<date>/

One source failing degrades the run (recorded in the manifest); it never
aborts it. Schedule with Windows Task Scheduler via _run_daily.bat.

Usage:
    python daily_ingest.py
    python daily_ingest.py --date 2026-05-22 --capital 1000000
    python daily_ingest.py --refresh-history
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR  # noqa: E402


@dataclass
class StepResult:
    name: str
    status: str   # ok | failed | skipped
    message: str = ""


def _run_module(module: str, args: list[str] | None = None,
                timeout: int = 600) -> tuple[bool, str]:
    """Run `python -m <module>` in a subprocess, isolated from this process."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-m", module] + (args or [])
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, timeout=timeout,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return False, " | ".join(tail) or f"exit {proc.returncode}"
    return True, "ok"


def step_macro() -> StepResult:
    ok, msg = _run_module("ingest.macro", timeout=120)
    return StepResult("macro", "ok" if ok else "failed", msg)


def step_news() -> StepResult:
    ok, msg = _run_module("ingest.news_scraper", timeout=600)
    return StepResult("news", "ok" if ok else "failed", msg)


def step_disclosures() -> StepResult:
    ok, msg = _run_module("ingest.fetch_disclosures", timeout=600)
    return StepResult("disclosures", "ok" if ok else "failed", msg)


def step_snapshot(date: str) -> StepResult:
    """Use today's manual snapshot if it exists; otherwise build from the API."""
    snap_path = DATA_DIR / "snapshots" / date / "snapshot.parquet"
    if snap_path.exists():
        return StepResult("snapshot", "skipped",
                          f"snapshot already present for {date} — using it")

    ok, msg = _run_module("ingest.build_snapshot", ["--date", date], timeout=120)
    if not ok:
        return StepResult("snapshot", "failed", f"base build failed: {msg}")

    # Enrich with the proven scanner-API modules (non-fatal if either fails).
    notes = ["base built from API"]
    for mod in ("ingest.tv_technicals", "ingest.tv_financials"):
        e_ok, e_msg = _run_module(mod, ["--date", date], timeout=300)
        notes.append(f"{mod.split('.')[-1]}: {'ok' if e_ok else e_msg}")
    return StepResult("snapshot", "ok", "; ".join(notes))


def step_history() -> StepResult:
    ok, msg = _run_module("download_bulk_historical", timeout=3600)
    return StepResult("history", "ok" if ok else "failed", msg)


def step_decisions(date: str, capital: float) -> tuple[StepResult, object]:
    """Run the decision pipeline and write the output artifacts."""
    try:
        from decision_system.pipeline import DecisionPipeline
        from decision_system.outputs import write_run

        # Resolve to the actual latest snapshot if the requested date has none.
        result = DecisionPipeline(capital=capital).run(
            date if (DATA_DIR / "snapshots" / date / "snapshot.parquet").exists()
            else None)
        out_dir = write_run(result)
        msg = (f"{result.stats.get('stocks_scored', 0)} scored, "
               f"{len(result.orders)} orders -> {out_dir}")
        return StepResult("decisions", "ok", msg), result
    except Exception as exc:  # noqa: BLE001
        return StepResult("decisions", "failed", str(exc)), None


def step_notebooks() -> StepResult:
    """Execute the curated daily notebooks in place. Best-effort per notebook."""
    try:
        from ingest.run_notebooks import run_notebooks
        results = run_notebooks()
    except Exception as exc:  # noqa: BLE001
        return StepResult("notebooks", "failed", str(exc))

    ok = sum(1 for r in results if r.status == "ok")
    failed = [r.name for r in results if r.status == "failed"]
    missing = [r.name for r in results if r.status == "missing"]
    msg = f"{ok}/{len(results)} ok"
    if failed:
        msg += f"; failed: {', '.join(failed)}"
    if missing:
        msg += f"; missing: {', '.join(missing)}"
    status = "failed" if failed else "ok"
    return StepResult("notebooks", status, msg)


def step_seasonality() -> StepResult:
    """Refresh the bot's seasonal_month_*.csv via run_analysis for this month."""
    month = str(datetime.now().month)  # run_analysis --seasonal-month accepts a number
    ok, msg = _run_module("run_analysis", ["--seasonal-month", month], timeout=600)
    return StepResult("seasonality", "ok" if ok else "failed", msg or f"month {month}")


def _write_report(steps: list[StepResult], date: str) -> None:
    """Write a summary of every step's status to data/pipeline_report.json."""
    report = {
        "date": date,
        "run_at": datetime.now().isoformat(),
        "steps": {s.name: {"status": s.status, "message": s.message} for s in steps},
    }
    path = DATA_DIR / "pipeline_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def step_notify(result) -> StepResult:
    """Broadcast the daily brief to all approved users. Best-effort — a
    delivery failure for one user never aborts the run."""
    import logging
    try:
        from config.settings import NotifyConfig
        if not NotifyConfig.ENABLED:
            return StepResult("notify", "skipped", "disabled in NotifyConfig")
        from notify import registry
        from notify.formatter import build_brief
        from notify.sender import send_telegram

        recipients = [str(u["user_id"]) for u in registry.list_users()
                      if u.get("approved")]
        if not recipients:
            # Registry empty — fall back to the admin chat from .env.
            admin_id = os.getenv("TELEGRAM_ADMIN_ID",
                                 os.getenv("TELEGRAM_CHAT_ID", ""))
            if not admin_id:
                return StepResult("notify", "failed",
                                  "no approved users and no TELEGRAM_CHAT_ID set")
            recipients = [admin_id]

        sent = 0
        for uid in recipients:
            try:
                if send_telegram(build_brief(uid), chat_id=uid):
                    sent += 1
            except Exception as exc:  # noqa: BLE001
                logging.warning("brief to %s failed: %s", uid, exc)
        return StepResult("notify", "ok" if sent else "failed",
                          f"brief sent to {sent}/{len(recipients)} users")
    except Exception as exc:  # noqa: BLE001
        return StepResult("notify", "failed", str(exc))


def main() -> None:
    ap = argparse.ArgumentParser(description="decision_system daily orchestrator.")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--capital", type=float, default=0.0,
                    help="New capital to deploy, naira (default: BacktestConfig).")
    ap.add_argument("--refresh-history", action="store_true",
                    help="Also refresh historical OHLCV (slow, browser-based).")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="Skip macro/news/disclosures/snapshot — decisions only.")
    ap.add_argument("--skip-notify", action="store_true",
                    help="Skip the Telegram brief broadcast.")
    args = ap.parse_args()

    date = args.date or datetime.now().strftime("%Y-%m-%d")
    print("=" * 64)
    print(f"  decision_system daily run — {date}")
    print("=" * 64)

    steps: list[StepResult] = []
    if not args.skip_ingest:
        for fn in (step_macro, step_news, step_disclosures):
            r = fn()
            steps.append(r)
            print(f"  [{r.status:7s}] {r.name:12s} {r.message}")
        r = step_snapshot(date)
        steps.append(r)
        print(f"  [{r.status:7s}] {r.name:12s} {r.message}")

    if args.refresh_history:
        r = step_history()
        steps.append(r)
        print(f"  [{r.status:7s}] {r.name:12s} {r.message}")

    dec_step, result = step_decisions(date, args.capital)
    steps.append(dec_step)
    print(f"  [{dec_step.status:7s}] {dec_step.name:12s} {dec_step.message}")

    nb_step = step_notebooks()
    steps.append(nb_step)
    print(f"  [{nb_step.status:7s}] {nb_step.name:12s} {nb_step.message}")

    seas_step = step_seasonality()
    steps.append(seas_step)
    print(f"  [{seas_step.status:7s}] {seas_step.name:12s} {seas_step.message}")

    # --- Telegram notification (best-effort) — last, so the brief reflects
    #     the freshly regenerated notebook/seasonality feeds. ---
    notify_step = (StepResult("notify", "skipped", "--skip-notify")
                   if args.skip_notify else step_notify(result))
    steps.append(notify_step)
    print(f"  [{notify_step.status:7s}] {notify_step.name:12s} {notify_step.message}")

    _write_report(steps, date)

    print("-" * 64)
    ok = sum(1 for s in steps if s.status == "ok")
    failed = [s.name for s in steps if s.status == "failed"]
    print(f"  Done: {ok}/{len(steps)} steps ok"
          + (f" — failed: {', '.join(failed)}" if failed else ""))
    macro_state = getattr(result, "macro_state", None)
    if macro_state is not None:
        print(f"  Macro regime: {macro_state.regime_label}")

    # Non-zero exit only if the decision run itself failed.
    sys.exit(0 if dec_step.status == "ok" else 1)


if __name__ == "__main__":
    main()
