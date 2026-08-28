"""
notify/tools.py
===============
Data-reading functions for the AI agent and formatter.
All functions are safe: they return empty structures on failure and log warnings.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date as _date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _decisions_dir() -> Path:
    return _root() / "outputs" / "decisions"


def get_latest_run_dir() -> Optional[Path]:
    base = _decisions_dir()
    if not base.exists():
        return None
    dirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
    return dirs[0] if dirs else None


def _latest_file(pattern: str) -> Optional[Path]:
    """Most recently modified file matching a glob under outputs/."""
    matches = sorted(
        (_root() / "outputs").glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _profiles_dir() -> Path:
    from config.settings import NotifyConfig
    p = _root() / NotifyConfig.PROFILES_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Decision outputs (shared across all users)
# ---------------------------------------------------------------------------

def get_decision_table() -> pd.DataFrame:
    run_dir = get_latest_run_dir()
    if run_dir is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(run_dir / "decision_table.csv")
    except Exception as e:
        logger.warning("decision_table.csv: %s", e)
        return pd.DataFrame()


def get_orders() -> list[dict]:
    run_dir = get_latest_run_dir()
    if run_dir is None:
        return []
    try:
        df = pd.read_csv(run_dir / "orders.csv")
        return df.to_dict("records") if not df.empty else []
    except Exception as e:
        logger.warning("orders.csv: %s", e)
        return []


def get_macro_state() -> str:
    run_dir = get_latest_run_dir()
    if run_dir is None:
        return ""
    try:
        return (run_dir / "macro_summary.md").read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("macro_summary.md: %s", e)
        return ""


def get_run_manifest() -> dict:
    run_dir = get_latest_run_dir()
    if run_dir is None:
        return {}
    try:
        return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("run_manifest.json: %s", e)
        return {}


# ---------------------------------------------------------------------------
# User-specific portfolio
# ---------------------------------------------------------------------------

def load_user_holdings(user_id: str) -> dict:
    """
    Load per-user holdings. Falls back to canonical holdings.json for admin.
    Returns {"holdings": {}} on failure.
    """
    profile_path = _profiles_dir() / str(user_id) / "holdings.json"
    if profile_path.exists():
        try:
            return json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("profile holdings %s: %s", user_id, e)

    from notify.registry import is_admin
    if is_admin(user_id):
        canonical = _root() / "data" / "portfolio" / "holdings.json"
        if canonical.exists():
            try:
                return json.loads(canonical.read_text(encoding="utf-8"))
            except Exception:
                pass

    return {"holdings": {}}


def save_user_holdings(user_id: str, holdings_data: dict) -> bool:
    try:
        d = _profiles_dir() / str(user_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "holdings.json").write_text(
            json.dumps(holdings_data, indent=2), encoding="utf-8"
        )
        return True
    except Exception as e:
        logger.error("save holdings %s: %s", user_id, e)
        return False


# ---------------------------------------------------------------------------
# User settings (available cash for the sized buy plan)
# ---------------------------------------------------------------------------

def load_user_settings(user_id: str) -> dict:
    """Load data/profiles/<user_id>/settings.json. Returns {} on failure."""
    path = _profiles_dir() / str(user_id) / "settings.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("settings %s: %s", user_id, e)
    return {}


def save_user_settings(user_id: str, settings: dict) -> bool:
    try:
        d = _profiles_dir() / str(user_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "settings.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8"
        )
        return True
    except Exception as e:
        logger.error("save settings %s: %s", user_id, e)
        return False


def get_available_cash(user_id: str) -> float:
    """User's available cash for deployment, in naira. 0.0 if unset."""
    try:
        return float(load_user_settings(user_id).get("available_cash", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def get_portfolio_summary(user_id: str) -> dict:
    """
    Per-user portfolio with live prices and conviction from decision_table.
    Positions with no price in decision table are flagged, not shown as -100%.
    """
    holdings_data = load_user_holdings(user_id)
    holdings = holdings_data.get("holdings", {})
    dt = get_decision_table()
    manifest = get_run_manifest()

    dt_idx: dict = {}
    if not dt.empty and "ticker" in dt.columns:
        dt_idx = dt.set_index("ticker").to_dict("index")

    positions = []
    unpriced = []

    for ticker, h in holdings.items():
        shares = float(h.get("shares", 0))
        avg_cost = float(h.get("avg_cost", 0))
        row = dt_idx.get(ticker, {})
        raw_price = row.get("price")

        try:
            price = float(raw_price) if raw_price is not None else None
            if price is not None and price <= 0:
                price = None
        except (TypeError, ValueError):
            price = None

        if price is None:
            # Cannot compute reliable P&L — separate list
            unpriced.append({
                "ticker": ticker,
                "shares": shares,
                "avg_cost": avg_cost,
                "action": row.get("action", "—"),
                "conviction": _safe_float(row.get("conviction")),
                "broker": h.get("broker", ""),
                "note": "price unavailable",
            })
            continue

        action = row.get("action", "—")
        conviction = _safe_float(row.get("conviction"))
        current_value = shares * price
        cost_basis = shares * avg_cost
        pnl = current_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0.0

        positions.append({
            "ticker": ticker,
            "shares": shares,
            "avg_cost": avg_cost,
            "price": price,
            "current_value": round(current_value, 2),
            "cost_basis": round(cost_basis, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "action": action,
            "conviction": conviction,
            "broker": h.get("broker", ""),
        })

    positions.sort(key=lambda x: x["current_value"], reverse=True)
    total_value = sum(p["current_value"] for p in positions)
    total_cost = sum(p["cost_basis"] for p in positions)
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0.0

    return {
        "positions": positions,
        "unpriced": unpriced,
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "snapshot_date": manifest.get("snapshot_date", "unknown"),
        "n_positions": len(positions),
        "n_unpriced": len(unpriced),
    }


def get_sized_buys(user_id: str, top_n: int = 6) -> list[dict]:
    """
    Conviction-weighted buy plan sized to the user's available cash.

    Uses STRONG_BUY / ADD candidates from the latest decision_table that the
    user does NOT already hold, with real current prices. Allocates the user's
    available cash across the top names weighted by conviction, then converts
    each allocation into whole shares.

    Returns [] when no cash is set or no candidates qualify.
    """
    cash = get_available_cash(user_id)
    if cash <= 0:
        return []
    dt = get_decision_table()
    if dt.empty or "action" not in dt.columns or "ticker" not in dt.columns:
        return []

    holdings = load_user_holdings(user_id).get("holdings", {})
    held = set(holdings.keys())

    cands = dt[dt["action"].isin(["STRONG_BUY", "ADD"])].copy()
    if cands.empty:
        return []
    cands = cands[~cands["ticker"].isin(held)]
    cands = cands[cands["price"].apply(
        lambda p: _safe_float(p) is not None and float(p) > 0)]
    if cands.empty:
        return []

    cands = cands.sort_values("conviction", ascending=False).head(top_n)
    convs = cands["conviction"].astype(float).clip(lower=1.0)
    wsum = float(convs.sum()) or 1.0

    out: list[dict] = []
    for (_, row), w in zip(cands.iterrows(), convs):
        alloc = cash * (float(w) / wsum)
        price = float(row["price"])
        shares = int(alloc // price)
        if shares <= 0:
            continue
        out.append({
            "ticker": row["ticker"],
            "action": row.get("action", ""),
            "conviction": _safe_float(row.get("conviction")),
            "price": round(price, 2),
            "shares": shares,
            "naira": round(shares * price, 2),
        })
    return out


def get_stock_detail(ticker: str) -> dict:
    dt = get_decision_table()
    if dt.empty or "ticker" not in dt.columns:
        return {}
    row = dt[dt["ticker"] == ticker.upper()]
    return row.iloc[0].to_dict() if not row.empty else {}


def _latest_snapshot_csv() -> Optional[Path]:
    """Newest full-universe snapshot CSV under data/snapshots/<date>/."""
    base = _root() / "data" / "snapshots"
    if not base.exists():
        return None
    for d in sorted((d for d in base.iterdir() if d.is_dir()), reverse=True):
        for name in ("snapshot_merged.csv", "snapshot.csv"):
            p = d / name
            if p.exists():
                return p
    return None


def get_stock_snapshot_detail(ticker: str) -> dict:
    """
    Rich per-stock detail used by the /why command.

    Returns any of: sector, price, coverage (0-1), perf_1m/3m/6m/1y,
    agg_rank/agg_score, guard_rank/guard_score. Missing pieces are simply
    absent from the dict.

    Sources, richest first:
      1. ngx_aggressive_top20_*.csv  — detail fields + aggressive rank/score
      2. ngx_guardrails_top20_*.csv  — guardrails rank/score (+ same detail)
      3. data/snapshots/<latest>/    — full-universe fallback (no coverage/rank)
    """
    ticker = ticker.upper()
    out: dict = {}

    def _row(path: Optional[Path]):
        if path is None:
            return None
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning("snapshot detail %s: %s", path, e)
            return None
        if "symbol" not in df.columns:
            return None
        r = df[df["symbol"] == ticker]
        return r.iloc[0] if not r.empty else None

    agg = _row(_latest_file("ngx_aggressive_top20_*.csv"))
    guard = _row(_latest_file("ngx_guardrails_top20_*.csv"))

    detail_row = agg if agg is not None else guard
    if detail_row is not None:
        for c in ("sector", "price", "coverage_score",
                  "perf_1m", "perf_3m", "perf_6m", "perf_1y"):
            if c in detail_row and pd.notna(detail_row[c]):
                out[c] = detail_row[c]

    # Full-universe fallback for sector/price/perf still missing.
    perf_cols = ("perf_1m", "perf_3m", "perf_6m", "perf_1y")
    if not out or any(c not in out for c in perf_cols):
        snap = _row(_latest_snapshot_csv())
        if snap is not None:
            for c in ("sector", "price") + perf_cols:
                if c not in out and c in snap and pd.notna(snap[c]):
                    out[c] = snap[c]

    if agg is not None and "rank" in agg and pd.notna(agg["rank"]):
        out["agg_rank"] = int(agg["rank"])
        score = agg.get("growth_potential_score_aggressive")
        if pd.notna(score):
            out["agg_score"] = float(score)
    if guard is not None and "rank" in guard and pd.notna(guard["rank"]):
        out["guard_rank"] = int(guard["rank"])
        score = guard.get("growth_potential_score_guardrails")
        if pd.notna(score):
            out["guard_score"] = float(score)

    # Normalise coverage to a 0-1 fraction under the "coverage" key.
    if "coverage_score" in out:
        out["coverage"] = out.pop("coverage_score")

    return out


# ---------------------------------------------------------------------------
# Intelligence sections (shared)
# ---------------------------------------------------------------------------

def get_momentum_picks(user_id: str, top_n: int = 5) -> list[dict]:
    """
    Top stocks from ngx_aggressive_top20_*.csv NOT in user's portfolio.
    Column: symbol, growth_potential_score_aggressive (verified schema).
    """
    path = _latest_file("ngx_aggressive_top20_*.csv")
    if path is None:
        return []
    try:
        df = pd.read_csv(path)
        holdings = load_user_holdings(user_id).get("holdings", {})
        held = set(holdings.keys())
        if "symbol" in df.columns:
            df = df[~df["symbol"].isin(held)]
        score_col = "growth_potential_score_aggressive"
        return df.head(top_n)[["symbol", score_col, "sector", "price"]].rename(
            columns={"symbol": "ticker", score_col: "score"}
        ).to_dict("records")
    except Exception as e:
        logger.warning("momentum picks: %s", e)
        return []


def get_gamble_punts(top_n: int = 2) -> list[dict]:
    """
    Top punt cards from data/punt_runs/*.json sorted by score descending.
    Real schema: ticker, score, tier, state, catalyst, buy_zone, stop.
    """
    punt_dir = _root() / "data" / "punt_runs"
    if not punt_dir.exists():
        return []
    try:
        files = sorted(punt_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return []
        data = json.loads(files[0].read_text(encoding="utf-8"))
        cards = data.get("cards", [])
        # Sort by score descending, filter to SETUP or active states
        active = sorted(
            [c for c in cards if c.get("state") in ("SETUP", "ACTIVE", "BUY")],
            key=lambda c: float(c.get("score", 0)),
            reverse=True,
        )
        if not active:
            active = sorted(cards, key=lambda c: float(c.get("score", 0)), reverse=True)
        return active[:top_n]
    except Exception as e:
        logger.warning("gamble punts: %s", e)
        return []


def get_seasonality(month: Optional[str] = None) -> list[dict]:
    """
    Current month seasonality. avg_return_pct is already a percentage value.
    """
    import calendar
    if month is None:
        month = calendar.month_name[_date.today().month].lower()
    path = _latest_file(f"seasonal_month_{month}_*.csv")
    if path is None:
        return []
    try:
        df = pd.read_csv(path)
        return df.to_dict("records")
    except Exception as e:
        logger.warning("seasonality %s: %s", month, e)
        return []


def run_fresh_analysis() -> str:
    """Trigger daily_ingest.py --skip-ingest. Admin-only; called only via bot."""
    import subprocess, sys
    root = _root()
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(sys.executable)
    try:
        result = subprocess.run(
            [str(python), "daily_ingest.py", "--skip-ingest"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        lines = (result.stdout or "").strip().splitlines()
        summary = "\n".join(lines[-5:]) if lines else "No output."
        return ("✅ Analysis complete:\n" if result.returncode == 0 else "⚠️ Completed with errors:\n") + summary
    except subprocess.TimeoutExpired:
        return "⏳ Analysis timed out after 5 minutes."
    except Exception as e:
        return f"❌ Could not run analysis: {e}"


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return round(f, 1) if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None
