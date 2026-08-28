"""
notify/formatter.py
===================
Builds the Telegram daily brief for a given user.
All strings that may contain user/market data are html.escape()'d.

Design model (Telegram HTML subset):
  • Header + triage line       → answer "what do I do today?" in one glance.
  • ACTIONS                    → the single emphasised <blockquote> (the focal box).
  • PORTFOLIO / MACRO          → light, un-boxed flowing sections.
  • MORE INTEL                 → one collapsed <blockquote expandable> (momentum,
                                 seasonality, speculative) so the brief stays short.

Column alignment: whole rows live inside one <code> span with fixed-width
columns, so monospace columns line up across rows. A single leading status
emoji per row keeps the <code> spans left-aligned with each other.
"""
from __future__ import annotations

import calendar
import html
from datetime import date
from typing import Optional

from notify.sender import escape
from notify.tools import (
    get_available_cash,
    get_decision_table,
    get_gamble_punts,
    get_macro_state,
    get_momentum_picks,
    get_orders,
    get_portfolio_summary,
    get_run_manifest,
    get_seasonality,
    get_sized_buys,
    load_user_holdings,
)

_LIMIT = 3800  # conservative buffer below Telegram's 4096
_TW = 10       # ticker column width (covers ZENITHBANK, TRANSCORP, …)


def _fmt_n(v: float) -> str:
    return f"₦{v:,.0f}"


def _cell(value, width: int, right: bool = False) -> str:
    """Pad a value to a minimum monospace column width (never truncates —
    long NGX tickers like ZENITHBANK keep their full name)."""
    s = str(value)
    return s.rjust(width) if right else s.ljust(width)


def _row(raw: str) -> str:
    """Wrap a pre-padded raw row in a monospace span (escaped for HTML)."""
    return f"<code>{escape(raw)}</code>"


def _user_orders(user_id: str) -> list:
    """Orders visible to this user: admins see all, others see only held tickers."""
    from notify.registry import is_admin
    orders = get_orders()
    if is_admin(user_id):
        return orders
    held = set(load_user_holdings(user_id).get("holdings", {}).keys())
    return [o for o in orders if o.get("ticker") in held]


# ---------------------------------------------------------------------------
# Tier 1 — ACTIONS  (the one emphasised box)
# ---------------------------------------------------------------------------

def _s_actions(user_id: str, max_sell: int = 5, max_buy: int = 5) -> str:
    """Everything the user should *do today*, grouped into one focal box."""
    user_orders = _user_orders(user_id)
    sells = [o for o in user_orders if o.get("side") in ("SELL", "TRIM")][:max_sell]
    buys = [o for o in user_orders if o.get("side") in ("BUY", "ADD")][:max_buy]

    body: list[str] = []

    if sells:
        body.append("🔴 <b>SELL / TRIM NOW</b>")
        for o in sells:
            t = _cell(o.get("ticker", "?"), _TW)
            side = _cell(o.get("side", ""), 4)
            sh = _cell(f"{int(o.get('shares', 0))}sh", 6, right=True)
            naira = _cell(_fmt_n(float(o.get("naira", 0))), 11, right=True)
            reason = escape(str(o.get("reason", ""))[:60])
            line = _row(f"{t} {side} {sh} {naira}")
            body.append(f"{line} <i>{reason}</i>" if reason else line)

    # Cash-sized buy plan (only when /cash AMOUNT is set) takes precedence over
    # raw buy signals — it's the actionable, quantified version of the same idea.
    sized = get_sized_buys(user_id, top_n=max_buy)
    if sized:
        if body:
            body.append("")
        cash = get_available_cash(user_id)
        body.append(f"💸 <b>SIZED BUY PLAN</b>  <i>(deploy {_fmt_n(cash)})</i>")
        for b in sized:
            t = _cell(b["ticker"], _TW)
            sh = _cell(f"{b['shares']}sh", 7, right=True)
            price = _cell(f"@{_fmt_n(b['price'])}", 9, right=True)
            conv = f"{b['conviction']:.0f}⚡" if b["conviction"] is not None else "—"
            body.append(f"{_row(f'{t} {sh} {price}')} {conv}")
    elif buys:
        if body:
            body.append("")
        body.append("🟢 <b>BUY SIGNALS</b>  <i>(verify cash first)</i>")
        for o in buys:
            t = _cell(o.get("ticker", "?"), _TW)
            side = _cell(o.get("side", ""), 4)
            body.append(_row(f"{t} {side}"))

    if not body:
        return ""
    return "<blockquote>" + "\n".join(body) + "</blockquote>"


def _s_watch(user_id: str, max_rows: int = 4) -> str:
    """Held names nearing the ADD threshold — a lighter, un-boxed heads-up."""
    from config.settings import NotifyConfig
    dt = get_decision_table()
    holdings = load_user_holdings(user_id).get("holdings", {})
    if dt.empty or not holdings:
        return ""
    held = set(holdings.keys())
    watch = dt[
        dt["ticker"].isin(held)
        & (dt["action"] == "HOLD")
        & (dt["conviction"] >= NotifyConfig.WATCH_LOWER)
        & (dt["conviction"] < NotifyConfig.WATCH_UPPER)
    ].sort_values("conviction", ascending=False).head(max_rows)
    if watch.empty:
        return ""
    lines = ["👀 <b>WATCHLIST</b>  <i>(nearing ADD @ 64)</i>"]
    for _, row in watch.iterrows():
        t = _cell(row["ticker"], _TW)
        conv = _cell(f"{row['conviction']:.0f}", 3, right=True)
        gap = NotifyConfig.WATCH_UPPER - row["conviction"]
        lines.append(f"{_row(f'{t} conv {conv}')} <i>+{gap:.0f} to go</i>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 2 — PORTFOLIO + MACRO  (light, un-boxed)
# ---------------------------------------------------------------------------

def _s_portfolio(user_id: str) -> str:
    s = get_portfolio_summary(user_id)
    if not s["positions"] and not s.get("unpriced"):
        return "🏦 <b>PORTFOLIO</b>\n<i>No positions yet — use /add to build one.</i>"

    pnl = s["total_pnl"]
    arrow = "🟩" if pnl >= 0 else "🟥"
    sign = "+" if pnl >= 0 else "−"
    lines = [
        f"🏦 <b>PORTFOLIO</b>  <i>{s['n_positions']} active · {s['snapshot_date']}</i>",
        f"{arrow} <b>{_fmt_n(s['total_value'])}</b>  "
        f"{sign}{_fmt_n(abs(pnl))} ({s['total_pnl_pct']:+.1f}%)",
    ]
    if s.get("n_unpriced", 0):
        lines.append(f"<i>+{s['n_unpriced']} awaiting prices</i>")

    for p in s["positions"][:5]:
        emoji = "🟩" if p["pnl"] >= 0 else "🟥"
        t = _cell(p["ticker"], _TW)
        action = _cell(p.get("action", "—"), 4)
        conv = _cell(f"{p['conviction']:.0f}" if p["conviction"] is not None else "—", 3, right=True)
        pct = _cell(f"{p['pnl_pct']:+.1f}%", 7, right=True)
        lines.append(f"{emoji} {_row(f'{t} {action} {conv}  {pct}')}")
    if s["n_positions"] > 5:
        lines.append(f"<i>… +{s['n_positions'] - 5} more · /portfolio for full view</i>")
    return "\n".join(lines)


def _s_macro() -> str:
    raw = get_macro_state()
    if not raw:
        return ""
    lines_in = raw.splitlines()

    def _find(key: str) -> str:
        return next((l for l in lines_in if key in l), "")

    rows = [_find("Regime:"), _find("Rate cycle"), _find("Naira trend"),
            _find("Oil trend"), _find("Inflation")]
    body = "\n".join(
        escape(l.strip().replace("**", "").lstrip("- "))
        for l in rows if l
    )
    return f"🌍 <b>MACRO REGIME</b>\n{body}"


# ---------------------------------------------------------------------------
# Tier 3 — MORE INTEL  (one collapsed expandable block)
# ---------------------------------------------------------------------------

def _intel_momentum(user_id: str, max_rows: int = 3) -> list[str]:
    picks = get_momentum_picks(user_id, top_n=max_rows)
    if not picks:
        return []
    out = ["⚡ <b>Momentum picks</b>"]
    for p in picks:
        t = _cell(p.get("ticker", "?"), _TW)
        try:
            score = _cell(f"{float(p.get('score', 0)):.1f}", 5, right=True)
        except (TypeError, ValueError):
            score = _cell(str(p.get("score", "")), 5, right=True)
        sector = escape(str(p.get("sector", ""))[:15])
        out.append(f"{_row(f'{t} {score}')} <i>{sector}</i>")
    return out


def _intel_seasonality(max_rows: int = 4) -> list[str]:
    rows = get_seasonality()
    if not rows:
        return []
    month = calendar.month_name[date.today().month].upper()
    ranked = sorted(rows, key=lambda r: float(r.get("avg_return_pct", 0)), reverse=True)
    bullish = ranked[:max_rows // 2 + 1]
    bull_tickers = {r.get("ticker") for r in bullish}
    # Worst performers, excluding any name already shown as a tailwind.
    bearish = [r for r in reversed(ranked) if r.get("ticker") not in bull_tickers][:max_rows // 2]
    out = [f"📅 <b>{month} seasonality</b>"]
    if bullish:
        chips = ", ".join(f"<code>{escape(r.get('ticker', '?'))}</code>" for r in bullish)
        out.append(f"🔼 Tailwinds: {chips}")
    if bearish:
        chips = ", ".join(f"<code>{escape(r.get('ticker', '?'))}</code>" for r in reversed(bearish))
        out.append(f"🔽 Headwinds: {chips}")
    return out


def _intel_punts(max_cards: int = 2) -> list[str]:
    cards = get_gamble_punts(top_n=max_cards)
    if not cards:
        return []
    out = ["🎲 <b>Speculative plays</b>"]
    for c in cards:
        t = _cell(c.get("ticker", "?"), _TW)
        score = _cell(f"{c.get('score', 0):.0f}", 3, right=True)
        state = escape(str(c.get("state", "")))
        bz = c.get("buy_zone", [])
        # Penny stocks need decimals; show them only below ₦10.
        zfmt = lambda x: f"{x:.2f}" if x < 10 else f"{x:.0f}"
        bz_str = f" · zone ₦{zfmt(bz[0])}–{zfmt(bz[1])}" if len(bz) == 2 else ""
        out.append(f"{_row(f'{t} {score}')} {state}{bz_str}")
        catalyst = escape(str(c.get("catalyst", ""))[:80])
        if catalyst and catalyst != "no scheduled catalyst":
            out.append(f"   <i>↳ {catalyst}</i>")
    return out


def _s_intel(user_id: str) -> str:
    """Momentum + seasonality + speculative, collapsed into one expandable box."""
    blocks = [
        _intel_momentum(user_id),
        _intel_seasonality(),
        _intel_punts(),
    ]
    body: list[str] = []
    for b in blocks:
        if not b:
            continue
        if body:
            body.append("")
        body.extend(b)
    if not body:
        return ""
    head = "📡 <b>MORE INTEL</b>  <i>(tap to expand)</i>"
    return f"<blockquote expandable>{head}\n" + "\n".join(body) + "</blockquote>"


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

def _triage_line(user_id: str) -> str:
    """One-glance chips: sells, buys, portfolio P&L."""
    uo = _user_orders(user_id)
    n_sell = len([o for o in uo if o.get("side") in ("SELL", "TRIM")])
    n_buy = len([o for o in uo if o.get("side") in ("BUY", "ADD")])

    chips: list[str] = []
    if n_sell:
        chips.append(f"🔴 <b>{n_sell}</b> sell")
    if n_buy:
        chips.append(f"🟢 <b>{n_buy}</b> buy")

    s = get_portfolio_summary(user_id)
    if s["positions"]:
        emoji = "🟩" if s["total_pnl"] >= 0 else "🟥"
        chips.append(f"{emoji} <b>{s['total_pnl_pct']:+.1f}%</b>")

    if not chips:
        return "<i>No actions today — hold steady.</i>"
    return "  ·  ".join(chips)


def build_brief(user_id: str) -> str:
    """
    Build the full daily brief.
    Order: Header → Triage → Actions → Watch → Portfolio → Macro → Intel.
    Earlier sections are never dropped; later ones stop once near the limit.
    """
    day = date.today().strftime("%a %d %b %Y")
    header = [
        "🦅 <b>NAIJA TRADES · NGX INTELLIGENCE</b>",
        f"<i>{day}</i>",
        _triage_line(user_id),
    ]

    # Priority order — first sections are never dropped.
    sections = [
        _s_actions(user_id),
        _s_watch(user_id),
        _s_portfolio(user_id),
        _s_macro(),
        _s_intel(user_id),
    ]

    body_lines = list(header)
    for section in sections:
        if not section:
            continue
        candidate = "\n".join(body_lines) + "\n\n" + section
        if len(candidate) > _LIMIT:
            break
        body_lines.append("")
        body_lines.append(section)

    manifest = get_run_manifest()
    scored = manifest.get("stats", {}).get("stocks_scored", "?")
    body_lines += [
        "",
        f"<blockquote>🤖 {scored} stocks scored · "
        f"<i>/portfolio /orders /cash</i></blockquote>",
    ]
    return "\n".join(body_lines)
