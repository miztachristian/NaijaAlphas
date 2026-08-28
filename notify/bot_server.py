"""
notify/bot_server.py — Two-way Telegram bot. Polling mode, no webhook needed.
All Ollama calls run in executor to avoid blocking the async event loop.
run_fresh_analysis is admin-only.
"""
from __future__ import annotations

import asyncio
import html
import logging
import os
from collections import defaultdict
from functools import partial

from dotenv import load_dotenv

# override=True: the .env file is the single source of truth for bot config
# (model, tokens). Without it, a stale ambient OLLAMA_MODEL in the launching
# shell would silently shadow the .env value.
load_dotenv(override=True)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CommandHandler, ContextTypes,
    MessageHandler, filters,
)

from notify import registry
from notify.agent import StockAgent
from notify.formatter import build_brief
from notify.sender import send_telegram
from notify.tools import (
    get_available_cash, get_gamble_punts, get_momentum_picks, get_macro_state,
    get_orders, get_portfolio_summary, get_seasonality, get_sized_buys,
    get_stock_detail, get_stock_snapshot_detail, load_user_holdings,
    load_user_settings, save_user_holdings, save_user_settings,
    run_fresh_analysis,
)

_history: dict[str, list] = defaultdict(list)
_agent = StockAgent()


def _uid(update: Update) -> str:
    return str(update.effective_user.id)


async def _check_registered(update: Update) -> bool:
    uid = _uid(update)
    if not registry.is_registered(uid):
        await update.message.reply_text(
            "You're not registered. Send /start to request access."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = _uid(update)
    user = update.effective_user
    name = user.full_name or "Friend"
    username = f"@{user.username}" if user.username else name
    result = registry.register(uid, name, username)
    admin_id = os.getenv("TELEGRAM_ADMIN_ID", os.getenv("TELEGRAM_CHAT_ID", ""))

    if result == "already_registered":
        await update.message.reply_text(
            f"Welcome back, {name}! 📊\n\n"
            "/brief — today's full brief\n"
            "/portfolio — holdings & P&L\n"
            "/orders — buy/sell list\n"
            "/cash AMOUNT — set available cash for a sized buy plan\n"
            "/gems — top momentum picks\n"
            "/seasonality — monthly edges\n"
            "/punt — speculative plays\n"
            "/why TICKER — explain conviction\n"
            "/add TICKER SHARES AVGCOST — add position\n"
            "/remove TICKER — remove position\n"
            "Or ask me anything in plain English."
        )
    elif result == "registered":
        await update.message.reply_text(f"✅ Welcome, {name}! Your account is active.")
    else:
        await update.message.reply_text(
            f"👋 Hi {name}! Access request sent. You'll hear back soon."
        )
        if admin_id:
            send_telegram(
                f"🔔 New access request\n{name} {username}\nID: {uid}\n"
                f"Approve: /approve {uid}",
                chat_id=admin_id,
            )


async def cmd_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not registry.is_admin(_uid(update)):
        await update.message.reply_text("⛔ Admin only.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /approve USER_ID")
        return
    target = context.args[0]
    ok = registry.approve(target)
    if ok:
        u = registry.get_user(target)
        await update.message.reply_text(f"✅ Approved {u.get('name','?')} ({target})")
        send_telegram("✅ Your NGX bot access is approved! Send /start to begin.", chat_id=target)
    else:
        await update.message.reply_text(f"❌ User {target} not found.")


async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not registry.is_admin(_uid(update)):
        await update.message.reply_text("⛔ Admin only.")
        return
    users = registry.list_users()
    if not users:
        await update.message.reply_text("No users registered.")
        return
    lines = ["<b>Registered users</b>"]
    for u in users:
        status = "✅" if u.get("approved") else "⏳"
        tag = " (admin)" if u.get("is_admin") else ""
        lines.append(f"{status} {u['name']} {u['username']}{tag} · {u['user_id']}")
    await update.message.reply_html("\n".join(lines))


async def cmd_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_registered(update):
        return
    await update.message.reply_chat_action(ChatAction.TYPING)
    brief = build_brief(_uid(update))
    await update.message.reply_html(brief)


async def cmd_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_registered(update):
        return
    uid = _uid(update)
    s = get_portfolio_summary(uid)
    if not s["positions"] and not s.get("unpriced"):
        await update.message.reply_text(
            "No positions. Use /add TICKER SHARES AVGCOST to build your portfolio."
        )
        return
    pnl_sign = "+" if s["total_pnl"] >= 0 else ""
    lines = [
        f"<b>Portfolio — {s['snapshot_date']}</b>",
        f"Value ₦{s['total_value']:,.0f}  P&amp;L {pnl_sign}₦{s['total_pnl']:,.0f} ({pnl_sign}{s['total_pnl_pct']:.1f}%)",
        "",
    ]
    for p in s["positions"]:
        emoji = "🟢" if p["pnl"] >= 0 else "🔴"
        conv = f"{p['conviction']:.0f}" if p["conviction"] is not None else "—"
        lines.append(
            f"{emoji} <code>{p['ticker']:<10}</code> {p['action']:<12} "
            f"conv {conv}  {p['pnl_pct']:+.1f}%"
        )
    if s.get("unpriced"):
        lines.append(f"\n<i>Unpriced ({s['n_unpriced']}): " +
                     ", ".join(p["ticker"] for p in s["unpriced"]) + "</i>")
    await update.message.reply_html("\n".join(lines))


async def cmd_orders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_registered(update):
        return
    uid = _uid(update)
    orders = get_orders()
    holdings = load_user_holdings(uid).get("holdings", {})
    held = set(holdings.keys())
    if held and not registry.is_admin(uid):
        orders = [o for o in orders if o.get("ticker") in held]
    sized = get_sized_buys(uid)
    if not orders and not sized:
        await update.message.reply_text("No orders today (or none matching your positions).")
        return
    lines = ["<b>Today's Orders</b>"]
    sells = [o for o in orders if o.get("side") in ("SELL", "TRIM")]
    buys = [o for o in orders if o.get("side") in ("BUY", "ADD")]
    for o in sells:
        lines.append(
            f"🔴 <code>{o['ticker']}</code> {o['side']} {int(o.get('shares',0))} "
            f"→ ₦{o.get('naira',0):,.0f}"
        )
    if buys:
        lines.append("\n<i>Buy signals (verify cash first):</i>")
        for o in buys:
            lines.append(f"🟢 <code>{o['ticker']}</code> {o['side']}")
    if sized:
        cash = get_available_cash(uid)
        lines.append(f"\n<b>💰 Sized buy plan</b> (deploying ₦{cash:,.0f}):")
        for b in sized:
            lines.append(
                f"   <code>{b['ticker']}</code> {b['shares']} sh @ ₦{b['price']:,.0f} "
                f"= ₦{b['naira']:,.0f}"
            )
    await update.message.reply_html("\n".join(lines))


async def cmd_cash(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set / view available cash for the sized buy plan: /cash AMOUNT"""
    if not await _check_registered(update):
        return
    uid = _uid(update)
    args = context.args or []
    if not args:
        cash = get_available_cash(uid)
        if cash > 0:
            await update.message.reply_text(
                f"Available cash: ₦{cash:,.0f}\n"
                "Update with /cash AMOUNT, or clear with /cash 0."
            )
        else:
            await update.message.reply_text(
                "No available cash set.\nUse /cash AMOUNT  e.g. /cash 500000"
            )
        return
    raw = args[0].replace(",", "").replace("₦", "")
    try:
        amount = float(raw)
    except ValueError:
        await update.message.reply_text("AMOUNT must be a number. e.g. /cash 500000")
        return
    if amount < 0:
        await update.message.reply_text("AMOUNT cannot be negative.")
        return
    settings = load_user_settings(uid)
    settings["available_cash"] = amount
    save_user_settings(uid, settings)
    if amount > 0:
        await update.message.reply_text(
            f"✅ Available cash set to ₦{amount:,.0f}.\n"
            "/orders and /brief now show a conviction-weighted sized buy plan."
        )
    else:
        await update.message.reply_text("✅ Available cash cleared. Buy plan hidden.")


async def cmd_gems(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_registered(update):
        return
    picks = get_momentum_picks(_uid(update), top_n=5)
    if not picks:
        await update.message.reply_text("No momentum data yet. Run daily_ingest.py first.")
        return
    lines = ["<b>📈 Top Momentum Picks</b>  (not in your portfolio)"]
    for p in picks:
        score = f"{float(p.get('score',0)):.1f}"
        lines.append(f"  <code>{p.get('ticker','?')}</code>  {score}  {p.get('sector','')}")
    await update.message.reply_html("\n".join(lines))


async def cmd_seasonality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_registered(update):
        return
    import calendar
    from datetime import date
    month_name = calendar.month_name[date.today().month]
    rows = get_seasonality()
    if not rows:
        await update.message.reply_text(f"No seasonality data for {month_name}.")
        return
    sorted_rows = sorted(rows, key=lambda r: float(r.get("avg_return_pct", 0)), reverse=True)
    lines = [f"<b>📅 {month_name} Seasonality</b>"]
    for r in sorted_rows[:10]:
        ret = float(r.get("avg_return_pct", 0))
        emoji = "📈" if ret > 0 else "📉"
        lines.append(f"{emoji} <code>{r.get('ticker','?'):<10}</code> {ret:+.1f}%")
    await update.message.reply_html("\n".join(lines))


async def cmd_punt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_registered(update):
        return
    cards = get_gamble_punts(top_n=3)
    if not cards:
        await update.message.reply_text("No punt data available.")
        return
    lines = ["<b>🎰 Gamble Punts</b>  (speculative / high-risk)"]
    for c in cards:
        t = c.get("ticker", "?")
        score = c.get("score", 0)
        tier = c.get("tier", "")
        state = c.get("state", "")
        catalyst = str(c.get("catalyst", ""))[:80]
        bz = c.get("buy_zone", [])
        bz_str = f"  zone ₦{bz[0]:.2f}–{bz[1]:.2f}" if len(bz) == 2 else ""
        lines.append(f"  <code>{t}</code>  {score:.0f}  {tier}/{state}{bz_str}")
        if catalyst and catalyst != "no scheduled catalyst":
            lines.append(f"  <i>↳ {catalyst}</i>")
    await update.message.reply_html("\n".join(lines))


def _perf_bar(pct: float, peak: float, max_blocks: int = 12) -> str:
    """Bar scaled so this stock's largest absolute return fills max_blocks.

    Sub-one-block (but non-zero) magnitudes render as a thin ▏.
    """
    if not peak:
        return ""
    n = round(abs(float(pct)) / peak * max_blocks)
    return "█" * min(n, max_blocks) if n > 0 else "▏"


def _perf_line(label: str, pct, peak: float) -> str:
    if pct is None:
        return f"  {label:<9}  {'N/A':>7}"
    pct = float(pct)
    arrow = "▲" if pct >= 0 else "▼"
    signed = f"{pct:+.1f}%"
    return f"  {label:<9}{arrow} {signed:>7}  {_perf_bar(pct, peak)}"


def _fmt(val, spec: str = "") -> str:
    """Format a value, returning 'N/A' when it is missing."""
    if val is None:
        return "N/A"
    try:
        return format(float(val), spec) if spec else str(val)
    except (TypeError, ValueError):
        return str(val)


def _build_why(ticker: str, detail: dict, snap: dict) -> str:
    # --- header (from decision_table) ---
    conf = detail.get("confidence")
    conf_str = f"{float(conf) * 100:.0f}%" if conf not in (None, "") else "?"
    head = [
        f"🔍 <b>{ticker}</b>",
        f"Action: <b>{detail.get('action', '?')}</b>   "
        f"Conviction: {_fmt(detail.get('conviction'), '.0f')}",
        f"Confidence: {conf_str}",
    ]

    # --- monospace stock-details block (from snapshot rankings) ---
    cov = snap.get("coverage")
    cov_str = f"{float(cov) * 100:.0f}%" if cov is not None else "N/A"
    # Bars scale relative to this stock's own largest absolute return.
    perfs = [snap.get(k) for k in ("perf_1m", "perf_3m", "perf_6m", "perf_1y")]
    peak = max((abs(float(p)) for p in perfs if p is not None), default=0.0)
    block = [
        f"STOCK DETAILS: {ticker}",
        "=" * 34,
        f"Sector:    {_fmt(snap.get('sector'))}",
        f"Price:     {_fmt(snap.get('price'), '.1f') if snap.get('price') is not None else 'N/A'}",
        f"Coverage:  {cov_str}",
        "",
        "Performance:",
        _perf_line("1 Month", snap.get("perf_1m"), peak),
        _perf_line("3 Month", snap.get("perf_3m"), peak),
        _perf_line("6 Month", snap.get("perf_6m"), peak),
        _perf_line("1 Year", snap.get("perf_1y"), peak),
        "",
    ]
    if "agg_rank" in snap:
        block.append(f"Aggressive Rank:  #{snap['agg_rank']} "
                     f"(Score {_fmt(snap.get('agg_score'), '.1f')})")
    else:
        block.append("Aggressive Rank:  Did not pass gates")
    if "guard_rank" in snap:
        block.append(f"Guardrails Rank:  #{snap['guard_rank']} "
                     f"(Score {_fmt(snap.get('guard_score'), '.1f')})")
    else:
        block.append("Guardrails Rank:  Did not pass gates")

    held = detail.get("held")
    shares = detail.get("current_shares")
    if held not in (None, "", False, 0, "0"):
        try:
            held_str = f"yes  ({int(float(shares)):,} shares)" if shares else "yes"
        except (TypeError, ValueError):
            held_str = "yes"
    else:
        held_str = "no"
    block += ["", f"Held: {held_str}"]

    parts = ["\n".join(head),
             "<pre>" + html.escape("\n".join(block)) + "</pre>"]

    reasons = str(detail.get("reasons", "")).strip()
    if reasons:
        parts.append(f"<i>{html.escape(reasons[:400])}</i>")
    return "\n\n".join(parts)


async def cmd_why(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_registered(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /why TICKER  e.g. /why MTNN")
        return
    ticker = context.args[0].upper()
    detail = get_stock_detail(ticker)
    snap = get_stock_snapshot_detail(ticker)
    if not detail and not snap:
        await update.message.reply_text(f"No data for {ticker} in latest run.")
        return
    await update.message.reply_html(_build_why(ticker, detail, snap))


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_registered(update):
        return
    args = context.args or []
    if len(args) < 3:
        await update.message.reply_text("Usage: /add TICKER SHARES AVGCOST\nExample: /add MTNN 500 819.00")
        return
    ticker = args[0].upper()
    try:
        shares = float(args[1])
        avg_cost = float(args[2])
    except ValueError:
        await update.message.reply_text("SHARES and AVGCOST must be numbers.")
        return
    uid = _uid(update)
    data = load_user_holdings(uid)
    holdings = data.get("holdings", {})
    holdings[ticker] = {"shares": shares, "avg_cost": avg_cost}
    data["holdings"] = holdings
    if save_user_holdings(uid, data):
        await update.message.reply_text(
            f"✅ {ticker}: {shares:.0f} shares @ ₦{avg_cost:,.2f}\n/portfolio to see snapshot."
        )
    else:
        await update.message.reply_text("❌ Save failed. Check bot logs.")


async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _check_registered(update):
        return
    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: /remove TICKER")
        return
    ticker = args[0].upper()
    uid = _uid(update)
    data = load_user_holdings(uid)
    holdings = data.get("holdings", {})
    if ticker not in holdings:
        await update.message.reply_text(f"{ticker} not in your portfolio.")
        return
    del holdings[ticker]
    data["holdings"] = holdings
    save_user_holdings(uid, data)
    await update.message.reply_text(f"✅ {ticker} removed.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Free-text → AI agent. Ollama runs in thread pool to avoid blocking."""
    if not await _check_registered(update):
        return
    uid = _uid(update)
    question = update.message.text or ""

    # Admin-only guard for fresh analysis trigger via plain text
    if any(kw in question.lower() for kw in ("run analysis", "fresh analysis", "run fresh")):
        if not registry.is_admin(uid):
            await update.message.reply_text(
                "⛔ Running a fresh analysis is admin-only.\n"
                "Use /brief to see the latest completed run."
            )
            return

    await update.message.reply_chat_action(ChatAction.TYPING)
    history = _history[uid]

    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None,
        partial(_agent.ask, question, uid, history),
    )

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    _history[uid] = history[-12:]

    await update.message.reply_text(answer)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set.")

    app = Application.builder().token(token).build()

    for name, handler in [
        ("start", cmd_start), ("approve", cmd_approve), ("users", cmd_users),
        ("brief", cmd_brief), ("portfolio", cmd_portfolio), ("orders", cmd_orders),
        ("cash", cmd_cash), ("gems", cmd_gems), ("seasonality", cmd_seasonality),
        ("punt", cmd_punt), ("why", cmd_why), ("add", cmd_add), ("remove", cmd_remove),
    ]:
        app.add_handler(CommandHandler(name, handler))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("NGX bot starting (polling)…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
