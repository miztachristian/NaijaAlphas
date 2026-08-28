"""
notify/agent.py — Ollama-powered AI analyst agent.
Context-stuffing approach: all portfolio + market data loaded into system prompt.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Static knowledge of how THIS project thinks. Kept in sync with
# config/settings.py (ConvictionConfig/ConvictionWeights), decision_system/
# conviction.py, and docs/hidden_gems_methodology.md. Update here when those
# change so the agent never explains the system incorrectly.
_PROJECT_KNOWLEDGE = """\
=== HOW THIS SYSTEM WORKS (background — always true) ===

PORTFOLIO PHILOSOPHY — two sleeves:
  • Core (~85%): long-term quality growth. Buy strong, profitable, growing
    businesses and hold. This is where conviction scores and STRONG_BUY/ADD
    calls apply.
  • Punt (~15%): short-term seasonal / momentum / catalyst trades ("gamble
    punts"). Speculative, position-sized small, with defined buy zones & stops.
  When advising, keep these buckets separate — never fund a punt from core.

CONVICTION SCORE (0-100): one explainable number per stock, in
decision_system/conviction.py. It is a weighted blend of up to 8 signals,
re-normalised over only the signals that have data (a missing signal is
excluded, never counted as zero), then shrunk toward 50 when data is thin
(low confidence). Signal weights:
  fundamental 30%, technical 20%, quality 15%, disclosure 10%,
  news sentiment 8%, seasonality 7%, report tone 5%, market context 5%.
A bounded macro/sector tilt (±6 pts) nudges but never dominates the score.

ACTION LABELS (from the conviction score; holding-aware):
  • STRONG_BUY  conviction ≥ 78  (downgraded to ADD if confidence < 0.55)
  • ADD         conviction ≥ 64
  • HOLD        conviction ≥ 48
  • TRIM        conviction ≥ 38  → TRIM if held, else AVOID
  • below 38                     → SELL if held, else AVOID
"Confidence" = how much signal coverage backs the score; low confidence pulls
the score toward neutral (50), so a thinly-covered stock is never over-rated.

MOMENTUM PICKS: top names from the aggressive growth screen
(growth_potential_score), shown only if NOT already held — core-sleeve buy ideas.

HIDDEN GEMS: strong fundamentals (EPS or revenue growth >15%, profitable) that
have NOT yet rallied — bought before momentum arrives. Heat status by 1-month
return: ❄️ Cold (<0%, best entry) · 🌤️ Warming (0-10%) · 🔥 Hot (10-20%) ·
🚀 Running (>20%, likely late). Colder + higher score = bigger position.

GAMBLE PUNTS: the punt sleeve. Each card has a score, tier, state
(SETUP/ACTIVE/BUY), a catalyst, a buy zone and a stop. High risk, sized small.

SEASONALITY: avg_return_pct is the historical average return for THIS calendar
month for that ticker (already a percentage). Positive = seasonal tailwind."""

_SYSTEM = """\
You are the analyst assistant for a Nigerian (NGX) stock decision system. \
Use the background below to explain the system correctly, but answer questions \
about specific positions using ONLY the live data provided. Reference specific \
tickers and numbers; never invent prices, scores or holdings. Answer concisely \
(3-6 sentences). Format Naira as ₦1,234. When the two sleeves are relevant, \
keep core and punt advice separate. \
IMPORTANT: only tickers under "## Portfolio" are positions the user actually \
HOLDS. Tickers under "Today's Orders", "Momentum Picks", "Gamble Punts" or any \
buy plan are NOT held — describe them as orders/ideas/candidates, never as \
positions the user owns. If asked to run fresh analysis, output exactly: \
TRIGGER_ANALYSIS

{knowledge}

=== PORTFOLIO & MARKET DATA (live, this user) ===
{context}
=== END ===
"""


def _build_context(user_id: str) -> str:
    from notify.tools import (
        get_gamble_punts, get_macro_state, get_momentum_picks,
        get_orders, get_portfolio_summary, get_seasonality,
    )
    parts = []

    macro = get_macro_state()
    if macro:
        parts.append(f"## Macro\n{macro[:500]}")

    try:
        s = get_portfolio_summary(user_id)
        pnl_sign = "+" if s["total_pnl_pct"] >= 0 else ""
        pos_lines = [
            f"  {p['ticker']}: {p['shares']:.0f}sh @ ₦{p['price']:,.0f} "
            f"| {p['action']} conv={p['conviction']} | P&L {p['pnl_pct']:+.1f}%"
            for p in s["positions"]
        ]
        parts.append(
            f"## Portfolio ({s['n_positions']} positions)\n"
            f"Value ₦{s['total_value']:,.0f} P&L {pnl_sign}{s['total_pnl_pct']:.1f}%\n"
            + "\n".join(pos_lines)
        )
        if s.get("unpriced"):
            parts.append("## Unpriced positions\n" + "\n".join(
                f"  {p['ticker']}: {p['shares']:.0f}sh | {p['action']}"
                for p in s["unpriced"]
            ))
    except Exception as e:
        logger.warning("context portfolio: %s", e)

    orders = get_orders()
    if orders:
        parts.append("## Today's Orders\n" + "\n".join(
            f"  {o['ticker']} {o['side']} {int(o.get('shares',0))} → ₦{o.get('naira',0):,.0f}"
            for o in orders[:10]
        ))

    picks = get_momentum_picks(user_id, top_n=5)
    if picks:
        parts.append("## Top Momentum Picks (not held)\n" + "\n".join(
            f"  {p['ticker']} score={p['score']:.1f} sector={p.get('sector','')}"
            for p in picks
        ))

    seasonal = get_seasonality()
    if seasonal:
        sorted_s = sorted(seasonal, key=lambda r: float(r.get("avg_return_pct", 0)), reverse=True)
        parts.append("## Seasonality (current month)\n" + "\n".join(
            f"  {r.get('ticker','?')} {float(r.get('avg_return_pct',0)):+.1f}%"
            for r in sorted_s[:8]
        ))

    punts = get_gamble_punts(top_n=3)
    if punts:
        parts.append("## Gamble Punts\n" + "\n".join(
            f"  {p['ticker']} score={p.get('score',0):.0f} state={p.get('state','')}"
            for p in punts
        ))

    return "\n\n".join(parts)


class StockAgent:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def _available(self) -> bool:
        import requests as _r
        try:
            return _r.get(f"{self.base_url}/api/tags", timeout=3).status_code == 200
        except Exception:
            return False

    def ask(self, question: str, user_id: str = "0",
            history: Optional[list[dict]] = None) -> str:
        if not self._available():
            return (
                "⚠️ AI agent offline (Ollama not running).\n"
                "Use /portfolio /orders /gems /why TICKER for instant data."
            )
        import requests as _r
        context = _build_context(user_id)
        system = _SYSTEM.format(knowledge=_PROJECT_KNOWLEDGE, context=context)
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": question})
        try:
            resp = _r.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 400},
                    # Keep the model resident between questions so users don't
                    # pay the multi-GB cold-load on every message (big win for
                    # 7B-class models on CPU).
                    "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
                },
                # Generous default: a cold load of a 7B model on CPU can exceed
                # 2 min. Override via OLLAMA_TIMEOUT for faster/slower hardware.
                timeout=int(os.getenv("OLLAMA_TIMEOUT", 300)),
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"].strip()
            if "TRIGGER_ANALYSIS" in answer:
                from notify.tools import run_fresh_analysis
                return "⏳ Running fresh analysis…\n\n" + run_fresh_analysis()
            return answer
        except Exception as exc:
            logger.error("Ollama: %s", exc)
            return f"⚠️ AI agent error: {exc}"
