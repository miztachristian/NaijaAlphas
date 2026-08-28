"""
Notebook Helpers — Report & News Integration
=============================================
Shared functions used by all 4 analysis notebooks to load annual report
data and news sentiment without duplicating import/loading logic.

Usage (in any notebook):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))
    from notebooks.nb_helpers import *
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from analysis.report_analyzer import ReportAnalyzer
from analysis.disclosure_analyzer import DisclosureAnalyzer, compute_disclosure_adjustment
from analysis.sentiment import aggregate_sentiment, sentiment_flag
from ingest.news_scraper import NewsScraper


# ═══════════════════════════════════════════════════════════════════════
# EMOJI MAPS — consistent across all notebooks
# ═══════════════════════════════════════════════════════════════════════

SIGNAL_EMOJI = {
    "expansion": "🏗️ expansion",
    "m_and_a": "🤝 M&A",
    "innovation": "💡 innovation",
    "share_buyback": "📈 buyback",
    "debt_reduction": "💳 debt reduction",
    "esg_focus": "🌱 ESG",
    "cost_optimization": "✂️ cost opt.",
}

DIVIDEND_DISPLAY = {
    "PROPOSED": "💰 PROPOSED",
    "BONUS_ISSUE": "🎁 BONUS ISSUE",
    "OMITTED": "❌ OMITTED",
    "MENTIONED": "💬 MENTIONED",
    "KPI_ONLY": "📊 KPI ONLY",
    "NOT_MENTIONED": "—",
}

TONE_EMOJI = {
    "BULLISH": "🟢",
    "CAUTIOUS": "🔴",
    "NEUTRAL": "🟡",
    "UNKNOWN": "⚪",
    "N/A": "⚪",
}


# ═══════════════════════════════════════════════════════════════════════
# REPORT CACHE — load annual report data
# ═══════════════════════════════════════════════════════════════════════

def load_report_cache() -> dict:
    """
    Load and merge the most recent AfricanFinancials + ProShare caches.

    Cache filename conventions:
      • AFF      : data/reports/YYYY-MM-DD.json
      • ProShare : data/reports/proshare-YYYY-MM-DD.json
      • Unified  : data/reports/unified-YYYY-MM-DD.json  (ignored here —
                   we re-merge live so we always pick each source's latest)

    Merge rule: AFF takes priority when both sources have the same
    ticker, because AFF summaries are AI-analysed while ProShare entries
    are metadata + PDF URL only.

    Returns
    -------
    dict : {ticker: report_dict}. Every record carries a "source" key
           ("africanfinancials" or "proshare").
    """
    reports_dir = _PROJECT_ROOT / "data" / "reports"
    if not reports_dir.exists():
        print("⚠️  No data/reports/ directory found.")
        return {}

    all_caches = sorted(reports_dir.glob("*.json"))
    if not all_caches:
        print("⚠️  No report cache files found in data/reports/.")
        return {}

    # Split by prefix. AFF caches are bare dates; ProShare/unified carry prefixes.
    aff_caches, ps_caches = [], []
    for p in all_caches:
        name = p.stem
        if name.startswith("proshare-"):
            ps_caches.append(p)
        elif name.startswith("unified-"):
            continue  # derived cache — skip, we merge live
        else:
            aff_caches.append(p)

    aff_data: dict = {}
    aff_date = "N/A"
    if aff_caches:
        latest = sorted(aff_caches)[-1]
        with open(latest, "r", encoding="utf-8") as f:
            aff_data = json.load(f)
        aff_date = latest.stem
        for rec in aff_data.values():
            rec.setdefault("source", "africanfinancials")

    ps_data: dict = {}
    ps_date = "N/A"
    if ps_caches:
        latest = sorted(ps_caches)[-1]
        with open(latest, "r", encoding="utf-8") as f:
            ps_data = json.load(f)
        ps_date = latest.stem.replace("proshare-", "")
        for rec in ps_data.values():
            rec.setdefault("source", "proshare")

    # AFF overwrites ProShare on key collisions (richer summary)
    merged = {**ps_data, **aff_data}

    aff_count = sum(1 for v in aff_data.values() if v.get("summary"))
    ps_count = len(ps_data)
    print(
        f"📄 Reports: AFF {aff_count} ({aff_date}) + ProShare {ps_count} ({ps_date}) "
        f"=> {len(merged)} merged"
    )
    return merged


# ═══════════════════════════════════════════════════════════════════════
# REPORT INSIGHTS — analyze individual or batch
# ═══════════════════════════════════════════════════════════════════════

_analyzer = ReportAnalyzer()


def get_report_insights(ticker: str, cache: dict) -> dict:
    """
    Analyze a single ticker's annual report from cache.

    Returns
    -------
    dict with keys: ticker, tone, tone_score, report_bonus, strategic_signals,
                    dividend_action, dividend_display, growth_metrics,
                    governance_flags, url, has_report
    """
    raw = cache.get(ticker)
    if not raw or not raw.get("summary"):
        return {
            "ticker": ticker,
            "tone": "N/A",
            "tone_score": 0.0,
            "report_bonus": 0,
            "strategic_signals": [],
            "dividend_action": "NOT_MENTIONED",
            "dividend_display": "—",
            "growth_metrics": {},
            "governance_flags": [],
            "url": "",
            "has_report": False,
        }

    analysis = _analyzer.analyze_report(raw)

    # Compute ±5pt bonus (same logic as growth.py)
    tone_score = analysis.get("tone_score", 0.0)
    if tone_score > 0.2:
        bonus = 5
    elif tone_score < -0.2:
        bonus = -5
    else:
        bonus = 0

    div_info = analysis.get("dividend", {})
    div_action = div_info.get("action", "NOT_MENTIONED")
    div_display = DIVIDEND_DISPLAY.get(div_action, "—")
    if div_action == "PROPOSED" and div_info.get("amount"):
        div_display += f" (₦{div_info['amount']})"
    elif div_action == "BONUS_ISSUE" and div_info.get("ratio"):
        div_display += f" ({div_info['ratio']})"

    return {
        "ticker": ticker,
        "tone": analysis.get("tone", "UNKNOWN"),
        "tone_score": tone_score,
        "report_bonus": bonus,
        "strategic_signals": analysis.get("strategic_signals", []),
        "dividend_action": div_action,
        "dividend_display": div_display,
        "growth_metrics": analysis.get("growth_metrics", {}),
        "governance_flags": analysis.get("governance_flags", []),
        "url": analysis.get("url", ""),
        "has_report": True,
    }


def get_batch_report_insights(tickers: list, cache: dict) -> pd.DataFrame:
    """
    Run report analysis for multiple tickers.

    Returns
    -------
    DataFrame with columns: symbol, tone, tone_score, bonus, strategic,
                            dividend, rev_growth_rpt, pat_growth_rpt, has_report
    """
    rows = []
    for t in tickers:
        ins = get_report_insights(t, cache)
        gm = ins["growth_metrics"]
        rows.append({
            "symbol": t,
            "tone": ins["tone"],
            "tone_score": ins["tone_score"],
            "bonus": ins["report_bonus"],
            "strategic": ", ".join(
                SIGNAL_EMOJI.get(s, s) for s in ins["strategic_signals"]
            ) if ins["strategic_signals"] else "—",
            "dividend": ins["dividend_display"],
            "rev_growth_rpt": gm.get("revenue_growth", "—"),
            "pat_growth_rpt": gm.get(
                "pat_growth", gm.get("net_profit_growth", "—")
            ),
            "has_report": ins["has_report"],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# NEWS — load from cache or fetch live
# ═══════════════════════════════════════════════════════════════════════

def load_or_fetch_news(
    tickers: list,
    cache_first: bool = True,
    descriptions: dict = None,
    max_per_ticker: int = 5,
) -> dict:
    """
    Load news articles, preferring cached data.

    Parameters
    ----------
    tickers : list of ticker symbols
    cache_first : if True (default), use cache if fresh (< 6hrs)
    descriptions : optional {TICKER: "Company Name"} for better Google search
    max_per_ticker : max headlines per ticker

    Returns
    -------
    dict : {TICKER: [{title, source, date, url}, ...]}
    """
    scraper = NewsScraper(
        cache_hours=6 if cache_first else 0,
        descriptions=descriptions,
    )
    articles = scraper.fetch_all(tickers, max_per_ticker=max_per_ticker)

    total = sum(len(v) for v in articles.values())
    with_news = sum(1 for v in articles.values() if v)
    print(f"📰 News: {total} articles for {with_news}/{len(tickers)} tickers", end="")

    # Show cache freshness
    cache_path = scraper._cache_path()
    if cache_path.exists():
        age_hrs = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        print(f" (cache: {age_hrs:.1f}h old)")
    else:
        print(" (live fetch)")

    return articles


# ═══════════════════════════════════════════════════════════════════════
# NEWS SENTIMENT — aggregate per ticker
# ═══════════════════════════════════════════════════════════════════════

def get_news_sentiment(tickers: list, articles: dict) -> pd.DataFrame:
    """
    Compute aggregate sentiment for each ticker.

    Returns
    -------
    DataFrame with columns: symbol, news_flag, news_score, headline_count,
                            latest_headline, sentiment_details
    """
    rows = []
    for t in tickers:
        arts = articles.get(t, [])
        agg = aggregate_sentiment(arts)
        latest = arts[0]["title"][:60] if arts else "—"
        rows.append({
            "symbol": t,
            "news_flag": sentiment_flag(agg["overall"]),
            "news_overall": agg["overall"],
            "news_score": agg["avg_score"],
            "headline_count": agg["total"],
            "pos_count": agg["positive_count"],
            "neg_count": agg["negative_count"],
            "latest_headline": latest,
            "sentiment_details": agg["details"],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# COMBINED CONVICTION — merge report + news signals
# ═══════════════════════════════════════════════════════════════════════

def combined_conviction(report_tone: str, report_score: float,
                        news_overall: str, news_score: float) -> str:
    """
    Merge report tone and news sentiment into a single conviction label.

    Logic:
        Both positive              → STRONG BUY
        Report positive, news ok   → BUY
        Report ok, news positive   → BUY
        Both neutral / mixed       → HOLD
        One negative               → CAUTION
        Both negative              → AVOID
        No data                    → —
    """
    r_pos = report_tone == "BULLISH"
    r_neg = report_tone == "CAUTIOUS"
    r_na = report_tone in ("N/A", "UNKNOWN")
    n_pos = news_overall == "POSITIVE"
    n_neg = news_overall == "NEGATIVE"
    n_na = news_overall in ("NO_NEWS", "")

    if r_na and n_na:
        return "—"
    if r_pos and n_pos:
        return "🟢 STRONG BUY"
    if r_pos and not n_neg:
        return "🟢 BUY"
    if n_pos and not r_neg:
        return "🟢 BUY"
    if r_neg and n_neg:
        return "🔴 AVOID"
    if r_neg or n_neg:
        return "🟠 CAUTION"
    return "🟡 HOLD"


def get_conviction_df(report_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge report and news DataFrames and add conviction column.

    Parameters
    ----------
    report_df : from get_batch_report_insights()
    news_df : from get_news_sentiment()

    Returns
    -------
    Merged DataFrame with all report + news columns + conviction
    """
    merged = report_df.merge(news_df, on="symbol", how="outer")
    merged["conviction"] = merged.apply(
        lambda r: combined_conviction(
            r.get("tone", "N/A"),
            r.get("tone_score", 0.0),
            r.get("news_overall", "NO_NEWS"),
            r.get("news_score", 0.0),
        ),
        axis=1,
    )
    return merged


# ═══════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS — rich printing for notebook cells
# ═══════════════════════════════════════════════════════════════════════

def print_report_block(ticker: str, cache: dict, context: str = "") -> None:
    """
    Print a rich report insight block for a single ticker.

    Parameters
    ----------
    ticker : stock symbol
    cache : report cache dict
    context : optional context string (e.g., "Bought at ₦176")
    """
    ins = get_report_insights(ticker, cache)

    if not ins["has_report"]:
        print(f"  📄 {ticker}: No annual report data available")
        return

    tone_e = TONE_EMOJI.get(ins["tone"], "⚪")
    print(f"\n  {'─'*58}")
    print(f"  📄 {ticker} — Annual Report Analysis")
    if context:
        print(f"     {context}")
    print(f"  {'─'*58}")
    print(f"  Tone       : {tone_e} {ins['tone']} (score: {ins['tone_score']:+.2f}, bonus: {ins['report_bonus']:+d})")
    print(f"  Dividend   : {ins['dividend_display']}")

    # Growth metrics
    gm = ins["growth_metrics"]
    if gm:
        metrics_parts = []
        if gm.get("revenue_growth"):
            metrics_parts.append(f"Rev +{gm['revenue_growth']}%")
        if gm.get("pat_growth"):
            metrics_parts.append(f"PAT +{gm['pat_growth']}%")
        elif gm.get("net_profit_growth"):
            metrics_parts.append(f"Net Profit +{gm['net_profit_growth']}%")
        if gm.get("eps_growth"):
            metrics_parts.append(f"EPS +{gm['eps_growth']}%")
        if gm.get("ebitda_growth"):
            metrics_parts.append(f"EBITDA +{gm['ebitda_growth']}%")
        if metrics_parts:
            print(f"  Growth     : {' | '.join(metrics_parts)}")

    # Strategic signals
    if ins["strategic_signals"]:
        tags = [SIGNAL_EMOJI.get(s, s) for s in ins["strategic_signals"]]
        print(f"  Strategy   : {' · '.join(tags)}")

    # Governance
    if ins["governance_flags"]:
        print(f"  Governance : {', '.join(ins['governance_flags'])}")

    if ins["url"]:
        print(f"  Source     : {ins['url']}")


def print_news_block(ticker: str, articles: dict, max_headlines: int = 3) -> None:
    """
    Print a news sentiment block for a single ticker.
    """
    arts = articles.get(ticker, [])
    agg = aggregate_sentiment(arts)
    flag = sentiment_flag(agg["overall"])

    print(f"\n  {'─'*58}")
    print(f"  📰 {ticker} — News Sentiment")
    print(f"  {'─'*58}")

    if not arts:
        print(f"  No recent news found")
        return

    print(f"  Overall    : {flag} {agg['overall']} (avg score: {agg['avg_score']:+.3f})")
    print(f"  Headlines  : {agg['total']} total — {agg['positive_count']}+ {agg['negative_count']}- {agg['neutral_count']}~")

    # Show top headlines
    for detail in agg["details"][:max_headlines]:
        h_flag = sentiment_flag(detail.get("sentiment", "NEUTRAL"))
        title = detail.get("title", "")[:65]
        source = detail.get("source", "")
        print(f"    {h_flag} {title}")
        if source:
            print(f"       — {source}")


def print_divergence_warnings(report_df: pd.DataFrame, news_df: pd.DataFrame) -> None:
    """
    Print warnings where report and news sentiment diverge.
    """
    merged = report_df.merge(news_df, on="symbol", how="inner")
    divergences = []
    for _, row in merged.iterrows():
        tone = row.get("tone", "N/A")
        news = row.get("news_overall", "NO_NEWS")
        if tone == "CAUTIOUS" and news == "POSITIVE":
            divergences.append(
                f"  ⚠️  {row['symbol']}: CAUTIOUS report but POSITIVE news — verify thesis"
            )
        elif tone == "BULLISH" and news == "NEGATIVE":
            divergences.append(
                f"  ⚠️  {row['symbol']}: BULLISH report but NEGATIVE news — check recent events"
            )

    if divergences:
        print(f"\n{'═'*60}")
        print(f"  ⚠️  DIVERGENCE ALERTS — Report vs News Mismatch")
        print(f"{'═'*60}")
        for d in divergences:
            print(d)
    else:
        print("\n  ✅ No report/news divergences detected")


# ═══════════════════════════════════════════════════════════════════════
# NGX DISCLOSURE INTELLIGENCE — events, forecasts, official fundamentals
# ═══════════════════════════════════════════════════════════════════════

def _load_latest_disclosure_json(pattern: str) -> dict:
    """Load the newest data/disclosures/<pattern> JSON; {} if none."""
    ddir = _PROJECT_ROOT / "data" / "disclosures"
    if not ddir.exists():
        return {}
    files = sorted(ddir.glob(pattern))
    if not files:
        return {}
    try:
        with open(files[-1], "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_batch_disclosure_insights(tickers: list) -> pd.DataFrame:
    """
    NGX disclosure intelligence for a list of tickers.

    Pulls Tier-A event signals + Tier-B forecasts from DisclosureAnalyzer and
    Tier-C official fundamentals from the parsed-statements cache. `score_impact`
    is the capped ±5 adjustment that growth.py folds into growth_score.

    Returns
    -------
    DataFrame with columns: symbol, catalyst, insider_90d, dividend,
        late_filer, days_since_results, forecast, ngx_roe, ngx_rev_growth,
        score_impact, impact_reasons
    """
    analyzer = DisclosureAnalyzer()
    statements = _load_latest_disclosure_json("statements-*.json")
    rows = []
    for t in tickers:
        sig = analyzer.analyze(t)
        impact, reasons = compute_disclosure_adjustment(sig)

        stmt = statements.get(t, {})
        ngx_roe = None
        if (stmt.get("parse_status") in ("ok", "partial")
                and stmt.get("pat") and stmt.get("total_equity")):
            ngx_roe = round(stmt["pat"] / stmt["total_equity"] * 100, 1)

        forecast = "—"
        if sig.get("forecast_pat") is not None:
            period = sig.get("forecast_period", "") or "next Q"
            forecast = f"{period}: PAT ₦{sig['forecast_pat'] / 1e6:.2f}bn"

        rows.append({
            "symbol": t,
            "catalyst": (f"⚡ {sig['catalyst_reason']}"
                         if sig.get("earnings_catalyst") else "—"),
            "insider_90d": sig.get("insider_count_90d", 0) or "—",
            "dividend": "💰 recent" if sig.get("dividend_recent") else "—",
            "late_filer": (f"🔴 {sig['late_filer_reason']}"
                           if sig.get("late_filer") else "—"),
            "days_since_results": sig.get("days_since_results"),
            "forecast": forecast,
            "ngx_roe": ngx_roe,
            "ngx_rev_growth": stmt.get("revenue_growth"),
            "score_impact": impact,
            "impact_reasons": "; ".join(reasons) if reasons else "—",
        })
    return pd.DataFrame(rows)


def print_disclosure_block(ticker: str) -> None:
    """Print a rich NGX disclosure intelligence block for one ticker."""
    analyzer = DisclosureAnalyzer()
    sig = analyzer.analyze(ticker)
    impact, reasons = compute_disclosure_adjustment(sig)

    print(f"\n  {'─'*58}")
    print(f"  🏛️  {ticker} — NGX Disclosure Intelligence")
    print(f"  {'─'*58}")
    print(f"  Catalyst        : "
          f"{sig['catalyst_reason'] if sig.get('earnings_catalyst') else 'none'}")
    print(f"  Insider filings : {sig.get('insider_count_90d', 0)} in last 90d")
    print(f"  Recent dividend : {'yes' if sig.get('dividend_recent') else 'no'}")
    if sig.get("late_filer"):
        print(f"  ⚠️  Late filer   : {sig['late_filer_reason']}")
    dsr = sig.get("days_since_results")
    if dsr is not None:
        print(f"  Last results    : {sig.get('last_results_date', '?')} ({dsr}d ago)")
    if sig.get("forecast_pat") is not None:
        pbt = sig.get("forecast_pbt") or 0
        print(f"  Forecast        : {sig.get('forecast_period', '')} "
              f"PAT ₦{sig['forecast_pat'] / 1e6:.2f}bn  PBT ₦{pbt / 1e6:.2f}bn")
    detail = f"  ({'; '.join(reasons)})" if reasons else ""
    print(f"  Score impact    : {impact:+.1f} pts{detail}")


def data_freshness_banner(report_cache: dict, news_articles: dict) -> None:
    """Print a data freshness indicator using HTML."""
    from IPython.display import HTML, display

    reports_dir = _PROJECT_ROOT / "data" / "reports"
    aff_files, ps_files = [], []
    if reports_dir.exists():
        for p in reports_dir.glob("*.json"):
            name = p.stem
            if name.startswith("proshare-"):
                ps_files.append(p)
            elif name.startswith("unified-"):
                continue
            else:
                aff_files.append(p)
    aff_date = sorted(aff_files)[-1].stem if aff_files else "N/A"
    ps_date = (
        sorted(ps_files)[-1].stem.replace("proshare-", "") if ps_files else "N/A"
    )

    # News age
    news_dir = _PROJECT_ROOT / "data" / "news"
    news_files = sorted(news_dir.glob("*.json"), reverse=True) if news_dir.exists() else []
    if news_files:
        age_hrs = (datetime.now().timestamp() - news_files[0].stat().st_mtime) / 3600
        news_age = f"{age_hrs:.1f}h ago"
    else:
        news_age = "N/A"

    # Per-source counts in the loaded cache
    aff_count = sum(
        1 for v in report_cache.values()
        if v.get("source") == "africanfinancials" and v.get("summary")
    )
    ps_count = sum(1 for v in report_cache.values() if v.get("source") == "proshare")
    news_count = sum(1 for v in news_articles.values() if v)

    html = f"""
    <div style="background: #1e1e1e; padding: 15px; border-radius: 8px; border-left: 5px solid #00E676; margin-bottom: 20px; font-family: Inter, sans-serif; color: #fff; width: fit-content; min-width: 400px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h4 style="margin: 0 0 10px 0; color: #00E676; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">System Data Freshness</h4>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <span><span style="font-size: 1.2em; margin-right: 8px;">📅</span> <b>AFF Reports:</b></span>
            <span style="color: #aaa; margin-left: 20px;">{aff_date} <span style="color: #888; font-size: 0.9em;">({aff_count} tickers)</span></span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <span><span style="font-size: 1.2em; margin-right: 8px;">📑</span> <b>ProShare Reports:</b></span>
            <span style="color: #aaa; margin-left: 20px;">{ps_date} <span style="color: #888; font-size: 0.9em;">({ps_count} tickers)</span></span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span><span style="font-size: 1.2em; margin-right: 8px;">📰</span> <b>News:</b></span>
            <span style="color: #aaa; margin-left: 20px;">{news_age} <span style="color: #888; font-size: 0.9em;">({news_count} coverage)</span></span>
        </div>
    </div>
    """
    display(HTML(html))
