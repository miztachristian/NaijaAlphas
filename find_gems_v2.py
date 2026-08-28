"""
Enhanced Gem Screener — with Liquidity & Sentiment
====================================================
Builds on the original find_more_gems.py scoring (Valuation + Growth + Quality = /15)
and adds:
  - Liquidity filter:  volume_1d >= MIN_VOLUME  (+2 pts for strong volume)
  - News sentiment:    aggregate headline sentiment per ticker (+2 / -2 pts)

Total max score: /19

Original find_more_gems.py is untouched — this is the v2 upgrade.

Usage:
    python find_gems_v2.py                  # full run (scrapes news)
    python find_gems_v2.py --skip-news      # skip news fetch (quant-only)
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Fix Windows encoding for emoji output (redirected streams use cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Project imports ──────────────────────────────────────────────
from ingest.news_scraper import NewsScraper
from analysis.sentiment import aggregate_sentiment, sentiment_flag

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────
SNAPSHOT = Path(r"data\snapshots\2026-08-20\snapshot.parquet")
MIN_VOLUME = 500_000        # minimum daily volume (shares, not naira)
STRONG_VOLUME = 2_000_000   # above this = strong liquidity
TOP_N = 15

SKIP_NEWS = "--skip-news" in sys.argv


# ── Helpers ──────────────────────────────────────────────────────
def get_pe(row):
    pe = row.get("pe_ratio", np.nan)
    if pd.isna(pe) or pe == 0:
        eps = row.get("eps_basic_ttm", np.nan)
        price = row.get("price", np.nan)
        if pd.notna(eps) and eps > 0 and pd.notna(price):
            pe = price / eps
    return pe


def safe_fmt(val, fmt=".1f", fallback="N/A"):
    """Safely format a numeric value."""
    if pd.notna(val):
        return f"{val:{fmt}}"
    return fallback


# ── Main ─────────────────────────────────────────────────────────
def run():
    df = pd.read_parquet(SNAPSHOT)
    logger.info("Loaded %d stocks from snapshot", len(df))

    # ── Phase 1: Quantitative scoring (same as original) ─────────
    results = []
    for _, row in df.iterrows():
        ticker = row["symbol"]

        pe = get_pe(row)
        roe = row.get("roe_ttm", np.nan)
        rev_g = row.get("revenue_growth_ttm", np.nan)
        net_inc_g = row.get("net_income_growth_ttm_yoy", np.nan)
        margin = row.get("net_margin_ttm", np.nan)
        debt_eq = row.get("debt_to_equity", np.nan)
        perf_1y = row.get("perf_1y", np.nan)
        volume = row.get("volume_1d", np.nan)
        rel_vol = row.get("relative_volume_1_day", np.nan)
        mkt_cap = row.get("market_cap", np.nan)

        # Core filters — same as find_more_gems.py
        if not (pd.notna(pe) and 0 < pe <= 20):
            continue
        if not (pd.notna(net_inc_g) and net_inc_g > 15):
            continue
        if not (pd.notna(roe) and roe > 15):
            continue

        # Liquidity gate — new filter
        if pd.notna(volume) and volume < MIN_VOLUME:
            continue  # Skip illiquid stocks

        score = 0

        # ── Valuation (max +5) ──
        if pe < 5:
            score += 5
        elif pe < 8:
            score += 3
        elif pe < 12:
            score += 1

        # ── Growth (max +6) ──
        if pd.notna(rev_g):
            if rev_g > 80:
                score += 3
            elif rev_g > 40:
                score += 2
            elif rev_g > 20:
                score += 1

        if net_inc_g > 150:
            score += 3
        elif net_inc_g > 80:
            score += 2
        elif net_inc_g > 30:
            score += 1

        # ── Quality/Profitability (max +4) ──
        if roe > 40:
            score += 3
        elif roe > 25:
            score += 2

        if pd.notna(debt_eq) and debt_eq < 0.5:
            score += 1

        # ── Liquidity bonus (max +2) ──
        liq_bonus = 0
        if pd.notna(volume) and volume >= STRONG_VOLUME:
            liq_bonus += 1
        if pd.notna(rel_vol) and rel_vol > 1.5:
            liq_bonus += 1
        score += liq_bonus

        results.append({
            "ticker": ticker,
            "desc": row.get("description", "N/A"),
            "price": row.get("price", np.nan),
            "pe": pe,
            "roe": roe,
            "rev_g": rev_g,
            "net_inc_g": net_inc_g,
            "margin": margin,
            "debt_eq": debt_eq,
            "perf_1y": perf_1y,
            "volume": volume,
            "rel_vol": rel_vol,
            "mkt_cap": mkt_cap,
            "quant_score": score,
            "liq_bonus": liq_bonus,
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No gems found.")
        return

    logger.info("Found %d stocks passing quantitative filters", len(results_df))

    # ── Phase 2: News sentiment ──────────────────────────────────
    tickers = results_df["ticker"].tolist()
    sentiment_map = {}

    # Build descriptions map for auto-alias fallback
    desc_map = dict(zip(df["symbol"], df["description"])) if "description" in df.columns else {}

    if not SKIP_NEWS:
        try:
            scraper = NewsScraper(cache_hours=6, descriptions=desc_map)
            news = scraper.fetch_all(tickers, max_per_ticker=5)

            for ticker in tickers:
                articles = news.get(ticker, [])
                agg = aggregate_sentiment(articles)
                sent_pts = 0
                if agg["overall"] == "POSITIVE":
                    sent_pts = 2
                elif agg["overall"] == "NEGATIVE":
                    sent_pts = -2
                # NEUTRAL and NO_NEWS = 0

                sentiment_map[ticker] = {
                    "overall": agg["overall"],
                    "avg_score": agg["avg_score"],
                    "total": agg["total"],
                    "sent_pts": sent_pts,
                    "flag": sentiment_flag(agg["overall"]),
                }
        except Exception as exc:
            logger.warning("News fetch failed, scoring without sentiment: %s", exc)
    else:
        logger.info("Skipping news fetch (--skip-news)")

    # Apply sentiment bonus/penalty
    def final_score(row):
        base = row["quant_score"]
        sent = sentiment_map.get(row["ticker"], {}).get("sent_pts", 0)
        return base + sent

    results_df["sent_pts"] = results_df["ticker"].map(
        lambda t: sentiment_map.get(t, {}).get("sent_pts", 0)
    )
    results_df["sent_flag"] = results_df["ticker"].map(
        lambda t: sentiment_map.get(t, {}).get("flag", "[ ]")
    )
    results_df["news_count"] = results_df["ticker"].map(
        lambda t: sentiment_map.get(t, {}).get("total", 0)
    )
    results_df["total_score"] = results_df.apply(final_score, axis=1)

    # Sort by total score desc, then PE asc
    results_df = results_df.sort_values(
        by=["total_score", "pe"], ascending=[False, True]
    ).head(TOP_N)

    # ── Phase 3: Display ─────────────────────────────────────────
    max_possible = 19  # 15 quant + 2 liq + 2 sentiment

    print("\n" + "=" * 70)
    print("  NGX HIDDEN GEMS v2 -- Quant + Liquidity + Sentiment")
    print("  " + f"{'='*66}")
    print(f"  Scoring: Valuation(/5) + Growth(/6) + Quality(/4) + Liq(/2) + News(+/-2) = /{max_possible}")
    print(f"  Liquidity gate: vol >= {MIN_VOLUME:,.0f} shares/day")
    print("=" * 70)

    for rank, (_, r) in enumerate(results_df.iterrows(), 1):
        flag = r["sent_flag"]
        ticker = r["ticker"]
        total = r["total_score"]
        quant = r["quant_score"]
        sent = r["sent_pts"]
        news_n = r["news_count"]

        vol_str = f"{r['volume']:,.0f}" if pd.notna(r["volume"]) else "N/A"
        rvol_str = safe_fmt(r["rel_vol"], ".2f")
        cap_str = f"N{r['mkt_cap']/1e9:.1f}B" if pd.notna(r["mkt_cap"]) else "N/A"

        print(f"\n  #{rank}  {flag} {ticker}  -- TOTAL SCORE: {total}/{max_possible}  (quant:{quant} sent:{sent:+d} news:{news_n})")
        print(f"       PE: {safe_fmt(r['pe'], '.2f')}  |  ROE: {safe_fmt(r['roe'], '.1f')}%  |  Margin: {safe_fmt(r['margin'], '.1f')}%")
        print(f"       Rev Growth: {safe_fmt(r['rev_g'], '.1f')}%  |  NI Growth: {safe_fmt(r['net_inc_g'], '.1f')}%")
        print(f"       D/E: {safe_fmt(r['debt_eq'], '.2f')}  |  1Y Perf: {safe_fmt(r['perf_1y'], '.1f')}%")
        print(f"       Volume: {vol_str}  |  Rel.Vol: {rvol_str}x  |  MktCap: {cap_str}")

    # ── Summary legend ─
    print(f"\n{'-'*70}")
    print("  [+] Positive news  [-] Negative news  [~] Neutral  [ ] No news")
    print(f"{'-'*70}\n")


if __name__ == "__main__":
    run()
