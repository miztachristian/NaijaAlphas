"""
Annual Report Sentiment & Signal Analyzer
==========================================
Processes the AI-generated summaries scraped from AfricanFinancials
and extracts structured sentiment signals for each company.

Signals extracted:
- Overall management tone (bullish/cautious/neutral)
- Revenue & profit growth mentions (with percentages)
- Dividend actions (increase, maintain, cut, proposed)
- Strategic outlook keywords (expansion, restructuring, risk)
- Governance / compliance flags

Usage:
    from analysis.report_analyzer import ReportAnalyzer
    analyzer = ReportAnalyzer()
    signals = analyzer.analyze_report(report_dict)
    batch   = analyzer.analyze_batch({"DANGCEM": report, "MTNN": report2})
"""

import re
from typing import Optional

# -- Tone keyword dictionaries ----------------------------------------
# Weighted keywords tuned for corporate annual report language.
# These are different from news headline sentiment - reports use
# more formal, measured language.

BULLISH_KEYWORDS: list[tuple[str, float]] = [
    # Strong growth language
    ("record profitability", 1.0),
    ("record-breaking", 1.0),
    ("exceptional performance", 1.0),
    ("outstanding performance", 0.9),
    ("exceptional financial", 0.9),
    ("impressive growth", 0.9),
    ("remarkable increase", 0.9),
    ("significant growth", 0.8),
    ("strong growth", 0.8),
    ("robust growth", 0.8),
    ("substantial growth", 0.8),
    ("landmark period", 0.9),
    ("transformational year", 0.9),
    ("historic milestone", 0.8),
    ("surged", 0.7),
    ("doubled", 0.7),
    ("tripled", 0.9),
    # Revenue / profit signals
    ("revenue increased", 0.6),
    ("profit increased", 0.6),
    ("profit grew", 0.6),
    ("revenue growth", 0.6),
    ("profit growth", 0.6),
    ("earnings growth", 0.6),
    ("ebitda growth", 0.6),
    # Shareholder returns
    ("dividend increase", 0.7),
    ("proposed dividend", 0.5),
    ("bonus issue", 0.6),
    ("share buy-back", 0.6),
    ("share buyback", 0.6),
    # Expansion / strategy
    ("capacity expansion", 0.6),
    ("market leadership", 0.6),
    ("market share", 0.5),
    ("new markets", 0.5),
    ("strategic investment", 0.5),
    ("strategic partnership", 0.5),
    ("future outlook", 0.3),
    ("strong confidence", 0.7),
    ("well-positioned", 0.5),
    ("positive outlook", 0.6),
    # General positive
    ("resilient", 0.4),
    ("sustainable growth", 0.5),
    ("unwavering", 0.5),
    ("strengthening", 0.4),
]

CAUTIOUS_KEYWORDS: list[tuple[str, float]] = [
    # Negative performance
    ("decline", 0.5),
    ("decreased", 0.5),
    ("loss for the year", 0.8),
    ("net loss", 0.9),
    ("operating loss", 0.8),
    ("impairment", 0.6),
    ("write-down", 0.6),
    ("write-off", 0.6),
    ("goodwill impairment", 0.7),
    # Adverse conditions
    ("challenging environment", 0.6),
    ("headwinds", 0.5),
    ("volatility", 0.4),
    ("uncertainty", 0.4),
    ("adverse", 0.5),
    ("inflationary pressure", 0.5),
    ("currency devaluation", 0.6),
    ("naira depreciation", 0.6),
    ("forex loss", 0.7),
    ("foreign exchange loss", 0.7),
    # Restructuring / risk
    ("restructuring", 0.5),
    ("cost-cutting", 0.5),
    ("right-sizing", 0.5),
    ("refinancing", 0.4),
    ("going concern", 0.9),
    ("material uncertainty", 0.8),
    ("qualified opinion", 0.9),
    # Dividend signals
    ("no dividend", 0.7),
    ("dividend omitted", 0.8),
    ("dividend suspended", 0.8),
    ("dividend cut", 0.8),
]


# -- Growth pattern extraction ----------------------------------------

GROWTH_PATTERNS = [
    # -- KPI bullet format (highest priority) --
    # "* Group Revenue Growth:20%"  /  "* Net Profit Growth:102%"
    (r"(?:Group\s+)?Revenue\s+Growth[:\s]+([\d.]+)\s*%", "revenue_growth"),
    (r"(?:Group\s+)?(?:Net\s+)?Profit\s+Growth[:\s]+([\d.]+)\s*%", "net_profit_growth"),
    (r"(?:Group\s+)?(?:PAT|Profit\s+After\s+Tax)\s+Growth[:\s]+([\d.]+)\s*%", "pat_growth"),
    (r"(?:Group\s+)?EBITDA\s+Growth[:\s]+([\d.]+)\s*%", "ebitda_growth"),
    (r"(?:Group\s+)?EPS\s+Growth[:\s]+([\d.]+)\s*%", "eps_growth"),

    # -- Prose format with optional Group prefix & colon --
    # "Group Revenue:Increased by 20% to N4.307 trillion"
    (r"(?:Group\s+)?Revenue[:\s]+(?:increased|grew|rose|up|surged|jumped|soared)\s+(?:by\s+)?(?:an?\s+\w+\s+)?(\d+(?:\.\d+)?)\s*%", "revenue_growth"),
    # "Profit After Tax grew 102%"  /  "Group PAT:Surged by 102%"
    (r"(?:Group\s+)?(?:Profit\s+(?:After|Before)\s+Tax|PAT|PBT)[:\s]+(?:increased|grew|rose|up|surged|jumped|soared)\s+(?:by\s+)?(?:an?\s+\w+\s+)?(\d+(?:\.\d+)?)\s*%", "pat_growth"),
    # "Net profit increased 102%"  /  "Group Net Profit:Surged by 102%"
    (r"(?:Group\s+)?Net\s+[Pp]rofit[:\s]+(?:increased|grew|rose|up|surged|jumped|soared)\s+(?:by\s+)?(?:an?\s+\w+\s+)?(\d+(?:\.\d+)?)\s*%", "net_profit_growth"),
    # "EPS increased 101%"  /  "Earnings Per Share rose by 60%"
    (r"(?:EPS|Earnings\s+[Pp]er\s+[Ss]hare)[:\s]+(?:increased|grew|rose|up|surged)\s+(?:by\s+)?(?:an?\s+\w+\s+)?(\d+(?:\.\d+)?)\s*%", "eps_growth"),

    # -- Qualitative growth (approximate) --
    # "more than doubled" / "more than tripled"
    (r"(?:Group\s+)?(?:Revenue|Net\s+Profit|PAT|Profit)[:\s]+.*?more\s+than\s+doubl(?:ed|ing)", "_qualitative_double"),
    (r"(?:Group\s+)?(?:Revenue|Net\s+Profit|PAT|Profit)[:\s]+.*?more\s+than\s+tripl(?:ed|ing)", "_qualitative_triple"),

    # -- Absolute value extractions --
    # "Earnings Per Share was N59.86"  /  "EPS of N53.07"
    (r"(?:Earnings\s+[Pp]er\s+[Ss]hare|EPS)[:\s]+(?:was|of|at)\s*[NN]?([\d.,]+)", "eps_value"),
    # "EBITDA: N1.981 trillion"  /  "EBITDA of N553.81 billion"
    (r"EBITDA[:\s]+(?:of\s+)?[NN]?([\d.,]+)\s*(billion|trillion|million)?", "ebitda"),
    # "Revenue: N4.307 trillion"  /  "Total Revenue:N5.20 trillion"
    (r"(?:Total\s+)?Revenue[:\s]+(?:of\s+)?[NN]?([\d.,]+)\s*(billion|trillion|million)?", "revenue"),
    # "a dividend of N45.00 per share"
    (r"dividend\s+of\s+[NN]?([\d.,]+)\s*(?:kobo|per\s+share)?", "dividend_proposed"),

    # -- Percentage decreases --
    (r"(?:Group\s+)?Revenue[:\s]+(?:decreased|declined|fell|dropped)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%", "revenue_decline"),
    (r"(?:Group\s+)?(?:Profit\s+(?:After|Before)\s+Tax|PAT|PBT)[:\s]+(?:decreased|declined|fell|dropped)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%", "pat_decline"),
    (r"(?:Group\s+)?Net\s+[Pp]rofit[:\s]+(?:decreased|declined|fell|dropped)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%", "net_profit_decline"),

    # -- Parenthetical KPI format --
    # "Gross Earnings:N2,148.34 billion(81.1% YoY growth)"
    # "Revenue:N4.307 trillion(20% YoY growth)"
    (r"(?:Gross\s+)?(?:Earnings|Revenue)[:\s]+[NN]?[\d.,]+\s*(?:billion|trillion|million)?\s*\((\d+(?:\.\d+)?)\s*%\s*(?:YoY\s+)?(?:growth|increase)\)", "revenue_growth"),
    # "Profit Before Tax:N1,266.24 billion(107.8% YoY growth)"
    (r"(?:Profit\s+(?:Before|After)\s+Tax|PBT|PAT)[:\s]+[NN]?[\d.,]+\s*(?:billion|trillion|million)?\s*\((\d+(?:\.\d+)?)\s*%\s*(?:YoY\s+)?(?:growth|increase)\)", "pat_growth"),
    # "Net Interest Income:N1,058.59 billion(142.4% YoY growth)"
    (r"Net\s+Interest\s+Income[:\s]+[NN]?[\d.,]+\s*(?:billion|trillion|million)?\s*\((\d+(?:\.\d+)?)\s*%\s*(?:YoY\s+)?(?:growth|increase)\)", "net_interest_growth"),
    # "Net Profit:NX billion(Y% increase)"
    (r"Net\s+[Pp]rofit[:\s]+[NN]?[\d.,]+\s*(?:billion|trillion|million)?\s*\((\d+(?:\.\d+)?)\s*%\s*(?:YoY\s+)?(?:growth|increase)\)", "net_profit_growth"),
    # "Total Assets:N14,795.71 billion(52.7% YoY growth)"
    (r"Total\s+Assets[:\s]+[NN]?[\d.,]+\s*(?:billion|trillion|million)?\s*\((\d+(?:\.\d+)?)\s*%\s*(?:YoY\s+)?(?:growth|increase)\)", "total_assets_growth"),
]


# -- Dividend action detection ----------------------------------------

def _detect_dividend_action(text: str, kpis: dict | None = None) -> dict:
    """Classify the dividend action from report text, with KPI fallback.

    Parameters
    ----------
    text : str  - the report summary prose
    kpis : dict - optional KPI dict from the scraper (may contain 'dividend')
    """
    text_lower = text.lower()

    if any(phrase in text_lower for phrase in ["no dividend", "dividend suspended", "dividend omitted"]):
        return {"action": "OMITTED", "confidence": 0.9}

    # Look for proposed dividend amount
    match = re.search(
        r"(?:proposed|recommended|declared)\s+(?:a\s+)?(?:final\s+)?dividend\s+of\s+[NN]?([\d.,]+)\s*(?:kobo|per\s*share)?",
        text,
        re.IGNORECASE,
    )
    if match:
        amount = match.group(1)
        return {"action": "PROPOSED", "amount": amount, "confidence": 0.8}

    # Bonus issue
    if "bonus issue" in text_lower:
        match_bonus = re.search(r"(\w+-for-\w+)\s+bonus", text_lower)
        ratio = match_bonus.group(1) if match_bonus else "unspecified"
        return {"action": "BONUS_ISSUE", "ratio": ratio, "confidence": 0.7}

    # Generic dividend mention
    if "dividend" in text_lower:
        return {"action": "MENTIONED", "confidence": 0.4}

    # KPI fallback: scraper found a dividend figure but prose didn't mention it
    if kpis and kpis.get("dividend"):
        return {"action": "KPI_ONLY", "amount": kpis["dividend"], "confidence": 0.6}

    return {"action": "NOT_MENTIONED", "confidence": 0.5}


# -- Main analyzer class ---------------------------------------------

class ReportAnalyzer:
    """
    Extracts structured signals from AfricanFinancials report summaries.
    """

    def analyze_report(self, report: dict) -> dict:
        """
        Analyze a single report dict (as returned by ReportScraper).

        Parameters
        ----------
        report : dict with at least 'summary' and 'ticker' keys

        Returns
        -------
        dict with:
            ticker, tone, tone_score, growth_metrics, dividend,
            strategic_signals, governance_flags, raw_kpis
        """
        summary = report.get("summary", "")
        ticker = report.get("ticker", "UNKNOWN")

        if not summary:
            return {
                "ticker": ticker,
                "tone": "UNKNOWN",
                "tone_score": 0.0,
                "error": "No summary text available",
            }

        tone, tone_score, tone_triggers = self._score_tone(summary)
        growth = self._extract_growth_metrics(summary)
        kpis = report.get("kpis", {})
        dividend = _detect_dividend_action(summary, kpis=kpis)
        strategic = self._extract_strategic_signals(summary)
        governance = self._extract_governance_flags(summary)

        return {
            "ticker": ticker,
            "doc_type": report.get("doc_type_label", ""),
            "year": report.get("year"),
            "tone": tone,
            "tone_score": round(tone_score, 3),
            "tone_triggers": tone_triggers,
            "growth_metrics": growth,
            "dividend": dividend,
            "strategic_signals": strategic,
            "governance_flags": governance,
            "raw_kpis": report.get("kpis", {}),
            "url": report.get("url", ""),
        }

    def analyze_batch(self, reports: dict[str, dict]) -> dict[str, dict]:
        """
        Analyze a batch of reports.

        Parameters
        ----------
        reports : dict mapping ticker -> report dict

        Returns
        -------
        dict mapping ticker -> analysis result
        """
        results = {}
        for ticker, report in reports.items():
            results[ticker] = self.analyze_report(report)
        return results

    def summarize_batch(self, analyses: dict[str, dict]) -> str:
        """
        Generate a human-readable summary table from batch analysis.
        """
        lines = []
        lines.append(f"{'Ticker':<14} {'Tone':<10} {'Score':>6} {'Rev Grw':>8} {'PAT Grw':>8} {'Dividend':<12}")
        lines.append("-" * 70)

        for ticker, a in sorted(analyses.items()):
            tone = a.get("tone", "?")
            score = a.get("tone_score", 0)
            rev_g = a.get("growth_metrics", {}).get("revenue_growth", "-")
            pat_g = a.get("growth_metrics", {}).get("pat_growth", a.get("growth_metrics", {}).get("net_profit_growth", "-"))
            div_action = a.get("dividend", {}).get("action", "-")

            if rev_g != "-":
                rev_g = f"{rev_g}%"
            if pat_g != "-":
                pat_g = f"{pat_g}%"

            lines.append(
                f"{ticker:<14} {tone:<10} {score:>6.2f} {rev_g:>8} {pat_g:>8} {div_action:<12}"
            )

        return "\n".join(lines)

    # -- Internal scoring ---------------------------------------------

    @staticmethod
    def _score_tone(text: str) -> tuple[str, float, list[str]]:
        """
        Score the overall management tone as BULLISH / CAUTIOUS / NEUTRAL.

        Returns (label, net_score, list_of_trigger_phrases)
        """
        text_lower = text.lower()
        bull_score = 0.0
        bear_score = 0.0
        triggers = []

        for phrase, weight in BULLISH_KEYWORDS:
            if phrase.lower() in text_lower:
                bull_score += weight
                triggers.append(f"+{phrase}")

        for phrase, weight in CAUTIOUS_KEYWORDS:
            if phrase.lower() in text_lower:
                bear_score += weight
                triggers.append(f"-{phrase}")

        net = bull_score - bear_score
        total = bull_score + bear_score

        if total == 0:
            return "NEUTRAL", 0.0, triggers

        # Normalize to [-1, 1] range
        normalized = net / max(total, 1)

        if normalized > 0.2:
            label = "BULLISH"
        elif normalized < -0.2:
            label = "CAUTIOUS"
        else:
            label = "NEUTRAL"

        return label, normalized, triggers

    @staticmethod
    def _extract_growth_metrics(text: str) -> dict[str, str]:
        """Extract growth percentages and financial metrics."""
        metrics: dict[str, str] = {}
        for pattern, key in GROWTH_PATTERNS:
            if key in metrics:          # first match wins per metric
                continue
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Handle qualitative patterns -> approximate %
                if key == "_qualitative_double":
                    metrics["net_profit_growth"] = "~100"
                elif key == "_qualitative_triple":
                    metrics["net_profit_growth"] = "~200"
                else:
                    metrics[key] = match.group(1)
        return metrics

    @staticmethod
    def _extract_strategic_signals(text: str) -> list[str]:
        """Detect strategic keywords and themes."""
        signals = []
        text_lower = text.lower()

        strategic_themes = {
            "expansion": ["capacity expansion", "new markets", "new facility", "geographical expansion"],
            "m_and_a": ["acquisition", "merger", "joint venture"],
            "innovation": ["innovation", "digital transformation", "technology investment", "r&d"],
            "share_buyback": ["share buy-back", "share buyback", "treasury shares"],
            "debt_reduction": ["deleveraging", "debt reduction", "gearing ratio"],
            "esg_focus": ["sustainability", "esg", "carbon", "renewable energy", "environmental"],
            "cost_optimization": ["cost optimization", "operational efficiency", "cost management"],
        }

        for theme, keywords in strategic_themes.items():
            for kw in keywords:
                if kw in text_lower:
                    signals.append(theme)
                    break  # one hit per theme is enough

        return signals

    @staticmethod
    def _extract_governance_flags(text: str) -> list[str]:
        """Flag governance concerns or highlights."""
        flags = []
        text_lower = text.lower()

        concerns = {
            "qualified_audit": ["qualified opinion", "emphasis of matter"],
            "going_concern": ["going concern", "material uncertainty"],
            "regulatory_issue": ["infringement", "fine", "penalty", "sanction"],
            "related_party": ["related party transaction"],
        }
        positives = {
            "clean_audit": ["unqualified opinion", "clean audit", "no infringements"],
            "full_compliance": ["full compliance", "compliant with"],
            "board_diversity": ["diversity", "gender balance", "inclusive workplace"],
        }

        for flag, keywords in concerns.items():
            for kw in keywords:
                if kw in text_lower:
                    flags.append(f"! {flag}")
                    break

        for flag, keywords in positives.items():
            for kw in keywords:
                if kw in text_lower:
                    flags.append(f"v {flag}")
                    break

        return flags


# -- CLI entry point --------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    # Try to load the most recent cached reports
    reports_dir = Path(__file__).parent.parent / "data" / "reports"
    cache_files = sorted(reports_dir.glob("*.json"), reverse=True)

    if not cache_files:
        print("No cached reports found. Run fetch_annual_reports.py first.")
        sys.exit(1)

    latest = cache_files[0]
    print(f"Loading reports from: {latest.name}\n")

    with open(latest, "r", encoding="utf-8") as f:
        reports = json.load(f)

    analyzer = ReportAnalyzer()
    analyses = analyzer.analyze_batch(reports)

    # Print summary table
    print(analyzer.summarize_batch(analyses))
    print()

    # Print detailed analysis for each
    for ticker, result in sorted(analyses.items()):
        print(f"\n{'='*60}")
        print(f"  {ticker} - {result.get('doc_type', '')} {result.get('year', '')}")
        print(f"{'='*60}")
        print(f"  Tone           : {result['tone']} ({result['tone_score']:+.2f})")
        print(f"  Triggers       : {', '.join(result.get('tone_triggers', []))}")
        print(f"  Growth Metrics : {result.get('growth_metrics', {})}")
        print(f"  Dividend       : {result.get('dividend', {})}")
        print(f"  Strategy       : {result.get('strategic_signals', [])}")
        print(f"  Governance     : {result.get('governance_flags', [])}")
        print(f"  KPIs           : {result.get('raw_kpis', {})}")
