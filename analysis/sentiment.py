"""
Headline Sentiment Classifier
==============================
Keyword-based sentiment scoring for Nigerian financial news headlines.
Zero external ML dependencies — fast, deterministic, and interpretable.

Returns one of: POSITIVE, NEGATIVE, NEUTRAL with a confidence score.

Usage:
    from analysis.sentiment import classify_headline, classify_batch
    result = classify_headline("Wema Bank posts 118% profit growth in Q3")
    # => {"sentiment": "POSITIVE", "score": 0.8, "triggers": ["profit", "growth"]}
"""

import re
from typing import Optional

# ------------------------------------------------------------------
# Keyword dictionaries — tuned for Nigerian financial headlines
# ------------------------------------------------------------------

# (word/phrase, weight) — higher weight = stronger signal
POSITIVE_KEYWORDS: list[tuple[str, float]] = [
    # Earnings / Revenue
    ("profit growth", 1.0),
    ("profit surges", 1.0),
    ("record profit", 1.0),
    ("revenue growth", 0.9),
    ("revenue surges", 0.9),
    ("earnings growth", 0.9),
    ("earnings beat", 0.9),
    ("impressive result", 0.8),
    ("strong result", 0.8),
    ("outperform", 0.7),
    ("beats estimate", 0.8),
    ("exceeds expectation", 0.8),
    ("robust growth", 0.8),
    ("double-digit growth", 0.7),
    ("triple-digit growth", 0.9),
    # Dividends
    ("declares dividend", 0.7),
    ("dividend increase", 0.8),
    ("final dividend", 0.6),
    ("interim dividend", 0.6),
    ("bonus issue", 0.6),
    # Expansion / Positive corporate
    ("expansion", 0.5),
    ("new facility", 0.5),
    ("acquisition", 0.4),
    ("partnership", 0.4),
    ("strategic investment", 0.5),
    ("capacity expansion", 0.6),
    ("market share", 0.5),
    ("recapitalization", 0.5),
    ("capital raise", 0.4),
    # Upgrades
    ("upgrade", 0.7),
    ("buy rating", 0.8),
    ("target price raised", 0.7),
    ("positive outlook", 0.7),
    ("bullish", 0.6),
    ("rally", 0.5),
    ("hits new high", 0.7),
    ("all-time high", 0.7),
    ("breakout", 0.5),
    # General positive
    ("growth", 0.3),
    ("profit", 0.3),
    ("gains", 0.3),
    ("rises", 0.3),
    ("surges", 0.5),
    ("jumps", 0.4),
    ("soars", 0.5),
    ("improves", 0.4),
    ("recovery", 0.4),
    ("turnaround", 0.6),
    ("milestone", 0.4),
]

NEGATIVE_KEYWORDS: list[tuple[str, float]] = [
    # Losses / Decline
    ("profit decline", 1.0),
    ("profit drops", 1.0),
    ("revenue decline", 0.9),
    ("revenue drops", 0.9),
    ("loss after tax", 1.0),
    ("net loss", 1.0),
    ("operating loss", 0.9),
    ("earnings miss", 0.9),
    ("misses estimate", 0.8),
    ("disappointing result", 0.8),
    ("weak result", 0.7),
    ("underperform", 0.7),
    # Negative corporate
    ("fraud", 1.0),
    ("scandal", 1.0),
    ("investigation", 0.7),
    ("lawsuit", 0.6),
    ("regulatory action", 0.7),
    ("cbn sanction", 0.9),
    ("sec sanction", 0.9),
    ("fined", 0.7),
    ("penalty", 0.6),
    ("suspension", 0.7),
    ("delisting", 1.0),
    ("default", 0.9),
    ("debt crisis", 0.9),
    ("insolvency", 1.0),
    ("liquidation", 1.0),
    # Downgrades
    ("downgrade", 0.8),
    ("sell rating", 0.8),
    ("target price cut", 0.7),
    ("negative outlook", 0.7),
    ("bearish", 0.6),
    # Management issues
    ("ceo resign", 0.7),
    ("ceo fired", 0.8),
    ("board crisis", 0.8),
    ("management shake", 0.6),
    ("audit concern", 0.7),
    ("qualified opinion", 0.8),
    # General negative
    ("decline", 0.3),
    ("drops", 0.3),
    ("falls", 0.3),
    ("crash", 0.6),
    ("plunges", 0.6),
    ("slumps", 0.5),
    ("tumbles", 0.5),
    ("shrinks", 0.4),
    ("warning", 0.4),
    ("risk", 0.2),
    ("concern", 0.3),
    ("cuts dividend", 0.8),
    ("scraps dividend", 0.9),
    ("no dividend", 0.5),
]

# Negation words that flip the next keyword's sentiment
NEGATION_WORDS = {"not", "no", "never", "despite", "without", "fails to", "unlikely"}

NEGATION_WINDOW = 3  # How many words ahead a negation affects


# ------------------------------------------------------------------
# Core classifier
# ------------------------------------------------------------------

def classify_headline(headline: str) -> dict:
    """
    Classify a single headline.

    Returns:
        {
            "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
            "score": float between -1.0 and 1.0,
            "triggers": list of matched keywords
        }
    """
    if not headline or not headline.strip():
        return {"sentiment": "NEUTRAL", "score": 0.0, "triggers": []}

    text = headline.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation
    words = text.split()

    # Find negation positions
    negation_positions = set()
    for i, w in enumerate(words):
        if w in NEGATION_WORDS:
            for j in range(i + 1, min(i + NEGATION_WINDOW + 1, len(words))):
                negation_positions.add(j)

    pos_score = 0.0
    neg_score = 0.0
    triggers = []

    # Score positive keywords
    for keyword, weight in POSITIVE_KEYWORDS:
        if keyword in text:
            # Check if keyword starts in a negated position
            kw_start = text.find(keyword)
            word_index = len(text[:kw_start].split()) - 1
            if word_index in negation_positions:
                neg_score += weight * 0.5  # Negated positive → weak negative
                triggers.append(f"NOT:{keyword}")
            else:
                pos_score += weight
                triggers.append(keyword)

    # Score negative keywords
    for keyword, weight in NEGATIVE_KEYWORDS:
        if keyword in text:
            kw_start = text.find(keyword)
            word_index = len(text[:kw_start].split()) - 1
            if word_index in negation_positions:
                pos_score += weight * 0.5  # Negated negative → weak positive
                triggers.append(f"NOT:{keyword}")
            else:
                neg_score += weight
                triggers.append(keyword)

    # Calculate net score
    total = pos_score + neg_score
    if total == 0:
        return {"sentiment": "NEUTRAL", "score": 0.0, "triggers": []}

    net_score = (pos_score - neg_score) / max(total, 1.0)
    net_score = max(-1.0, min(1.0, net_score))  # Clamp

    # Classify
    if net_score >= 0.15:
        sentiment = "POSITIVE"
    elif net_score <= -0.15:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    return {
        "sentiment": sentiment,
        "score": round(net_score, 3),
        "triggers": triggers,
    }


def classify_batch(headlines: list[str]) -> list[dict]:
    """Classify a list of headlines."""
    return [classify_headline(h) for h in headlines]


def aggregate_sentiment(articles: list[dict]) -> dict:
    """
    Given a list of articles (each with a "title" key),
    return an aggregate sentiment summary.

    Returns:
        {
            "overall": "POSITIVE" | "NEGATIVE" | "NEUTRAL" | "NO_NEWS",
            "avg_score": float,
            "positive_count": int,
            "negative_count": int,
            "neutral_count": int,
            "total": int,
            "details": [ { headline, sentiment, score, triggers }, ... ]
        }
    """
    if not articles:
        return {
            "overall": "NO_NEWS",
            "avg_score": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "total": 0,
            "details": [],
        }

    details = []
    scores = []
    pos_count = neg_count = neu_count = 0

    for article in articles:
        title = article.get("title", "")
        result = classify_headline(title)
        scores.append(result["score"])
        details.append({**article, **result})

        if result["sentiment"] == "POSITIVE":
            pos_count += 1
        elif result["sentiment"] == "NEGATIVE":
            neg_count += 1
        else:
            neu_count += 1

    avg = sum(scores) / len(scores) if scores else 0.0

    if avg >= 0.15:
        overall = "POSITIVE"
    elif avg <= -0.15:
        overall = "NEGATIVE"
    else:
        overall = "NEUTRAL"

    return {
        "overall": overall,
        "avg_score": round(avg, 3),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "total": len(articles),
        "details": details,
    }


# ------------------------------------------------------------------
# Sentiment flag helper (for integration with screener)
# ------------------------------------------------------------------

SENTIMENT_FLAGS = {
    "POSITIVE": "[+]",
    "NEGATIVE": "[-]",
    "NEUTRAL": "[~]",
    "NO_NEWS": "[ ]",
}


def sentiment_flag(overall: str) -> str:
    """Return a text flag for the overall sentiment."""
    return SENTIMENT_FLAGS.get(overall, "[ ]")


# ------------------------------------------------------------------
# CLI test
# ------------------------------------------------------------------

if __name__ == "__main__":
    test_headlines = [
        "Wema Bank posts 118% profit growth in Q3 2025",
        "Lafarge Africa revenue drops 15% amid rising costs",
        "Custodian Investment declares N2.50 final dividend",
        "CBN sanctions three banks over forex violations",
        "MTN Nigeria earnings beat analyst estimates",
        "Dangote Cement slumps to 52-week low on weak demand",
        "Stanbic IBTC reports robust growth in fee income",
        "NPF Microfinance Bank announces expansion to 12 new states",
        "SEC investigates insider trading at unnamed Nigerian firm",
        "TIP (Initiates PLC) hits all-time high after record profit",
        "Despite profit growth, analysts remain cautious on FCMB",
    ]

    print("=" * 70)
    print("  SENTIMENT CLASSIFIER TEST")
    print("=" * 70)
    for h in test_headlines:
        r = classify_headline(h)
        flag = sentiment_flag(r["sentiment"])
        print(f"\n{flag} [{r['sentiment']:>8}] (score: {r['score']:+.3f})")
        print(f"   \"{h}\"")
        if r["triggers"]:
            print(f"   triggers: {', '.join(r['triggers'])}")
