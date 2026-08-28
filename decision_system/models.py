"""
Data models for the decision system.
=====================================
Pure dataclasses — no logic. Every scoring/decision computation lives in the
sibling modules (signals, confidence, conviction, portfolio, macro_regime).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


# Signal names the conviction engine expects. The single source of truth for
# "what is a signal" — signals.py populates these, conviction.py weights them.
SIGNAL_NAMES = (
    "fundamental",
    "technical",
    "quality",
    "seasonality",
    "disclosure",
    "report_tone",
    "sentiment",
    "market_context",
)


def _is_present(value: Optional[float]) -> bool:
    """A sub-score counts as present only if it is a real, finite number."""
    return value is not None and not (isinstance(value, float) and math.isnan(value))


@dataclass
class SignalSet:
    """
    Raw, normalized sub-scores for one stock — each on a 0-100 scale, or NaN
    when the underlying data is missing. `details` keeps the human-readable
    breakdown each analyzer produced, for explainability.
    """

    ticker: str
    sector: str = "OTHER"
    sub_scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, dict] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def present_signals(self) -> List[str]:
        """Signal names that carry real data (used for weight re-normalization)."""
        return [n for n in SIGNAL_NAMES
                if _is_present(self.sub_scores.get(n))]

    def coverage(self) -> float:
        """Fraction of expected signals that have data (0-1)."""
        return len(self.present_signals()) / len(SIGNAL_NAMES)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "sector": self.sector,
            "sub_scores": self.sub_scores,
            "errors": self.errors,
        }


@dataclass
class MacroState:
    """Snapshot of the Nigeria macro environment and the tilts it implies."""

    asof: str = ""
    regime_label: str = "Unknown"

    # Sub-regimes
    rate_cycle: str = "HOLDING"        # CUTTING | HOLDING | HIKING
    naira_trend: str = "STABLE"        # STRENGTHENING | STABLE | WEAKENING
    oil_trend: str = "FLAT"            # RISING | FLAT | FALLING
    yield_trend: str = "STABLE_YIELDS"  # RISING_YIELDS | STABLE_YIELDS | FALLING_YIELDS

    # Raw levels
    mpr: float = float("nan")          # CBN Monetary Policy Rate, %
    inflation: float = float("nan")    # headline CPI YoY, %
    usdngn: float = float("nan")       # USD/NGN spot
    brent: float = float("nan")        # Brent crude, USD/bbl
    bond_10y: float = float("nan")     # 10-Year FGN bond yield, %

    # sector -> additive tilt in conviction points (bounded by MacroConfig)
    sector_tilts: Dict[str, float] = field(default_factory=dict)

    # field name -> {"value", "source", "asof"} provenance
    sources: Dict[str, dict] = field(default_factory=dict)
    degraded: bool = False             # True if any input fell back to a constant

    def to_dict(self) -> dict:
        return {
            "asof": self.asof,
            "regime_label": self.regime_label,
            "rate_cycle": self.rate_cycle,
            "naira_trend": self.naira_trend,
            "oil_trend": self.oil_trend,
            "yield_trend": self.yield_trend,
            "mpr": self.mpr,
            "inflation": self.inflation,
            "usdngn": self.usdngn,
            "brent": self.brent,
            "bond_10y": self.bond_10y,
            "sector_tilts": self.sector_tilts,
            "sources": self.sources,
            "degraded": self.degraded,
        }


@dataclass
class ConvictionScore:
    """The fused, explainable conviction result for one stock."""

    ticker: str
    conviction: float = 0.0            # final 0-100 score
    base_score: float = 0.0            # weighted sum before macro + confidence
    macro_adjustment: float = 0.0      # bounded macro tilt actually applied
    confidence: float = 0.0            # 0-1 data-coverage confidence
    action: str = "HOLD"               # STRONG_BUY | ADD | HOLD | TRIM | SELL | AVOID

    sub_scores: Dict[str, float] = field(default_factory=dict)
    effective_weights: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "conviction": round(self.conviction, 2),
            "base_score": round(self.base_score, 2),
            "macro_adjustment": round(self.macro_adjustment, 2),
            "confidence": round(self.confidence, 3),
            "action": self.action,
            "sub_scores": {k: (round(v, 1) if _is_present(v) else None)
                           for k, v in self.sub_scores.items()},
            "effective_weights": {k: round(v, 3)
                                  for k, v in self.effective_weights.items()},
            "reasons": self.reasons,
        }


@dataclass
class Decision:
    """A ConvictionScore placed in portfolio context (held / sleeve / price)."""

    ticker: str
    sector: str = "OTHER"
    price: float = float("nan")
    score: Optional[ConvictionScore] = None

    held: bool = False
    current_shares: float = 0.0
    current_value: float = 0.0
    sleeve: str = "long_term"          # long_term | short_term

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "sector": self.sector,
            "price": self.price,
            "held": self.held,
            "current_shares": self.current_shares,
            "current_value": round(self.current_value, 2),
            "sleeve": self.sleeve,
            "score": self.score.to_dict() if self.score else None,
        }


@dataclass
class OrderItem:
    """One concrete buy / trim / sell instruction from portfolio construction."""

    ticker: str
    side: str                          # BUY | TRIM | SELL
    shares: int = 0
    naira: float = 0.0
    price: float = float("nan")
    sleeve: str = "long_term"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "side": self.side,
            "shares": self.shares,
            "naira": round(self.naira, 2),
            "price": self.price,
            "sleeve": self.sleeve,
            "reason": self.reason,
        }
