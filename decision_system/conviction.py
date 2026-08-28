"""
conviction.py — fuse all signals into ONE explainable 0-100 conviction score.

This is the decision system's core. It extends the transparent weighted-scoring
design of analysis/growth.py:

  1. weighted sum of the present signals, with weights RE-NORMALIZED over only
     the signals that have data (a missing signal is excluded, never zero-filled);
  2. a bounded macro regime tilt added on top;
  3. confidence shrinkage — a thinly-covered stock is pulled toward neutral (50)
     instead of being penalised;
  4. a holding-aware action label.

Owns the weight set (config ConvictionWeights) and action thresholds
(config ConvictionConfig). No ML — with only ~30 historical snapshots a model
would fit noise; a documented weighted scheme is auditable and calibratable.
"""

from __future__ import annotations

from config.settings import ConvictionWeights, ConvictionConfig
from decision_system.confidence import compute_confidence, confidence_factor
from decision_system.models import ConvictionScore, SignalSet


_SIGNAL_LABELS = {
    "fundamental": "fundamentals",
    "technical": "technicals",
    "quality": "quality",
    "seasonality": "seasonality",
    "disclosure": "disclosures",
    "report_tone": "report tone",
    "sentiment": "news sentiment",
    "market_context": "market context",
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class ConvictionEngine:
    """Turns a SignalSet (+ macro tilt) into a ConvictionScore."""

    def __init__(self, weights: dict | None = None):
        ConvictionWeights.validate()
        self.weights = weights or ConvictionWeights.as_dict()

    def score(self, signal_set: SignalSet, macro_tilt: float = 0.0,
              held: bool = False) -> ConvictionScore:
        """Fuse `signal_set` into a ConvictionScore. `macro_tilt` is the raw
        sector tilt (clamped here); `held` makes the action holding-aware."""
        present = signal_set.present_signals()
        result = ConvictionScore(ticker=signal_set.ticker,
                                 sub_scores=dict(signal_set.sub_scores))

        # No data at all — neutral, zero confidence.
        if not present:
            result.conviction = 50.0
            result.base_score = 50.0
            result.confidence = 0.0
            result.action = "HOLD" if held else "AVOID"
            result.reasons = ["No signal data available — neutral by default."]
            return result

        # 1. Re-normalize weights over only the present signals.
        present_weight = sum(self.weights.get(n, 0.0) for n in present)
        eff = {n: self.weights.get(n, 0.0) / present_weight for n in present}
        result.effective_weights = eff

        # 2. Weighted base score.
        base = sum(eff[n] * signal_set.sub_scores[n] for n in present)
        result.base_score = base

        # 3. Bounded macro tilt.
        cap = ConvictionConfig.MACRO_TILT_CAP
        macro_adj = _clamp(macro_tilt, -cap, cap)
        result.macro_adjustment = macro_adj

        # 4. Confidence shrinkage toward neutral.
        confidence = compute_confidence(signal_set, self.weights)
        result.confidence = confidence
        factor = confidence_factor(confidence)

        pre = _clamp(base + macro_adj, 0.0, 100.0)
        final = 50.0 + (pre - 50.0) * factor
        result.conviction = _clamp(final, 0.0, 100.0)

        # 5. Holding-aware action.
        result.action = self._action(result.conviction, confidence, held)
        result.reasons = self._reasons(signal_set, eff, macro_adj, confidence,
                                       result.action)
        return result

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _action(conviction: float, confidence: float, held: bool) -> str:
        cfg = ConvictionConfig
        if conviction >= cfg.STRONG_BUY:
            # A strong-buy needs enough data behind it.
            if confidence < cfg.STRONG_BUY_MIN_CONFIDENCE:
                return "ADD"
            return "STRONG_BUY"
        if conviction >= cfg.ADD:
            return "ADD"
        if conviction >= cfg.HOLD:
            return "HOLD"
        if conviction >= cfg.TRIM:
            return "TRIM" if held else "AVOID"
        return "SELL" if held else "AVOID"

    def _reasons(self, signal_set: SignalSet, eff: dict, macro_adj: float,
                 confidence: float, action: str) -> list:
        """Human-readable explanation of what drove the score."""
        reasons: list = []

        # Rank present signals by their contribution to the base score.
        contributions = sorted(
            ((n, eff[n] * signal_set.sub_scores[n]) for n in eff),
            key=lambda kv: kv[1], reverse=True,
        )
        if contributions:
            top = contributions[0]
            reasons.append(
                f"Led by {_SIGNAL_LABELS.get(top[0], top[0])} "
                f"({signal_set.sub_scores[top[0]]:.0f}/100).")
        # Flag a notably weak present signal.
        weak = [n for n in eff if signal_set.sub_scores[n] < 35.0]
        if weak:
            reasons.append(
                "Weak: " + ", ".join(_SIGNAL_LABELS.get(n, n) for n in weak) + ".")

        if abs(macro_adj) >= 0.5:
            direction = "tailwind" if macro_adj > 0 else "headwind"
            reasons.append(f"Macro {direction} {macro_adj:+.1f} pts.")

        if confidence < 0.999:
            missing = [n for n in signal_set.sub_scores
                       if n not in eff]
            pct = round(confidence * 100)
            note = f"Confidence {pct}%"
            if missing:
                note += " — missing: " + ", ".join(
                    _SIGNAL_LABELS.get(n, n) for n in missing)
            reasons.append(note + ".")

        if signal_set.errors:
            reasons.append("Signal errors: " + ", ".join(signal_set.errors.keys()) + ".")

        return reasons


def score_signal_set(signal_set: SignalSet, macro_tilt: float = 0.0,
                     held: bool = False) -> ConvictionScore:
    """Convenience one-shot wrapper around ConvictionEngine.score()."""
    return ConvictionEngine().score(signal_set, macro_tilt, held)
