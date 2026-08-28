"""
confidence.py — how much to trust a stock's conviction score.

Confidence is the importance-weighted fraction of signals that actually have
data: missing the fundamental signal (weight 0.30) costs far more confidence
than missing report tone (0.05). The conviction engine then shrinks a
low-confidence score toward neutral (50) rather than penalising it outright —
the structural fix for sparse-data stocks being unfairly marked down.
"""

from __future__ import annotations

from config.settings import ConvictionWeights, ConvictionConfig
from decision_system.models import SignalSet


def compute_confidence(signal_set: SignalSet,
                       weights: dict | None = None) -> float:
    """
    Confidence in [0, 1] — the summed weight of the signals that have data.
    1.0 means every signal fired; 0.30 means only the fundamental signal did.
    """
    weights = weights or ConvictionWeights.as_dict()
    present = signal_set.present_signals()
    confidence = sum(weights.get(name, 0.0) for name in present)
    return max(0.0, min(1.0, confidence))


def confidence_factor(confidence: float) -> float:
    """
    Map confidence (0-1) to the shrinkage factor applied to the score's
    distance from neutral. Scales linearly from CONFIDENCE_FACTOR_FLOOR (no
    data) to 1.0 (full coverage), so even a thin stock keeps most of its
    signal while a no-data stock collapses toward 50.
    """
    floor = ConvictionConfig.CONFIDENCE_FACTOR_FLOOR
    confidence = max(0.0, min(1.0, confidence))
    return floor + (1.0 - floor) * confidence
