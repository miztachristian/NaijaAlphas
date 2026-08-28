"""
decision_system — unified NGX decision engine.

Fuses every existing analysis signal (fundamental, technical, quality,
seasonality, disclosure, report tone, news sentiment) plus a Nigeria macro
regime into ONE 0-100 conviction score + action per stock, then constructs a
concrete portfolio. Designed to be orchestrated by daily_ingest.py.

The engine *orchestrates* the existing analyzers in analysis/ — it never
re-implements their scoring math.
"""

from decision_system.models import (
    SignalSet,
    ConvictionScore,
    Decision,
    OrderItem,
    MacroState,
)

__all__ = [
    "SignalSet",
    "ConvictionScore",
    "Decision",
    "OrderItem",
    "MacroState",
]
