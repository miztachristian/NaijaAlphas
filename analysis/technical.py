"""
Technical Analysis Module
=========================
Comprehensive technical indicators, momentum, trend, and volatility analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import NGXDataLoader
from core.models import TechnicalMetrics
from config.settings import TechnicalParams, AnalysisConfig


class TechnicalAnalyzer:
    """
    Technical analysis engine for NGX stocks.
    
    Analyzes:
    - Trend indicators (Moving averages, MACD)
    - Momentum (RSI, Stochastic, Price momentum)
    - Volatility (ATR, Bollinger Bands, Historical volatility)
    - Risk metrics (Sharpe, Sortino, Max Drawdown)
    """
    
    def __init__(self, loader: NGXDataLoader = None):
        """
        Initialize analyzer.
        
        Args:
            loader: Data loader instance
        """
        self.loader = loader or NGXDataLoader()
        self.index_data: Optional[pd.Series] = None
    
    def analyze(self, ticker: str) -> TechnicalMetrics:
        """
        Perform complete technical analysis on a stock.
        
        Args:
            ticker: NGX ticker symbol
        
        Returns:
            TechnicalMetrics with all calculated values
        """
        metrics = TechnicalMetrics(ticker=ticker)
        
        # Get price data
        prices = self.loader.get_price_series(ticker)
        
        if prices is None or len(prices) < TechnicalParams.SMA_SHORT:
            return metrics
        
        df = self.loader.get_price_data(ticker)
        if df is None:
            return metrics
        
        # Current price
        metrics.price = float(prices.iloc[-1])
        metrics.price_change_1d = float((prices.iloc[-1] / prices.iloc[-2] - 1) * 100) if len(prices) > 1 else 0
        
        # Calculate all indicators
        self._calculate_moving_averages(metrics, prices)
        self._calculate_trend_signals(metrics, prices)
        self._calculate_rsi(metrics, prices)
        self._calculate_macd(metrics, prices)
        self._calculate_stochastic(metrics, df)
        self._calculate_momentum(metrics, prices)
        self._calculate_volatility(metrics, prices)
        self._calculate_bollinger_bands(metrics, prices)
        self._calculate_risk_metrics(metrics, prices)
        self._calculate_volume_analysis(metrics, df)
        
        return metrics
    
    def _calculate_moving_averages(self, metrics: TechnicalMetrics, prices: pd.Series):
        """Calculate moving averages"""
        # SMA
        if len(prices) >= 20:
            metrics.sma_20 = float(prices.rolling(20).mean().iloc[-1])
        if len(prices) >= TechnicalParams.SMA_SHORT:
            metrics.sma_50 = float(prices.rolling(TechnicalParams.SMA_SHORT).mean().iloc[-1])
        if len(prices) >= TechnicalParams.SMA_LONG:
            metrics.sma_200 = float(prices.rolling(TechnicalParams.SMA_LONG).mean().iloc[-1])
        
        # EMA
        if len(prices) >= TechnicalParams.EMA_SHORT:
            metrics.ema_12 = float(prices.ewm(span=TechnicalParams.EMA_SHORT, adjust=False).mean().iloc[-1])
        if len(prices) >= TechnicalParams.EMA_LONG:
            metrics.ema_26 = float(prices.ewm(span=TechnicalParams.EMA_LONG, adjust=False).mean().iloc[-1])
    
    def _calculate_trend_signals(self, metrics: TechnicalMetrics, prices: pd.Series):
        """Calculate trend-based signals"""
        # Price vs moving averages
        if not np.isnan(metrics.sma_50):
            metrics.above_sma50 = metrics.price > metrics.sma_50
        if not np.isnan(metrics.sma_200):
            metrics.above_sma200 = metrics.price > metrics.sma_200
        
        # Golden/Death cross
        if len(prices) >= TechnicalParams.SMA_LONG:
            sma_50 = prices.rolling(TechnicalParams.SMA_SHORT).mean()
            sma_200 = prices.rolling(TechnicalParams.SMA_LONG).mean()
            
            # Current state
            current_50_above_200 = sma_50.iloc[-1] > sma_200.iloc[-1]
            prev_50_above_200 = sma_50.iloc[-2] > sma_200.iloc[-2] if len(sma_50) > 1 else current_50_above_200
            
            # Golden cross: SMA50 crosses above SMA200
            metrics.golden_cross = current_50_above_200 and not prev_50_above_200
            # Death cross: SMA50 crosses below SMA200
            metrics.death_cross = not current_50_above_200 and prev_50_above_200
    
    def _calculate_rsi(self, metrics: TechnicalMetrics, prices: pd.Series, period: int = None):
        """Calculate RSI"""
        period = period or TechnicalParams.RSI_PERIOD
        
        if len(prices) < period + 1:
            return
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        metrics.rsi_14 = float(rsi.iloc[-1]) if not rsi.empty else np.nan
    
    def _calculate_macd(self, metrics: TechnicalMetrics, prices: pd.Series):
        """Calculate MACD indicator"""
        if len(prices) < TechnicalParams.MACD_SLOW + TechnicalParams.MACD_SIGNAL:
            return
        
        ema_fast = prices.ewm(span=TechnicalParams.MACD_FAST, adjust=False).mean()
        ema_slow = prices.ewm(span=TechnicalParams.MACD_SLOW, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=TechnicalParams.MACD_SIGNAL, adjust=False).mean()
        histogram = macd_line - signal_line
        
        metrics.macd = float(macd_line.iloc[-1])
        metrics.macd_signal = float(signal_line.iloc[-1])
        metrics.macd_histogram = float(histogram.iloc[-1])
    
    def _calculate_stochastic(self, metrics: TechnicalMetrics, df: pd.DataFrame, period: int = 14):
        """Calculate Stochastic oscillator"""
        if len(df) < period:
            return
        
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=3).mean()
        
        metrics.stochastic_k = float(stoch_k.iloc[-1]) if not stoch_k.empty else np.nan
        metrics.stochastic_d = float(stoch_d.iloc[-1]) if not stoch_d.empty else np.nan
    
    def _calculate_momentum(self, metrics: TechnicalMetrics, prices: pd.Series):
        """Calculate price momentum over various periods"""
        def momentum(days):
            if len(prices) < days:
                return np.nan
            return float((prices.iloc[-1] / prices.iloc[-days] - 1) * 100)
        
        metrics.momentum_1m = momentum(TechnicalParams.MOMENTUM_1M)
        metrics.momentum_3m = momentum(TechnicalParams.MOMENTUM_3M)
        metrics.momentum_6m = momentum(TechnicalParams.MOMENTUM_6M)
        metrics.momentum_12m = momentum(TechnicalParams.MOMENTUM_12M)
        
        # 12-1 momentum factor (skip most recent month)
        if len(prices) >= TechnicalParams.MOMENTUM_12M:
            price_12m_ago = prices.iloc[-TechnicalParams.MOMENTUM_12M]
            price_1m_ago = prices.iloc[-TechnicalParams.MOMENTUM_1M]
            metrics.momentum_12_1 = float((price_1m_ago / price_12m_ago - 1) * 100)
    
    def _calculate_volatility(self, metrics: TechnicalMetrics, prices: pd.Series):
        """Calculate volatility measures"""
        returns = prices.pct_change().dropna()
        
        if len(returns) >= 20:
            metrics.volatility_20d = float(returns.iloc[-20:].std() * np.sqrt(252) * 100)
        if len(returns) >= 60:
            metrics.volatility_60d = float(returns.iloc[-60:].std() * np.sqrt(252) * 100)
        if len(returns) >= TechnicalParams.VOLATILITY_PERIOD:
            metrics.volatility_252d = float(returns.iloc[-TechnicalParams.VOLATILITY_PERIOD:].std() * np.sqrt(252) * 100)
    
    def _calculate_atr(self, metrics: TechnicalMetrics, df: pd.DataFrame, period: int = 14):
        """Calculate Average True Range"""
        if len(df) < period + 1:
            return
        
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        metrics.atr_14 = float(atr.iloc[-1]) if not atr.empty else np.nan
    
    def _calculate_bollinger_bands(self, metrics: TechnicalMetrics, prices: pd.Series):
        """Calculate Bollinger Bands"""
        period = TechnicalParams.BB_PERIOD
        std_dev = TechnicalParams.BB_STD
        
        if len(prices) < period:
            return
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        metrics.bb_upper = float(sma.iloc[-1] + std_dev * std.iloc[-1])
        metrics.bb_middle = float(sma.iloc[-1])
        metrics.bb_lower = float(sma.iloc[-1] - std_dev * std.iloc[-1])
        
        # Bandwidth and position
        if metrics.bb_middle > 0:
            metrics.bb_width = (metrics.bb_upper - metrics.bb_lower) / metrics.bb_middle * 100
            if metrics.bb_upper != metrics.bb_lower:
                metrics.bb_position = (metrics.price - metrics.bb_lower) / (metrics.bb_upper - metrics.bb_lower)
    
    def _calculate_risk_metrics(self, metrics: TechnicalMetrics, prices: pd.Series):
        """Calculate risk-adjusted return metrics"""
        returns = prices.pct_change().dropna()
        days = TechnicalParams.VOLATILITY_PERIOD
        
        if len(returns) < days:
            days = len(returns)
        
        recent_returns = returns.iloc[-days:]
        
        # Max Drawdown
        rolling_max = prices.iloc[-days:].cummax()
        drawdown = (prices.iloc[-days:] - rolling_max) / rolling_max
        metrics.max_drawdown_1y = float(drawdown.min() * 100)
        
        # Sharpe Ratio
        risk_free_daily = AnalysisConfig.RISK_FREE_RATE / 252
        excess_returns = recent_returns - risk_free_daily
        
        if recent_returns.std() > 0:
            metrics.sharpe_1y = float(
                (excess_returns.mean() * 252) / (recent_returns.std() * np.sqrt(252))
            )
        
        # Sortino Ratio (downside deviation)
        downside_returns = recent_returns[recent_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics.sortino_1y = float(
                (excess_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252))
            )
    
    def _calculate_volume_analysis(self, metrics: TechnicalMetrics, df: pd.DataFrame):
        """Calculate volume-based indicators"""
        if "Volume" not in df.columns or len(df) < 20:
            return
        
        volume = df["Volume"]
        
        # Volume SMA
        metrics.volume_sma_20 = float(volume.rolling(20).mean().iloc[-1])
        
        # Volume ratio (current vs average)
        if metrics.volume_sma_20 > 0:
            metrics.volume_ratio = float(volume.iloc[-1] / metrics.volume_sma_20)
    
    def calculate_relative_strength(
        self, 
        ticker: str, 
        index_prices: pd.Series
    ) -> float:
        """
        Calculate relative strength vs index.
        
        Args:
            ticker: Stock ticker
            index_prices: Index price series
        
        Returns:
            Relative strength (positive = outperforming)
        """
        stock_prices = self.loader.get_price_series(ticker)
        
        if stock_prices is None or index_prices is None:
            return np.nan
        
        # Align dates
        common_dates = stock_prices.index.intersection(index_prices.index)
        
        if len(common_dates) < 20:
            return np.nan
        
        stock = stock_prices.loc[common_dates]
        index = index_prices.loc[common_dates]
        
        # Calculate relative returns
        stock_return = (stock.iloc[-1] / stock.iloc[0] - 1) * 100
        index_return = (index.iloc[-1] / index.iloc[0] - 1) * 100
        
        return stock_return - index_return
    
    def score_technicals(self, metrics: TechnicalMetrics) -> Dict[str, float]:
        """
        Score technical metrics on 0-1 scale.
        
        Args:
            metrics: TechnicalMetrics object
        
        Returns:
            Dictionary of scores
        """
        scores = {}
        
        # Momentum Score (12-1 month strategy)
        scores["momentum_score"] = self._score_momentum(metrics)
        
        # Trend Score
        scores["trend_score"] = self._score_trend(metrics)
        
        # RSI Score
        scores["rsi_score"] = self._score_rsi(metrics.rsi_14)
        
        # MACD Score
        scores["macd_score"] = self._score_macd(metrics)
        
        # Volatility Score
        scores["volatility_score"] = self._score_volatility(metrics.volatility_252d)
        
        # Sharpe Score
        scores["sharpe_score"] = self._score_sharpe(metrics.sharpe_1y)
        
        # Drawdown Score
        scores["drawdown_score"] = self._score_drawdown(metrics.max_drawdown_1y)
        
        # Bollinger Score
        scores["bb_score"] = self._score_bollinger(metrics.bb_position)
        
        return scores
    
    def _score_momentum(self, metrics: TechnicalMetrics) -> float:
        """Score momentum indicators"""
        # Use 12-1 month momentum (excluding recent month to avoid reversal)
        mom = metrics.momentum_12_1
        
        if np.isnan(mom):
            mom = metrics.momentum_12m
        
        if np.isnan(mom):
            return 0.3
        
        if mom > 50:
            return 1.0
        elif mom > 30:
            return 0.9
        elif mom > 15:
            return 0.8
        elif mom > 5:
            return 0.7
        elif mom > 0:
            return 0.6
        elif mom > -10:
            return 0.4
        elif mom > -20:
            return 0.3
        else:
            return 0.2
    
    def _score_trend(self, metrics: TechnicalMetrics) -> float:
        """Score trend indicators"""
        score = 0.5  # Neutral base
        
        if metrics.golden_cross:
            score = 1.0
        elif metrics.death_cross:
            score = 0.2
        elif metrics.above_sma200 and metrics.above_sma50:
            score = 0.9
        elif metrics.above_sma200:
            score = 0.7
        elif metrics.above_sma50:
            score = 0.6
        else:
            score = 0.3
        
        return score
    
    def _score_rsi(self, rsi: float) -> float:
        """Score RSI (neutral zone is best for growth)"""
        if np.isnan(rsi):
            return 0.5
        
        if rsi < 30:
            return 0.8  # Oversold - potential bounce
        elif rsi < 40:
            return 0.7
        elif rsi < 60:
            return 0.9  # Neutral - healthy
        elif rsi < 70:
            return 0.7
        else:
            return 0.4  # Overbought - risk of pullback
    
    def _score_macd(self, metrics: TechnicalMetrics) -> float:
        """Score MACD"""
        if np.isnan(metrics.macd) or np.isnan(metrics.macd_signal):
            return 0.5
        
        # MACD above signal is bullish
        if metrics.macd > metrics.macd_signal:
            if metrics.macd_histogram > 0:
                return 0.9  # Strong bullish
            return 0.7
        else:
            if metrics.macd_histogram < 0:
                return 0.3  # Strong bearish
            return 0.4
    
    def _score_volatility(self, vol: float) -> float:
        """Score volatility (moderate is ideal for growth)"""
        if np.isnan(vol):
            return 0.5
        
        if vol < 15:
            return 0.7  # Low vol - stable but limited upside
        elif vol < 25:
            return 0.9  # Moderate - ideal
        elif vol < 35:
            return 0.8  # Good
        elif vol < 50:
            return 0.6  # High
        else:
            return 0.3  # Very high - risky
    
    def _score_sharpe(self, sharpe: float) -> float:
        """Score Sharpe ratio"""
        if np.isnan(sharpe):
            return 0.3
        
        if sharpe > 2:
            return 1.0  # Excellent
        elif sharpe > 1.5:
            return 0.9
        elif sharpe > 1:
            return 0.8
        elif sharpe > 0.5:
            return 0.7
        elif sharpe > 0:
            return 0.5
        elif sharpe > -0.5:
            return 0.3
        else:
            return 0.1  # Poor risk-adjusted returns
    
    def _score_drawdown(self, dd: float) -> float:
        """Score max drawdown (smaller is better)"""
        if np.isnan(dd):
            return 0.5
        
        dd = abs(dd)  # Ensure positive
        
        if dd < 10:
            return 1.0  # Excellent
        elif dd < 15:
            return 0.9
        elif dd < 20:
            return 0.8
        elif dd < 30:
            return 0.6
        elif dd < 40:
            return 0.4
        else:
            return 0.2  # Large drawdown
    
    def _score_bollinger(self, position: float) -> float:
        """Score Bollinger Band position"""
        if np.isnan(position):
            return 0.5
        
        # Position 0 = at lower band, 1 = at upper band
        if position < 0.1:
            return 0.8  # Near lower band - potential bounce
        elif position < 0.3:
            return 0.7
        elif position < 0.7:
            return 0.9  # Middle - healthy
        elif position < 0.9:
            return 0.6
        else:
            return 0.4  # Near upper band - extended
    
    def get_composite_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate composite technical score.
        
        Weights:
        - Momentum: 30%
        - Trend: 25%
        - RSI: 15%
        - Sharpe: 15%
        - Drawdown: 10%
        - MACD: 5%
        
        Args:
            scores: Dictionary of individual scores
        
        Returns:
            Composite score (0-1)
        """
        weights = {
            "momentum_score": 0.30,
            "trend_score": 0.25,
            "rsi_score": 0.15,
            "sharpe_score": 0.15,
            "drawdown_score": 0.10,
            "macd_score": 0.05,
        }
        
        composite = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in scores:
                composite += scores[key] * weight
                total_weight += weight
        
        return composite / total_weight if total_weight > 0 else 0.0
    
    def get_signals(self, metrics: TechnicalMetrics) -> Dict[str, str]:
        """
        Generate trading signals based on technical analysis.
        
        Args:
            metrics: TechnicalMetrics object
        
        Returns:
            Dictionary with signal types and strengths
        """
        signals = {
            "overall": "NEUTRAL",
            "trend": "NEUTRAL",
            "momentum": "NEUTRAL",
            "oscillator": "NEUTRAL",
        }
        
        # Trend signals
        if metrics.golden_cross:
            signals["trend"] = "STRONG_BUY"
        elif metrics.death_cross:
            signals["trend"] = "STRONG_SELL"
        elif metrics.above_sma200 and metrics.above_sma50:
            signals["trend"] = "BUY"
        elif not metrics.above_sma200 and not metrics.above_sma50:
            signals["trend"] = "SELL"
        
        # Momentum signals
        mom = metrics.momentum_12_1 if not np.isnan(metrics.momentum_12_1) else metrics.momentum_6m
        if not np.isnan(mom):
            if mom > 30:
                signals["momentum"] = "STRONG_BUY"
            elif mom > 10:
                signals["momentum"] = "BUY"
            elif mom < -20:
                signals["momentum"] = "STRONG_SELL"
            elif mom < -5:
                signals["momentum"] = "SELL"
        
        # Oscillator signals
        if not np.isnan(metrics.rsi_14):
            if metrics.rsi_14 < 30:
                signals["oscillator"] = "BUY"  # Oversold
            elif metrics.rsi_14 > 70:
                signals["oscillator"] = "SELL"  # Overbought
        
        # Overall signal (voting system)
        buy_count = sum(1 for s in signals.values() if "BUY" in s)
        sell_count = sum(1 for s in signals.values() if "SELL" in s)
        
        if buy_count >= 2:
            signals["overall"] = "BUY" if buy_count == 2 else "STRONG_BUY"
        elif sell_count >= 2:
            signals["overall"] = "SELL" if sell_count == 2 else "STRONG_SELL"
        
        return signals
    
    def generate_analysis_summary(
        self, 
        ticker: str, 
        metrics: TechnicalMetrics, 
        scores: Dict[str, float]
    ) -> str:
        """Generate human-readable analysis summary"""
        signals = self.get_signals(metrics)
        
        lines = [
            f"\n{'='*50}",
            f"TECHNICAL ANALYSIS: {ticker}",
            f"{'='*50}",
            "",
            f"PRICE: ₦{metrics.price:,.2f}" if not np.isnan(metrics.price) else "PRICE: N/A",
            "",
            "TREND:",
            f"  SMA 50: ₦{metrics.sma_50:,.2f}" if not np.isnan(metrics.sma_50) else "  SMA 50: N/A",
            f"  SMA 200: ₦{metrics.sma_200:,.2f}" if not np.isnan(metrics.sma_200) else "  SMA 200: N/A",
            f"  Above SMA50: {'Yes' if metrics.above_sma50 else 'No'}",
            f"  Above SMA200: {'Yes' if metrics.above_sma200 else 'No'}",
            f"  Golden Cross: {'Yes' if metrics.golden_cross else 'No'}",
            "",
            "MOMENTUM:",
            f"  1-Month: {metrics.momentum_1m:.1f}%" if not np.isnan(metrics.momentum_1m) else "  1-Month: N/A",
            f"  3-Month: {metrics.momentum_3m:.1f}%" if not np.isnan(metrics.momentum_3m) else "  3-Month: N/A",
            f"  6-Month: {metrics.momentum_6m:.1f}%" if not np.isnan(metrics.momentum_6m) else "  6-Month: N/A",
            f"  12-Month: {metrics.momentum_12m:.1f}%" if not np.isnan(metrics.momentum_12m) else "  12-Month: N/A",
            "",
            "OSCILLATORS:",
            f"  RSI (14): {metrics.rsi_14:.1f}" if not np.isnan(metrics.rsi_14) else "  RSI (14): N/A",
            f"  MACD: {metrics.macd:.4f}" if not np.isnan(metrics.macd) else "  MACD: N/A",
            "",
            "RISK METRICS:",
            f"  Volatility (252d): {metrics.volatility_252d:.1f}%" if not np.isnan(metrics.volatility_252d) else "  Volatility: N/A",
            f"  Max Drawdown (1Y): {metrics.max_drawdown_1y:.1f}%" if not np.isnan(metrics.max_drawdown_1y) else "  Max Drawdown: N/A",
            f"  Sharpe Ratio (1Y): {metrics.sharpe_1y:.2f}" if not np.isnan(metrics.sharpe_1y) else "  Sharpe Ratio: N/A",
            "",
            "SIGNALS:",
            f"  Trend: {signals['trend']}",
            f"  Momentum: {signals['momentum']}",
            f"  Oscillator: {signals['oscillator']}",
            f"  OVERALL: {signals['overall']}",
            "",
            "SCORES:",
        ]
        
        for key, score in scores.items():
            clean_key = key.replace("_score", "").replace("_", " ").title()
            lines.append(f"  {clean_key}: {score:.2f}")
        
        composite = self.get_composite_score(scores)
        lines.extend([
            "",
            f"COMPOSITE TECHNICAL SCORE: {composite:.2f}",
            f"{'='*50}",
        ])
        
        return "\n".join(lines)
