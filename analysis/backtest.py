"""
Backtesting Engine Module
=========================
Backtests growth-based stock selection strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import NGXDataLoader
from core.models import BacktestResult
from config.settings import (
    BacktestConfig,
    BACKTEST_DIR,
    AnalysisConfig,
)


@dataclass
class Trade:
    """Represents a single trade"""
    ticker: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    shares: float
    position_value: float
    return_pct: Optional[float] = None
    holding_days: Optional[int] = None
    
    def close(self, exit_date: datetime, exit_price: float):
        """Close the trade"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.return_pct = (exit_price / self.entry_price - 1) * 100
        self.holding_days = (exit_date - self.entry_date).days


class Portfolio:
    """Manages portfolio holdings and transactions"""
    
    def __init__(
        self,
        initial_capital: float = BacktestConfig.INITIAL_CAPITAL,
        transaction_cost: float = BacktestConfig.TRANSACTION_COST,
        slippage: float = BacktestConfig.SLIPPAGE
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        self.holdings: Dict[str, Dict] = {}  # ticker -> {shares, avg_price, value}
        self.trades: List[Trade] = []
        self.value_history: List[Dict] = []
        self.holdings_history: List[Dict] = []
    
    def buy(
        self,
        ticker: str,
        price: float,
        amount: float,
        date: datetime
    ) -> bool:
        """
        Buy shares of a stock.
        
        Args:
            ticker: Stock ticker
            price: Price per share
            amount: Dollar amount to invest
            date: Trade date
        
        Returns:
            True if successful
        """
        # Apply slippage (pay slightly more)
        effective_price = price * (1 + self.slippage)
        
        # Calculate shares and cost
        shares = amount / effective_price
        total_cost = amount + (amount * self.transaction_cost)
        
        if total_cost > self.cash:
            return False
        
        # Update cash
        self.cash -= total_cost
        
        # Update holdings
        if ticker in self.holdings:
            # Average up
            old_shares = self.holdings[ticker]["shares"]
            old_avg = self.holdings[ticker]["avg_price"]
            new_shares = old_shares + shares
            new_avg = (old_shares * old_avg + shares * effective_price) / new_shares
            self.holdings[ticker] = {
                "shares": new_shares,
                "avg_price": new_avg,
                "value": new_shares * price
            }
        else:
            self.holdings[ticker] = {
                "shares": shares,
                "avg_price": effective_price,
                "value": shares * price
            }
        
        # Record trade
        trade = Trade(
            ticker=ticker,
            entry_date=date,
            exit_date=None,
            entry_price=effective_price,
            exit_price=None,
            shares=shares,
            position_value=amount
        )
        self.trades.append(trade)
        
        return True
    
    def sell(
        self,
        ticker: str,
        price: float,
        shares: float,
        date: datetime
    ) -> float:
        """
        Sell shares of a stock.
        
        Args:
            ticker: Stock ticker
            price: Price per share
            shares: Number of shares to sell
            date: Trade date
        
        Returns:
            Proceeds from sale
        """
        if ticker not in self.holdings:
            return 0.0
        
        available_shares = self.holdings[ticker]["shares"]
        shares_to_sell = min(shares, available_shares)
        
        # Apply slippage (receive slightly less)
        effective_price = price * (1 - self.slippage)
        
        # Calculate proceeds
        gross_proceeds = shares_to_sell * effective_price
        net_proceeds = gross_proceeds * (1 - self.transaction_cost)
        
        # Update cash
        self.cash += net_proceeds
        
        # Update holdings
        remaining_shares = available_shares - shares_to_sell
        if remaining_shares < 0.001:
            del self.holdings[ticker]
        else:
            self.holdings[ticker]["shares"] = remaining_shares
            self.holdings[ticker]["value"] = remaining_shares * price
        
        # Close trade
        for trade in reversed(self.trades):
            if trade.ticker == ticker and trade.exit_date is None:
                trade.close(date, effective_price)
                break
        
        return net_proceeds
    
    def sell_all(self, ticker: str, price: float, date: datetime) -> float:
        """Sell all shares of a stock"""
        if ticker not in self.holdings:
            return 0.0
        return self.sell(ticker, price, self.holdings[ticker]["shares"], date)
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        holdings_value = sum(
            holding["shares"] * prices.get(ticker, holding["value"] / holding["shares"])
            for ticker, holding in self.holdings.items()
        )
        return self.cash + holdings_value
    
    def update_values(self, prices: Dict[str, float], date: datetime):
        """Update portfolio values and record history"""
        for ticker in self.holdings:
            if ticker in prices:
                self.holdings[ticker]["value"] = (
                    self.holdings[ticker]["shares"] * prices[ticker]
                )
        
        total_value = self.get_total_value(prices)
        
        self.value_history.append({
            "date": date,
            "total_value": total_value,
            "cash": self.cash,
            "holdings_value": total_value - self.cash
        })
        
        self.holdings_history.append({
            "date": date,
            "holdings": {
                ticker: {
                    "shares": h["shares"],
                    "value": h["value"]
                }
                for ticker, h in self.holdings.items()
            }
        })
    
    def get_allocation(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Get current allocation percentages"""
        total_value = self.get_total_value(prices)
        
        if total_value == 0:
            return {}
        
        allocation = {"cash": self.cash / total_value * 100}
        
        for ticker, holding in self.holdings.items():
            price = prices.get(ticker, holding["value"] / holding["shares"])
            value = holding["shares"] * price
            allocation[ticker] = value / total_value * 100
        
        return allocation


class BacktestEngine:
    """
    Backtesting engine for growth strategies.
    
    Supports:
    - Multiple rebalancing frequencies
    - Transaction costs and slippage
    - Stop-loss and take-profit
    - Performance metrics
    """
    
    def __init__(
        self,
        initial_capital: float = BacktestConfig.INITIAL_CAPITAL,
        rebalance_frequency: str = BacktestConfig.REBALANCE_FREQUENCY,
        transaction_cost: float = BacktestConfig.TRANSACTION_COST,
        slippage: float = BacktestConfig.SLIPPAGE,
        max_position_size: float = BacktestConfig.MAX_POSITION_SIZE,
        stop_loss: float = BacktestConfig.STOP_LOSS,
        take_profit: float = BacktestConfig.TAKE_PROFIT
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting portfolio value
            rebalance_frequency: monthly, quarterly, annually
            transaction_cost: Transaction cost percentage
            slippage: Slippage percentage
            max_position_size: Maximum position size as fraction
            stop_loss: Stop loss threshold (negative)
            take_profit: Take profit threshold (positive)
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        self.loader = NGXDataLoader()
    
    def run_backtest(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        selection_fn: Callable[[List[str], datetime], List[str]],
        n_holdings: int = 10,
        strategy_name: str = "Growth Strategy"
    ) -> BacktestResult:
        """
        Run backtest with a stock selection strategy.
        
        Args:
            tickers: Universe of tickers
            start_date: Backtest start date
            end_date: Backtest end date
            selection_fn: Function that selects top stocks at each rebalance
            n_holdings: Number of stocks to hold
            strategy_name: Name for the strategy
        
        Returns:
            BacktestResult with performance metrics
        """
        print(f"\nRunning backtest: {strategy_name}")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Universe: {len(tickers)} stocks, Holdings: {n_holdings}")
        print("-" * 50)
        
        # Load price data
        price_data = self._load_price_data(tickers, start_date, end_date)
        
        if not price_data:
            print("No price data available for backtest")
            return self._create_empty_result(strategy_name, start_date, end_date)
        
        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            slippage=self.slippage
        )
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        
        # Get all trading dates
        all_dates = self._get_trading_dates(price_data, start_date, end_date)
        
        # Run simulation
        current_holdings = []
        
        for i, date in enumerate(all_dates):
            if i % 50 == 0:
                print(f"  Processing {date.strftime('%Y-%m-%d')}...", end="\r")
            
            # Get current prices
            prices = self._get_prices_for_date(price_data, date)
            
            if not prices:
                continue
            
            # Check stop-loss / take-profit
            self._check_exits(portfolio, prices, date)
            
            # Rebalance if needed
            if date in rebalance_dates or not current_holdings:
                current_holdings = selection_fn(tickers, date)[:n_holdings]
                self._rebalance(portfolio, current_holdings, prices, date)
            
            # Update portfolio values
            portfolio.update_values(prices, date)
        
        print(f"  Completed simulation" + " " * 30)
        
        # Calculate results
        result = self._calculate_results(
            portfolio, strategy_name, start_date, end_date
        )
        
        return result
    
    def _load_price_data(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Load price data for all tickers"""
        price_data = {}
        
        for ticker in tickers:
            df = self.loader.get_price_data(ticker)
            
            if df is None or df.empty:
                continue
            
            df = df.set_index("Date")
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if not df.empty:
                price_data[ticker] = df
        
        return price_data
    
    def _get_rebalance_dates(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[datetime]:
        """Generate rebalance dates based on frequency"""
        dates = []
        current = start_date
        
        if self.rebalance_frequency == "monthly":
            delta = timedelta(days=30)
        elif self.rebalance_frequency == "quarterly":
            delta = timedelta(days=91)
        else:  # annually
            delta = timedelta(days=365)
        
        while current <= end_date:
            dates.append(current)
            current = current + delta
        
        return dates
    
    def _get_trading_dates(
        self,
        price_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> List[datetime]:
        """Get all trading dates in period"""
        all_dates = set()
        
        for df in price_data.values():
            dates = df[(df.index >= start_date) & (df.index <= end_date)].index
            all_dates.update(dates)
        
        return sorted(list(all_dates))
    
    def _get_prices_for_date(
        self,
        price_data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> Dict[str, float]:
        """Get closing prices for a specific date"""
        prices = {}
        
        for ticker, df in price_data.items():
            if date in df.index:
                price_col = "Adj_Close" if "Adj_Close" in df.columns else "Close"
                prices[ticker] = float(df.loc[date, price_col])
        
        return prices
    
    def _check_exits(
        self,
        portfolio: Portfolio,
        prices: Dict[str, float],
        date: datetime
    ):
        """Check stop-loss and take-profit conditions"""
        tickers_to_sell = []
        
        for ticker, holding in portfolio.holdings.items():
            if ticker not in prices:
                continue
            
            current_price = prices[ticker]
            avg_price = holding["avg_price"]
            return_pct = (current_price / avg_price - 1)
            
            if return_pct <= self.stop_loss:
                tickers_to_sell.append((ticker, "stop_loss"))
            elif return_pct >= self.take_profit:
                tickers_to_sell.append((ticker, "take_profit"))
        
        for ticker, reason in tickers_to_sell:
            portfolio.sell_all(ticker, prices[ticker], date)
    
    def _rebalance(
        self,
        portfolio: Portfolio,
        target_holdings: List[str],
        prices: Dict[str, float],
        date: datetime
    ):
        """Rebalance portfolio to target holdings"""
        # Sell positions not in target
        current_tickers = list(portfolio.holdings.keys())
        for ticker in current_tickers:
            if ticker not in target_holdings and ticker in prices:
                portfolio.sell_all(ticker, prices[ticker], date)
        
        # Calculate target allocation
        total_value = portfolio.get_total_value(prices)
        n_holdings = len(target_holdings)
        
        if n_holdings == 0:
            return
        
        target_per_stock = min(
            total_value / n_holdings,
            total_value * self.max_position_size
        )
        
        # Buy target holdings
        for ticker in target_holdings:
            if ticker not in prices:
                continue
            
            current_value = 0
            if ticker in portfolio.holdings:
                current_value = portfolio.holdings[ticker]["shares"] * prices[ticker]
            
            target_value = target_per_stock
            diff = target_value - current_value
            
            if diff > 100:  # Minimum trade size
                portfolio.buy(ticker, prices[ticker], diff, date)
    
    def _calculate_results(
        self,
        portfolio: Portfolio,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate backtest performance metrics"""
        result = BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date
        )
        
        if not portfolio.value_history:
            return result
        
        # Extract value series
        values = pd.DataFrame(portfolio.value_history)
        values = values.set_index("date")
        
        # Basic returns
        result.final_value = values["total_value"].iloc[-1]
        result.peak_value = values["total_value"].max()
        result.total_return = (result.final_value / self.initial_capital - 1) * 100
        
        # Annualized return
        years = (end_date - start_date).days / 365.25
        if years > 0:
            result.annualized_return = (
                (result.final_value / self.initial_capital) ** (1/years) - 1
            ) * 100
        
        # Returns for risk metrics
        daily_returns = values["total_value"].pct_change().dropna()
        
        # Volatility
        if len(daily_returns) > 0:
            result.volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe Ratio
        if result.volatility > 0:
            excess_return = result.annualized_return - (AnalysisConfig.RISK_FREE_RATE * 100)
            result.sharpe_ratio = excess_return / result.volatility
        
        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(252) * 100
            if downside_vol > 0:
                excess_return = result.annualized_return - (AnalysisConfig.RISK_FREE_RATE * 100)
                result.sortino_ratio = excess_return / downside_vol
        
        # Max Drawdown
        rolling_max = values["total_value"].cummax()
        drawdown = (values["total_value"] - rolling_max) / rolling_max
        result.max_drawdown = drawdown.min() * 100
        
        # Calmar Ratio
        if result.max_drawdown != 0:
            result.calmar_ratio = result.annualized_return / abs(result.max_drawdown)
        
        # Trade statistics
        closed_trades = [t for t in portfolio.trades if t.exit_date is not None]
        result.total_trades = len(closed_trades)
        
        if closed_trades:
            returns = [t.return_pct for t in closed_trades if t.return_pct is not None]
            winning = [r for r in returns if r > 0]
            losing = [r for r in returns if r <= 0]
            
            result.winning_trades = len(winning)
            result.losing_trades = len(losing)
            result.win_rate = len(winning) / len(returns) * 100 if returns else 0
            result.avg_win = np.mean(winning) if winning else 0
            result.avg_loss = np.mean(losing) if losing else 0
            
            if result.avg_loss != 0:
                result.profit_factor = abs(sum(winning) / sum(losing)) if losing else 0
        
        # Store histories
        result.portfolio_values = portfolio.value_history
        result.holdings_history = portfolio.holdings_history
        
        return result
    
    def _create_empty_result(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Create empty result when no data"""
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date
        )
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate backtest report"""
        lines = [
            "\n" + "=" * 60,
            f"BACKTEST REPORT: {result.strategy_name}",
            "=" * 60,
            "",
            "PERIOD:",
            f"  Start: {result.start_date.strftime('%Y-%m-%d')}",
            f"  End: {result.end_date.strftime('%Y-%m-%d')}",
            "",
            "RETURNS:",
            f"  Total Return: {result.total_return:.2f}%",
            f"  Annualized Return: {result.annualized_return:.2f}%",
            f"  Initial Capital: ₦{self.initial_capital:,.0f}",
            f"  Final Value: ₦{result.final_value:,.0f}",
            f"  Peak Value: ₦{result.peak_value:,.0f}",
            "",
            "RISK METRICS:",
            f"  Volatility: {result.volatility:.2f}%",
            f"  Max Drawdown: {result.max_drawdown:.2f}%",
            f"  Sharpe Ratio: {result.sharpe_ratio:.2f}",
            f"  Sortino Ratio: {result.sortino_ratio:.2f}",
            f"  Calmar Ratio: {result.calmar_ratio:.2f}",
            "",
            "TRADE STATISTICS:",
            f"  Total Trades: {result.total_trades}",
            f"  Winning Trades: {result.winning_trades}",
            f"  Losing Trades: {result.losing_trades}",
            f"  Win Rate: {result.win_rate:.1f}%",
            f"  Avg Win: {result.avg_win:.2f}%",
            f"  Avg Loss: {result.avg_loss:.2f}%",
            f"  Profit Factor: {result.profit_factor:.2f}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def save_results(self, result: BacktestResult, filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{result.strategy_name.replace(' ', '_')}_{timestamp}.json"
        
        filepath = BACKTEST_DIR / filename
        
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        print(f"Results saved to: {filepath}")


def momentum_selection(tickers: List[str], date: datetime, n: int = 10) -> List[str]:
    """
    Simple momentum-based stock selection.
    Selects top N stocks by 12-month momentum.
    """
    loader = NGXDataLoader()
    momentum_scores = []
    
    for ticker in tickers:
        prices = loader.get_price_series(ticker)
        
        if prices is None or len(prices) < 252:
            continue
        
        # Filter to before selection date
        prices = prices[prices.index < date]
        
        if len(prices) < 252:
            continue
        
        # Calculate 12-month momentum (excluding last month)
        price_12m_ago = prices.iloc[-252]
        price_1m_ago = prices.iloc[-21]
        
        momentum = (price_1m_ago / price_12m_ago - 1) * 100
        momentum_scores.append((ticker, momentum))
    
    # Sort by momentum and return top N
    momentum_scores.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in momentum_scores[:n]]
