"""
Visualization Module
====================
Charts and plots for stock analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import OUTPUT_DIR


class StockVisualizer:
    """
    Visualization tools for stock analysis.
    
    Supports both matplotlib and plotly backends.
    """
    
    def __init__(self, backend: str = "plotly"):
        """
        Initialize visualizer.
        
        Args:
            backend: 'plotly' or 'matplotlib'
        """
        self.backend = backend
        
        if backend == "plotly" and not HAS_PLOTLY:
            print("Warning: plotly not installed, falling back to matplotlib")
            self.backend = "matplotlib"
        
        if self.backend == "matplotlib" and not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
    
    def plot_price_chart(
        self,
        prices: pd.Series,
        ticker: str,
        show_ma: bool = True,
        save: bool = False
    ):
        """
        Plot price chart with moving averages.
        
        Args:
            prices: Price series
            ticker: Stock ticker
            show_ma: Show moving averages
            save: Save to file
        """
        if self.backend == "plotly":
            self._plot_price_plotly(prices, ticker, show_ma, save)
        else:
            self._plot_price_matplotlib(prices, ticker, show_ma, save)
    
    def _plot_price_plotly(
        self,
        prices: pd.Series,
        ticker: str,
        show_ma: bool,
        save: bool
    ):
        """Plotly price chart"""
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices.values,
            mode='lines',
            name='Price',
            line=dict(color='#2E86AB', width=2)
        ))
        
        # Moving averages
        if show_ma and len(prices) >= 50:
            sma_50 = prices.rolling(50).mean()
            fig.add_trace(go.Scatter(
                x=sma_50.index,
                y=sma_50.values,
                mode='lines',
                name='SMA 50',
                line=dict(color='#F18F01', width=1.5)
            ))
        
        if show_ma and len(prices) >= 200:
            sma_200 = prices.rolling(200).mean()
            fig.add_trace(go.Scatter(
                x=sma_200.index,
                y=sma_200.values,
                mode='lines',
                name='SMA 200',
                line=dict(color='#C73E1D', width=1.5)
            ))
        
        fig.update_layout(
            title=f'{ticker} Price Chart',
            xaxis_title='Date',
            yaxis_title='Price (₦)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        if save:
            fig.write_html(OUTPUT_DIR / f"{ticker}_price_chart.html")
        
        fig.show()
    
    def _plot_price_matplotlib(
        self,
        prices: pd.Series,
        ticker: str,
        show_ma: bool,
        save: bool
    ):
        """Matplotlib price chart"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(prices.index, prices.values, label='Price', color='#2E86AB', linewidth=2)
        
        if show_ma and len(prices) >= 50:
            sma_50 = prices.rolling(50).mean()
            ax.plot(sma_50.index, sma_50.values, label='SMA 50', color='#F18F01', linewidth=1.5)
        
        if show_ma and len(prices) >= 200:
            sma_200 = prices.rolling(200).mean()
            ax.plot(sma_200.index, sma_200.values, label='SMA 200', color='#C73E1D', linewidth=1.5)
        
        ax.set_title(f'{ticker} Price Chart')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (₦)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            plt.savefig(OUTPUT_DIR / f"{ticker}_price_chart.png", dpi=300)
        
        plt.show()
    
    def plot_growth_scores(
        self,
        df: pd.DataFrame,
        top_n: int = 10,
        save: bool = False
    ):
        """
        Plot growth scores for top stocks.
        
        Args:
            df: Analysis results DataFrame
            top_n: Number of stocks to show
            save: Save to file
        """
        top_df = df.head(top_n)
        
        if self.backend == "plotly":
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=top_df['ticker'],
                y=top_df['growth_score'],
                marker_color=top_df['growth_score'],
                marker_colorscale='Viridis',
                text=top_df['growth_score'].round(1),
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f'Top {top_n} Growth Scores',
                xaxis_title='Stock',
                yaxis_title='Growth Score',
                template='plotly_white'
            )
            
            if save:
                fig.write_html(OUTPUT_DIR / "growth_scores.html")
            
            fig.show()
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = plt.cm.viridis(top_df['growth_score'] / 100)
            bars = ax.bar(top_df['ticker'], top_df['growth_score'], color=colors)
            
            ax.set_title(f'Top {top_n} Growth Scores')
            ax.set_xlabel('Stock')
            ax.set_ylabel('Growth Score')
            ax.bar_label(bars, fmt='%.1f')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save:
                plt.savefig(OUTPUT_DIR / "growth_scores.png", dpi=300)
            
            plt.show()
    
    def plot_score_breakdown(
        self,
        df: pd.DataFrame,
        ticker: str,
        save: bool = False
    ):
        """
        Plot score breakdown for a stock.
        
        Args:
            df: Analysis results DataFrame
            ticker: Stock ticker
            save: Save to file
        """
        stock = df[df['ticker'] == ticker].iloc[0]
        
        categories = [
            'Earnings', 'Appreciation', 'Momentum',
            'Health', 'Sector', 'Risk-Adj'
        ]
        values = [
            stock['earnings_score'],
            stock['appreciation_score'],
            stock['momentum_score'],
            stock['health_score'],
            stock['sector_score'],
            stock['risk_score']
        ]
        
        if self.backend == "plotly":
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=ticker
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                title=f'{ticker} Score Breakdown (Total: {stock["growth_score"]:.1f})'
            )
            
            if save:
                fig.write_html(OUTPUT_DIR / f"{ticker}_breakdown.html")
            
            fig.show()
        else:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values_plot = values + [values[0]]  # Close the polygon
            angles += angles[:1]
            
            ax.plot(angles, values_plot)
            ax.fill(angles, values_plot, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)
            ax.set_title(f'{ticker} Score Breakdown (Total: {stock["growth_score"]:.1f})')
            
            if save:
                plt.savefig(OUTPUT_DIR / f"{ticker}_breakdown.png", dpi=300)
            
            plt.show()
    
    def plot_sector_distribution(
        self,
        df: pd.DataFrame,
        save: bool = False
    ):
        """
        Plot sector distribution of top picks.
        
        Args:
            df: Analysis results DataFrame
            save: Save to file
        """
        sector_counts = df.head(20)['sector'].value_counts()
        
        if self.backend == "plotly":
            fig = go.Figure(data=[go.Pie(
                labels=sector_counts.index,
                values=sector_counts.values,
                hole=0.4
            )])
            
            fig.update_layout(title='Sector Distribution (Top 20)')
            
            if save:
                fig.write_html(OUTPUT_DIR / "sector_distribution.html")
            
            fig.show()
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            ax.pie(sector_counts.values, labels=sector_counts.index, autopct='%1.1f%%')
            ax.set_title('Sector Distribution (Top 20)')
            
            if save:
                plt.savefig(OUTPUT_DIR / "sector_distribution.png", dpi=300)
            
            plt.show()
    
    def plot_backtest_results(
        self,
        portfolio_values: List[Dict],
        benchmark_values: Optional[List[Dict]] = None,
        save: bool = False
    ):
        """
        Plot backtest portfolio performance.
        
        Args:
            portfolio_values: List of {date, total_value} dicts
            benchmark_values: Optional benchmark values
            save: Save to file
        """
        df = pd.DataFrame(portfolio_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        if self.backend == "plotly":
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['total_value'],
                mode='lines',
                name='Portfolio',
                line=dict(color='#2E86AB', width=2)
            ))
            
            if benchmark_values:
                bm_df = pd.DataFrame(benchmark_values)
                bm_df['date'] = pd.to_datetime(bm_df['date'])
                fig.add_trace(go.Scatter(
                    x=bm_df['date'],
                    y=bm_df['value'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#C73E1D', width=1.5, dash='dash')
                ))
            
            fig.update_layout(
                title='Backtest Performance',
                xaxis_title='Date',
                yaxis_title='Portfolio Value (₦)',
                template='plotly_white',
                hovermode='x unified'
            )
            
            if save:
                fig.write_html(OUTPUT_DIR / "backtest_performance.html")
            
            fig.show()
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(df.index, df['total_value'], label='Portfolio', color='#2E86AB', linewidth=2)
            
            if benchmark_values:
                bm_df = pd.DataFrame(benchmark_values)
                bm_df['date'] = pd.to_datetime(bm_df['date'])
                ax.plot(bm_df['date'], bm_df['value'], label='Benchmark',
                       color='#C73E1D', linewidth=1.5, linestyle='--')
            
            ax.set_title('Backtest Performance')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value (₦)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save:
                plt.savefig(OUTPUT_DIR / "backtest_performance.png", dpi=300)
            
            plt.show()
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        metrics: List[str] = None,
        save: bool = False
    ):
        """
        Plot correlation matrix of metrics.
        
        Args:
            df: Analysis results DataFrame
            metrics: List of metrics to include
            save: Save to file
        """
        if metrics is None:
            metrics = [
                'growth_score', 'earnings_score', 'appreciation_score',
                'momentum_score', 'pe_ratio', 'roe', 'momentum_12m'
            ]
        
        # Filter to numeric columns that exist
        available = [m for m in metrics if m in df.columns]
        corr_df = df[available].corr()
        
        if self.backend == "plotly":
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(title='Metric Correlations')
            
            if save:
                fig.write_html(OUTPUT_DIR / "correlations.html")
            
            fig.show()
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(corr_df.values, cmap='RdBu', vmin=-1, vmax=1)
            
            ax.set_xticks(range(len(corr_df.columns)))
            ax.set_yticks(range(len(corr_df.index)))
            ax.set_xticklabels(corr_df.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_df.index)
            
            plt.colorbar(im)
            ax.set_title('Metric Correlations')
            plt.tight_layout()
            
            if save:
                plt.savefig(OUTPUT_DIR / "correlations.png", dpi=300)
            
            plt.show()
