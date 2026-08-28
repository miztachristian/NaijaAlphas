# Hidden Gems Methodology
## Finding the Next REDSTAREX Before It Rallies 78%

### 📖 Core Philosophy

**The Hidden Gems strategy identifies stocks with strong fundamentals that haven't yet attracted momentum traders.**

This is the inverse of momentum investing:
- **Momentum traders** chase stocks already running (+20% in 1 month)
- **Hidden Gems hunters** find stocks BEFORE they run (flat or negative short-term)

The key insight: **Strong fundamentals eventually drive price appreciation, but there's a lag.** During this lag period, you can accumulate at discounted prices.

---

## 🎯 The REDSTAREX Case Study

**Why this strategy works:**

| Date | Status | Metrics | Action |
|------|--------|---------|--------|
| **Dec 17, 2025** | ❄️ Cold | Price: ₦9.20<br>1M: 0%<br>3M: -16%<br>EPS Growth: +159% | **BUY** - Strong fundamentals, weak momentum |
| **Dec 23, 2025** | ❄️ Cold | Price: ₦9.01 (user entry) | **ACCUMULATE** - Perfect timing |
| **Jan 20, 2026** | 🚀 Running | Price: ₦15.90<br>1M: +78%<br>**Return: +76.5%** | **HOLD/TRIM** - Momentum caught up |

**The pattern:** Fundamentals led price by ~1 month. This delay is your opportunity window.

---

## 🔍 The Four-Step Filter System

### Step 1: Fundamental Strength Filter

**At least ONE of these must be true:**
```python
eps_growth_ttm > 15%        # Earnings growing
OR
revenue_growth_ttm > 15%    # Sales growing
```

**Why:** Growth companies eventually get re-rated by the market. EPS growth >50% is especially powerful.

### Step 2: "Hidden" Filter (Not Hot Yet)

**The stock must be overlooked:**
```python
perf_1m < 15%          # Not rallying hard yet
OR
perf_3m < 20%          # 3-month not crazy either
```

**Why:** If it's already running +30% in a month, you're late. We want stocks BEFORE momentum kicks in.

### Step 3: Quality Filter

**Basic profitability:**
```python
net_margin_ttm > 0     # At least profitable
```

**Optional (stricter):**
```python
roe_ttm > 15%          # Strong return on equity
debt_to_equity < 1.0   # Not overleveraged
```

**Why:** We want undiscovered gems, not value traps. Profitability ensures underlying business quality.

### Step 4: Combine Filters

```python
gems = df[
    (fundamental_filter) &    # Strong growth
    (hidden_filter) &          # Not hot yet  
    (quality_filter)           # Profitable
]
```

---

## 📊 The Scoring System

**Score Components (0-100 scale):**

### 1. Fundamentals (50% weight)
- **EPS Growth TTM** (25%): Primary driver of long-term returns
- **Revenue Growth TTM** (25%): Confirms earnings quality

### 2. Quality Indicators (25% weight)
- **ROE TTM** (15%): Capital efficiency
- **Net Margin TTM** (10%): Profitability strength

### 3. Long-term Momentum (25% weight)
- **1-Year Performance** (15%): Shows underlying strength
- **YTD Performance** (10%): Recent trend confirmation

**Implementation:**
```python
hidden_gem_score = (
    # Clip extreme outliers first
    eps_growth.clip(-100, 500) * 0.25 +
    revenue_growth.clip(-100, 300) * 0.25 +
    roe.clip(-50, 100) * 0.15 +
    margin.clip(-50, 50) * 0.10 +
    perf_1y.clip(-50, 500) * 0.15 +
    perf_ytd.clip(-50, 300) * 0.10
)

# Normalize to 0-100
score = (score - min) / (max - min) * 100
```

**Why this works:**
- Weights fundamentals heavily (50%) since we're looking for undervalued growth
- Long-term performance (1Y) validates the business model
- Short-term momentum is intentionally excluded from scoring

---

## 🌡️ Heat Status Classification

After scoring, classify each gem by "temperature":

| Status | 1-Month Perf | Meaning | Action |
|--------|-------------|---------|---------|
| ❄️ **Cold** | < 0% | **Best entry** - Fundamentals strong, price weak | BUY 10-15% position |
| 🌤️ **Warming** | 0-10% | Good entry - Early momentum building | BUY 8-12% position |
| 🔥 **Hot** | 10-20% | Fair entry - Momentum accelerating | BUY 5-8% position |
| 🚀 **Running** | > 20% | Already discovered - May be late | HOLD or wait for pullback |

**Key insight:** The colder the stock, the better the risk/reward ratio. REDSTAREX was ❄️ Cold when we bought it.

---

## 💎 Output Format

**Essential columns for Hidden Gems report:**

```python
output_columns = [
    'gem_rank',              # 1-20 ranking
    'symbol',
    'description',
    'sector',
    'price',                 # Current price
    'hidden_gem_score',      # 0-100 score
    'heat_status',           # ❄️🌤️🔥🚀
    
    # Momentum (for timing)
    'perf_1m',
    'perf_3m', 
    'perf_6m',
    'perf_1y',
    
    # Fundamentals (for conviction)
    'eps_growth_ttm',
    'revenue_growth_ttm',
    'roe_ttm',
    'net_margin_ttm',
    
    # Risk indicators
    'debt_to_equity',
    'volume_1d',
    'liquidity_value_1d',
]
```

---

## 🔄 Bonus: Value Turnaround Candidates

**A complementary strategy for contrarian plays:**

### Filter Criteria:
```python
beaten_down = perf_3m < 0      # Negative 3-month (everyone selling)
profitable = net_margin > 0     # But still making money
has_history = perf_1y.notna()   # Proven track record
```

### Scoring:
```python
turnaround_score = (
    eps_growth * 0.30 +
    revenue_growth * 0.20 +
    roe * 0.20 +
    margin * 0.15 +
    perf_1m.clip(-30, 0) * 0.15  # Penalize steep falls
)
```

**Use case:** Higher risk, higher reward. These are "falling knives" but with fundamental support. Size smaller (5-8% positions).

---

## 🛠️ Implementation Checklist

### Required Data Fields:
- [ ] `price` - Current stock price
- [ ] `perf_1m`, `perf_3m`, `perf_6m`, `perf_1y`, `perf_ytd` - Price returns
- [ ] `eps_growth_ttm` - Earnings growth (trailing twelve months)
- [ ] `revenue_growth_ttm` - Revenue growth TTM
- [ ] `roe_ttm` - Return on equity TTM
- [ ] `net_margin_ttm` - Net profit margin TTM
- [ ] `debt_to_equity` - Leverage ratio
- [ ] `volume_1d` - Daily trading volume
- [ ] `sector`, `description` - Classification

### Python Implementation:
```python
def find_hidden_gems(df, top_n=20):
    # Step 1: Filter fundamentally strong stocks
    fundamental = (
        (df['eps_growth_ttm'] > 15) | 
        (df['revenue_growth_ttm'] > 15)
    )
    
    # Step 2: Filter "hidden" (not hot yet)
    hidden = (
        (df['perf_1m'] < 15) | 
        (df['perf_3m'] < 20)
    )
    
    # Step 3: Quality filter
    quality = (df['net_margin_ttm'] > 0)
    
    # Combine
    gems = df[fundamental & hidden & quality].copy()
    
    # Step 4: Score
    gems['score'] = (
        gems['eps_growth_ttm'].clip(-100, 500) * 0.25 +
        gems['revenue_growth_ttm'].clip(-100, 300) * 0.25 +
        gems['roe_ttm'].clip(-50, 100) * 0.15 +
        gems['net_margin_ttm'].clip(-50, 50) * 0.10 +
        gems['perf_1y'].clip(-50, 500) * 0.15 +
        gems['perf_ytd'].clip(-50, 300) * 0.10
    )
    
    # Normalize
    gems['score'] = (
        (gems['score'] - gems['score'].min()) / 
        (gems['score'].max() - gems['score'].min()) * 100
    )
    
    # Sort and rank
    gems = gems.sort_values('score', ascending=False)
    gems['rank'] = range(1, len(gems) + 1)
    
    # Add heat status
    def heat(perf_1m):
        if perf_1m < 0: return "❄️ Cold"
        elif perf_1m < 10: return "🌤️ Warming"
        elif perf_1m < 20: return "🔥 Hot"
        else: return "🚀 Running"
    
    gems['heat_status'] = gems['perf_1m'].apply(heat)
    
    return gems.head(top_n)
```

---

## 📈 Portfolio Integration Strategy

### Weekly Workflow:
1. **Run Hidden Gems scan** on latest data
2. **Focus on ❄️ Cold opportunities** (best risk/reward)
3. **Research top 5-10 gems** for fundamental validation
4. **Size positions by conviction:**
   - High conviction + Cold = 10-15% of portfolio
   - Medium conviction + Warming = 8-12%
   - Low conviction + Hot = 5-8%
5. **Monitor heat status** - as gems warm up, consider trimming

### Position Management:
```python
if pnl > +50% and heat == "🚀 Running":
    action = "TAKE PROFIT (trim 30-50%)"
elif pnl < -10% and eps_growth > 50:
    action = "ADD MORE (fundamentals intact)"
elif heat == "❄️ Cold" and score > 80:
    action = "ACCUMULATE (perfect setup)"
```

---

## 🌍 Adapting for US Stocks

### Data Source Adjustments:

**Nigerian Stocks (current):**
- Source: Manual CSV from financial websites
- Update: Weekly snapshots
- Fields: Basic fundamentals + price history

**US Stocks (adaptation):**
- **Recommended sources:**
  - `yfinance` - Free, good for price/volume
  - `Alpha Vantage` - Free tier for fundamentals
  - `Financial Modeling Prep` - Better fundamental data
  - `Seeking Alpha` - For earnings estimates
- **Update frequency:** Daily (more liquid market)
- **Additional fields to consider:**
  - `forward_pe` - Market expectations
  - `earnings_surprise` - Analyst beat/miss
  - `institutional_ownership` - Smart money position
  - `short_interest` - Contrarian signal

### Market Differences:

| Factor | Nigerian Market | US Market | Adjustment |
|--------|----------------|-----------|------------|
| **Liquidity** | Lower | Much higher | Can be more aggressive with position sizes |
| **Volatility** | Higher | Lower (generally) | May need tighter stop losses |
| **Information** | Sparse | Abundant | Add sentiment/analyst data to filters |
| **Market Cap** | Small-mid cap focus | All sizes available | Filter by market cap tiers |

### Enhanced US Filters:

```python
# Additional US-specific filters
us_quality_filters = (
    (df['market_cap'] > 100_000_000) &      # Min $100M cap
    (df['avg_volume_3m'] > 100_000) &       # Minimum liquidity
    (df['institutional_ownership'] > 0.05) & # Some smart money
    (df['price'] > 5)                        # Avoid penny stocks
)

# Optional: Sentiment enhancement
if 'analyst_rating' in df.columns:
    # Look for positive revisions while price is weak
    sentiment_boost = (
        (df['analyst_rating_change'] > 0) &  # Upgrades
        (df['perf_1m'] < 5)                   # But price flat
    )
```

### Recommended Parameter Tweaks for US:

```python
# More competitive market = higher bar
find_hidden_gems_us(
    df,
    min_eps_growth=20,        # Higher (vs 15 for NG)
    min_revenue_growth=15,    # Same
    max_1m_perf=10,           # Stricter (vs 15 for NG)
    require_positive_margin=True,
    min_market_cap=100_000_000,  # $100M minimum
    top_n=30                  # More stocks available
)
```

---

## 📊 Success Metrics

**Track your hidden gems performance:**

```python
# When you buy a gem
entry_metrics = {
    'symbol': 'TICKER',
    'entry_date': '2026-01-23',
    'entry_price': 10.00,
    'gem_score': 85.3,
    'heat_status': '❄️ Cold',
    'eps_growth': 159.0,
    'position_size': 0.12  # 12% of portfolio
}

# Review after 1 month, 3 months, 6 months
review_metrics = {
    'days_held': 30,
    'exit_price': 15.90,
    'return_pct': 59.0,
    'heat_at_exit': '🚀 Running',
    'max_drawdown': -5.2,  # Worst drop during hold
}
```

**Target metrics:**
- Win rate: >60% of gems show positive returns in 3 months
- Average winner: +30-50% within 6 months
- Average loser: -10-15% (tight stops)
- Cold gems should outperform Warming/Hot by 2:1

---

## ⚠️ Risk Management

### Position Sizing Rules:
```python
if heat_status == "❄️ Cold" and score > 80:
    max_position = 0.15  # 15% of portfolio
elif heat_status == "🌤️ Warming" and score > 70:
    max_position = 0.12  # 12%
elif heat_status == "🔥 Hot":
    max_position = 0.08  # 8%
else:  # Running
    max_position = 0.05  # 5% or skip
```

### Stop Loss Guidelines:
- **Cold gems:** -15% stop (needs room to work)
- **Warming gems:** -12% stop
- **Hot gems:** -8% stop (less room for error)

### Diversification:
- Minimum 5 hidden gems across different sectors
- Maximum 40% of portfolio in hidden gems strategy
- Keep 20% cash for new opportunities

---

## 🎓 Key Takeaways

1. **Fundamentals lead price** - There's a lag you can exploit
2. **❄️ Cold is best** - Negative sentiment + strong fundamentals = opportunity
3. **Score objectively** - Don't fall in love with a story, trust the numbers
4. **Size for conviction** - Higher score + colder = bigger position
5. **Monitor heat** - As gems warm up, trim and recycle capital
6. **Be patient** - The lag can be 1-3 months. Don't chase.
7. **Review weekly** - New gems emerge, old gems heat up

---

## 📚 References & Resources

**Original implementation:**
- `analysis/hidden_gems.py` - Core logic
- `notebooks/hidden_gems.ipynb` - Interactive analysis
- `backtest_gems.py` - Historical validation

**For US stocks adaptation:**
- Start with `yfinance` for data pipeline
- Consider `FinanceToolkit` library for fundamental ratios
- Use `Pandas` for all calculations (same as Nigerian version)
- Build weekly snapshot system similar to current workflow

**Next steps:**
1. Copy this methodology to `us_stock/docs/`
2. Adapt data ingestion for US market sources
3. Implement `us_stock/analysis/hidden_gems.py`
4. Create `us_stock/notebooks/hidden_gems.ipynb`
5. Backtest on US historical data (2020-2025)
6. Start weekly scanning!

---

**Remember:** The best time to buy a hidden gem is when it's ❄️ Cold and everyone else is ignoring it. The worst time is when it's 🚀 Running and everyone is chasing it.

*"Be fearful when others are greedy, and greedy when others are fearful."* - Warren Buffett

This is that philosophy, quantified.
