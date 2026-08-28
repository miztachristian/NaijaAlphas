#!/usr/bin/env python
# coding: utf-8

# # 🛡️ Stop-Loss & Risk Tracker
# 
# Monitor your positions against stop-loss levels to protect your capital.

# In[1]:


# Setup
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from IPython.display import display, HTML

# Add parent directory to path
sys.path.insert(0, str(Path.cwd().parent))

# Configuration — use relative path from project root
DATA_DIR = Path(__file__).parent.parent / 'data' / 'snapshots'

print(f"📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"📁 Data Directory: {DATA_DIR}")


# ## ⚙️ Configure Your Stop-Losses
# 
# **Stop-loss strategies:**
# - **Fixed %**: Sell if price drops X% from your cost
# - **Trailing %**: Sell if price drops X% from recent high
# - **Support level**: Sell if price breaks below key support

# In[2]:


# ==========================================
# 📝 CONFIGURE YOUR STOP-LOSSES HERE
# ==========================================

STOP_LOSS_CONFIG = {
    # Illustrative config so the notebook runs standalone.
    # In production, positions load from PORTFOLIO_DIR/holdings.json (git-ignored).
    'GTCO': {
        'shares': 1000,
        'avg_cost': 50.00,
        'stop_loss_price': 60.00,      # trailing stop
        'stop_loss_pct': None,
        'action': 'TRAIL - lock in gains',
        'notes': 'Sample position'
    },

    'DANGCEM': {
        'shares': 100,
        'avg_cost': 450.00,
        'stop_loss_price': None,
        'stop_loss_pct': -8.0,         # percentage stop
        'action': 'HOLD',
        'notes': 'Sample position'
    },
}


# In[3]:


# Load latest snapshot
all_snaps = sorted(DATA_DIR.glob('*/snapshot.parquet'))
if not all_snaps:
    raise FileNotFoundError(f"No snapshots found in {DATA_DIR}")
latest_snap = all_snaps[-1]
snapshot_date = latest_snap.parent.name

# Warn if data is stale (> 5 days old)
days_old = (datetime.now() - datetime.strptime(snapshot_date, '%Y-%m-%d')).days
if days_old > 5:
    print(f"⚠️  WARNING: Snapshot is {days_old} days old ({snapshot_date}). Prices may be stale!")
else:
    print(f"✅ Snapshot date: {snapshot_date} ({days_old} days old)")

df = pd.read_parquet(latest_snap)
print(f"📊 Loaded: {snapshot_date} ({len(df)} stocks)")


# ## 🚨 Stop-Loss Alert Dashboard

# In[4]:


def check_stop_losses(df, config):
    """Check all positions against their stop-loss levels."""

    alerts = []

    for symbol, pos in config.items():
        row = df[df['symbol'] == symbol]
        if len(row) == 0:
            continue

        current_price = row['price'].values[0]
        stop_price = pos['stop_loss_price']
        cost = pos['avg_cost']
        shares = pos['shares']

        # Calculate metrics
        pnl_pct = ((current_price - cost) / cost) * 100 if cost > 0 else 0
        pnl_value = shares * (current_price - cost)
        # Distance = how far current price is above stop, as % of stop price
        distance_to_stop = ((current_price - stop_price) / stop_price) * 100
        loss_at_stop = shares * (stop_price - cost)

        # Determine alert level
        if current_price <= stop_price:
            status = '🔴 TRIGGERED'
            urgency = 0
        elif distance_to_stop < 5:
            status = '🟠 WARNING'
            urgency = 1
        elif distance_to_stop < 10:
            status = '🟡 WATCH'
            urgency = 2
        else:
            status = '🟢 OK'
            urgency = 3

        alerts.append({
            'symbol': symbol,
            'status': status,
            'urgency': urgency,
            'current_price': current_price,
            'stop_price': stop_price,
            'distance_to_stop': distance_to_stop,
            'cost': cost,
            'pnl_pct': pnl_pct,
            'pnl_value': pnl_value,
            'loss_at_stop': loss_at_stop,
            'action': pos['action'],
            'notes': pos['notes']
        })

    return pd.DataFrame(alerts).sort_values('urgency')

# Run the check
alerts_df = check_stop_losses(df, STOP_LOSS_CONFIG)

print("=" * 80)
print("🛡️ STOP-LOSS ALERT DASHBOARD")
print("=" * 80)
print()

for _, row in alerts_df.iterrows():
    print(f"{row['status']} {row['symbol']}")
    print(f"   Price: ₦{row['current_price']:.2f} | Stop: ₦{row['stop_price']:.2f} | Distance: {row['distance_to_stop']:.1f}%")
    print(f"   P&L: ₦{row['pnl_value']:,.0f} ({row['pnl_pct']:+.1f}%) | Loss at stop: ₦{row['loss_at_stop']:,.0f}")
    print(f"   → {row['action']}")
    print()


# ## 📊 Risk Summary

# In[5]:


# Calculate total risk exposure
triggered = alerts_df[alerts_df['status'] == '🔴 TRIGGERED']
warning = alerts_df[alerts_df['status'] == '🟠 WARNING']
watch = alerts_df[alerts_df['status'] == '🟡 WATCH']

total_current_pnl = alerts_df['pnl_value'].sum()
total_loss_at_stops = alerts_df['loss_at_stop'].sum()

print("=" * 60)
print("📊 RISK SUMMARY")
print("=" * 60)
print(f"🔴 Triggered stops:  {len(triggered)}")
print(f"🟠 Warning (<5%):    {len(warning)}")
print(f"🟡 Watch (<10%):     {len(watch)}")
print(f"🟢 OK:               {len(alerts_df) - len(triggered) - len(warning) - len(watch)}")
print()
print(f"Current P&L (tracked): ₦{total_current_pnl:,.0f}")
print(f"Max loss if all stops hit: ₦{total_loss_at_stops:,.0f}")
print("=" * 60)

if len(triggered) > 0:
    print("\n⚠️ ACTION REQUIRED: You have triggered stop-losses!")
    for sym in triggered['symbol'].values:
        print(f"   → Consider selling {sym}")


# ## 📋 Position Heat Map

# In[6]:


def style_alerts(df):
    """Style the alerts dataframe."""
    display_df = df[['symbol', 'status', 'current_price', 'stop_price', 
                     'distance_to_stop', 'pnl_pct', 'action']].copy()
    display_df.columns = ['Symbol', 'Status', 'Price', 'Stop', 'Dist%', 'P&L%', 'Action']

    def color_pnl(val):
        if val > 0:
            return 'color: green; font-weight: bold'
        elif val < -10:
            return 'color: red; font-weight: bold'
        elif val < 0:
            return 'color: orange'
        return ''

    def color_distance(val):
        if val < 5:
            return 'background-color: #ffcccc'
        elif val < 10:
            return 'background-color: #fff3cd'
        return 'background-color: #d4edda'

    styled = display_df.style\
        .map(color_pnl, subset=['P&L%'])\
        .map(color_distance, subset=['Dist%'])\
        .format({'Price': '₦{:.2f}', 'Stop': '₦{:.2f}', 'Dist%': '{:.1f}%', 'P&L%': '{:+.1f}%'})

    return styled

display(style_alerts(alerts_df))


# ---
# ## 💡 Tips for Managing Risk
# 
# 1. **Never risk more than 2-3% of portfolio on a single trade**
# 2. **Move stops to breakeven once a stock is up 10%+**
# 3. **Trail stops on winners (like WAPCO) to lock in gains**
# 4. **Cut losers quickly, let winners run**
# 5. **Re-run this notebook daily to check your positions**
