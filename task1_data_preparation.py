"""
Task 1: Data Preparation
========================
Collects financial time series data for 3 companies:
  - TCS (Tata Consultancy Services) - NASDAQ/NSE
  - INFY (Infosys)
  - WIPRO

Aligns all variables to a common time scale and normalizes the data.

Run Requirements:
    pip install yfinance pandas numpy matplotlib scikit-learn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ─── Try to import yfinance; fall back to synthetic data ───────────────────────
try:
    import yfinance as yf
    USE_REAL_DATA = True
    print("✅ yfinance found — fetching real data from Yahoo Finance.")
except ImportError:
    USE_REAL_DATA = False
    print("⚠️  yfinance not installed. Using synthetic data that mimics real stock behaviour.")
    print("   Install with: pip install yfinance")

os.makedirs("data",    exist_ok=True)
os.makedirs("plots",   exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─── Configuration ─────────────────────────────────────────────────────────────
TICKERS   = ["TCS.NS", "INFY.NS", "WIPRO.NS"]
NAMES     = ["TCS", "Infosys", "Wipro"]
START     = "2020-01-01"
END       = "2024-12-31"
FEATURES  = ["Close", "Volume", "Open", "High", "Low"]

# ─── Synthetic Data Generator ─────────────────────────────────────────────────
def generate_synthetic_stock(name, n_days=1200, seed=42):
    """Generates a realistic-looking stock price series using GBM + seasonality."""
    rng = np.random.default_rng(seed)
    params = {
        "TCS":     dict(s0=2200, mu=0.0003, sigma=0.015),
        "Infosys": dict(s0=900,  mu=0.0002, sigma=0.016),
        "Wipro":   dict(s0=300,  mu=0.00025, sigma=0.018),
    }
    p = params.get(name, dict(s0=1000, mu=0.0003, sigma=0.015))

    dt      = 1
    returns = rng.normal(p["mu"] * dt, p["sigma"] * np.sqrt(dt), n_days)
    prices  = p["s0"] * np.cumprod(1 + returns)

    # Add mild yearly seasonality
    t         = np.arange(n_days)
    seasonal  = 0.05 * np.sin(2 * np.pi * t / 252)
    prices   *= (1 + seasonal)

    dates   = pd.bdate_range(start=START, periods=n_days)[:n_days]
    volume  = rng.integers(500_000, 5_000_000, n_days).astype(float)
    open_p  = prices * (1 + rng.normal(0, 0.003, n_days))
    high_p  = np.maximum(prices, open_p) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low_p   = np.minimum(prices, open_p) * (1 - np.abs(rng.normal(0, 0.005, n_days)))

    df = pd.DataFrame({
        "Close":  prices,
        "Volume": volume,
        "Open":   open_p,
        "High":   high_p,
        "Low":    low_p,
    }, index=dates)
    df.index.name = "Date"
    return df


# ─── Load / Download Data ─────────────────────────────────────────────────────
raw_data = {}

for ticker, name in zip(TICKERS, NAMES):
    print(f"\n📥 Loading data for {name} ({ticker}) ...")
    if USE_REAL_DATA:
        try:
            df = yf.download(ticker, start=START, end=END, progress=False)
            df = df[FEATURES].dropna()
            print(f"   ✅ Downloaded {len(df)} rows.")
        except Exception as e:
            print(f"   ⚠️  Failed ({e}). Falling back to synthetic data.")
            df = generate_synthetic_stock(name)
    else:
        df = generate_synthetic_stock(name)
        print(f"   🔧 Synthetic: {len(df)} rows generated.")

    raw_data[name] = df
    df.to_csv(f"data/{name}_raw.csv")
    print(f"   💾 Saved → data/{name}_raw.csv")


# ─── Task 1A: Align to Common Time Scale ──────────────────────────────────────
print("\n📐 Aligning all tickers to a common date range...")

common_index = raw_data[NAMES[0]].index
for name in NAMES[1:]:
    common_index = common_index.intersection(raw_data[name].index)

print(f"   Common dates: {common_index[0].date()} → {common_index[-1].date()}  ({len(common_index)} trading days)")

aligned = {}
for name in NAMES:
    aligned[name] = raw_data[name].loc[common_index].copy()

# Build a combined multi-level DataFrame
combined = pd.concat(aligned, axis=1)
combined.columns = [f"{name}_{feat}" for name in NAMES for feat in FEATURES]
combined.to_csv("data/aligned_combined.csv")
print("   💾 Saved → data/aligned_combined.csv")


# ─── Task 1B: Normalization (Min-Max per feature per ticker) ──────────────────
print("\n📊 Normalizing data (Min-Max scaling)...")

from sklearn.preprocessing import MinMaxScaler

normalized = {}
scalers    = {}

for name in NAMES:
    df     = aligned[name].copy()
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    normalized[name] = df_norm
    scalers[name]    = scaler
    df_norm.to_csv(f"data/{name}_normalized.csv")
    print(f"   💾 Saved → data/{name}_normalized.csv")

# Save scalers (min/max) for inverse transform later
scaler_info = {}
for name in NAMES:
    s = scalers[name]
    scaler_info[name] = {
        "data_min":  s.data_min_.tolist(),
        "data_max":  s.data_max_.tolist(),
        "features":  FEATURES,
    }

import json
with open("data/scalers.json", "w") as f:
    json.dump(scaler_info, f, indent=2)
print("   💾 Saved → data/scalers.json")


# ─── Task 1C: Visualizations ──────────────────────────────────────────────────
print("\n🎨 Plotting time series...")

fig, axes = plt.subplots(len(NAMES), 1, figsize=(14, 9), sharex=True)
fig.suptitle("Closing Price Time Series (Normalized)", fontsize=14, fontweight="bold")

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for ax, name, color in zip(axes, NAMES, colors):
    ax.plot(normalized[name].index, normalized[name]["Close"],
            color=color, linewidth=1.0, label=name)
    ax.set_ylabel("Norm. Close", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("plots/task1_time_series.png", dpi=150)
plt.close()
print("   💾 Saved → plots/task1_time_series.png")

# Volume subplot
fig, axes = plt.subplots(len(NAMES), 1, figsize=(14, 9), sharex=True)
fig.suptitle("Trading Volume Over Time (Normalized)", fontsize=14, fontweight="bold")

for ax, name, color in zip(axes, NAMES, colors):
    ax.bar(normalized[name].index, normalized[name]["Volume"],
           color=color, alpha=0.6, width=1, label=f"{name} Volume")
    ax.set_ylabel("Norm. Volume", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("plots/task1_volume.png", dpi=150)
plt.close()
print("   💾 Saved → plots/task1_volume.png")

print("\n✅ Task 1 Complete!")
print("   Files → data/  |  Plots → plots/task1_*.png")
