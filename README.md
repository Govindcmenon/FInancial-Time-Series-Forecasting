# Assignment 2 — Pattern Recognition for Financial Time Series Forecasting

## Overview
This project explores how **time-frequency signal processing** and **deep learning (CNN)** can be combined to predict stock prices using financial time series data.

Three Indian IT stocks are used:
| Company | NSE Ticker | Description |
|---------|-----------|-------------|
| TCS     | TCS.NS    | Tata Consultancy Services |
| Infosys | INFY.NS   | Infosys Limited |
| Wipro   | WIPRO.NS  | Wipro Limited |

---

## Project Structure

```
stock_assignment/
│
├── task1_data_preparation.py     # Task 1: Download, align, normalize data
├── task2_signal_processing.py    # Task 2: FFT, STFT, spectrograms
├── task3_cnn_model.py            # Task 3: CNN design, training, prediction
├── task4_analysis.py             # Task 4: Evaluation, feature impact analysis
├── cnn_architecture_diagram.py  # Standalone CNN diagram generator
├── run_all.py                    # Run ALL tasks in sequence
│
├── data/                         # Auto-created — CSV files
│   ├── TCS_raw.csv
│   ├── Infosys_raw.csv
│   ├── Wipro_raw.csv
│   ├── aligned_combined.csv
│   ├── TCS_normalized.csv
│   ├── Infosys_normalized.csv
│   ├── Wipro_normalized.csv
│   └── scalers.json
│
├── outputs/                      # Auto-created — arrays and reports
│   ├── TCS_spectrogram_Sxx.npy
│   ├── TCS_close_normalized.npy
│   ├── results_summary.json
│   └── results_report.txt
│
├── models/                       # Auto-created — saved CNN weights
│   └── TCS_cnn_model.keras
│
└── plots/                        # Auto-created — all figures
    ├── task1_time_series.png
    ├── task1_volume.png
    ├── task2_fft_spectrum.png
    ├── task2_fft_comparison.png
    ├── task2_spectrograms.png
    ├── task2_TCS_full_view.png
    ├── task3_TCS_training_curve.png
    ├── task3_TCS_prediction.png
    ├── task4_metrics_comparison.png
    ├── task4_scatter_actual_vs_predicted.png
    ├── task4_feature_impact.png
    ├── task4_window_sensitivity.png
    └── cnn_architecture_diagram.png
```

---

## Setup

### 1. Install Python (3.9+)
Download from https://www.python.org/downloads/

### 2. Install Dependencies

```bash
pip install numpy pandas scipy matplotlib scikit-learn yfinance tensorflow
```

> **Note:** `yfinance` fetches real market data from Yahoo Finance.
> If it is unavailable (e.g., no internet), the scripts automatically fall back
> to **synthetic data** that mirrors real stock behaviour.

> **Note:** `tensorflow` is required for the CNN model in Task 3.
> Without it, a NumPy-based ridge regression baseline runs instead.

---

## How to Run

### Option A — Run Everything at Once
```bash
cd stock_assignment
python run_all.py
```

### Option B — Run Each Task Separately
```bash
python task1_data_preparation.py
python task2_signal_processing.py
python task3_cnn_model.py
python task4_analysis.py
python cnn_architecture_diagram.py    # optional: architecture figure
```

---

## Task Descriptions

### Task 1: Data Preparation
- Downloads closing price, open, high, low, and volume for TCS, Infosys, and Wipro
  from Yahoo Finance (2020–2024)
- Aligns all tickers to a **common date range** (intersection of trading days)
- Applies **Min-Max normalization** per feature per stock
- Saves raw and normalized CSVs to `data/`
- Generates `plots/task1_time_series.png` and `plots/task1_volume.png`

### Task 2: Signal Processing
The closing price signal is treated as a 1-D time series.

#### 2a. Fourier Transform (FFT)
Decomposes the full signal into frequency components.
- Horizontal axis: frequency (cycles/year)
- Vertical axis: amplitude
- Dominant peaks correspond to **annual and quarterly cycles**

#### 2b. Short-Time Fourier Transform (STFT)
Computes frequency content over a **sliding window** to capture
time-varying behaviour in the non-stationary financial signal.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Window Length (L) | 64 | Samples per segment |
| Hop Size (H) | 8 | Shift between windows |
| Overlap | 56 | L − H |
| Window type | Hann | Reduces spectral leakage |

The spectrogram S(t, f) = |STFT(t, f)|² is saved as a 2D array
and treated as an **image** for CNN input.

### Task 3: CNN Model

**Input:** Spectrogram patch of shape `(n_freq, 32, 1)` — a 32-column slice of the spectrogram

**Output:** Normalized closing price 5 trading days ahead

**Architecture:**
```
Input (F × 32 × 1)
    → Conv2D(32, 3×3, ReLU) + BatchNorm + MaxPool(2×2)
    → Conv2D(64, 3×3, ReLU) + BatchNorm + MaxPool(2×2)
    → Conv2D(128, 3×3, ReLU) + BatchNorm + GlobalAvgPool
    → Dense(128, ReLU) + Dropout(0.3)
    → Dense(64, ReLU)
    → Dense(1)                          ← predicted price
```

**Training details:**
| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss function | MSE |
| Batch size | 32 |
| Epochs | 50 (early stop) |
| Train/Val/Test | 64% / 16% / 20% |

### Task 4: Analysis
- Bar chart comparing **MSE, RMSE, MAE** across all three stocks
- Scatter plot of **Actual vs Predicted** values
- **Feature ablation study**: Close-only vs OHLCV
- **Window length sensitivity**: L = 16, 32, 64, 128, 256

---

## Key Equations

### STFT
$$\text{STFT}(t, f) = \int_{-\infty}^{\infty} X(\tau)\, w(\tau - t)\, e^{-j2\pi f\tau}\, d\tau$$

### Spectrogram
$$S(t, f) = |\text{STFT}(t, f)|^2$$

### Prediction Model
$$\hat{p}(t + \Delta t) = f_\theta(S_t)$$

where $f_\theta$ is the trained CNN.

---

## Data Sources
| Source | URL |
|--------|-----|
| Yahoo Finance | https://finance.yahoo.com |
| NSE India | https://www.nseindia.com |
| BSE India | https://www.bseindia.com |
| Kaggle | https://www.kaggle.com |

---

## Expected Figures (for Report)

| Figure | File | Task |
|--------|------|------|
| Time series plot | task1_time_series.png | Task 1 |
| Volume plot | task1_volume.png | Task 1 |
| FFT Spectrum (each stock) | task2_fft_spectrum.png | Task 2 |
| Spectrograms | task2_spectrograms.png | Task 2 |
| Full signal view | task2_TCS_full_view.png | Task 2 |
| CNN architecture | cnn_architecture_diagram.png | Task 3 |
| Training curves | task3_*_training_curve.png | Task 3 |
| Prediction vs actual | task3_*_prediction.png | Task 3 |
| Metrics comparison | task4_metrics_comparison.png | Task 4 |
| Scatter: actual vs pred | task4_scatter_actual_vs_predicted.png | Task 4 |
| Feature impact | task4_feature_impact.png | Task 4 |
| Window sensitivity | task4_window_sensitivity.png | Task 4 |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: yfinance` | `pip install yfinance` — or let it use synthetic data |
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` — or numpy baseline runs instead |
| Yahoo Finance rate limit / timeout | Re-run after a few minutes; synthetic fallback activates automatically |
| CUDA/GPU issues | Add `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` at top of task3 to force CPU |

---

## References
1. Y. Zhang and C. Aggarwal, "Stock Market Prediction Using Deep Learning," IEEE Access.
2. A. Tsantekidis et al., "Deep Learning for Financial Time Series Forecasting."
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
4. A. Borovykh et al., "Conditional Time Series Forecasting with CNNs."
