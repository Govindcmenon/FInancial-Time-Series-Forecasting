"""
Task 2: Signal Processing
=========================
- Applies Fourier Transform (FFT) to each stock's closing price signal
- Generates Spectrograms using Short-Time Fourier Transform (STFT)
- Visualizes frequency spectra and spectrograms

Run AFTER task1_data_preparation.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import stft, get_window
from scipy.fft import fft, fftfreq

os.makedirs("plots",   exist_ok=True)
os.makedirs("outputs", exist_ok=True)

NAMES     = ["TCS", "Infosys", "Wipro"]
COLORS    = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# STFT parameters
WINDOW_LENGTH = 64    # L: number of samples per window
HOP_SIZE      = 8     # H: step size between windows
WINDOW_TYPE   = "hann"
FS            = 1     # 1 sample per trading day

# ─── Load Normalized Data ──────────────────────────────────────────────────────
print("📂 Loading normalized data...")
normalized = {}
for name in NAMES:
    df = pd.read_csv(f"data/{name}_normalized.csv", index_col=0, parse_dates=True)
    normalized[name] = df
    print(f"   {name}: {len(df)} rows")

print(f"\n⚙️  STFT Parameters:")
print(f"   Window Length (L) = {WINDOW_LENGTH}")
print(f"   Hop Size      (H) = {HOP_SIZE}")
print(f"   Overlap           = {WINDOW_LENGTH - HOP_SIZE}")
print(f"   Window Type       = {WINDOW_TYPE}")


# ─── Task 2A: Fourier Transform (Frequency Spectrum) ─────────────────────────
print("\n🔬 Computing FFT (Frequency Spectrum)...")

fig, axes = plt.subplots(len(NAMES), 1, figsize=(14, 9))
fig.suptitle("Frequency Spectrum of Closing Price (FFT)", fontsize=14, fontweight="bold")

fft_results = {}

for ax, name, color in zip(axes, NAMES, COLORS):
    signal = normalized[name]["Close"].values
    N      = len(signal)

    # Detrend: remove mean to focus on oscillations
    signal_detrended = signal - np.mean(signal)

    # Compute FFT
    yf = fft(signal_detrended)
    xf = fftfreq(N, d=1.0 / FS)   # frequency in cycles/day

    # Positive frequencies only
    pos_mask     = xf > 0
    freqs        = xf[pos_mask]
    amplitudes   = (2.0 / N) * np.abs(yf[pos_mask])

    # Convert to cycles/year for readability
    freqs_year   = freqs * 252

    fft_results[name] = {"freqs": freqs_year, "amplitudes": amplitudes}

    ax.plot(freqs_year, amplitudes, color=color, linewidth=0.8, alpha=0.9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_xlabel("Frequency (cycles/year)", fontsize=9)
    ax.set_title(f"{name} — Frequency Spectrum", fontsize=10)
    ax.set_xlim([0, 100])
    ax.grid(True, alpha=0.3)

    # Annotate top peaks
    top_n = 5
    top_idx = np.argsort(amplitudes)[-top_n:][::-1]
    for idx in top_idx:
        ax.annotate(
            f"{freqs_year[idx]:.1f}",
            xy=(freqs_year[idx], amplitudes[idx]),
            xytext=(0, 6), textcoords="offset points",
            fontsize=7, ha="center", color=color
        )

plt.tight_layout()
plt.savefig("plots/task2_fft_spectrum.png", dpi=150)
plt.close()
print("   💾 Saved → plots/task2_fft_spectrum.png")


# ─── Task 2B: Combined Spectrum Comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_title("Frequency Spectrum Comparison — All Stocks", fontsize=13, fontweight="bold")

for name, color in zip(NAMES, COLORS):
    r = fft_results[name]
    ax.plot(r["freqs"], r["amplitudes"], color=color, linewidth=0.9, alpha=0.8, label=name)

ax.set_xlabel("Frequency (cycles/year)", fontsize=10)
ax.set_ylabel("Amplitude", fontsize=10)
ax.set_xlim([0, 100])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/task2_fft_comparison.png", dpi=150)
plt.close()
print("   💾 Saved → plots/task2_fft_comparison.png")


# ─── Task 2C: Spectrogram via STFT ────────────────────────────────────────────
print("\n🌈 Computing Spectrograms (STFT)...")

spectrograms = {}   # {name: (f, t, Sxx)}

fig, axes = plt.subplots(len(NAMES), 1, figsize=(14, 12))
fig.suptitle("Spectrograms (STFT) of Closing Price", fontsize=14, fontweight="bold")

for ax, name, color in zip(axes, NAMES, COLORS):
    signal = normalized[name]["Close"].values
    N      = len(signal)

    # Scipy STFT
    f, t, Zxx = stft(
        signal,
        fs=FS,
        window=WINDOW_TYPE,
        nperseg=WINDOW_LENGTH,
        noverlap=WINDOW_LENGTH - HOP_SIZE,
        boundary=None,
        padded=False
    )

    # Power spectrogram
    Sxx = np.abs(Zxx) ** 2

    spectrograms[name] = {"f": f, "t": t, "Sxx": Sxx}

    # Convert t from sample index to approximate day index
    t_days = t * 1.0

    im = ax.pcolormesh(
        t_days, f * 252, 10 * np.log10(Sxx + 1e-10),
        shading="gouraud",
        cmap="inferno"
    )
    plt.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_ylabel("Frequency (cycles/year)", fontsize=9)
    ax.set_title(f"{name} Spectrogram", fontsize=10)
    ax.set_ylim([0, 50])

axes[-1].set_xlabel("Time (trading days from start)", fontsize=10)
plt.tight_layout()
plt.savefig("plots/task2_spectrograms.png", dpi=150)
plt.close()
print("   💾 Saved → plots/task2_spectrograms.png")


# ─── Task 2D: Individual high-res spectrogram ─────────────────────────────────
for name in NAMES:
    r   = spectrograms[name]
    Sxx = r["Sxx"]
    f   = r["f"] * 252
    t   = r["t"]

    fig, axes2 = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"{name} — Full Signal Processing View", fontsize=13, fontweight="bold")

    # Time series
    axes2[0].plot(normalized[name]["Close"].values, color=COLORS[NAMES.index(name)], linewidth=0.8)
    axes2[0].set_title("Normalized Closing Price (Time Domain)")
    axes2[0].set_ylabel("Amplitude")
    axes2[0].grid(True, alpha=0.3)

    # Frequency spectrum
    r2 = fft_results[name]
    axes2[1].plot(r2["freqs"], r2["amplitudes"], color="purple", linewidth=0.8)
    axes2[1].set_title("Frequency Spectrum (FFT)")
    axes2[1].set_ylabel("Amplitude")
    axes2[1].set_xlabel("Frequency (cycles/year)")
    axes2[1].set_xlim([0, 80])
    axes2[1].grid(True, alpha=0.3)

    # Spectrogram
    im2 = axes2[2].pcolormesh(
        t, f, 10 * np.log10(Sxx + 1e-10),
        shading="gouraud", cmap="inferno"
    )
    plt.colorbar(im2, ax=axes2[2], label="Power (dB)")
    axes2[2].set_title("Spectrogram (STFT)")
    axes2[2].set_ylabel("Frequency (cycles/year)")
    axes2[2].set_xlabel("Time (trading days)")
    axes2[2].set_ylim([0, 50])

    plt.tight_layout()
    plt.savefig(f"plots/task2_{name}_full_view.png", dpi=150)
    plt.close()
    print(f"   💾 Saved → plots/task2_{name}_full_view.png")


# ─── Task 2E: Save Spectrogram Arrays for Task 3 ──────────────────────────────
print("\n💾 Saving spectrogram arrays for CNN training...")

for name in NAMES:
    r = spectrograms[name]
    np.save(f"outputs/{name}_spectrogram_f.npy",   r["f"])
    np.save(f"outputs/{name}_spectrogram_t.npy",   r["t"])
    np.save(f"outputs/{name}_spectrogram_Sxx.npy", r["Sxx"])

# Also save the close price arrays (targets for CNN)
for name in NAMES:
    close = normalized[name]["Close"].values
    np.save(f"outputs/{name}_close_normalized.npy", close)

print("   💾 Spectrogram .npy files saved in outputs/")
print("\n✅ Task 2 Complete!")
print("   Plots → plots/task2_*.png")
print("   Arrays → outputs/*_spectrogram_*.npy")
