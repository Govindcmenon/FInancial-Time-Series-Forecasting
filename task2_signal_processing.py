import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.fft import fft, fftfreq

os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

NAMES  = ["TCS", "Infosys", "Wipro"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

WINDOW_LENGTH = 64
HOP_SIZE      = 8
WINDOW_TYPE   = "hann"
FS            = 1

# ─── Load Data ─────────────────────────────────────────────
print("📂 Loading normalized data...")
normalized = {}

for name in NAMES:
    df = pd.read_csv(f"data/{name}_normalized.csv", index_col=0, skiprows=1)

    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors='coerce')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    normalized[name] = df
    print(f"   {name}: {len(df)} rows")

print("\n⚙️ STFT Parameters:")
print(f"L={WINDOW_LENGTH}, H={HOP_SIZE}, Overlap={WINDOW_LENGTH - HOP_SIZE}")

# ─── FFT ───────────────────────────────────────────────────
print("\n🔬 Computing FFT...")

fig, axes = plt.subplots(len(NAMES), 1, figsize=(14, 9))
fft_results = {}

for ax, name, color in zip(axes, NAMES, COLORS):
    signal = normalized[name].iloc[:, 0].values
    N = len(signal)

    signal = signal - np.mean(signal)

    yf = fft(signal)
    xf = fftfreq(N, d=1.0 / FS)

    mask = xf > 0
    freqs = xf[mask] * 252
    amps  = (2.0 / N) * np.abs(yf[mask])

    fft_results[name] = {"freqs": freqs, "amps": amps}

    ax.plot(freqs, amps, color=color)
    ax.set_title(name)
    ax.grid()

plt.tight_layout()
plt.savefig("plots/task2_fft_spectrum.png")
plt.close()

# ─── STFT ──────────────────────────────────────────────────
print("\n🌈 Computing STFT...")

spectrograms = {}

fig, axes = plt.subplots(len(NAMES), 1, figsize=(14, 12))

for ax, name, color in zip(axes, NAMES, COLORS):
    signal = normalized[name].iloc[:, 0].values

    f, t, Zxx = stft(signal, fs=FS,
                     window=WINDOW_TYPE,
                     nperseg=WINDOW_LENGTH,
                     noverlap=WINDOW_LENGTH - HOP_SIZE)

    Sxx = np.abs(Zxx) ** 2
    spectrograms[name] = {"f": f, "t": t, "Sxx": Sxx}

    im = ax.pcolormesh(t, f * 252, 10*np.log10(Sxx+1e-10), shading="gouraud")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("plots/task2_spectrograms.png")
plt.close()

# ─── FULL VIEW ─────────────────────────────────────────────
for name in NAMES:
    fig, axes2 = plt.subplots(3, 1, figsize=(12, 8))

    signal = normalized[name].iloc[:, 0].values

    # Time
    axes2[0].plot(signal)
    axes2[0].set_title("Time Domain")

    # FFT
    axes2[1].plot(fft_results[name]["freqs"], fft_results[name]["amps"])
    axes2[1].set_title("FFT")

    # Spectrogram
    r = spectrograms[name]
    im = axes2[2].pcolormesh(r["t"], r["f"]*252,
                             10*np.log10(r["Sxx"]+1e-10),
                             shading="gouraud")
    plt.colorbar(im, ax=axes2[2])

    plt.tight_layout()
    plt.savefig(f"plots/task2_{name}.png")
    plt.close()

# ─── SAVE ARRAYS ───────────────────────────────────────────
print("\n💾 Saving outputs...")

for name in NAMES:
    r = spectrograms[name]

    np.save(f"outputs/{name}_Sxx.npy", r["Sxx"])
    np.save(f"outputs/{name}_f.npy", r["f"])
    np.save(f"outputs/{name}_t.npy", r["t"])

    close = normalized[name].iloc[:, 0].values
    np.save(f"outputs/{name}_close.npy", close)

print("\n✅ Task 2 COMPLETE")