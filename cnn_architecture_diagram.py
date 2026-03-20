"""
cnn_architecture_diagram.py
============================
Generates a clean CNN architecture diagram for inclusion in the report.
No TensorFlow required — pure matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(-1.5, 6)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# ─── Color palette ─────────────────────────────────────────────────────────────
C_INPUT    = "#4A90D9"
C_CONV     = "#E67E22"
C_BN       = "#8E44AD"
C_POOL     = "#27AE60"
C_DENSE    = "#C0392B"
C_OUT      = "#16A085"
C_ARROW    = "#BDC3C7"
C_TEXT     = "#ECF0F1"
C_SUBTITLE = "#95A5A6"

def draw_block(ax, x, y, w, h, label, sublabel="", color="#4A90D9", text_color="white"):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor="white",
        facecolor=color, alpha=0.9
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.05, label,    ha="center", va="center",
            fontsize=9, fontweight="bold", color=text_color)
    if sublabel:
        ax.text(x, y - 0.35, sublabel, ha="center", va="center",
                fontsize=7, color=C_SUBTITLE)

def draw_arrow(ax, x1, x2, y=2.5):
    ax.annotate("",
        xy=(x2 - 0.05, y), xytext=(x1 + 0.05, y),
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.5)
    )

# ─── Blocks ────────────────────────────────────────────────────────────────────
blocks = [
    # x     label              sublabel                color
    (1.0,   "Input",           "Spectrogram\n(F×32×1)",  C_INPUT),
    (3.0,   "Conv2D #1",       "32 filters 3×3\nReLU",   C_CONV),
    (4.5,   "BatchNorm\n+MaxPool", "(2×2)",              C_BN),
    (6.5,   "Conv2D #2",       "64 filters 3×3\nReLU",   C_CONV),
    (8.0,   "BatchNorm\n+MaxPool", "(2×2)",              C_BN),
    (10.0,  "Conv2D #3",       "128 filters 3×3\nReLU",  C_CONV),
    (11.5,  "GlobalAvgPool",   "",                        C_POOL),
    (13.0,  "Dense(128)\nDropout 0.3", "ReLU",           C_DENSE),
    (14.5,  "Dense(64)",       "ReLU",                    C_DENSE),
    (15.8,  "Output",          "Price\nPrediction",       C_OUT),
]

y_center = 2.5
for (x, label, sublabel, color) in blocks:
    draw_block(ax, x, y_center, 1.5, 1.2, label, sublabel, color)

# Arrows
xs = [b[0] for b in blocks]
for i in range(len(xs) - 1):
    draw_arrow(ax, xs[i], xs[i+1], y=y_center)

# ─── Title ─────────────────────────────────────────────────────────────────────
ax.text(8.0, 5.2, "CNN Architecture for Stock Price Prediction",
        ha="center", va="center", fontsize=14, fontweight="bold", color=C_TEXT)
ax.text(8.0, 4.7, "Input: STFT Spectrogram Patch  →  Output: Future Closing Price",
        ha="center", va="center", fontsize=10, color=C_SUBTITLE)

# ─── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C_INPUT, "Input"),
    (C_CONV,  "Convolutional Layer"),
    (C_BN,    "BatchNorm + Pooling"),
    (C_POOL,  "Global Avg Pooling"),
    (C_DENSE, "Fully Connected"),
    (C_OUT,   "Output"),
]
for i, (color, label) in enumerate(legend_items):
    patch = mpatches.Patch(color=color, label=label, alpha=0.9)
    ax.add_patch(FancyBboxPatch((0.3 + i * 2.7, -1.2), 0.4, 0.4,
                                boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor="white", lw=0.8))
    ax.text(0.85 + i * 2.7, -1.0, label, fontsize=7.5, color=C_TEXT, va="center")

plt.tight_layout()
plt.savefig("plots/cnn_architecture_diagram.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("✅ Saved → plots/cnn_architecture_diagram.png")
