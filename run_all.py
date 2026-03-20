"""
run_all.py
==========
Runs all four assignment tasks in sequence.

Usage:
    python run_all.py
"""

import subprocess
import sys
import time

tasks = [
    ("Task 1 — Data Preparation",   "task1_data_preparation.py"),
    ("Task 2 — Signal Processing",  "task2_signal_processing.py"),
    ("Task 3 — CNN Model Training", "task3_cnn_model.py"),
    ("Task 4 — Analysis",           "task4_analysis.py"),
]

print("=" * 60)
print("  STOCK PRICE FORECASTING ASSIGNMENT — FULL PIPELINE")
print("=" * 60)

for title, script in tasks:
    print(f"\n{'─'*60}")
    print(f"▶  {title}")
    print(f"{'─'*60}")
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n❌ {script} failed (code {result.returncode}). Fix and re-run.")
        sys.exit(1)
    print(f"\n⏱  Completed in {elapsed:.1f}s")

print("\n" + "=" * 60)
print("  ✅ ALL TASKS COMPLETE")
print("=" * 60)
print("\nOutputs:")
print("  data/     — raw and normalized CSV files")
print("  plots/    — all figures (time series, FFT, spectrograms, CNN)")
print("  models/   — saved CNN model(s)")
print("  outputs/  — spectrogram arrays, results, report")
