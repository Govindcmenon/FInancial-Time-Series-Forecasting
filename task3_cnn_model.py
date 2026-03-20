"""
Task 3: CNN Model Development
==============================
- Builds spectrogram patches → target price samples
- Designs a 2D CNN regression model
- Trains on spectrograms to predict future stock price
- Saves trained model and predictions

Run AFTER task1 and task2.

Requirements:
    pip install tensorflow scikit-learn numpy matplotlib
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import json

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

NAMES         = ["TCS", "Infosys", "Wipro"]
FUTURE_STEPS  = 5
PATCH_WIDTH   = 32
TEST_RATIO    = 0.2
EPOCHS        = 50
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3

# ─── TensorFlow ───────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    USE_TF = True
    print(f"✅ TensorFlow {tf.__version__} found.")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
except ImportError:
    USE_TF = False
    print("⚠️ TensorFlow not installed. Using NumPy model.")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ─── Dataset Builder ──────────────────────────────────────
def build_dataset(Sxx, close_prices):
    n_freq, n_time = Sxx.shape

    Sxx = 10 * np.log10(Sxx + 1e-10)
    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min() + 1e-8)

    n_close = len(close_prices)
    t_map = np.linspace(0, n_close - 1, n_time)

    X, y = [], []

    for i in range(n_time - PATCH_WIDTH - 1):
        patch = Sxx[:, i:i+PATCH_WIDTH]
        idx = int(t_map[i + PATCH_WIDTH]) + FUTURE_STEPS

        if idx >= n_close:
            break

        X.append(patch[..., np.newaxis])
        y.append(close_prices[idx])

    return np.array(X), np.array(y)

# ─── CNN Model ────────────────────────────────────────────
def build_model(shape):
    inp = layers.Input(shape=shape)

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='mse',
                  metrics=['mae'])
    return model

# ─── NumPy fallback ───────────────────────────────────────
class SimpleModel:
    def fit(self, X, y):
        self.W = np.linalg.pinv(X.reshape(len(X), -1)) @ y

    def predict(self, X):
        return X.reshape(len(X), -1) @ self.W

    def save(self, path):
        np.save(path, self.W)

# ─── MAIN ─────────────────────────────────────────────────
results = {}

for name in NAMES:
    print(f"\n==== Training {name} ====")

    # ✅ FIXED FILE NAMES
    Sxx   = np.load(f"outputs/{name}_Sxx.npy")
    close = np.load(f"outputs/{name}_close.npy")

    X, y = build_dataset(Sxx, close)

    if len(X) == 0:
        print("❌ No data, skipping...")
        continue

    print(f"Data: X={X.shape}, y={y.shape}")

    split = int(len(X)*(1-TEST_RATIO))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False
    )

    if USE_TF:
        model = build_model(X.shape[1:])

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )

        model.save(f"models/{name}.keras")
        train_loss = history.history["loss"]
        val_loss   = history.history["val_loss"]

    else:
        model = SimpleModel()
        model.fit(X_tr, y_tr)

        train_loss = []
        val_loss = []

    y_pred = model.predict(X_test).flatten()

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(y_pred - y_test))

    print(f"MSE={mse:.6f} RMSE={rmse:.6f} MAE={mae:.6f}")

    results[name] = {"mse":float(mse),"rmse":float(rmse),"mae":float(mae)}

    # Plot
    plt.plot(y_test[:200], label="Actual")
    plt.plot(y_pred[:200], label="Pred")
    plt.legend()
    plt.savefig(f"plots/task3_{name}.png")
    plt.close()

# Save results
with open("outputs/results.json","w") as f:
    json.dump(results,f,indent=2)

print("\n✅ TASK 3 DONE")