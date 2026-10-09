import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# =========================
# 1️⃣ Setup Dataset & Label
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_fix.csv")

# -------------------------
# Fungsi Hitung EWMA & Z-score (untuk analisis tren)
# -------------------------
def ewma(series, span=10):
    """Menghitung rata-rata bergerak eksponensial (EWMA)."""
    return series.ewm(span=span, adjust=False).mean()

def zscore(series):
    """Menghitung Z-score standar."""
    return (series - series.mean()) / series.std(ddof=0)

# -------------------------
# Labeling menggunakan Percentile (utama)
# -------------------------
def label_from_row(df, sentiment):
    """
    Melabeli data menggunakan metode percentile:
    - ≥ P90  → Pump
    - ≤ P10  → Dump
    - sentimen positif di tengah → SellTheNews
    - sisanya → Stable
    """
    df = df.copy()
    if df.empty:
        return df

    p90 = np.percentile(df["change"], 90)
    p10 = np.percentile(df["change"], 10)

    labels = []
    for _, row in df.iterrows():
        c = row["change"]
        s = row["sentiment"]
        if c >= p90:
            labels.append(1)  # Pump
        elif c <= p10:
            labels.append(2)  # Dump
        elif -0.3 < s < 0.3 and abs(c) < (p90 - p10) / 3:
            labels.append(3)  # SellTheNewsSmall
        elif s > 0.05 and abs(c) < (p90 - p10) / 3:
            labels.append(4)  # SellTheNewsBig
        else:
            labels.append(0)  # Stable

    df["label"] = labels
    return df

# -------------------------
# Bangun dataset dari CSV
# -------------------------
def build_dataset():
    df = pd.read_csv(DATA_PATH)

    # Pastikan kolom penting ada
    for col in ['change', 'volatility', 'volume', 'sentiment']:
        if col not in df.columns:
            df[col] = 0.0

    # Urutkan berdasarkan waktu
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

    # Label utama dengan percentile
    if 'label' not in df.columns:
        df = label_from_row(df, df['sentiment'].mean())

    # Tambahan analisis tren (tidak dipakai untuk training RF)
    df["ewma_change"] = ewma(df["change"])
    df["ewma_volatility"] = ewma(df["volatility"])
    df["z_change"] = zscore(df["change"])
    df["z_volatility"] = zscore(df["volatility"])

    return df

# =========================
# 2️⃣ Train Random Forest
# =========================
def train_random_forest(df):
    X = df[["change", "volatility", "volume", "sentiment"]]
    y = df["label"]

    # Oversampling minor label agar dataset seimbang
    train_df = pd.concat([X, y], axis=1)
    df_majority = train_df[train_df.label == train_df.label.value_counts().idxmax()]
    dfs_minority = []

    for l in train_df.label.unique():
        if l == df_majority.label.iloc[0]:
            continue
        df_min = train_df[train_df.label == l]
        n_samples = max(int(len(df_majority) * 0.3), len(df_min), 20)
        df_min_up = resample(df_min, replace=True, n_samples=n_samples, random_state=42)
        dfs_minority.append(df_min_up)

    train_bal = pd.concat([df_majority] + dfs_minority).sample(frac=1, random_state=42)
    X_train_bal = train_bal[["change", "volatility", "volume", "sentiment"]]
    y_train_bal = train_bal["label"]

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_bal, y_train_bal)
    return rf

# =========================
# 3️⃣ Flask App
# =========================
app = Flask(__name__)

df = build_dataset()
rf_model = train_random_forest(df)

label_mapping = {
    0: "Stable 😐",
    1: "Pump 🚀",
    2: "Dump 💥",
    3: "SellTheNewsSmall ⚠️",
    4: "SellTheNewsBig 🚨"
}

color_mapping = {
    0: "green",
    1: "blue",
    2: "red",
    3: "orange",
    4: "darkred"
}

@app.route('/')
def index():
    coins = sorted(df['cryptocurrency'].unique()) if 'cryptocurrency' in df.columns else []
    return render_template("index_dataset_percen&EWMA.html", coins=coins)

@app.route('/detect', methods=['POST'])
def detect():
    coin = request.form.get('coin')
    if 'cryptocurrency' not in df.columns:
        return jsonify({"rf_status": "Invalid dataset", "rule_status": "-", "latest_news": []})

    df_coin = df[df['cryptocurrency'] == coin].copy()
    if df_coin.empty:
        return jsonify({"rf_status": "No data", "rule_status": "-", "latest_news": []})

    # Prediksi Random Forest
    features = df_coin[["change", "volatility", "volume", "sentiment"]].values
    rf_preds = rf_model.predict(features)
    df_coin['rf_label'] = [label_mapping[p] for p in rf_preds]
    df_coin['rf_color'] = [color_mapping[p] for p in rf_preds]

    # Ambil status RF terakhir
    rf_status = df_coin['rf_label'].iloc[-1]

    # Analisis tambahan: rata-rata Z-score & EWMA terakhir
    trend_analysis = {
        "avg_z_change": round(df_coin["z_change"].iloc[-10:].mean(), 3),
        "avg_z_volatility": round(df_coin["z_volatility"].iloc[-10:].mean(), 3),
        "ewma_change_last": round(df_coin["ewma_change"].iloc[-1], 4),
        "ewma_volatility_last": round(df_coin["ewma_volatility"].iloc[-1], 4)
    }

    latest_news = df_coin.to_dict(orient="records")

    return jsonify({
        "rf_status": rf_status,
        "trend_analysis": trend_analysis,
        "latest_news": latest_news
    })

if __name__ == "__main__":
    app.run(debug=True)
