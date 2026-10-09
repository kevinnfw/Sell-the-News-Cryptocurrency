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
# Fungsi Hitung EWMA Z-Score
# -------------------------
def ewma_zscore(series, span=10):
    """Menghitung Z-score berbasis EWMA."""
    mean_ewma = series.ewm(span=span, adjust=False).mean()
    std_ewma = series.ewm(span=span, adjust=False).std()
    zscore = (series - mean_ewma) / std_ewma
    return zscore.fillna(0)

# -------------------------
# Labeling menggunakan EWMA Z-score
# -------------------------
def label_from_row(df, sentiment):
    df = df.copy()
    if df.empty:
        return df

    # Hitung Z-score EWMA untuk kolom 'change' dan 'volatility'
    df["z_change"] = ewma_zscore(df["change"])
    df["z_volatility"] = ewma_zscore(df["volatility"])

    labels = []
    for _, row in df.iterrows():
        zc = row["z_change"]
        zv = row["z_volatility"]
        s = sentiment

        # Logika labeling berdasarkan kombinasi Z-score & sentimen
        if zc > 1.5 and s > 0.05:
            labels.append(1)  # Pump
        elif zc < -1.5 and s < -0.05:
            labels.append(2)  # Dump
        elif -0.5 < zc < 0.5 and s > 0.03 and zv <= 0.2:
            labels.append(3)  # SellTheNewsSmall
        elif -0.5 < zc < 0.5 and s > 0.03 and zv > 0.2:
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

    # Gunakan rata-rata sentimen untuk labeling global
    sentiment_value = df['sentiment'].mean() if 'sentiment' in df.columns else 0.0

    # Buat label jika belum ada
    if 'label' not in df.columns:
        df = label_from_row(df, sentiment_value)

    return df

# =========================
# 2️⃣ Train Random Forest
# =========================
def train_random_forest(df):
    X = df[["change", "volatility", "volume", "sentiment"]]
    y = df["label"]

    # Oversampling minor label
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
    return render_template("index_dataset_EWMA.html", coins=coins)

@app.route('/detect', methods=['POST'])
def detect():
    coin = request.form.get('coin')
    if 'cryptocurrency' not in df.columns:
        return jsonify({"rf_status": "Invalid dataset", "rule_status": "-", "latest_news": []})

    df_coin = df[df['cryptocurrency'] == coin].copy()
    if df_coin.empty:
        return jsonify({"rf_status": "No data", "rule_status": "-", "latest_news": []})

    # Prediksi RF untuk semua baris
    features = df_coin[["change", "volatility", "volume", "sentiment"]].values
    rf_preds = rf_model.predict(features)
    df_coin['rf_label'] = [label_mapping[p] for p in rf_preds]
    df_coin['rf_color'] = [color_mapping[p] for p in rf_preds]

    # Ambil status RF terakhir
    rf_status = df_coin['rf_label'].iloc[-1]
    latest_news = df_coin.to_dict(orient="records")

    return jsonify({
        "rf_status": rf_status,
        "rule_status": "-",
        "latest_news": latest_news
    })

if __name__ == "__main__":
    app.run(debug=True)
