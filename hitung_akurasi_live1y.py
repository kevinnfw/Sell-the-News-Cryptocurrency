import pandas as pd
import numpy as np
import requests
from binance.client import Client
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from textblob import TextBlob
import datetime

# ==========================================================
# 1️⃣ NEWS SENTIMENT (ambil 30 hari terakhir + proporsi sentimen)
# ==========================================================
def get_news_with_real_sentiment():
    print("📰 Mengambil news real-time dari NewsAPI (30 hari terakhir)...")
    api_key = "34225cf5d726481eb90e67cbb8d04b57"
    url = "https://newsapi.org/v2/everything"
    query = "crypto OR bitcoin OR ethereum OR xrp OR dogecoin"

    # Gunakan timezone-aware UTC datetime
    today = datetime.datetime.now(datetime.UTC).date()
    from_date = today - datetime.timedelta(days=30)
    params = {
        "q": query,
        "language": "en",
        "pageSize": 50,
        "sortBy": "publishedAt",
        "from": from_date.isoformat(),
        "to": today.isoformat(),
        "apiKey": api_key
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        sentiments = []

        for art in data.get("articles", []):
            text = f"{art.get('title','')}. {art.get('description','')}"
            if text.strip():
                sentiments.append(TextBlob(text).sentiment.polarity)

        if not sentiments:
            print("⚠️ Tidak ada berita valid ditemukan.")
            return 0.0

        avg_sent = np.mean(sentiments)
        pos = sum(s > 0.05 for s in sentiments)
        neg = sum(s < -0.05 for s in sentiments)
        neu = len(sentiments) - pos - neg

        print(f"🧠 Total berita: {len(sentiments)}")
        print(f"🙂 Positif: {pos} | 😐 Netral: {neu} | 🙁 Negatif: {neg}")
        print(f"🧮 Rata-rata sentimen (TextBlob): {avg_sent:.4f}")

        return avg_sent
    except Exception as e:
        print("❌ Gagal ambil news:", e)
        return 0.0

# ==========================================================
# 2️⃣ BINANCE HISTORICAL DATA (1 tahun)
# ==========================================================
client = Client("", "")  # Masukkan API Key Binance jika perlu

def build_dataset_for_coin(symbol="BTCUSDT", interval="1h", limit=8760):
    print(f"📈 Mengambil data historis untuk {symbol} ({interval}, 1 tahun)...")
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            return pd.DataFrame()
        df = pd.DataFrame(klines, columns=[
            "Open time","Open","High","Low","Close","Volume",
            "Close time","Quote asset volume","Number of trades",
            "Taker buy base","Taker buy quote","Ignore"
        ])
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = df[col].astype(float)

        df["change"] = df["Close"].pct_change() * 100
        df["volatility"] = (df["High"] - df["Low"]) / df["Open"] * 100
        df["volume"] = df["Volume"]
        df.dropna(inplace=True)
        df["change_lag1"] = df["change"].shift(1)
        df["change_lag2"] = df["change"].shift(2)
        df.dropna(inplace=True)
        return df[["change","volatility","volume","change_lag1","change_lag2"]]
    except Exception as e:
        print(f"❌ Gagal ambil data {symbol}:", e)
        return pd.DataFrame()

# ==========================================================
# 3️⃣ LABELING DATA-DRIVEN
# ==========================================================
def label_from_row(df, sentiment):
    labels = []
    p90 = np.percentile(df["change"], 90)
    p10 = np.percentile(df["change"], 10)
    for _, row in df.iterrows():
        c, v = row["change"], row["volatility"]
        if c >= p90 and sentiment > 0.05:
            labels.append(1)  # Pump
        elif c <= p10 and sentiment < -0.05:
            labels.append(2)  # Dump
        elif abs(c) < 1.0 and sentiment > 0.03 and v <= 0.25:
            labels.append(3)  # Sell the News small
        elif abs(c) < 1.0 and sentiment > 0.03 and v > 0.2:
            labels.append(4)  # Sell the News big
        else:
            labels.append(0)  # Stable
    df["label"] = labels
    return df

# ==========================================================
# 4️⃣ BANGUN DATASET MULTI-COIN
# ==========================================================
def build_multi_coin_dataset():
    coins = ["BTCUSDT","ETHUSDT","XRPUSDT","DOGEUSDT"]
    all_data = []
    sentiment_value = get_news_with_real_sentiment()

    for coin in coins:
        df = build_dataset_for_coin(coin)
        if df.empty:
            continue
        df = label_from_row(df, sentiment_value)
        df["sentiment"] = sentiment_value
        df["coin"] = coin
        all_data.append(df)

    if not all_data:
        raise ValueError("❌ Tidak ada data coin yang berhasil diambil.")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"✅ Total data gabungan: {len(combined)} baris")
    print(f"Distribusi label:\n{combined['label'].value_counts()}")
    return combined

# ==========================================================
# 5️⃣ TRAIN & CROSS-VALIDATION (Terminal Confusion)
# ==========================================================
def train_random_forest_cv(df):
    X = df[["change","volatility","volume","change_lag1","change_lag2","sentiment"]]
    y = df["label"]

    print("\nDistribusi label:\n", y.value_counts())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    f1_macro = []
    feature_importances = []
    all_true = []
    all_pred = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Oversample minor labels
        train_df = pd.concat([X_train, y_train], axis=1)
        df_majority = train_df[train_df.label == 4]
        minor_labels = [l for l in train_df.label.unique() if l != 4]
        dfs_minority = []
        for l in minor_labels:
            df_min = train_df[train_df.label == l]
            n_samples = max(int(len(df_majority)*0.3/len(minor_labels)), len(df_min), 20)
            df_min_up = resample(df_min, replace=True, n_samples=n_samples, random_state=42)
            dfs_minority.append(df_min_up)
        train_bal = pd.concat([df_majority]+dfs_minority).sample(frac=1, random_state=42)

        X_train_bal = train_bal[["change","volatility","volume","change_lag1","change_lag2","sentiment"]]
        y_train_bal = train_bal["label"]

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train_bal, y_train_bal)

        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1_macro.append(classification_report(y_test, y_pred, output_dict=True)["macro avg"]["f1-score"])
        feature_importances.append(model.feature_importances_)

        all_true.extend(y_test)
        all_pred.extend(y_pred)

    # Print terminal confusion matrix
    print("\n✅ Confusion Matrix (gabungan semua fold):")
    cm = confusion_matrix(all_true, all_pred)
    print(cm)

    # Feature importance
    fi_avg = np.mean(feature_importances, axis=0)
    fi_df = pd.DataFrame({"feature": X.columns, "importance": fi_avg}).sort_values(by="importance", ascending=False)
    print("\n✅ Feature Importances:")
    print(fi_df)

    print("\n✅ Cross-Validation Results:")
    print(f"Accuracy (avg): {np.mean(accs)*100:.2f}%")
    print(f"Macro F1-score (avg): {np.mean(f1_macro)*100:.2f}%")

# ==========================================================
# 6️⃣ JALANKAN
# ==========================================================
if __name__ == "__main__":
    df = build_multi_coin_dataset()
    train_random_forest_cv(df)
