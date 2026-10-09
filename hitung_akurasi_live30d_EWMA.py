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
import time

# ==========================================================
# 1️⃣ NEWS SENTIMENT HARIAN (30 hari terakhir, per tanggal)
# ==========================================================
def get_daily_news_sentiment():
    print("📰 Mengambil news real-time dari NewsAPI (30 hari terakhir, per tanggal)...")
    api_key = "399c17b27f414ee5a4dc39b57e5441b5"
    url = "https://newsapi.org/v2/everything"
    query = "crypto OR bitcoin OR ethereum OR xrp OR dogecoin"

    today = datetime.datetime.now(datetime.timezone.utc).date()
    from_date = today - datetime.timedelta(days=30)
    headers = {"User-Agent": "Mozilla/5.0"}

    daily_sentiments = {}

    for i in range(30):
        day = from_date + datetime.timedelta(days=i)
        next_day = day + datetime.timedelta(days=1)

        params = {
            "q": query,
            "language": "en",
            "pageSize": 50,
            "sortBy": "publishedAt",
            "from": day.isoformat(),
            "to": next_day.isoformat(),
            "apiKey": api_key
        }

        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            sentiments = []
            for art in data.get("articles", []):
                text = f"{art.get('title','')}. {art.get('description','')}"
                if text.strip():
                    sentiments.append(TextBlob(text).sentiment.polarity)

            daily_sentiments[str(day)] = np.mean(sentiments) if sentiments else 0.0
            print(f"📅 {day}: {len(sentiments)} berita → rata-rata {daily_sentiments[str(day)]:.4f}")
        except Exception as e:
            print(f"⚠️ Gagal ambil berita {day}: {e}")
            daily_sentiments[str(day)] = 0.0

        time.sleep(1)

    print("✅ Pengambilan sentimen harian selesai.")
    return daily_sentiments

# ==========================================================
# 2️⃣ BINANCE HISTORICAL DATA (30 hari terakhir)
# ==========================================================
client = Client("", "")  # Masukkan API Key Binance jika perlu

def build_dataset_for_coin(symbol="BTCUSDT", interval="1h"):
    print(f"📈 Mengambil data historis untuk {symbol} ({interval}, 30 hari)...")
    try:
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(days=30)

        klines = client.get_historical_klines(
            symbol,
            interval,
            start_str=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_str=end_time.strftime("%Y-%m-%d %H:%M:%S")
        )

        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            "Open time","Open","High","Low","Close","Volume",
            "Close time","Quote asset volume","Number of trades",
            "Taker buy base","Taker buy quote","Ignore"
        ])
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = df[col].astype(float)

        df["time"] = pd.to_datetime(df["Open time"], unit="ms")
        df["date"] = df["time"].dt.date

        df["change"] = df["Close"].pct_change() * 100
        df["volatility"] = (df["High"] - df["Low"]) / df["Open"] * 100
        df["volume"] = df["Volume"]
        df.dropna(inplace=True)
        df["change_lag1"] = df["change"].shift(1)
        df["change_lag2"] = df["change"].shift(2)
        df.dropna(inplace=True)

        # EWMA (exponential weighted moving average) dan Z-score
        df["ewma"] = df["change"].ewm(span=10, adjust=False).mean()
        df["zscore"] = (df["change"] - df["change"].mean()) / df["change"].std()

        return df[["date","change","volatility","volume","change_lag1","change_lag2","ewma","zscore"]]
    except Exception as e:
        print(f"❌ Gagal ambil data {symbol}:", e)
        return pd.DataFrame()

# ==========================================================
# 3️⃣ LABELING DATA DENGAN ZSCORE + EWMA
# ==========================================================
def label_from_row(df, sentiment):
    labels = []
    z_thresh = 2.0
    for _, row in df.iterrows():
        z = row["zscore"]
        ewma = row["ewma"]
        c = row["change"]
        v = row["volatility"]
        s = sentiment  # Bisa juga row["sentiment"]

        if z <= -z_thresh and s > 0.03:
            labels.append(4)  # Sell the News Big
        elif z < 0 and z > -z_thresh and s > 0.03:
            labels.append(3)  # Sell the News Small
        elif z >= z_thresh and s > 0.03:
            labels.append(1)  # Pump
        elif z <= -z_thresh and s <= 0.03:
            labels.append(2)  # Dump
        else:
            labels.append(0)  # Stable
    df["label"] = labels
    return df

# ==========================================================
# 4️⃣ BANGUN DATASET MULTI-COIN + MERGE SENTIMEN HARIAN
# ==========================================================
def build_multi_coin_dataset():
    coins = ["BTCUSDT","ETHUSDT","XRPUSDT","DOGEUSDT"]
    all_data = []
    daily_sentiments = get_daily_news_sentiment()
    sent_df = pd.DataFrame(list(daily_sentiments.items()), columns=["date_str","sentiment"])
    sent_df["date"] = pd.to_datetime(sent_df["date_str"]).dt.date

    for coin in coins:
        df = build_dataset_for_coin(coin)
        if df.empty:
            continue

        df = df.merge(sent_df, on="date", how="left")
        df["sentiment"].fillna(0.0, inplace=True)

        df = label_from_row(df, df["sentiment"].mean())  # Label berbasis Z-score + EWMA
        df["coin"] = coin
        all_data.append(df)

    if not all_data:
        raise ValueError("❌ Tidak ada data coin yang berhasil diambil.")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"✅ Total data gabungan: {len(combined)} baris")
    print(f"Distribusi label:\n{combined['label'].value_counts()}")
    return combined

# ==========================================================
# 5️⃣ TRAIN & CROSS-VALIDATION RANDOM FOREST
# ==========================================================
def train_random_forest_cv(df):
    X = df[["change","volatility","volume","change_lag1","change_lag2","sentiment","ewma","zscore"]]
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

        # Oversampling minor label
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

        X_train_bal = train_bal[["change","volatility","volume","change_lag1","change_lag2","sentiment","ewma","zscore"]]
        y_train_bal = train_bal["label"]

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train_bal, y_train_bal)

        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1_macro.append(classification_report(y_test, y_pred, output_dict=True)["macro avg"]["f1-score"])
        feature_importances.append(model.feature_importances_)

        all_true.extend(y_test)
        all_pred.extend(y_pred)

    print("\n✅ Confusion Matrix (gabungan semua fold):")
    print(confusion_matrix(all_true, all_pred))

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
