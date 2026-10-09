import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from binance.client import Client
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from termcolor import colored
from textblob import TextBlob
import datetime
import time
import csv

# ==========================================================
# SETUP
# ==========================================================
client = Client("", "")   # Replace with your API keys


# ==========================================================
# 1) SENTIMEN HARIAN — DIPERTAHANKAN
# ==========================================================
def get_daily_news_sentiment():
    print("Mengambil news…")
    api_key = "782119e826ad4550b1d63662c70bfcc1"
    url = "https://newsapi.org/v2/everything"
    query = "crypto OR bitcoin OR ethereum OR xrp OR dogecoin"

    today = datetime.datetime.utcnow().date()
    start_date = today - datetime.timedelta(days=30)

    daily = {}

    for i in range(30):
        d1 = start_date + datetime.timedelta(days=i)
        d2 = d1 + datetime.timedelta(days=1)

        params = {
            "q": query,
            "language": "en",
            "pageSize": 50,
            "sortBy": "publishedAt",
            "from": d1.isoformat(),
            "to": d2.isoformat(),
            "apiKey": api_key
        }

        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            vals = []

            for art in data.get("articles", []):
                txt = f"{art.get('title','')} {art.get('description','')}"
                if txt.strip():
                    vals.append(TextBlob(txt).sentiment.polarity)

            daily[str(d1)] = np.mean(vals) if vals else 0.0
            print(f"{d1} → {daily[str(d1)]:.4f}")

        except:
            daily[str(d1)] = 0.0

        time.sleep(1)

    return daily


# ==========================================================
# 2) GET CANDLES BINANCE
# ==========================================================
def get_candles(symbol="BTCUSDT", interval="1h", days=30):
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=days)

    try:
        raw = client.get_historical_klines(
            symbol, interval,
            start_str=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_str=end.strftime("%Y-%m-%d %H:%M:%S")
        )
    except:
        return []

    candles = []
    for c in raw:
        candles.append({
            "time": int(c[0]),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5])
        })
    return candles


# ==========================================================
# 3) LABEL — SESUAI APP.PY1
# ==========================================================
def label_from_row(df, sentiment):
    if df.empty:
        df["label"] = 0
        return df

    changes = df["change"].values
    p90 = float(np.percentile(changes, 90))
    p10 = float(np.percentile(changes, 10))

    labels = []
    for _, row in df.iterrows():
        c = float(row["change"])
        s = float(sentiment)

        if s > 0.03 and c <= p10:
            labels.append(4)    # STN Big
        elif c >= p90:
            labels.append(1)    # Pump
        elif c <= p10:
            labels.append(2)    # Dump
        elif c < 0 and s > 0.03 and c > p10:
            labels.append(3)    # STN Small
        else:
            labels.append(0)    # Stable

    df["label"] = labels
    return df


# ==========================================================
# 4) WINDOW-BASED DATASET (MATCH app.py1)
# ==========================================================
def build_window_dataset(symbol="BTCUSDT", interval="1h",
                         window=30, step=5, df_raw=None):

    candles = get_candles(symbol, interval)
    if len(candles) < window + 2:
        return pd.DataFrame()

    rows = []

    for start in range(0, len(candles) - window - 1, step):
        win = candles[start:start+window+1]

        old = win[0]["close"]
        new = win[-1]["close"]
        change = ((new - old) / old) * 100

        closes = np.array([c["close"] for c in win])
        returns = np.diff(closes) / closes[:-1]
        vol = float(np.std(returns) * 100)

        volm = win[-1]["volume"]

        # ==========================
        # TAMBAHAN SENTIMENT
        # ==========================
        if df_raw is not None:
            dates = [datetime.datetime.fromtimestamp(c["time"]/1000).date() for c in win]
            sent_vals = df_raw[df_raw["date"].isin(dates)]["sentiment"]
            sentiment = float(sent_vals.mean()) if not sent_vals.empty else 0.0
        else:
            sentiment = 0.0

        rows.append([change, vol, volm, sentiment])

    df = pd.DataFrame(rows, columns=["change","volatility","volume","sentiment"])
    return df


# ==========================================================
# 5) MULTI-COIN DATASET
# ==========================================================
def build_multi_coin_dataset():
    coins = ["BTCUSDT","ETHUSDT","XRPUSDT","DOGEUSDT"]
    result = []

    daily_sent = get_daily_news_sentiment()
    sent_df = pd.DataFrame(daily_sent.items(), columns=["date_str","sentiment"])
    sent_df["date"] = pd.to_datetime(sent_df["date_str"]).dt.date

    for coin in coins:

        raw = get_candles(coin, "1h")
        if len(raw) < 50:
            continue

        df_raw = pd.DataFrame(raw)
        df_raw["date"] = pd.to_datetime(df_raw["time"], unit="ms").dt.date
        df_raw = df_raw.merge(sent_df, on="date", how="left")
        df_raw["sentiment"].fillna(0.0, inplace=True)

        # ==========================
        # PASS df_raw KE WINDOW
        # ==========================
        df_win = build_window_dataset(coin, df_raw=df_raw)

        if df_win.empty:
            continue

        avg_sent = df_raw["sentiment"].mean()
        df_win = label_from_row(df_win, avg_sent)
        df_win["coin"] = coin

        result.append(df_win)

    df_final = pd.concat(result, ignore_index=True)
    print("Dataset ready:", df_final.shape)
    print(df_final["label"].value_counts())
    return df_final


# ==========================================================
# 6) TRAIN RF + OUTPUT
# ==========================================================
def train_random_forest_cv(df):

    # ==========================
    # TAMBAH SENTIMENT KE MODEL
    # ==========================
    X = df[["change","volatility","volume","sentiment"]]
    y = df["label"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_true, all_pred = [], []
    accs, f1s, feat_imps = [], [], []

    print("\n====================")
    print("  CROSS VALIDATION  ")
    print("====================")

    for fold,(tr,ts) in enumerate(skf.split(X,y),1):
        Xtr, Xts = X.iloc[tr], X.iloc[ts]
        Ytr, Yts = y.iloc[tr], y.iloc[ts]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42
        )
        model.fit(Xtr, Ytr)

        pred = model.predict(Xts)
        acc = accuracy_score(Yts, pred) * 100
        f1 = classification_report(Yts, pred, output_dict=True)["macro avg"]["f1-score"] * 100

        accs.append(acc)
        f1s.append(f1)
        feat_imps.append(model.feature_importances_)

        acc_color = "green" if acc >= 80 else "red"

        print(colored(
            f"Fold {fold} → ACC={acc:.2f}% | F1={f1:.2f}%",
            acc_color
        ))

        all_true.extend(Yts)
        all_pred.extend(pred)

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(all_true, all_pred))

    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)

    print("\n====================")
    print(colored(f"MEAN ACCURACY : {mean_acc:.2f}%", "green"))
    print(colored(f"MEAN MACRO-F1 : {mean_f1:.2f}%", "green"))
    print("====================")

    # ==========================
    # UPDATE FEATURE IMPORTANCE
    # ==========================
    fi = np.mean(feat_imps, axis=0)
    fi_dict = dict(zip(["change","volatility","volume","sentiment"], fi))

    print("\nFEATURE IMPORTANCE (%):")
    for k,v in fi_dict.items():
        print(f"{k} : {v*100:.2f}%")

    return mean_acc, mean_f1, fi_dict


# ==========================================================
# 7) EWMA + Z-SCORE
# ==========================================================
def analyze_trends_app1(df):
    df = df.copy()

    df["ewma_change"] = df["change"].ewm(span=10, adjust=False).mean()
    df["zscore_change"] = (df["change"] - df["change"].mean()) / df["change"].std()

    df["ewma_vol"] = df["volatility"].ewm(span=10, adjust=False).mean()
    df["zscore_vol"] = (df["volatility"] - df["volatility"].mean()) / df["volatility"].std()

    print("\nEWMA Change Latest:", df["ewma_change"].iloc[-1])
    print("EWMA Vol Latest:", df["ewma_vol"].iloc[-1])
    print("Zscore Change Latest:", df["zscore_change"].iloc[-1])
    print("Zscore Vol Latest:", df["zscore_vol"].iloc[-1])


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    df = build_multi_coin_dataset()
    train_random_forest_cv(df)
    analyze_trends_app1(df)