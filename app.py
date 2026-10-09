from flask import Flask, render_template, request, jsonify, send_file
import requests, datetime, time, os, sqlite3, csv, io, joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from textblob import TextBlob
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "crypto_news.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_realtime.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

NEWS_API_KEY = "399c17b27f414ee5a4dc39b57e5441b5"

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

rf_model_cache = {}
recent_data_buffer = deque(maxlen=300)
recent_news_buffer = deque(maxlen=50)

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        url TEXT,
        published TEXT,
        source TEXT
    )
    """)
    conn.commit()
    conn.close()

# ---------------- NEWSAPI ----------------
def get_news_with_real_sentiment(limit=20):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "crypto OR bitcoin OR ethereum OR cryptocurrency",
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": limit,
        "sortBy": "publishedAt"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        news_list = []
        for item in data.get("articles", []):
            title = item.get("title", "")
            url_news = item.get("url", "#")
            published_at = item.get("publishedAt", "Unknown Time")
            source = item.get("source", {}).get("name", "Unknown")
            sentiment = analyze_sentiment_real(title)
            news_list.append({
                "title": title,
                "url": url_news,
                "published_at": published_at,
                "source": source,
                "sentiment": sentiment
            })
        for n in news_list:
            recent_news_buffer.append(n)
        return news_list
    except Exception as e:
        print("❌ NewsAPI error:", e)
        return []

def get_sentiment_summary_from_db(limit=50):
    news = list(recent_news_buffer)[-limit:]
    if not news:
        return {"positive": 0, "negative": 0, "neutral": 0}
    pos = sum(1 for n in news if n["sentiment"] > 0)
    neg = sum(1 for n in news if n["sentiment"] < 0)
    neu = len(news) - pos - neg
    total = len(news)
    return {
        "positive": round(pos / total * 100),
        "negative": round(neg / total * 100),
        "neutral": round(neu / total * 100)
    }

# ---------------- Binance ----------------
def get_price_change_binance(symbol="DOGEUSDT", minutes_window=15):
    interval, limit = select_binance_interval(minutes_window)
    candles = get_candlestick_data_binance(symbol, interval, limit)
    if len(candles) < 2:
        return 0.0, 0.0, 0.0
    price_old = candles[0]['close']
    price_now = candles[-1]['close']
    change = ((price_now - price_old) / price_old) * 100
    return round(change, 4), round(price_now, 4), round(price_old, 4)

def select_binance_interval(minutes_window):
    if minutes_window <= 60:
        return "1m", min(minutes_window, 1000)
    elif minutes_window <= 180:
        return "3m", min(minutes_window // 3, 1000)
    elif minutes_window <= 600:
        return "5m", min(minutes_window // 5, 1000)
    elif minutes_window <= 1800:
        return "15m", min(minutes_window // 15, 1000)
    elif minutes_window <= 3600:
        return "30m", min(minutes_window // 30, 1000)
    elif minutes_window <= 7200:
        return "1h", min(minutes_window // 60, 1000)
    else:
        return "4h", 100

def get_candlestick_data_binance(symbol="DOGEUSDT", interval="1m", limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        candles = [{
            "time": int(d[0]),
            "open": float(d[1]),
            "high": float(d[2]),
            "low": float(d[3]),
            "close": float(d[4]),
            "volume": float(d[5])
        } for d in data]
        return candles
    except Exception as e:
        print("⚠️ Binance API error:", e)
        return []

# ---------------- Utility ----------------
def calculate_volatility(candles):
    if not candles or len(candles) < 2:
        return 0.0
    closes = np.array([c["close"] for c in candles])
    returns = np.diff(closes) / closes[:-1]
    return round(np.std(returns) * 100, 3)

def analyze_sentiment_real(text):
    if not text or text.strip() == "":
        return 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity

def average_sentiment(news_list):
    if not news_list:
        return 0.0
    sentiments = [n["sentiment"] for n in news_list]
    return np.mean(sentiments)

# ---------------- Label dari revisi (percentile-based) ----------------
def label_from_row(df, sentiment):
    """
    Labeling dengan percentile-based:
    0: Stable
    1: Pump
    2: Dump
    3: Sell the News Small
    4: Sell the News Big
    """
    labels = []
    if df.empty:
        return df
    # Percentile threshold
    p90 = np.percentile(df["change"], 90)
    p10 = np.percentile(df["change"], 10)
    for _, row in df.iterrows():
        c = row["change"]
        s = sentiment
        if c >= p90:
            labels.append(1)  # Pump
        elif c <= p10:
            labels.append(2)  # Dump
        elif c < 0 and s > 0.03 and c > p10:
            labels.append(3)  # Sell the News Small
        elif c <= p10 and s > 0.03:
            labels.append(4)  # Sell the News Big
        else:
            labels.append(0)  # Stable
    df["label"] = labels
    return df

# ---------------- Random Forest ----------------
def build_realtime_features(symbol="BTCUSDT", interval="1m"):
    candles = get_candlestick_data_binance(symbol, interval, 200)
    if not candles:
        return None, None
    lookback = min(30, len(candles)-1)
    price_old = candles[-1 - lookback]["close"]
    price_now = candles[-1]["close"]
    change = ((price_now - price_old) / price_old) * 100
    volatility = calculate_volatility(candles[-lookback-1:])
    volume = candles[-1]["volume"] if candles else 0.0
    avg_sent = average_sentiment(list(recent_news_buffer)[-20:])
    features = np.array([[round(change, 6), round(volatility, 6), round(volume, 6), round(avg_sent, 6)]])
    details = {
        "change": round(change, 6),
        "volatility": round(volatility, 6),
        "volume": volume,
        "sentiment": avg_sent,
        "latest_price": price_now,
        "old_price": price_old
    }
    return features, details

def build_training_dataset_from_candles(symbol="BTCUSDT", interval="1m", candles_limit=200, window_size=30, step=5):
    candles = get_candlestick_data_binance(symbol, interval, candles_limit)
    if not candles or len(candles) < window_size + 1:
        return None, None

    df_rows = []
    for start in range(0, len(candles) - window_size, step):
        window = candles[start:start + window_size + 1]
        price_old = window[0]["close"]
        price_new = window[-1]["close"]
        change = ((price_new - price_old) / price_old) * 100
        volatility = calculate_volatility(window)
        volume = window[-1]["volume"]
        df_rows.append([change, volatility, volume])

    df = pd.DataFrame(df_rows, columns=["change", "volatility", "volume"])
    avg_sent = average_sentiment(list(recent_news_buffer)[-20:])
    df = label_from_row(df, avg_sent)
    X = df[["change", "volatility", "volume"]].values
    y = df["label"].values
    return X, y

def train_rf_on_demand(symbol="BTCUSDT", interval="1m"):
    X, y = build_training_dataset_from_candles(symbol, interval)
    if X is None or len(X) < 10:
        return None

    df = pd.DataFrame(X, columns=["change", "volatility", "volume"])
    df["label"] = y

    majority = df[df.label == df.label.value_counts().idxmax()]
    minority = df[df.label != df.label.value_counts().idxmax()]

    if len(minority) == 0:
        Xb = X
        yb = y
    else:
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        df_bal = pd.concat([majority, minority_upsampled])
        Xb = df_bal[["change", "volatility", "volume"]].values
        yb = df_bal["label"].values

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(Xb, yb)
    rf_model_cache["rf_real_model"] = model

    try:
        joblib.dump(model, MODEL_PATH)
    except Exception:
        pass

    return model

def random_forest_predict(symbol="BTCUSDT", interval="1m"):
    model = train_rf_on_demand(symbol, interval)
    if model is None:
        return "Stable"
    features, details = build_realtime_features(symbol, interval)
    if features is None:
        return "Stable"
    try:
        pred = model.predict(features)[0]
    except Exception:
        return "Stable"
    mapping = {0: "Stable", 1: "Pump", 2: "Dump", 3: "SellTheNewsSmall", 4: "SellTheNewsBig"}
    return mapping.get(pred, "Stable")

# ---------------- Flask Routes ----------------
@app.route("/")
def index():
    latest_news = get_news_with_real_sentiment(limit=20)
    return render_template("index.html", latest_news=latest_news)

@app.route("/predict", methods=["POST"])
def predict():
    coin = request.form.get("coin", "") or "BTCUSDT"
    meme_coin = request.form.get("meme_coin", "")
    selected_coin = meme_coin or coin or "BTCUSDT"
    exchange = request.form.get("exchange", "Binance")
    try:
        window = int(request.form.get("window", 10))
    except:
        window = 10
    chart_interval = request.form.get("chart_interval", "1m")

    # Harga & candles
    change, latest_price, old_price = get_price_change_binance(selected_coin, window)
    candles = get_candlestick_data_binance(selected_coin, chart_interval, 200)
    volatility = calculate_volatility(candles)

    # Random Forest prediksi label terbaru
    rf_label = random_forest_predict(selected_coin, interval=chart_interval)

    # Mapping RF label ke teks & warna
    label_mapping = {
        "Stable": "Stable",
        "Pump": "Pump 🚀",
        "Dump": "Dump 💥",
        "SellTheNewsSmall": "Sell the News (Small)",
        "SellTheNewsBig": "Sell the News (Big)"
    }
    color_mapping = {
        "Stable": "gray",
        "Pump": "green",
        "Dump": "red",
        "SellTheNewsSmall": "red",
        "SellTheNewsBig": "darkred"
    }
    rf_prediction_text = f"Random Forest: {label_mapping.get(rf_label, 'Stable')}"
    rf_color = color_mapping.get(rf_label, "gray")
    prediction = rf_label  # gunakan RF sebagai status utama

    # Hitung percentiles
    candles_for_percentile = get_candlestick_data_binance(selected_coin, chart_interval, 200)
    if len(candles_for_percentile) >= 2:
        changes_for_percentile = [((candles_for_percentile[i]["close"] - candles_for_percentile[i-1]["close"]) / candles_for_percentile[i-1]["close"]) * 100
                                for i in range(1, len(candles_for_percentile))]
        p10 = np.percentile(changes_for_percentile, 10)
    else:
        p10 = -100.0  # default sangat rendah jika tidak ada data

    # Ambil rata-rata sentimen dari 20 berita terakhir
    avg_sentiment = average_sentiment(list(recent_news_buffer)[-20:])

    # Hitung status Sell the News berdasarkan formula
    stn_status = calculate_sell_the_news_status(change, p10, avg_sentiment)
    stn_text = "Saat ini belum mendekati Sell the News."
    if stn_status == "SellTheNewsSmall":
        stn_text = "Prediksi: Sell the News (Small) mungkin terjadi jika harga turun sedikit lagi."
    elif stn_status == "SellTheNewsBig":
        stn_text = "Prediksi: Sell the News (Big) mungkin terjadi karena harga sudah ekstrem turun."

    # Ringkasan status pasar
    summary = {
        "positive": get_sentiment_summary_from_db()["positive"],
        "negative": get_sentiment_summary_from_db()["negative"],
        "pump_dump": prediction,
        "most_volatile": selected_coin
    }

    latest_news = get_news_with_real_sentiment(limit=50)
    sentiment_trend = generate_sentiment_trend(list(recent_news_buffer))

    return jsonify({
    "coin": selected_coin,
    "exchange": exchange,
    "window": window,
    "latest_price": latest_price,
    "old_price": old_price,
    "change": change,
    "volatility": volatility,
    "prediction": prediction,
    "rf_prediction": rf_prediction_text,
    "rf_label": rf_label,
    "rf_color": rf_color,
    "candles": candles,
    "sentiment_trend": sentiment_trend,
    "summary": summary,
    "historical_news": latest_news,
    "sell_the_news_status": stn_status,
    "sell_the_news_text": stn_text,
    "p10_percentile": round(p10, 4),
    "avg_sentiment": round(avg_sentiment, 4)
    })

    # ---------------- Sell the News Realtime ----------------
def calculate_sell_the_news_status(change, p10, sentiment):
    """
    Hitung status Sell the News berdasarkan harga, p10 percentile, dan sentimen.
    Return: "None", "SellTheNewsSmall", "SellTheNewsBig"
    """
    if sentiment <= 0.03:
        return None  # Sentimen terlalu rendah, STN kemungkinan kecil

    if change < 0 and change > p10:
        return "SellTheNewsSmall"
    elif change <= p10:
        return "SellTheNewsBig"
    else:
        return None

@app.route("/download_csv")
def download_csv():
    news = get_news_with_real_sentiment(limit=10)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Time", "Title", "Source", "Sentiment"])
    for n in news:
        writer.writerow([n["published_at"], n["title"], n["source"], n["sentiment"]])
    output.seek(0)
    return send_file(io.BytesIO(output.read().encode()), mimetype="text/csv",
                     as_attachment=True, download_name="historical_data.csv")

@app.route("/percentiles", methods=["GET"])
def percentiles():
    coin = request.args.get("coin", "BTCUSDT")
    interval = request.args.get("interval", "1m")
    candles_limit = int(request.args.get("limit", 200))

    candles = get_candlestick_data_binance(coin, interval, candles_limit)
    if not candles or len(candles) < 2:
        return jsonify({
            "error": f"Tidak ada data candlestick untuk {coin}."
        })

    # Hitung perubahan harga per candlestick
    changes = [((candles[i]["close"] - candles[i-1]["close"]) / candles[i-1]["close"]) * 100
               for i in range(1, len(candles))]

    p90 = np.percentile(changes, 90)
    p10 = np.percentile(changes, 10)
    latest_change = changes[-1]
    latest_price = candles[-1]["close"]

    return jsonify({
        "coin": coin,
        "latest_price": latest_price,
        "p90": round(p90, 4),
        "p10": round(p10, 4),
        "latest_change": round(latest_change, 4)
    })

# ---------------- Helpers ----------------
def pump_dump_alert(change, volatility, coin="BTCUSDT"):
    if change >= 5.0:
        return "Pump 🚀"
    elif change <= -5.0:
        return "Dump 💥"
    elif volatility > 2.0:
        return "Volatile ⚠️"
    return "Stable"

def generate_sentiment_trend(news_list):
    news_list = news_list[-50:]
    pos = sum(1 for n in news_list if n.get("sentiment", 0) > 0)
    neg = sum(1 for n in news_list if n.get("sentiment", 0) < 0)
    neu = len(news_list) - pos - neg
    return [{"positive": pos, "negative": neg, "neutral": neu}]

    # ---------------- Percentile Realtime Monitor ----------------
def print_realtime_percentiles(symbol="BTCUSDT", interval="1m", candles_limit=200):
    """
    Cetak nilai percentile 10 dan 90 secara realtime untuk koin tertentu.
    Ini membantu melihat batas Dump / Pump.
    """
    candles = get_candlestick_data_binance(symbol, interval, candles_limit)
    if not candles or len(candles) < 2:
        print(f"Tidak ada data candlestick untuk {symbol}.")
        return

    # Hitung perubahan harga per candlestick
    changes = []
    for i in range(1, len(candles)):
        old = candles[i-1]["close"]
        new = candles[i]["close"]
        change = ((new - old) / old) * 100
        changes.append(change)

    # Hitung percentile
    p90 = np.percentile(changes, 90)
    p10 = np.percentile(changes, 10)

    latest_price = candles[-1]["close"]

    print(f"Realtime Percentiles untuk {symbol}:")
    print(f"  Latest Price          : {latest_price}")
    print(f"  p90 (Pump threshold)  : {p90:.4f}%")
    print(f"  p10 (Dump threshold)  : {p10:.4f}%")
    print(f"  Current Change        : {changes[-1]:.4f}%")
    print("-" * 40)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
