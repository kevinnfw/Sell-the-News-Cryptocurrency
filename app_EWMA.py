from flask import Flask, render_template, request, jsonify, send_file
import requests, datetime, time, os, sqlite3, csv, io, joblib, threading
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from textblob import TextBlob
from collections import deque

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "crypto_news.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_realtime.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

NEWS_API_KEY = "782119e826ad4550b1d63662c70bfcc1"

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# ---------------- Globals ----------------
rf_model_cache = {}
recent_data_buffer = deque(maxlen=300)
recent_news_buffer = deque(maxlen=200)

# ---------------- Database ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        url TEXT,
        published TEXT,
        source TEXT,
        sentiment REAL
    )
    """)
    conn.commit()
    conn.close()

# ---------------- NewsAPI & Sentiment ----------------
def analyze_sentiment_real(text):
    if not text or str(text).strip() == "":
        return 0.0
    try:
        blob = TextBlob(str(text))
        return float(blob.sentiment.polarity)
    except Exception:
        return 0.0

def get_news_with_real_sentiment(limit=20):
    news_list = []
    if not NEWS_API_KEY:
        return news_list

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

        for item in data.get("articles", []):
            title = item.get("title", "") or item.get("description", "") or ""
            url_news = item.get("url", "#")
            published_at = item.get("publishedAt", "Unknown Time")
            source = item.get("source", {}).get("name", "Unknown")
            sentiment = analyze_sentiment_real(title)

            obj = {
                "title": title,
                "url": url_news,
                "published_at": published_at,
                "source": source,
                "sentiment": sentiment
            }

            news_list.append(obj)
            recent_news_buffer.append(obj)
            print("News buffer size:", len(recent_news_buffer))

        return news_list

    except Exception as e:
        print("❌ NewsAPI error:", e)
        return []

def seed_news_buffer():
    try:
        n = get_news_with_real_sentiment(limit=30)
        if n:
            print(f"✅ Seeded news buffer with {len(n)} items")
    except Exception as e:
        print("Seed news error:", e)

def get_sentiment_summary_from_buffer(limit=50):
    news = list(recent_news_buffer)[-limit:]

    if not news:
        return {"positive": 0, "negative": 0, "neutral": 0}

    pos = sum(1 for n in news if n.get("sentiment", 0) > 0)
    neg = sum(1 for n in news if n.get("sentiment", 0) < 0)
    neu = len(news) - pos - neg
    total = max(1, len(news))

    return {
        "positive": round(pos / total * 100, 1),
        "negative": round(neg / total * 100, 1),
        "neutral": round(neu / total * 100, 1)
    }

# ---------------- Binance ----------------
def get_candlestick_data_binance(symbol="DOGEUSDT", interval="1m", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, 1000)
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        candles = []
        for d in data:
            candles.append({
                "time": int(d[0]),
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5])
            })

        return candles

    except Exception as e:
        print("⚠️ Binance API error:", e)
        return []

def calculate_volatility(candles):
    if not candles or len(candles) < 2:
        return 0.0

    closes = np.array([c["close"] for c in candles], dtype=float)
    returns = np.diff(closes) / closes[:-1]
    return float(np.std(returns) * 100.0)

def average_sentiment(news_list):
    if not news_list:
        return 0.0

    sentiments = [
        n.get("sentiment", 0.0)
        for n in news_list
        if n.get("sentiment") is not None
    ]

    if not sentiments:
        return 0.0

    return float(np.mean(sentiments))

def select_binance_interval(minutes_window):
    if minutes_window <= 60:
        return "1m", min(minutes_window, 1000)
    elif minutes_window <= 180:
        return "3m", min((minutes_window + 2) // 3, 1000)
    elif minutes_window <= 600:
        return "5m", min((minutes_window + 4) // 5, 1000)
    elif minutes_window <= 1800:
        return "15m", min((minutes_window + 14) // 15, 1000)
    elif minutes_window <= 3600:
        return "30m", min((minutes_window + 29) // 30, 1000)
    elif minutes_window <= 7200:
        return "1h", min((minutes_window + 59) // 60, 1000)
    else:
        return "4h", 100

def get_price_change_binance(symbol="DOGEUSDT", minutes_window=15):
    interval, limit = select_binance_interval(minutes_window)
    candles = get_candlestick_data_binance(symbol, interval, limit)

    if len(candles) < 2:
        return 0.0, 0.0, 0.0

    price_old = candles[0]["close"]
    price_now = candles[-1]["close"]
    change = ((price_now - price_old) / price_old) * 100.0

    return round(change, 6), round(price_now, 6), round(price_old, 6)

# ---------------- Random Forest ----------------
def build_realtime_features(symbol="BTCUSDT", interval="1m"):
    candles = get_candlestick_data_binance(symbol, interval, 300)

    if not candles or len(candles) < 5:
        return None, None

    lookback = min(30, len(candles) - 1)
    price_old = candles[-1 - lookback]["close"]
    price_now = candles[-1]["close"]

    change = ((price_now - price_old) / price_old) * 100.0
    volatility = calculate_volatility(candles[-lookback - 1:])
    volume = float(candles[-1]["volume"]) if candles else 0.0
    avg_sent = average_sentiment(list(recent_news_buffer)[-20:])

    features = np.array([[
        round(change, 6),
        round(volatility, 6),
        round(volume, 6),
        round(avg_sent, 6)
    ]])

    details = {
        "change": round(change, 6),
        "volatility": round(volatility, 6),
        "volume": volume,
        "sentiment": avg_sent,
        "latest_price": price_now,
        "old_price": price_old
    }

    return features, details

def build_training_dataset_from_candles(
    symbol="BTCUSDT",
    interval="1m",
    candles_limit=500,
    window_size=30,
    step=5
):
    candles = get_candlestick_data_binance(symbol, interval, candles_limit)

    if not candles or len(candles) < window_size + 1:
        return None, None

    df_rows = []

    for start in range(0, len(candles) - window_size, step):
        window = candles[start:start + window_size + 1]
        price_old = window[0]["close"]
        price_new = window[-1]["close"]

        change = ((price_new - price_old) / price_old) * 100.0
        volatility = calculate_volatility(window)
        volume = float(window[-1]["volume"])

        df_rows.append([change, volatility, volume])

    if not df_rows:
        return None, None

    df = pd.DataFrame(df_rows, columns=["change", "volatility", "volume"])
    avg_sent = average_sentiment(list(recent_news_buffer)[-20:])
    df = label_from_row(df, avg_sent)

    if "label" not in df.columns or df.empty:
        return None, None

    X = df[["change", "volatility", "volume"]].values
    y = df["label"].values

    return X, y

def label_from_row(df, sentiment):
    if df.empty:
        return df

    df = df.copy()
    changes = df["change"].dropna().values

    if len(changes) == 0:
        df["label"] = 0
        return df

    p90 = float(np.percentile(changes, 90))
    p10 = float(np.percentile(changes, 10))

    labels = []

    for _, row in df.iterrows():
        c = float(row["change"])
        s = float(sentiment)

        if s > 0.03 and c <= p10:
            labels.append(4)
        elif c >= p90:
            labels.append(1)
        elif c <= p10:
            labels.append(2)
        elif c < 0 and s > 0.03 and c > p10:
            labels.append(3)
        else:
            labels.append(0)

    df["label"] = labels
    return df

def train_rf_on_demand(symbol="BTCUSDT", interval="1m"):
    print("🔵 TRAIN ON DEMAND DIMULAI")

    X, y = build_training_dataset_from_candles(symbol, interval)

    if X is None or len(X) < 10:
        if os.path.exists(MODEL_PATH):
            try:
                m = joblib.load(MODEL_PATH)
                rf_model_cache["rf_real_model"] = m
                return m
            except Exception:
                return None
        return None

    df = pd.DataFrame(X, columns=["change", "volatility", "volume"])
    df["label"] = y

    majority_label = df["label"].value_counts().idxmax()
    majority = df[df.label == majority_label]
    minority = df[df.label != majority_label]

    if len(minority) == 0:
        Xb = df[["change", "volatility", "volume"]].values
        yb = df["label"].values
    else:
        try:
            minority_upsampled = resample(
                minority,
                replace=True,
                n_samples=len(majority),
                random_state=42
            )
            df_bal = pd.concat([majority, minority_upsampled])
            Xb = df_bal[["change", "volatility", "volume"]].values
            yb = df_bal["label"].values
        except Exception:
            Xb = df[["change", "volatility", "volume"]].values
            yb = df["label"].values

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )

    try:
        print("⚙️ Training Random Forest...")
        model.fit(Xb, yb)
        print("✅ TRAINING SELESAI")

        rf_model_cache["rf_real_model"] = model

        try:
            joblib.dump(model, MODEL_PATH)
        except Exception:
            pass

        return model

    except Exception as e:
        print("RF train error:", e)
        return None

def random_forest_predict(symbol="BTCUSDT", interval="1m"):
    model = rf_model_cache.get("rf_real_model")

    if model is None:
        model = train_rf_on_demand(symbol, interval)

    features, details = build_realtime_features(symbol, interval)

    if features is None or model is None:
        return "Stable"

    try:
        feat = features

        if hasattr(model, "n_features_in_") and model.n_features_in_ == 3:
            feat = features[:, :3]

        pred = model.predict(feat)[0]

    except Exception as e:
        print("RF predict error:", e)
        return "Stable"

    mapping = {
        0: "Stable",
        1: "Pump",
        2: "Dump",
        3: "Sell the News (Small)",
        4: "Sell the News (Big)"
    }

    return mapping.get(int(pred), "Stable")

# ---------------- Sell the News ----------------
def calculate_sell_the_news_status(change, p10, sentiment):
    try:
        if sentiment is None:
            sentiment = 0.0

        if sentiment <= 0.03:
            return None

        if change < 0 and change > p10:
            return "SellTheNewsSmall"
        elif change <= p10:
            return "SellTheNewsBig"
        else:
            return None

    except Exception:
        return None

# ---------------- Helpers ----------------
def generate_sentiment_trend(news_list):
    news_list = news_list[-100:]
    trend_list = []

    for i in range(len(news_list)):
        window = news_list[max(0, i - 9):i + 1]
        pos = sum(1 for n in window if n.get("sentiment", 0) > 0)
        neg = sum(1 for n in window if n.get("sentiment", 0) < 0)
        neu = len(window) - pos - neg

        trend_list.append({
            "positive": pos,
            "negative": neg,
            "neutral": neu
        })

    return trend_list

# ---------------- Flask Routes ----------------
@app.route("/")
def index():
    latest_news = list(recent_news_buffer)[-20:]

    try:
        return render_template("index_EWMA.html", latest_news=latest_news)
    except Exception:
        return """
        <html><body><h3>Sell-the-News Detector</h3>
        <p>Open the frontend index.html in the project root or place the template in templates/index_EWMA.html</p>
        </body></html>
        """

@app.route("/predict", methods=["POST"])
def predict():
    coin = (request.form.get("coin", "") or "BTCUSDT").upper()
    exchange = request.form.get("exchange", "Binance")

    try:
        window = int(request.form.get("window", 10) or 10)
    except ValueError:
        window = 10

    chart_interval = request.form.get("chart_interval", "1m")

    print(f"🔹 Memaksa training Random Forest untuk {coin} interval {chart_interval}")

    model = train_rf_on_demand(coin, chart_interval)

    if model:
        print("✅ Model siap untuk prediksi")
    else:
        print("⚠️ Model gagal dibuat, prediksi fallback ke 'Stable'")

    change, latest_price, old_price = get_price_change_binance(coin, window)
    candles = get_candlestick_data_binance(coin, chart_interval, 300)
    volatility = calculate_volatility(candles)

    rf_label = random_forest_predict(coin, chart_interval)

    if candles and len(candles) > window:
        window_changes = []

        for i in range(0, len(candles) - window):
            old_p = candles[i]["close"]
            new_p = candles[i + window]["close"]

            if old_p == 0:
                continue

            window_changes.append(((new_p - old_p) / old_p) * 100)

        if window_changes:
            p90 = float(np.percentile(window_changes, 90))
            p10 = float(np.percentile(window_changes, 10))
        else:
            p90 = 0.0
            p10 = -100.0

    else:
        if candles and len(candles) >= 2:
            changes_for_percentile = [
                ((candles[i]["close"] - candles[i - 1]["close"]) / candles[i - 1]["close"]) * 100
                for i in range(1, len(candles))
            ]
            p90 = float(np.percentile(changes_for_percentile, 90))
            p10 = float(np.percentile(changes_for_percentile, 10))
        else:
            p90 = 0.0
            p10 = -100.0

    recent_news = list(recent_news_buffer)[-50:]
    avg_sentiment = average_sentiment(recent_news)
    stn_status = calculate_sell_the_news_status(change, p10, avg_sentiment)

    stn_text = {
        None: "Saat ini belum mendekati Sell the News.",
        "SellTheNewsSmall": "Prediksi: Sell the News (Small) — harga turun (change < 0) dan sentimen positif.",
        "SellTheNewsBig": "Prediksi: Sell the News (Big) — harga turun tajam (change ≤ p10) dan sentimen positif."
    }.get(stn_status)

    summary = {
        "positive": get_sentiment_summary_from_buffer()["positive"],
        "negative": get_sentiment_summary_from_buffer()["negative"],
        "pump_dump": rf_label,
        "most_volatile": coin
    }

    latest_news_full = recent_news
    sentiment_trend = generate_sentiment_trend(list(recent_news_buffer))

    return jsonify({
        "coin": coin,
        "exchange": exchange,
        "latest_price": latest_price,
        "old_price": old_price,
        "change": round(change, 6),
        "volatility": round(volatility, 6),
        "prediction": rf_label,
        "rf_prediction": rf_label,
        "candles": candles,
        "sentiment_trend": sentiment_trend,
        "summary": summary,
        "historical_news": latest_news_full,
        "sell_the_news_status": stn_status,
        "sell_the_news_text": stn_text,
        "p90_percentile": round(p90, 6),
        "p10_percentile": round(p10, 6),
        "avg_sentiment": round(avg_sentiment, 6)
    })

@app.route("/zscore")
def zscore():
    coin = (request.args.get("coin", "BTCUSDT") or "BTCUSDT").upper()
    chart_interval = request.args.get("interval", "1m")

    candles = get_candlestick_data_binance(coin, chart_interval, 500)

    if not candles or len(candles) < 3:
        return jsonify({"error": "Tidak ada data candlestick"}), 404

    closes = np.array([c["close"] for c in candles], dtype=float)
    returns = np.diff(closes) / closes[:-1]

    mean_r = np.mean(returns)
    std_r = np.std(returns) + 1e-12

    z_scores = (returns - mean_r) / std_r
    ewma = pd.Series(closes).ewm(span=10, adjust=False).mean().tolist()

    avg_sent = average_sentiment(list(recent_news_buffer)[-50:])

    return jsonify({
        "coin": coin,
        "latest_price": float(closes[-1]),
        "z_score": float(z_scores[-1]),
        "ewma": float(ewma[-1]),
        "change": float(((closes[-1] - closes[-2]) / closes[-2]) * 100),
        "sentiment": avg_sent
    })

@app.route("/download_csv")
def download_csv():
    coin = (request.args.get("coin", "BTCUSDT") or "BTCUSDT").upper()
    interval = request.args.get("interval", "1m") or "1m"

    try:
        window = int(request.args.get("window", 10) or 10)
    except ValueError:
        window = 10

    try:
        candles_limit = int(request.args.get("limit", 300) or 300)
    except ValueError:
        candles_limit = 300

    candles = get_candlestick_data_binance(coin, interval, candles_limit)

    if not candles or len(candles) < 3:
        return jsonify({
            "error": f"Data candlestick untuk {coin} tidak tersedia atau kurang dari 3 candle."
        }), 404

    change, latest_price, old_price = get_price_change_binance(coin, window)
    volatility = calculate_volatility(candles)

    if len(candles) > window:
        window_changes = []

        for i in range(0, len(candles) - window):
            old_p = candles[i]["close"]
            new_p = candles[i + window]["close"]

            if old_p == 0:
                continue

            window_changes.append(((new_p - old_p) / old_p) * 100.0)

    else:
        window_changes = []

        for i in range(1, len(candles)):
            old_p = candles[i - 1]["close"]
            new_p = candles[i]["close"]

            if old_p == 0:
                continue

            window_changes.append(((new_p - old_p) / old_p) * 100.0)

    if window_changes:
        p90 = float(np.percentile(window_changes, 90))
        p10 = float(np.percentile(window_changes, 10))
    else:
        p90 = 0.0
        p10 = -100.0

    closes = np.array([c["close"] for c in candles], dtype=float)
    returns = np.diff(closes) / closes[:-1]

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns) + 1e-12)

    z_scores = (returns - mean_r) / std_r
    latest_zscore = float(z_scores[-1])

    ewma_values = pd.Series(closes).ewm(span=10, adjust=False).mean().tolist()
    latest_ewma = float(ewma_values[-1])

    recent_news = list(recent_news_buffer)[-50:]
    avg_sentiment = average_sentiment(recent_news)
    sentiment_summary = get_sentiment_summary_from_buffer(limit=50)

    rf_prediction = random_forest_predict(coin, interval)

    stn_status = calculate_sell_the_news_status(change, p10, avg_sentiment)
    stn_status_text = stn_status if stn_status else "Stable"

    sentiment_positive = avg_sentiment > 0.03
    statistical_status = "Stable"

    if latest_zscore <= -2 and sentiment_positive:
        statistical_status = "Sell the News (Big)"
    elif latest_zscore < 0 and latest_zscore > -2 and sentiment_positive:
        statistical_status = "Sell the News (Small)"
    elif latest_zscore >= 2 and sentiment_positive:
        statistical_status = "Pump"
    elif latest_zscore <= -2 and not sentiment_positive:
        statistical_status = "Dump"

    generated_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Generated At",
        "Coin",
        "Interval",
        "Window",
        "Old Price",
        "Latest Price",
        "Change (%)",
        "Volatility (%)",
        "Average Sentiment",
        "Positive Sentiment (%)",
        "Negative Sentiment (%)",
        "Neutral Sentiment (%)",
        "P90 Percentile",
        "P10 Percentile",
        "Z-score",
        "EWMA",
        "Random Forest Prediction",
        "Sell The News Rule Status",
        "Z-score EWMA Dashboard Status",
        "News Time",
        "News Title",
        "News Source",
        "News Sentiment",
        "News URL"
    ])

    if not recent_news:
        writer.writerow([
            generated_at,
            coin,
            interval,
            window,
            old_price,
            latest_price,
            round(change, 6),
            round(volatility, 6),
            round(avg_sentiment, 6),
            sentiment_summary.get("positive", 0),
            sentiment_summary.get("negative", 0),
            sentiment_summary.get("neutral", 0),
            round(p90, 6),
            round(p10, 6),
            round(latest_zscore, 6),
            round(latest_ewma, 6),
            rf_prediction,
            stn_status_text,
            statistical_status,
            "",
            "",
            "",
            "",
            ""
        ])
    else:
        for n in recent_news:
            writer.writerow([
                generated_at,
                coin,
                interval,
                window,
                old_price,
                latest_price,
                round(change, 6),
                round(volatility, 6),
                round(avg_sentiment, 6),
                sentiment_summary.get("positive", 0),
                sentiment_summary.get("negative", 0),
                sentiment_summary.get("neutral", 0),
                round(p90, 6),
                round(p10, 6),
                round(latest_zscore, 6),
                round(latest_ewma, 6),
                rf_prediction,
                stn_status_text,
                statistical_status,
                n.get("published_at", ""),
                n.get("title", ""),
                n.get("source", ""),
                n.get("sentiment", ""),
                n.get("url", "")
            ])

    output.seek(0)

    filename = f"{coin}_sell_the_news_analysis.csv"

    return send_file(
        io.BytesIO(output.read().encode("utf-8-sig")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )

@app.route("/percentiles", methods=["GET"])
def percentiles():
    coin = (request.args.get("coin", "BTCUSDT") or "BTCUSDT").upper()
    interval = request.args.get("interval", "1m")

    try:
        candles_limit = int(request.args.get("limit", 200))
    except ValueError:
        candles_limit = 200

    try:
        window = int(request.args.get("window", 10)) if request.args.get("window") else 10
    except ValueError:
        window = 10

    candles = get_candlestick_data_binance(coin, interval, candles_limit)

    if not candles or len(candles) < 2:
        return jsonify({"error": f"Tidak ada data candlestick untuk {coin}."})

    if len(candles) > window:
        window_changes = []

        for i in range(0, len(candles) - window):
            old_p = candles[i]["close"]
            new_p = candles[i + window]["close"]

            if old_p == 0:
                continue

            ch = ((new_p - old_p) / old_p) * 100.0
            window_changes.append(ch)

        if window_changes:
            p90 = np.percentile(window_changes, 90)
            p10 = np.percentile(window_changes, 10)
        else:
            p90 = 0.0
            p10 = -100.0

    else:
        changes = [
            ((candles[i]["close"] - candles[i - 1]["close"]) / candles[i - 1]["close"]) * 100
            for i in range(1, len(candles))
        ]

        p90 = np.percentile(changes, 90)
        p10 = np.percentile(changes, 10)

    if len(candles) > window:
        latest_change = (
            (candles[-1]["close"] - candles[-1 - window]["close"])
            / candles[-1 - window]["close"]
        ) * 100
    else:
        latest_change = (
            (candles[-1]["close"] - candles[-2]["close"])
            / candles[-2]["close"]
        ) * 100

    latest_price = candles[-1]["close"]

    return jsonify({
        "coin": coin,
        "latest_price": latest_price,
        "p90": float(round(p90, 6)),
        "p10": float(round(p10, 6)),
        "latest_change": float(round(latest_change, 6))
    })

if __name__ == "__main__":
    init_db()

    try:
        t = threading.Thread(target=seed_news_buffer, daemon=True)
        t.start()
    except Exception:
        pass

    if os.path.exists(MODEL_PATH):
        try:
            rf_model_cache["rf_real_model"] = joblib.load(MODEL_PATH)
            print("✅ Loaded cached RF model.")
        except Exception:
            pass

    app.run(host="0.0.0.0", port=5000, debug=True)