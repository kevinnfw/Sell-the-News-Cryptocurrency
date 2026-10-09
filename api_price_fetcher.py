# api_price_fetcher.py
import requests, time, random

# ---------------- Binance ----------------
def get_price_change_binance(symbol="DOGEUSDT", minutes_window=15):
    url="https://api.binance.com/api/v3/klines"
    params={"symbol":symbol,"interval":"1m","limit":minutes_window+1}
    try:
        r=requests.get(url,params=params,timeout=8)
        r.raise_for_status()
        data=r.json()
        if len(data) < 2:
            return 0.0, 0.0, 0.0
        price_old = float(data[0][4])
        price_now = float(data[-1][4])
        change = ((price_now - price_old) / price_old) * 100 if price_old != 0 else 0.0
        return round(change, 4), round(price_now, 8), round(price_old, 8)
    except Exception as e:
        # print("get_price_change_binance error:", e)
        return 0.0, 0.0, 0.0

def get_candlestick_binance(symbol="BTCUSDT", interval="1m", limit=100):
    url="https://api.binance.com/api/v3/klines"
    params={"symbol":symbol,"interval":interval,"limit":limit}
    try:
        r=requests.get(url,params=params,timeout=8)
        r.raise_for_status()
        data=r.json()
        candles=[]
        for c in data:
            candles.append({
                "time": int(c[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4])
            })
        return candles
    except Exception:
        return []

# ---------------- OKX ----------------
def get_price_change_okx(symbol="DOGE-USDT", minutes_window=15):
    url="https://www.okx.com/api/v5/market/history-candles"
    params={"instId":symbol,"bar":"1m","limit":minutes_window+1}
    try:
        r=requests.get(url,params=params,timeout=8)
        r.raise_for_status()
        data=r.json().get("data",[])
        if len(data) < 2:
            return 0.0, 0.0, 0.0
        # OKX returns newest first; reverse for chronological
        rows = list(reversed(data))
        price_old = float(rows[0][4])
        price_now = float(rows[-1][4])
        change = ((price_now - price_old) / price_old) * 100 if price_old != 0 else 0.0
        return round(change, 4), round(price_now, 8), round(price_old, 8)
    except Exception:
        return 0.0, 0.0, 0.0

def get_candlestick_okx(symbol="DOGE-USDT", interval="1m", limit=100):
    url="https://www.okx.com/api/v5/market/history-candles"
    params={"instId":symbol,"bar":interval,"limit":limit}
    try:
        r=requests.get(url,params=params,timeout=8)
        r.raise_for_status()
        data=r.json().get("data",[])
        candles=[]
        for c in reversed(data):
            candles.append({
                "time": int(c[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4])
            })
        return candles
    except Exception:
        return []

# ---------------- Unified Interface ----------------
def normalize_coin_for_binance(coin):
    coin_upper = coin.upper().replace("-","").replace("/","")
    if not coin_upper.endswith("USDT"):
        coin_upper += "USDT"
    return coin_upper

def normalize_coin_for_okx(coin):
    coin_upper = coin.upper().replace("/","").replace("USDT","")
    # transform e.g. BTCUSDT -> BTC-USDT
    if "-" not in coin_upper and len(coin_upper) > 3:
        coin_upper = coin_upper[:3] + "-" + coin_upper[3:]
    if not coin_upper.endswith("USDT"):
        if not coin_upper.endswith("USDT"):
            coin_upper = coin_upper
    return coin_upper

def get_price_change(coin, exchange="binance", window=15):
    try:
        coin_upper = coin.upper().replace(" ", "")
        if exchange.lower() == "binance":
            symbol = normalize_coin_for_binance(coin_upper)
            return get_price_change_binance(symbol, window)
        elif exchange.lower() == "okx":
            symbol = normalize_coin_for_okx(coin_upper)
            return get_price_change_okx(symbol, window)
        else:
            return 0.0, 0.0, 0.0
    except Exception:
        return 0.0, 0.0, 0.0

def get_candlestick_data(coin, exchange="binance", interval="1m", limit=100):
    try:
        coin_upper = coin.upper().replace(" ", "")
        if exchange.lower() == "binance":
            symbol = normalize_coin_for_binance(coin_upper)
            return get_candlestick_binance(symbol, interval, limit)
        elif exchange.lower() == "okx":
            symbol = normalize_coin_for_okx(coin_upper)
            return get_candlestick_okx(symbol, interval, limit)
        else:
            return []
    except Exception:
        return []
