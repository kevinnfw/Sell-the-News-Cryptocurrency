import feedparser
import sqlite3
import time

# ======= SUMBER BERITA REAL (RSS) =======
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://news.bitcoin.com/feed/",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptonews.com/news/feed/",
    "https://u.today/rss"
]

# ======= BUAT DATABASE =======
conn = sqlite3.connect("crypto_news.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    url TEXT,
    published TEXT,
    sentiment TEXT,
    source TEXT
)''')

# ======= ANALISIS SENTIMEN SEDERHANA =======
def analyze_sentiment(title: str):
    t = title.lower()
    if any(w in t for w in ["surge", "up", "rise", "gain", "bull", "pump", "breakout"]):
        return "positive"
    elif any(w in t for w in ["fall", "drop", "down", "bear", "crash", "dump", "collapse"]):
        return "negative"
    else:
        return "neutral"

# ======= FETCH DAN SIMPAN BERITA =======
total_added = 0
for feed in RSS_FEEDS:
    print(f"Fetching from {feed}")
    parsed = feedparser.parse(feed)
    count = 0
    for entry in parsed.entries[:80]:  # ambil 80 per sumber
        title = entry.title.strip()
        url = entry.link
        published = entry.get("published", time.strftime("%Y-%m-%d"))
        sentiment = analyze_sentiment(title)
        source = feed.split("/")[2]

        c.execute("SELECT id FROM news WHERE url = ?", (url,))
        if c.fetchone():
            continue  # skip duplikat

        c.execute(
            "INSERT INTO news (title, url, published, sentiment, source) VALUES (?, ?, ?, ?, ?)",
            (title, url, published, sentiment, source)
        )
        total_added += 1
        count += 1

    print(f"  ✅ {count} berita dari {feed}")
    time.sleep(1)

conn.commit()
conn.close()

print(f"\n🎉 Selesai! Total berita disimpan: {total_added}")
print("File database tersimpan sebagai crypto_news.db")
