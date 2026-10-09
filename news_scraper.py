import feedparser
import sqlite3
import time

# Sumber berita kripto legal (RSS)
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://news.bitcoin.com/feed/",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptonews.com/news/feed/",
    "https://u.today/rss",
]

# Buat database SQLite
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

def analyze_sentiment(title):
    """Dummy sentiment sederhana (bisa diganti model AI nanti)"""
    title_lower = title.lower()
    if any(w in title_lower for w in ["surge", "bull", "gain", "rise", "up", "pump"]):
        return "positive"
    elif any(w in title_lower for w in ["fall", "down", "crash", "bear", "dump"]):
        return "negative"
    else:
        return "neutral"

added = 0
for feed in RSS_FEEDS:
    print(f"Fetching: {feed}")
    parsed = feedparser.parse(feed)
    for entry in parsed.entries[:50]:  # ambil 50 per sumber
        title = entry.title
        url = entry.link
        published = entry.get("published", time.strftime("%Y-%m-%d"))
        sentiment = analyze_sentiment(title)
        source = feed.split("/")[2]
        c.execute("INSERT INTO news (title, url, published, sentiment, source) VALUES (?, ?, ?, ?, ?)",
                  (title, url, published, sentiment, source))
        added += 1
    time.sleep(1)

conn.commit()
conn.close()
print(f"✅ Selesai! {added} berita disimpan ke crypto_news.db")
