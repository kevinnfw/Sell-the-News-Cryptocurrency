import feedparser
import sqlite3
import time
import os
from datetime import datetime, timezone

DB_PATH = "crypto_news.db"

RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://news.bitcoin.com/feed/",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptonews.com/news/feed/",
    "https://u.today/rss"
]


def init_db():
    """Inisialisasi database jika belum ada."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            link TEXT UNIQUE,
            published TEXT,
            source TEXT,
            image_url TEXT
        )
    ''')
    conn.commit()
    conn.close()


def fetch_and_store():
    """Ambil berita dari semua RSS dan simpan ke database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            source = feed.split("/")[2]
            count_new = 0

            for entry in parsed.entries[:30]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                if not title or not link:
                    continue

                # Published date
                published = getattr(entry, "published", None)
                if not published:
                    published = datetime.now(timezone.utc).isoformat()

                # Coba ambil gambar
                image_url = ""
                if "media_content" in entry:
                    try:
                        image_url = entry.media_content[0].get("url", "")
                    except Exception:
                        image_url = ""

                try:
                    c.execute(
                        """
                        INSERT OR IGNORE INTO news (title, link, published, source, image_url)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (title, link, published, source, image_url)
                    )
                    if c.rowcount > 0:
                        count_new += 1
                except Exception as e:
                    print(f"⚠️ Error insert dari {source}: {e}")

            if count_new > 0:
                print(f"📰 {source}: {count_new} berita baru ditambahkan.")
            else:
                print(f"ℹ️ {source}: tidak ada berita baru.")

        except Exception as e:
            print(f"❌ Gagal ambil feed {feed}: {e}")

    conn.commit()
    conn.close()


def rotate_oldest_to_top():
    """Putar berita tertua ke atas untuk menjaga dinamika tampilan."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM news ORDER BY datetime(published) ASC LIMIT 1")
    row = c.fetchone()
    if row:
        now = datetime.now(timezone.utc).isoformat()
        c.execute("UPDATE news SET published = ? WHERE id = ?", (now, row[0]))
    conn.commit()
    conn.close()


def main():
    """Loop utama untuk update berkala."""
    init_db()
    while True:
        print("⏳ Memperbarui database berita crypto...")
        fetch_and_store()
        rotate_oldest_to_top()
        print("✅ DB diperbarui & rotasi berhasil.\n")
        time.sleep(180)  # update setiap 3 menit


if __name__ == "__main__":
    main()
