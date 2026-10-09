import sqlite3

db_path = "crypto_news.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Cek apakah kolom url sudah ada
cur.execute("PRAGMA table_info(news)")
cols = [c[1] for c in cur.fetchall()]

if "url" not in cols:
    cur.execute("ALTER TABLE news ADD COLUMN url TEXT")

if "source" not in cols:
    cur.execute("ALTER TABLE news ADD COLUMN source TEXT")

if "published_at" not in cols:
    cur.execute("ALTER TABLE news ADD COLUMN published_at TEXT")

if "sentiment" not in cols:
    cur.execute("ALTER TABLE news ADD COLUMN sentiment TEXT")

conn.commit()
conn.close()
print("✅ Struktur tabel news diperbarui tanpa mengubah data lama.")
