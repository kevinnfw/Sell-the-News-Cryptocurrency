# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# download vader lexicon jika belum ada
nltk.download('vader_lexicon')

# ========== 1. Load dataset ==========
# data.csv harus punya kolom: text,price_change,label
# contoh: "Elon Musk menyebut DOGE",8.5,1
DATA_PATH = "data.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} tidak ditemukan. Buat file data.csv di folder project.")

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text', 'price_change', 'label'])
df['text'] = df['text'].astype(str)
df['price_change'] = df['price_change'].astype(float)
df['label'] = df['label'].astype(int)

# ========== 2. Feature engineering - Sentiment (VADER) ==========
sia = SentimentIntensityAnalyzer()
df['sentiment_compound'] = df['text'].apply(lambda t: sia.polarity_scores(t)['compound'])

# ========== 3. TF-IDF for text ==========
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_text = vectorizer.fit_transform(df['text'])  # sparse

# ========== 4. Combine features ==========
# numeric features: price_change, sentiment_compound
X_numeric = df[['price_change', 'sentiment_compound']].values  # shape (n,2)
# convert sparse to dense (ok for small datasets)
X = np.hstack((X_text.toarray(), X_numeric))
y = df['label'].values

# ========== 5. Train/test split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None
)

# ========== 6. Train RandomForest ==========
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ========== 7. Evaluate ==========
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi pada data test: {acc*100:.2f}%")
print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ========== 8. Simpan model & vectorizer ==========
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")
print("✅ Model dan Vectorizer tersimpan di folder 'model/'")
