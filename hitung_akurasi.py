import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
import os

# ==========================================================
# 1️⃣ LOAD DATASET OFFLINE
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_fix.csv")

def build_multi_coin_dataset():
    print("📂 Memuat dataset offline dari:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Pastikan kolom wajib ada
    for col in ['change', 'volatility', 'volume', 'sentiment']:
        if col not in df.columns:
            df[col] = 0.0

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

    # Tentukan nilai rata-rata sentiment keseluruhan (sebagai representasi umum)
    sentiment_value = df['sentiment'].mean() if 'sentiment' in df.columns else 0.0

    # Tambahkan label jika belum ada
    if 'label' not in df.columns:
        print("🏷️  Kolom label belum ada, membuat otomatis dengan aturan percentile-based...")
        df = label_from_row(df, sentiment_value)

    print(f"✅ Total data: {len(df)} baris")
    print(f"Distribusi label:\n{df['label'].value_counts()}")
    return df

# ==========================================================
# 2️⃣ LABELING (versi percentile-based seperti kode sebelumnya)
# ==========================================================
def label_from_row(df, sentiment):
    labels = []
    p90 = np.percentile(df["change"], 90)
    p10 = np.percentile(df["change"], 10)
    for _, row in df.iterrows():
        c, v = row["change"], row["volatility"]
        if c >= p90 and sentiment > 0.05:
            labels.append(1)  # Pump
        elif c <= p10 and sentiment < -0.05:
            labels.append(2)  # Dump
        elif abs(c) < 1.0 and sentiment > 0.03 and v <= 0.25:
            labels.append(3)  # Sell the News small
        elif abs(c) < 1.0 and sentiment > 0.03 and v > 0.2:
            labels.append(4)  # Sell the News big
        else:
            labels.append(0)  # Stable
    df["label"] = labels
    return df

# ==========================================================
# 3️⃣ TRAIN + STRATIFIED K-FOLD CV
# ==========================================================
def train_random_forest_cv(df):
    X = df[["change", "volatility", "volume", "sentiment"]]
    y = df["label"]

    print("\nDistribusi label:\n", y.value_counts())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1_macro = [], []
    feature_importances = []
    all_true, all_pred = [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Oversampling minor label agar tidak terlalu timpang
        train_df = pd.concat([X_train, y_train], axis=1)
        df_majority = train_df[train_df.label == train_df.label.value_counts().idxmax()]
        dfs_minority = []

        for l in train_df.label.unique():
            if l == df_majority.label.iloc[0]:
                continue
            df_min = train_df[train_df.label == l]
            n_samples = max(int(len(df_majority)*0.3), len(df_min), 20)
            df_min_up = resample(df_min, replace=True, n_samples=n_samples, random_state=42)
            dfs_minority.append(df_min_up)

        train_bal = pd.concat([df_majority] + dfs_minority).sample(frac=1, random_state=42)
        X_train_bal = train_bal[["change", "volatility", "volume", "sentiment"]]
        y_train_bal = train_bal["label"]

        # Train Random Forest
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train_bal, y_train_bal)

        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1_macro.append(classification_report(y_test, y_pred, output_dict=True)["macro avg"]["f1-score"])
        feature_importances.append(model.feature_importances_)

        all_true.extend(y_test)
        all_pred.extend(y_pred)

    # Hasil evaluasi gabungan
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
# 4️⃣ MAIN
# ==========================================================
if __name__ == "__main__":
    df = build_multi_coin_dataset()
    train_random_forest_cv(df)
