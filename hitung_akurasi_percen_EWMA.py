import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import os

# ==========================================================
# 1️⃣ LOAD DATASET
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

    if 'label' not in df.columns:
        print("🏷️ Membuat label otomatis (Percentile + Z-score)...")
        df = label_from_percentile_zscore(df)

    print(f"✅ Total data: {len(df)} baris")
    print(f"Distribusi label:\n{df['label'].value_counts()}")
    return df

# ==========================================================
# 2️⃣ LABELING: Percentile + Z-score
# ==========================================================
def label_from_percentile_zscore(df):
    df = df.copy()
    p90 = np.percentile(df['change'], 90)
    p10 = np.percentile(df['change'], 10)

    mean_c = df['change'].mean()
    std_c = df['change'].std(ddof=0)
    df['z_score'] = (df['change'] - mean_c) / (std_c + 1e-9)

    labels = []
    for _, row in df.iterrows():
        c, s, z, v = row['change'], row['sentiment'], row['z_score'], row['volatility']

        if c >= p90 and z > 1.0 and s > 0.05:
            labels.append(1)  # Pump 🚀
        elif c <= p10 and z < -1.0 and s < -0.05:
            labels.append(2)  # Dump 💥
        elif -0.5 < z < 0.5 and s > 0.03 and v <= df['volatility'].mean():
            labels.append(3)  # SellSmall ⚠️
        elif -0.5 < z < 0.5 and s > 0.03 and v > df['volatility'].mean():
            labels.append(4)  # SellBig 🚨
        else:
            labels.append(0)  # Stable 😐
    df['label'] = labels
    return df

# ==========================================================
# 3️⃣ TIME SERIES TRAINING + VALIDATION
# ==========================================================
def train_random_forest_timeseries(df, n_splits=4):
    X = df[["change", "volatility", "volume", "sentiment"]]
    y = df["label"]

    # Pisahkan out-of-sample (20% terakhir)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\n📆 Data training: {len(X_train)} | Testing (out-of-sample): {len(X_test)}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, f1_macro = [], []
    feature_importances = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Oversampling minoritas
        df_tr = pd.concat([X_tr, y_tr], axis=1)
        df_majority = df_tr[df_tr.label == df_tr.label.value_counts().idxmax()]
        dfs_minority = []

        for l in df_tr.label.unique():
            if l == df_majority.label.iloc[0]:
                continue
            df_min = df_tr[df_tr.label == l]
            n_samples = max(int(len(df_majority) * 0.3), len(df_min), 20)
            df_min_up = resample(df_min, replace=True, n_samples=n_samples, random_state=42)
            dfs_minority.append(df_min_up)

        train_bal = pd.concat([df_majority] + dfs_minority).sample(frac=1, random_state=42)
        X_tr_bal = train_bal[["change", "volatility", "volume", "sentiment"]]
        y_tr_bal = train_bal["label"]

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_tr_bal, y_tr_bal)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        accs.append(acc)
        f1_macro.append(f1)
        feature_importances.append(model.feature_importances_)

        print(f"Fold {fold} ✅ Accuracy: {acc*100:.2f}% | Macro-F1: {f1*100:.2f}%")

    # Out-of-sample test
    final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)

    print("\n🧩 Evaluasi Out-of-Sample (20% data terakhir):")
    print(confusion_matrix(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred, digits=3))

    fi_avg = np.mean(feature_importances, axis=0)
    fi_df = pd.DataFrame({"feature": X.columns, "importance": fi_avg}).sort_values(by="importance", ascending=False)

    print("\n✅ Feature Importances:")
    print(fi_df)

    print("\n✅ Cross-Validation Results:")
    print(f"Rata-rata Akurasi  : {np.mean(accs)*100:.2f}%")
    print(f"Rata-rata Macro F1 : {np.mean(f1_macro)*100:.2f}%")

# ==========================================================
# 4️⃣ EWMA ANALISIS + VISUALISASI
# ==========================================================
def analyze_trend_ewma(df, span=10):
    df = df.copy()
    df["ewma_change"] = df["change"].ewm(span=span, adjust=False).mean()
    df["ewma_volatility"] = df["volatility"].ewm(span=span, adjust=False).mean()

    df["z_change"] = (df["change"] - df["change"].mean()) / (df["change"].std(ddof=0) + 1e-9)
    df["z_volatility"] = (df["volatility"] - df["volatility"].mean()) / (df["volatility"].std(ddof=0) + 1e-9)

    avg_z_change = df["z_change"].tail(10).mean()
    avg_z_volatility = df["z_volatility"].tail(10).mean()
    last_ewma_change = df["ewma_change"].iloc[-1]
    last_ewma_volatility = df["ewma_volatility"].iloc[-1]

    print("\n📈 Analisis Tren (EWMA + Z-score)")
    print(f"Rata-rata Z-score Change (10 terakhir): {avg_z_change:.3f}")
    print(f"Rata-rata Z-score Volatility (10 terakhir): {avg_z_volatility:.3f}")
    print(f"EWMA Change terakhir: {last_ewma_change:.4f}")
    print(f"EWMA Volatility terakhir: {last_ewma_volatility:.4f}")

    # Visualisasi
    plt.figure(figsize=(10, 5))
    plt.plot(df['change'], label='Change', alpha=0.5)
    plt.plot(df['ewma_change'], label=f'EWMA Change (span={span})', linewidth=2)
    plt.title("Tren EWMA Change")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================================
# 5️⃣ MAIN
# ==========================================================
if __name__ == "__main__":
    df = build_multi_coin_dataset()
    train_random_forest_timeseries(df)
    analyze_trend_ewma(df)
