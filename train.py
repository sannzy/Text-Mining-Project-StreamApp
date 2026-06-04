import os
import pickle
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Menggunakan fungsi preprocessing yang seragam dari modul streamlit_app
from streamlit_app import preprocess

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data(path="data.csv"):
    """
    Format dataset wajib memiliki kolom 'text' dan 'label' (0 = negatif, 1 = positif).
    """
    if path and os.path.exists(path):
        print(f"[+] Memuat dataset dari file: {path}")
        df = pd.read_csv(path)
        df = df.rename(columns={c: c.lower() for c in df.columns})
        assert "text" in df.columns and "label" in df.columns, \
            "CSV harus memiliki kolom 'text' dan 'label'"
        return df[["text", "label"]].dropna()

    print("[!] data.csv tidak ditemukan di root folder, memakai dummy dataset untuk demo.")
    pos = [
        "barang bagus banget kualitas mantap recommended seller",
        "pelayanan ramah pengiriman cepat saya puas sekali",
        "produk sesuai deskripsi harga murah worth it",
        "suka banget sama kualitasnya keren dan awet",
        "mantap jiwa cepat sampai packing rapi terima kasih",
        "aplikasinya enak dipakai fiturnya lengkap dan membantu",
        "makanannya enak porsinya banyak tempatnya nyaman",
        "filmnya seru alurnya bagus aktingnya memukau",
    ] * 30
    neg = [
        "barang jelek tidak sesuai gambar mengecewakan sekali",
        "pengiriman lama banget pelayanan buruk tidak ramah",
        "produk rusak pas datang kualitas murahan kecewa",
        "aplikasinya lemot sering error bikin kesal",
        "makanannya hambar mahal pelayanan lambat",
        "filmnya membosankan jalan cerita berantakan tidak rekomen",
        "barang palsu tidak original parah banget penipuan",
        "kecewa berat produk cacat tidak bisa dipakai",
    ] * 30
    df = pd.DataFrame({
        "text": pos + neg,
        "label": [1] * len(pos) + [0] * len(neg),
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main(data_path="data.csv"):
    print("=" * 50)
    print("1. Load data")
    df = load_data(data_path)
    print(f"Total data: {len(df)} | Distribusi label:\n{df['label'].value_counts()}")

    print("\n2-3. Cleaning + Preprocessing")
    df["clean"] = df["text"].apply(preprocess)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)

    X_text = df["clean"].values
    y = df["label"].values

    print("\n5. Split data (80/20)")
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # ========================================================
    # MODEL 1: MACHINE LEARNING (TF-IDF + Logistic Regression)
    # ========================================================
    print("\n[ML] 4. Text representation: TF-IDF")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_txt)
    X_test_tfidf = tfidf.transform(X_test_txt)

    print("[ML] 6. Train Logistic Regression")
    ml_model = LogisticRegression(max_iter=1000, C=1.0)
    ml_model.fit(X_train_tfidf, y_train)

    ml_pred = ml_model.predict(X_test_tfidf)
    print(f"[ML] Akurasi: {accuracy_score(y_test, ml_pred):.4f}")
    print(classification_report(y_test, ml_pred, target_names=["negatif", "positif"]))

    print("[ML] 7. Export ML artifacts")
    joblib.dump(ml_model, os.path.join(MODEL_DIR, "ml_model.pkl"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    # ========================================================
    # MODEL 2: DEEP LEARNING (Embedding + LSTM) - FIXED FOR KERAS 3
    # ========================================================
    print("\n[DL] 4. Text representation: Tokenizer + Padding")
    MAX_WORDS = 5000
    MAX_LEN = 30
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_txt)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_txt),
                                maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_txt),
                               maxlen=MAX_LEN, padding="post", truncating="post")

    print("[DL] 6. Train LSTM Model")
    # Menggunakan tf.keras.Input secara eksplisit agar kompatibel universal
    dl_model = Sequential([
        tf.keras.Input(shape=(MAX_LEN,)), 
        Embedding(input_dim=MAX_WORDS, output_dim=64),
        LSTM(64), # Tetap 64 unit tanpa recurrent_dropout agar GPU ngebut
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])

    dl_model.compile(loss="binary_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                     metrics=["accuracy"])

    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                             restore_best_weights=True)

    dl_model.fit(X_train_seq, y_train, validation_split=0.1,
                 epochs=15, batch_size=32, callbacks=[early], verbose=2)

    dl_eval = dl_model.evaluate(X_test_seq, y_test, verbose=0)
    print(f"[DL] Akurasi Uji LSTM: {dl_eval[1]:.4f}")

    print("[DL] 7. Export DL artifacts (.keras, tokenizer, config)")
    # UBAH DISINI: Simpan dengan format .keras agar aman di Streamlit Cloud
    dl_model.save(os.path.join(MODEL_DIR, "dl_model.keras"))

    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    # Simpan config terpadu
    with open(os.path.join(MODEL_DIR, "config.pkl"), "wb") as f:
        pickle.dump({"MAX_LEN": MAX_LEN, "MAX_WORDS": MAX_WORDS}, f)

    print("\n[✓] Selesai. Semua artifact sukses tersimpan di folder ./models")

if __name__ == "__main__":
    import sys
    # Memeriksa argumen manual atau default ke data.csv
    path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    main(path)

