import os
import pickle
import joblib
import streamlit as st
from preprocessing import preprocess

# Konstanta
MODEL_DIR = "models"

# =================================
# PAGE CONFIG
# =================================
st.set_page_config(
    page_title="SentimenAnalytica - Integrated System", 
    layout="centered"
)

# Menghilangkan padding atas bawaan Streamlit agar layout lebih rapi
st.html("""
    <style>
    .block-container {
        padding-top: 2rem;
    }
    </style>
""")

# =================================
# LOAD ASSETS & PATCHES (Anti Bug Keras 3)
# =================================
@st.cache_resource
def load_ml():
    """Memuat model Machine Learning (Logistic Regression) dan TF-IDF Vectorizer."""
    model = joblib.load(os.path.join(MODEL_DIR, "ml_model.pkl"))
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    return model, tfidf


@st.cache_resource
def load_dl():
    """Memuat model Deep Learning (LSTM) dengan membersihkan semua bug Keras 3 secara global."""
    import tensorflow as tf
    import h5py
    import json
    
    try:
        model_path = os.path.join(MODEL_DIR, "dl_model.h5")
        
        # --- GLOBAL PATCH UNTUK KORUP CONFIG KERAS 3 ---
        def clean_quantization_config(config):
            if isinstance(config, dict):
                config.pop('quantization_config', None)
                for key, value in config.items():
                    clean_quantization_config(value)
            elif isinstance(config, list):
                for item in config:
                    clean_quantization_config(item)
            return config

        with h5py.File(model_path, 'r') as f:
            model_config_raw = f.attrs.get('model_config')
            if model_config_raw is None:
                raise ValueError("Format file .h5 tidak mengenali metadata model_config.")
            
            if isinstance(model_config_raw, bytes):
                model_config_raw = model_config_raw.decode('utf-8')
                
            model_config = json.loads(model_config_raw)
            cleaned_config = clean_quantization_config(model_config)
            model = tf.keras.models.model_from_config(cleaned_config)
            
            for layer in model.layers:
                layer_name = layer.name
                if f"model_weights/{layer_name}" in f:
                    weight_names = f[f"model_weights/{layer_name}"].attrs.get('weight_names')
                    weights = []
                    for weight_name in weight_names:
                        if isinstance(weight_name, bytes):
                            weight_name = weight_name.decode('utf-8')
                        weights.append(f[f"model_weights/{layer_name}/{weight_name}"][()])
                    layer.set_weights(weights)

        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
            tok = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
            cfg = pickle.load(f)
            
        return model, tok, cfg, None
    except Exception as e:
        return None, None, None, str(e)


# =================================
# PREDICTION FUNCTIONS
# =================================
def predict_ml(text):
    model, tfidf = load_ml()
    vec = tfidf.transform([preprocess(text)])
    proba = float(model.predict_proba(vec)[0][1])
    label = "Positif" if proba >= 0.5 else "Negatif"
    return label, proba


def predict_dl(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    model, tok, cfg, error_msg = load_dl()
    if error_msg:
        raise RuntimeError(error_msg)
        
    seq = pad_sequences(
        tok.texts_to_sequences([preprocess(text)]),
        maxlen=cfg["MAX_LEN"], 
        padding="post", 
        truncating="post"
    )
    proba = float(model.predict(seq, verbose=0)[0][0])
    label = "Positif" if proba >= 0.5 else "Negatif"
    return label, proba


# =================================
# SIDEBAR (Identitas Tanpa Markdown HTML)
# =================================
with st.sidebar:
    st.subheader("Informasi Proyek")
    st.text("Pengembang:\nSanly - 2702271474")
    st.text("Sistem Informasi:\nProyek Akhir Text Mining")
    st.text("Model: ML & Deep Learning LSTM")
    st.divider()
    st.caption("Deep Learning Project - 2026")


# =================================
# MAIN CONTENT AREA
# =================================
# Menggunakan fungsi bawaan murni (Aman dari bug st.markdown HTML)
st.header("Analisis Sentimen Teks", divider="blue")
st.caption("Sistem klasifikasi teks otomatis berbasis Machine Learning dan Deep Learning LSTM.")
st.write("")

model_choice = st.radio(
    "Pilih Model Analisis:",
    ["Machine Learning (Logistic Regression)", "Deep Learning (LSTM)"],
    horizontal=False,
)

text_input = st.text_area(
    "Masukkan teks ulasan yang ingin dianalisis:", 
    height=140,
    placeholder="Tulis ulasan Anda di sini tanpa simbol khusus..."
)

if st.button("Proses Analisis", type="primary"):
    if not text_input.strip():
        st.info("Pesan: Teks tidak boleh kosong. Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Menganalisis ulasan..."):
            try:
                if model_choice.startswith("Deep"):
                    label, proba = predict_dl(text_input)
                else:
                    label, proba = predict_ml(text_input)
                
                conf = proba if label == "Positif" else 1 - proba
                cleaned_text = preprocess(text_input)
                
                st.write("---")
                st.subheader("Hasil Klasifikasi")
                
                # Menggunakan st.html khusus untuk cetak kotak hasil agar terisolasi dari st.metric
                if label == "Positif":
                    st.html(f"""
                        <div style="padding: 20px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #1E40AF; background-color: #E0F2FE;">
                            <p style="margin:0; font-size: 0.9rem; color: #0369A1;">Prediksi:</p>
                            <p style="color: #1E40AF; font-weight: bold; font-size: 1.2rem; margin: 5px 0;">SENTIMEN POSITIF</p>
                            <p style="margin:0; font-size: 0.9rem; color: #0369A1;">Teks Bersih: <i>"{cleaned_text}"</i></p>
                        </div>
                    """)
                else:
                    st.html(f"""
                        <div style="padding: 20px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #B91C1C; background-color: #FEE2E2;">
                            <p style="margin:0; font-size: 0.9rem; color: #991B1B;">Prediksi:</p>
                            <p style="color: #B91C1C; font-weight: bold; font-size: 1.2rem; margin: 5px 0;">SENTIMEN NEGATIF</p>
                            <p style="margin:0; font-size: 0.9rem; color: #991B1B;">Teks Bersih: <i>"{cleaned_text}"</i></p>
                        </div>
                    """)

                # Menampilkan Metrik Komponen
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Tingkat Keyakinan", value=f"{conf * 100:.1f}%")
                with col2:
                    st.caption(f"Probabilitas ke arah Positif: {proba:.4f}")
                    st.progress(proba)
                    
            except Exception as e:
                st.error("Terjadi kesalahan internal pada pemrosesan model:")
                st.code(str(e), language="text")

# Footer menggunakan teks murni
st.divider()
st.caption("Created by Sanly - 2702271474")