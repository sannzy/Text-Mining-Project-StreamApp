import os
import pickle
import joblib
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import preprocess

# Konstanta
MODEL_DIR = "models"

# =================================
# PAGE CONFIG & TARGETED THEME
# =================================
st.set_page_config(
    page_title="SentimenAnalytica - Integrated System", 
    layout="centered"
)

# Injeksi CSS spesifik: memisahkan styling Main Content dan Sidebar
st.html("""
    <style>
    /* 1. MAIN CONTENT AREA SPECIFIC */
    .stMain, .stApp {
        background-color: #F8FAFC !important;
    }
    
    /* Memaksa warna teks di area konten utama saja menjadi gelap */
    .stMain p, .stMain label, .stMain span, div[data-testid="stWidgetLabel"] p {
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    
    /* Judul Utama */
    h2 {
        color: #1E40AF !important;
        font-weight: 700 !important;
    }
    
    /* Tombol Utama Paksa Warna Biru */
    button[data-testid="stBaseButton-primary"] {
        background-color: #1E40AF !important;
        color: white !important;
        border: none !important;
        width: 100% !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    
    /* 2. SIDEBAR AREA SPECIFIC (Memperbaiki teks yang hilang) */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
    }
    
    /* Memaksa teks di dalam sidebar menjadi putih terang agar kontras */
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #F8FAFC !important;
    }
    
    /* 3. RADIO BUTTON (BULLET) BLUE THEME */
    div[data-testid="stRadio"] label[data-baseweb="radio"] div div {
        border-color: #1E40AF !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] input[type="radio"]:checked + div div {
        background-color: #1E40AF !important;
        background-image: radial-gradient(circle, #1E40AF 0%, #1E40AF 40%, transparent 50%) !important;
    }
    
    /* Padding Atas */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""")

# =================================
# LOAD ASSETS & PATCHES 
# =================================
@st.cache_resource
def load_ml():
    """Memuat model Machine Learning (Logistic Regression) dan TF-IDF Vectorizer."""
    model = joblib.load(os.path.join(MODEL_DIR, "ml_model.pkl"))
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    return model, tfidf


@st.cache_resource
def load_dl():
    """Memuat model Deep Learning (LSTM) dengan perbaikan arsitektur."""
    import h5py
    import json
    
    try:
        model_path = os.path.join(MODEL_DIR, "dl_model.h5")
        
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
    model, tok, cfg, error_msg = load_dl()
    if error_msg:
        raise RuntimeError(error_msg)
        
    cleaned = preprocess(text)
    seq = tok.texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq,
        maxlen=int(cfg["MAX_LEN"]), 
        padding="post", 
        truncating="post"
    )
    
    raw_prediction = model.predict(padded, verbose=0)
    proba = float(raw_prediction[0][0])
    label = "Positif" if proba >= 0.5 else "Negatif"
    return label, proba


# =================================
# SIDEBAR
# =================================
with st.sidebar:
    st.markdown("### Informasi Proyek")
    st.markdown("**Pengembang:**")
    st.markdown("Sanly - 2702271474")
    st.markdown("**Sistem Informasi:**")
    st.markdown("Proyek Akhir Text Mining")
    st.markdown("**Model:**")
    st.markdown("ML & Deep Learning LSTM")
    st.divider()
    st.caption("Deep Learning Project - 2026")


# =================================
# MAIN CONTENT AREA
# =================================
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
                
                if label == "Positif":
                    st.html(f"""
                        <div style="padding: 20px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #1E40AF; background-color: #E0F2FE;">
                            <p style="margin:0; font-size: 0.9rem; color: #0369A1 !important;">Prediksi:</p>
                            <p style="color: #1E40AF !important; font-weight: bold; font-size: 1.2rem; margin: 5px 0;">SENTIMEN POSITIF</p>
                            <p style="margin:0; font-size: 0.9rem; color: #0369A1 !important;">Teks Bersih: <span style="color: #0F172A !important;"><i>"{cleaned_text}"</i></span></p>
                        </div>
                    """)
                else:
                    st.html(f"""
                        <div style="padding: 20px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #06B6D4; background-color: #ECFEFF;">
                            <p style="margin:0; font-size: 0.9rem; color: #0891B2 !important;">Prediksi:</p>
                            <p style="color: #0E7490 !important; font-weight: bold; font-size: 1.2rem; margin: 5px 0;">SENTIMEN NEGATIF</p>
                            <p style="margin:0; font-size: 0.9rem; color: #0891B2 !important;">Teks Bersih: <span style="color: #0F172A !important;"><i>"{cleaned_text}"</i></span></p>
                        </div>
                    """)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Tingkat Keyakinan", value=f"{conf * 100:.1f}%")
                with col2:
                    st.caption(f"Probabilitas ke arah Positif: {proba:.4f}")
                    st.progress(proba)
                    
            except Exception as e:
                st.error("Terjadi kesalahan internal pada pemrosesan model:")
                st.code(str(e), language="text")

# Footer
st.divider()
st.caption("Created by Sanly - 2702271474")