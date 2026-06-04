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
# PAGE CONFIG & PREMIUM DARK THEME
# =================================
st.set_page_config(
    page_title="SentimenAnalytica", 
    layout="centered"
)

# Kustomisasi Full Dark Mode secara aman tanpa merusak teks aplikasi
st.html("""
    <style>
    .stApp {
        background-color: #0F172A !important;
    }
        
    .stApp p, .stApp label, .stApp span, div[data-testid="stWidgetLabel"] p {
        color: #F8FAFC !important;
    }
    
    h2 {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p span {
        background-color: transparent !important;
        color: #F8FAFC !important;
    }
    
    div[data-testid="stRadio"] label[data-baseweb="radio"] div div {
        border-color: #38BDF8 !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] input[type="radio"]:checked + div div {
        background-color: #38BDF8 !important;
        background-image: radial-gradient(circle, #38BDF8 0%, #38BDF8 40%, transparent 50%) !important;
    }
    
    div.stButton > button {
        width: 100% !important;
        background-color: #1E40AF !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(30, 64, 175, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 10px 15px -3px rgba(30, 64, 175, 0.5) !important;
        border: none !important;
        color: white !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #020617 !important;
    }
    section[data-testid="stSidebar"] .sidebar-title {
        color: #64748B !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 2px !important;
        margin-top: 15px !important;
    }
    section[data-testid="stSidebar"] .sidebar-value {
        color: #F1F5F9 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-bottom: 12px !important;
    }
    
    .block-container {
        padding-top: 3rem;
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
    """Memuat model Deep Learning (LSTM) dengan aman."""
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
    st.subheader("Dashboard Proyek")
    
    st.html("""
        <div class="sidebar-title">Nama Pengembang</div>
        <div class="sidebar-value">Sanly</div>
            
        <div class="sidebar-title">NIM / Student ID</div>
        <div class="sidebar-value">2702271474</div>
            
        <div class="sidebar-title">Sistem Informasi</div>
        <div class="sidebar-value">Proyek Akhir Text Mining</div>
            
        <div class="sidebar-title">Arsitektur Model</div>
        <div class="sidebar-value">Logistic Reg. & LSTM</div>
    """)
    st.divider()
    st.caption("Deep Learning System - 2026")


# =================================
# MAIN CONTENT AREA
# =================================
st.header("Analisis Sentimen Teks", divider="blue")
st.caption("Aplikasi penentu polaritas sentimen otomatis berbasis kecerdasan buatan.")
st.write("")

model_choice = st.radio(
    "Pilih Model Analisis:",
    ["Machine Learning (Logistic Regression)", "Deep Learning (LSTM)"],
    horizontal=False,
)

text_input = st.text_area(
    "Masukkan teks ulasan yang ingin dianalisis:", 
    height=130,
    placeholder="Tulis ulasan Anda di sini..."
)

st.write("")

if st.button("Proses Analisis", type="primary"):
    if not text_input.strip():
        st.info("Pesan: Teks input kosong. Silakan masukkan ulasan teks terlebih dahulu.")
    else:
        with st.spinner("Menghitung probabilitas..."):
            try:
                if model_choice.startswith("Deep"):
                    label, proba = predict_dl(text_input)
                else:
                    label, proba = predict_ml(text_input)
                
                conf = proba if label == "Positif" else 1 - proba
                cleaned_text = preprocess(text_input)
                
                st.write("---")
                st.subheader("Hasil Klasifikasi")
                
                # Desain Box Hasil Kontras dalam Mode Gelap
                if label == "Positif":
                    st.html(f"""
                        <div style="padding: 18px; border-radius: 6px; margin-bottom: 20px; border-left: 5px solid #38BDF8; background-color: #1E293B;">
                            <p style="margin:0; font-size: 0.85rem; color: #38BDF8 !important;">PREDIKSI SISTEM</p>
                            <p style="color: #0EA5E9 !important; font-weight: 700; font-size: 1.25rem; margin: 4px 0;">SENTIMEN POSITIF</p>
                            <p style="margin:0; font-size: 0.9rem; color: #E2E8F0 !important;">Teks Bersih: <i>"{cleaned_text}"</i></p>
                        </div>
                    """)
                else:
                    st.html(f"""
                        <div style="padding: 18px; border-radius: 6px; margin-bottom: 20px; border-left: 5px solid #22D3EE; background-color: #1E293B;">
                            <p style="margin:0; font-size: 0.85rem; color: #22D3EE !important;">PREDIKSI SISTEM</p>
                            <p style="color: #06B6D4 !important; font-weight: 700; font-size: 1.25rem; margin: 4px 0;">SENTIMEN NEGATIF</p>
                            <p style="margin:0; font-size: 0.9rem; color: #E2E8F0 !important;">Teks Bersih: <i>"{cleaned_text}"</i></p>
                        </div>
                    """)

                # Indikator Nilai Kepercayaan
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Tingkat Keyakinan", value=f"{conf * 100:.1f}%")
                with col2:
                    st.caption(f"Probabilitas Sentimen Positif: {proba:.4f}")
                    st.progress(proba)
                    
            except Exception as e:
                st.error("Terjadi kesalahan internal pada pemrosesan model:")
                st.code(str(e), language="text")

# Footer
st.divider()
st.caption("Created by Sanly - 2702271474")