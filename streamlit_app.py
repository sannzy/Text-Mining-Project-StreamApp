import os
import pickle
import joblib
import streamlit as st
from preprocessing import preprocess

# Konstanta
MODEL_DIR = "models"

# =================================
# PAGE CONFIG & STYLING (Professional Blue)
# =================================
st.set_page_config(
    page_title="SentimenAnalytica - Integrated System", 
    layout="centered"
)

# Custom CSS untuk tema biru profesional dan menghilangkan emoticons bawaan
st.markdown("""
    <style>
    /* Background utama */
    .stApp {
        background-color: #F0F4F8;
    }
    
    /* Tombol Proses Biru */
    .stButton>button {
        width: 100%;
        background-color: #1E40AF;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.6rem;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1E3A8A;
        border: none;
        color: white;
    }

    /* Container Hasil */
    .result-box {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        border-left: 5px solid #1E40AF;
        background-color: #FFFFFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .positive-text {
        color: #1E40AF;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .negative-text {
        color: #B91C1C;
        font-weight: bold;
        font-size: 1.2rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #FFFFFF;
        color: #475569;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allowed_html=True)


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
            """Menghapus parameter quantization_config di semua layer secara rekursif."""
            if isinstance(config, dict):
                config.pop('quantization_config', None)
                for key, value in config.items():
                    clean_quantization_config(value)
            elif isinstance(config, list):
                for item in config:
                    clean_quantization_config(item)
            return config

        # Buka file .h5 secara manual untuk mengekstrak dan memperbaiki arsitektur model
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
    """Prediksi menggunakan model Machine Learning."""
    model, tfidf = load_ml()
    vec = tfidf.transform([preprocess(text)])
    proba = float(model.predict_proba(vec)[0][1])
    label = "Positif" if proba >= 0.5 else "Negatif"
    return label, proba


def predict_dl(text):
    """Prediksi menggunakan model Deep Learning (LSTM) dengan deteksi error internal."""
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
# SIDEBAR (Identitas Kamu)
# =================================
with st.sidebar:
    st.markdown("### Informasi Proyek")
    st.markdown("""
    **Pengembang:** Sanly - 2702271474  
    
    **Sistem Informasi:** Proyek Akhir Text Mining  
    Dual Model Option System  
    """)
    st.divider()
    st.caption("Deep Learning Project - 2026")


# =================================
# MAIN CONTENT AREA
# =================================
st.markdown("<h2 style='text-align: center; color: #1E40AF;'>Analisis Sentimen Teks</h2>", unsafe_allowed_html=True)
st.markdown("<p style='text-align: center; color: #475569;'>Sistem klasifikasi teks otomatis berbasis Machine Learning dan Deep Learning.</p>", unsafe_allowed_html=True)
st.write("")

# Input Pilihan Model
model_choice = st.radio(
    "Pilih Model Analisis:",
    ["Machine Learning (Logistic Regression)", "Deep Learning (LSTM)"],
    horizontal=False,
)

# Input Teks dari Pengguna
text_input = st.text_area(
    "Masukkan teks ulasan yang ingin dianalisis:", 
    height=140,
    placeholder="Tulis ulasan Anda di sini tanpa simbol khusus..."
)

# Tombol Aksi
if st.button("Proses Analisis"):
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
                st.markdown("#### Hasil Klasifikasi")
                
                if label == "Positif":
                    label_display = "SENTIMEN POSITIF"
                    css_class = "positive-text"
                else:
                    label_display = "SENTIMEN NEGATIF"
                    css_class = "negative-text"
                
                st.markdown(f"""
                    <div class="result-box">
                        <p style="margin:0; font-size: 0.9rem; color: #64748B;">Prediksi:</p>
                        <p class="{css_class}">{label_display}</p>
                        <p style="margin:0; font-size: 0.9rem; color: #64748B;">Teks Bersih: <i>"{cleaned_text}"</i></p>
                    </div>
                """, unsafe_allowed_html=True)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{conf * 100:.1f}%")
                with col2:
                    st.caption(f"Probabilitas ke arah Positif: P(positif) = {proba:.4f}")
                    st.progress(proba)
                    
            except Exception as e:
                st.error("Terjadi kesalahan internal pada pemrosesan model:")
                st.code(str(e), language="text")

# Footer
st.markdown('<div class="footer">Created by Sanly - 2702271474</div>', unsafe_allowed_html=True)