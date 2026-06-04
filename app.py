import streamlit as st
import pickle
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import preprocess

# Konstanta
MODEL_DIR = "models"

# =================================
# PAGE CONFIG & STYLING
# =================================
st.set_page_config(
    page_title="SentimenAnalytica - LSTM System",
    layout="centered"
)

st.markdown("""
    <style>
    .stApp { background-color: #F0F4F8; }
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
    .stButton>button:hover { background-color: #1E3A8A; color: white; }
    .result-box {
        padding: 20px; border-radius: 5px; margin-bottom: 20px;
        border-left: 5px solid #1E40AF; background-color: #FFFFFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .positive-text { color: #1E40AF; font-weight: bold; font-size: 1.2rem; }
    .negative-text { color: #B91C1C; font-weight: bold; font-size: 1.2rem; }
    section[data-testid="stSidebar"] { background-color: #1E293B; }
    section[data-testid="stSidebar"] * { color: #F8FAFC !important; }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #FFFFFF; color: #475569; text-align: center;
        padding: 10px; font-size: 12px; border-top: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allowed_html=True)

# =================================
# LOAD ASSETS
# =================================
@st.cache_resource
def load_assets():
    # Menggunakan pemanggilan aman Keras bawaan
    model_path = os.path.join(MODEL_DIR, "dl_model.h5")
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, tokenizer, config["MAX_LEN"]

try:
    model, tokenizer, max_len = load_assets()
except Exception as e:
    st.error("Sistem gagal memuat komponen model standar.")
    st.stop()

# =================================
# SIDEBAR
# =================================
with st.sidebar:
    st.markdown("### Informasi Proyek")
    st.markdown("""
    **Pengembang:** Sanly - 2702271474  
    **Arsitektur Model:** LSTM Network Dedicated
    """)
    st.divider()
    st.caption("Deep Learning Project - 2026")

# =================================
# MAIN CONTENT AREA
# =================================
st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>Analisis Sentimen Teks (LSTM)</h2>", unsafe_allowed_html=True)
st.markdown("<p style='text-align: center; color: #475569;'>Sistem klasifikasi teks otomatis menggunakan arsitektur neural network LSTM.</p>", unsafe_allowed_html=True)
st.write("")

text_input = st.text_area("Masukkan Teks Ulasan:", height=150, placeholder="Tulis ulasan Anda di sini...")

if st.button("Proses Analisis"):
    if not text_input.strip():
        st.info("Pesan: Masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Menganalisis data..."):
            cleaned_text = preprocess(text_input)
            if not cleaned_text.strip():
                st.warning("Pesan: Teks tidak valid untuk dianalisis.")
            else:
                seq = tokenizer.texts_to_sequences([cleaned_text])
                padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
                score = float(model.predict(padded, verbose=0)[0][0])
                
                st.write("---")
                st.markdown("#### Hasil Klasifikasi")
                confidence = score if score >= 0.5 else (1 - score)
                
                if score >= 0.5:
                    label = "SENTIMEN POSITIF"
                    css_class = "positive-text"
                else:
                    label = "SENTIMEN NEGATIF"
                    css_class = "negative-text"
                
                st.markdown(f"""
                    <div class="result-box">
                        <p style="margin:0; font-size: 0.9rem; color: #64748B;">Prediksi:</p>
                        <p class="{css_class}">{label}</p>
                        <p style="margin:0; font-size: 0.9rem; color: #64748B;">Teks Bersih: <i>"{cleaned_text}"</i></p>
                    </div>
                """, unsafe_allowed_html=True)
                st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence * 100:.2f}%")

st.markdown('<div class="footer">Created by Sanly - 2702271474</div>', unsafe_allowed_html=True)