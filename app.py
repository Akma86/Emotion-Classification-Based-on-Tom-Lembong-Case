import time
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ---------------------------- CONFIG / STYLE ----------------------------
st.set_page_config(
    page_title="Emotion Mining - Tom Lembong Case",
    page_icon="💬",
    layout="wide",
)

st.markdown("""
<style>
/* Hero Section */
.hero {
    background: linear-gradient(135deg, #3b82f6 0%, #9333ea 100%);
    padding: 40px 20px; border-radius: 24px; color: white;
    text-align: center; margin-bottom: 30px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}
.hero h1 {margin: 0; font-size: 2.8rem; text-shadow: 1px 1px 4px rgba(0,0,0,0.3);}
.hero p {margin-top: 10px; font-size: 1.2rem; opacity: 0.95;}

/* Card */
.card {
    border-radius: 24px; padding: 28px;
    background: rgba(255,255,255,0.85); margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.4);
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}

/* Probability bars */
.prob-bar {height:16px; background:#e5e7eb; border-radius:999px; overflow:hidden; margin-bottom:8px;}
.prob-fill {
    height:100%; border-radius:999px; 
    background: linear-gradient(90deg,#3b82f6,#9333ea);
    transition: width 1s ease-in-out;
}

/* Footer */
footer {visibility: hidden;}

/* Button hover */
.stButton>button:hover {
    background: linear-gradient(135deg,#3b82f6,#9333ea);
    color: white; border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------- HERO SECTION ----------------------------
st.markdown("""
<div class="hero">
    <h1>💬 Emotion Mining</h1>
    <p>Klasifikasi emosi komentar Instagram terkait <b>kasus Tom Lembong</b></p>
</div>
""", unsafe_allow_html=True)

# ---------------------------- MODEL LOADER + QUANTIZATION ----------------------------
MODEL_PATH = "AkmalYafa/Emotion-Classification-Tom-Lembong-Case"
LABELS = ["😢 SADNESS", "😡 ANGER", "🙏 HOPE", "😞 DISAPPOINTMENT", "🤝 SUPPORT"]

@st.cache_resource
def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    # DYNAMIC QUANTIZATION
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return tokenizer, model

tokenizer, model = load_model(MODEL_PATH)

# ---------------------------- INFERENCE ----------------------------
def infer_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    dt = (time.time()-t0)*1000
    probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
    pred_idx = int(torch.argmax(outputs.logits))
    return probs, pred_idx, dt

# ---------------------------- UI ----------------------------
tab1, tab2 = st.tabs(["📝 Input Komentar", "📚 Tentang Dataset"])

with tab1:
    st.subheader("📝 Masukkan Komentar Instagram")
    text = st.text_area(
        "Tulis komentar publik di sini...", height=150,
        placeholder="Contoh: 'Parah banget kasus ini, kecewa banget sama pemerintah.'"
    )

    if st.button("🚀 Klasifikasikan Emosi"):
        if not text.strip():
            st.warning("⚠️ Masukkan komentar terlebih dahulu!")
        else:
            probs, pred_idx, latency = infer_text(text)

            # Prediksi Utama
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center;'>🎯 Prediksi Utama</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align:center; font-size:2.4rem;'>{LABELS[pred_idx]}</h1>", unsafe_allow_html=True)
            st.caption(f"⚡ Latency: {latency:.1f} ms (quantized)")
            st.markdown('</div>', unsafe_allow_html=True)

            # Distribusi Probabilitas
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Distribusi Emosi")
            for i, label in enumerate(LABELS):
                st.markdown(f"<b>{label}</b>: {probs[i]*100:.2f}%", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="prob-bar">
                        <div class="prob-fill" style="width:{probs[i]*100:.2f}%;"></div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("📚 Tentang Dataset")
    st.markdown("""
    **Latar Belakang Kasus:**  
    Tom Lembong, mantan Menteri Perdagangan, pada tahun 2024 ditetapkan sebagai tersangka kasus korupsi impor gula.  
    Ia sempat dijatuhi hukuman, namun kemudian mendapat abolisi dari Presiden.  

    **Kategori Emosi:**  
    - 😢 **SADNESS**  
    - 😡 **ANGER**  
    - 🙏 **HOPE**  
    - 😞 **DISAPPOINTMENT**  
    - 🤝 **SUPPORT**  

    **Evaluasi:**  
    Model dievaluasi dengan **Macro F1-Score** untuk menjaga keseimbangan antar kelas.
    """)

# ---------------------------- FOOTER ----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Made with ❤️ by Akmal | Powered by HuggingFace & Streamlit | Quantized for faster inference</p>",
    unsafe_allow_html=True
)
