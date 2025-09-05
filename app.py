import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ================= CONFIG STREAMLIT =================
st.set_page_config(
    page_title="Emotion Mining - Tom Lembong Case",
    page_icon="💬",
    layout="wide"
)

# ================= MODEL CONFIG =================
MODEL_PATH = "AkmalYafa/Emotion-Classification-Tom-Lembong-Case"
LABELS = ["SADNESS", "ANGER", "HOPE", "DISAPPOINTMENT", "SUPPORT"]

# ================= LOAD MODEL =================
@st.cache_resource(show_spinner=True)
def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model(MODEL_PATH)

# ================= SIDEBAR =================
st.sidebar.title("💬 Emotion Mining")
st.sidebar.markdown("Masukkan komentar Instagram terkait kasus **Tom Lembong** dan dapatkan klasifikasi emosi.")

input_text = st.sidebar.text_area("📝 Komentar Instagram", height=150)
predict_btn = st.sidebar.button("🚀 Klasifikasikan")

# ================= MAIN PAGE =================
st.title("📊 Emotion Mining - Kasus Tom Lembong")

with st.expander("ℹ️ Tentang Dataset", expanded=True):
    st.markdown("""
    **Latar Belakang Kasus:**  
    Tom Lembong, mantan Menteri Perdagangan, pada tahun 2024 ditetapkan sebagai tersangka kasus korupsi impor gula. 
    Ia sempat dijatuhi hukuman, namun kemudian mendapat abolisi dari Presiden. 
    Kasus ini menimbulkan berbagai emosi publik di media sosial, khususnya Instagram.

    **Tujuan Dataset:**  
    Mengklasifikasikan komentar publik ke dalam **5 kategori emosi**:
    - 😢 **SADNESS**
    - 😡 **ANGER**
    - 🙏 **HOPE**
    - 😞 **DISAPPOINTMENT**
    - 🤝 **SUPPORT**

    **Evaluasi:**  
    Model dievaluasi menggunakan **Macro F1-Score**, agar performa tetap seimbang meskipun distribusi label tidak merata.
    """)

st.markdown("---")

# ================= PREDIKSI =================
if predict_btn:
    if not input_text.strip():
        st.warning("⚠️ Masukkan komentar terlebih dahulu!")
    else:
        # Tokenisasi
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()

        # Ambil hasil
        pred_idx = int(torch.argmax(outputs.logits))
        pred_label = LABELS[pred_idx]

        # Layout 2 kolom: Hasil + Chart
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📌 Hasil Prediksi")
            st.success(f"**Emosi:** {pred_label}")

        with col2:
            st.markdown("### 📊 Distribusi Probabilitas")
            st.bar_chart({label: prob for label, prob in zip(LABELS, probs)})

st.markdown("---")
st.caption("Made with ❤️ by Akmal | Powered by HuggingFace & Streamlit")
