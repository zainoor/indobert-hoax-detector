import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect
import re
import summa

# Konfigurasi halaman
st.set_page_config(page_title="IndoHoaxDetector 🔎", layout="centered")

# --- Fungsi Utilitas ---
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9À-ÿ\s.,!?;:'\"()-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_input(text):
    return len(text.strip()) >= 30 and re.search(r"[a-zA-Z]{3,}", text)

def describe_confidence(score):
    if score > 0.95:
        return "Sangat Yakin"
    elif score > 0.85:
        return "Cukup Yakin"
    elif score > 0.6:
        return "Yakin"
    else:
        return "Kurang Yakin"

def summarize_text(text, max_sentences=3):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return '. '.join(sentences[:max_sentences]) + '.' if len(sentences) > max_sentences else text

def summarize_with_summa(text):
    try:
        from summa.summarizer import summarize
        summary = summarize(text, ratio=0.2)
        return summary.strip() if summary else summarize_text(text)
    except Exception:
        return summarize_text(text)

# Load model dari HuggingFace (caching diaktifkan)
@st.cache_resource
def load_model():
    model_id = "zainoor/indo-hoax-detector-indobert-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return tokenizer, model

tokenizer, model = load_model()

# --- CSS Kustom ---
st.markdown("""
<style>
.custom-box {
    background-color: #fff;
    border: 2px solid #b52f2f;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
    margin: 20px 0;
}
.stTextArea textarea {
    background-color: #f9f9f9 !important;
    border: 1px solid #ccc !important;
    border-radius: 10px !important;
    padding: 15px !important;
    font-size: 16px !important;
}
.stButton > button {
    background-color: #b52f2f;
    color: white;
    font-weight: bold;
    font-size: 16px;
    border-radius: 8px;
    padding: 0.6em 1.5em;
    border: none;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #ebebeb;
    border: 2px solid #b52f2f;
    transform: scale(1.04);
}
           
textarea::placeholder {
    font-size: 25px !important;
    color: #999 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Judul Aplikasi ---
st.markdown("<h1 style='text-align: center;'>IndoHoaxDetector 🔎</h1>", unsafe_allow_html=True)

# --- Form Input ---
st.markdown("<div class='custom-box'><h3>📰 Masukkan Artikel Berita</h3>", unsafe_allow_html=True)

st.markdown("""
<p style='margin-top: -10px; font-size: 15px; color: #555;'>
🔹 Tempel teks artikel dari berita online atau sumber lainnya.<br>
🔹 Minimal 30 karakter dan gunakan <b>Bahasa Indonesia</b>.<br>
🔹 Pastikan teks cukup lengkap untuk dianalisis (bukan hanya judul).
</p>
""", unsafe_allow_html=True)

text = st.text_area(
    label="Masukkan teks artikel",
    height=200,
    placeholder="Berita apa yang mau di cek?",
    label_visibility="collapsed"
)
submit = st.button("🔍 Periksa")
st.markdown("</div>", unsafe_allow_html=True)

# --- Proses Deteksi ---
if submit:
    text = text.strip()
    if not text:
        st.warning("⚠️ Masukkan teks terlebih dahulu.")
    elif not is_valid_input(text):
        st.warning("⚠️ Teks terlalu pendek atau tidak valid.")
    else:
        with st.spinner("⏳ Menganalisis artikel..."):
            try:
                if detect(text) != "id":
                    st.warning("⚠️ Artikel harus menggunakan Bahasa Indonesia.")
                    st.stop()
            except Exception as e:
                st.warning(f"Gagal mendeteksi bahasa: {e}")

            cleaned = clean_text(text)
            inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=256)

            with torch.no_grad():
                logits = model(**inputs).logits / 1.5  # Penyesuaian suhu
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

            # Hasil klasifikasi
            label_text = "🚨 **Hoax**" if pred == 1 else "✅ **Valid**"
            st.markdown("## 🔎 Hasil Deteksi")
            st.success(f"Hasil Deteksi: {label_text}")
            st.markdown(f"**Tingkat Keyakinan:** {confidence:.2%} ({describe_confidence(confidence)})")

            if confidence < 0.60:
                st.warning("⚠️ Hasil deteksi kurang meyakinkan. Harap cek ulang ke sumber terpercaya.")

            # Ringkasan artikel jika cukup panjang
            if len(cleaned.split()) > 50:
                st.markdown("### ✂️ Ringkasan Artikel:")
                summary = summarize_with_summa(cleaned)
                st.info(summary)

# --- Footer ---
st.markdown("""
<hr style='border-top: 1px solid #bbb;'>
<div style='text-align: center; font-size: 14px;'>
    Dibuat oleh <b>Mohammad Ramadhan Zainoor</b> · © 2025 ·
    <a href="https://github.com/zainoor" target="_blank">GitHub Repo</a>
</div>
""", unsafe_allow_html=True)
