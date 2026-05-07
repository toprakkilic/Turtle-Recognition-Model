import streamlit as st
import PIL.Image as Image
import os
import tempfile
from main import TurtleRecognizer
from config import TrainingConfig

st.set_page_config(page_title="Turtle ID Expert", page_icon="🐢", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 12px; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 2rem; color: #1f77b4; }
    .prediction-card {
        padding: 20px;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .header-box {
        padding: 10px;
        background: linear-gradient(90deg, #007bff, #00d4ff);
        color: white;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_model():
    cfg = TrainingConfig()
    return TurtleRecognizer(cfg)

recognizer = get_model()

st.markdown('<div class="header-box"><h1>🐢 Deniz Kaplumbağası Biyometrik Tanımlama Sistemi</h1></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Analiz edilecek kaplumbağa fotoğrafını buraya bırakın (v5 Fine-Tuned Model)", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")
    
    img = Image.open(uploaded_file)
    
    with col1:
        st.markdown('### 🖼️ Giriş Görüntüsü')
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('### 📊 Analiz ve Sonuç')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            img.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            with st.spinner('Yapay zeka desenleri analiz ediyor...'):
                name, confidence = recognizer.predict(temp_path)
            
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            if confidence >= 60.0:
                st.success(f"### ✅ SİSTEM ONAYI\n**Tespit Edilen Kimlik:** {name}")
            else:
                st.warning(f"### ⚠️ DÜŞÜK GÜVEN SEVİYESİ\n**Tahmin Edilen:** {name}")
                st.info("Güven oranı %60 eşik değerinin altında olduğu için doğrulama yapılamadı.")
            
            st.metric("Doğruluk Payı", f"%{confidence:.2f}")
            st.progress(int(confidence))
            
            st.caption(f"Kullanılan Model: MobileNetV2 | Sınıf Sayısı: 19 | Eşik: %{recognizer.threshold}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
else:
    st.info("Sistemi başlatmak için lütfen bir fotoğraf yükleyin.")
    st.image("https://images.unsplash.com/photo-1544924405-4533bbb1f06d?auto=format&fit=crop&q=80&w=1000", width=800)