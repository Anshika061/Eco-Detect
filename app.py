import gdown
import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
st.set_page_config(page_title="EcoDetect AI", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #0a0f1c, #12192b, #1a2238);
    color: white;
}

/* HERO TITLE */
.title {
    text-align:center;
    font-size:50px;
    font-weight:700;
    background: linear-gradient(90deg, #8e44ad, #00c9ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* HERO SUBTITLE */
.subtitle {
    text-align:center;
    font-size:18px;
    color:#bbb;
    margin-bottom:30px;
}

/* HERO */
.hero {
    text-align:center;
    padding: 20px;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    transition: 0.3s;
    border: 1px solid rgba(255,255,255,0.1);
}

.card:hover {
    transform: translateY(-10px);
    box-shadow: 0 0 30px rgba(142,68,173,0.8);
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* LABEL */
.label {
    color:#00c9ff;
    font-weight:bold;
}

/* CONFIDENCE */
.confidence {
    color:#9b59b6;
    font-weight:bold;
}

/* BUTTON */
.stButton button {
    background: linear-gradient(45deg, #8e44ad, #00c9ff);
    color: white;
    border-radius: 20px;
}

/* FOOTER */
.footer {
    text-align:center;
    color:#aaa;
    margin-top:60px;
}

/* IMAGE HOVER */
img {
    transition: 0.3s;
}
img:hover {
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = load_model("ecodetect_model.h5")
classes = ["cardboard","glass","metal","paper","plastic","trash"]

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class='hero'>
    <div class='title'>🌱 EcoDetect AI</div>
    <div class='subtitle'>
        AI-powered waste classification for a smarter and cleaner planet 🚀
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- CHARACTER + UPLOAD ----------------
colA, colB = st.columns([1,2])

with colA:
    st.image("ecodetectimg.jpeg", width=220)
    #st.image("https://cdn-icons-png.flaticon.com/512/4140/4140048.png", width=220)
    st.markdown("### ♻ Smart Waste Sorting Assistant")

with colB:
    st.markdown("### ✨ Upload your waste image and let AI analyze it instantly")
    uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

# ---------------- MAIN LOGIC ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # PREPROCESS
    img = image.resize((150,150))  # match training size
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # PREDICTION
    prediction = model.predict(img_array)
    confidence = np.max(prediction)*100
    class_idx = np.argmax(prediction)
    label = classes[class_idx]

    probs = prediction[0]
    sorted_idx = np.argsort(probs)[::-1]

    col1, col2 = st.columns([1,2])

    # IMAGE CARD
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # RESULT CARD
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown("## 🧾 Result")

        st.markdown(f"### 🏷 Label: <span class='label'>{label.upper()}</span>", unsafe_allow_html=True)

        if label == "trash":
            st.error("Non-Recyclable ❌")
        else:
            st.success("Recyclable ✅")

        st.markdown("### 📊 Confidence")
        st.progress(int(confidence))
        st.markdown(f"<span class='confidence'>{round(confidence,2)}%</span>", unsafe_allow_html=True)

        st.markdown("### 🔎 Other Possibilities")
        for i in sorted_idx[1:3]:
            st.write(f"{classes[i]} ({round(probs[i]*100,2)}%)")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
© 2026 EcoDetect AI | Built by Anshika 🚀
</div>
""", unsafe_allow_html=True)