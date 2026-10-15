import os
import streamlit as st
import numpy as np
import cv2
import imutils
from PIL import Image

from gradcam import make_gradcam_heatmap, overlay_gradcam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(
    page_title="Brain Tumor Detection & Grad-CAM AI",
    page_icon="🧠",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        text-align: center;
        color: #4B5563;
        margin-bottom: 1.8rem;
    }
    .result-card-yes {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.2rem;
        color: #991B1B;
        text-align: center;
        font-size: 1.35rem;
        font-weight: 700;
    }
    .result-card-no {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        border: 2px solid #22C55E;
        border-radius: 12px;
        padding: 1.2rem;
        color: #166534;
        text-align: center;
        font-size: 1.35rem;
        font-weight: 700;
    }
    .info-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    candidates = [
        'models/cnn-parameters-improvement-23-0.91.model',
        'models/cnn-parameters-improvement-23-0.91.h5'
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                import tf_keras as keras
                return keras.models.load_model(candidate)
            except Exception:
                from tensorflow.keras.models import load_model
                return load_model(candidate)
    return None

def crop_brain_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if not cnts:
        return image

    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    cropped = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    if cropped.size == 0:
        return image
    return cropped

def process_and_predict(image_bgr, model):
    cropped_bgr = crop_brain_contour(image_bgr)
    resized = cv2.resize(cropped_bgr, (240, 240), interpolation=cv2.INTER_CUBIC)
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

    raw_pred = model.predict(input_tensor, verbose=0)
    prob = float(raw_pred[0][0])
    has_tumor = prob > 0.5
    confidence = prob if has_tumor else (1.0 - prob)
    return has_tumor, prob, confidence, cropped_bgr, input_tensor

st.markdown('<div class="main-title">🧠 Brain Tumor Detection & Grad-CAM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">MRI Image Classification with Explainable Visual Attention Heatmaps</div>', unsafe_allow_html=True)

model = get_model()
if model is None:
    st.error("Model could not be loaded. Please verify that 'models/cnn-parameters-improvement-23-0.91.h5' exists.")
    st.stop()

# Sidebar Controls
sidebar = st.sidebar
sidebar.header("📁 Scan Input Selection")
input_option = sidebar.radio("Select input mode:", ["Select Sample Scan", "Upload Custom MRI Scan"])

selected_img_bgr = None

if input_option == "Select Sample Scan":
    sample_category = sidebar.selectbox("Category:", ["Tumor Present (Yes)", "No Tumor (No)"])
    folder = "yes" if "Yes" in sample_category else "no"
    if os.path.exists(folder):
        sample_files = os.listdir(folder)[:15]
        selected_file = sidebar.selectbox("Sample Image:", sample_files)
        if selected_file:
            path = os.path.join(folder, selected_file)
            selected_img_bgr = cv2.imread(path)
else:
    uploaded = sidebar.file_uploader("Upload Brain MRI Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        selected_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

sidebar.markdown("---")
sidebar.header("🔍 Grad-CAM Settings")
show_gradcam = sidebar.checkbox("Enable Grad-CAM Heatmap", value=True)
cam_alpha = sidebar.slider("Heatmap Blending Opacity (α)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
cam_threshold = sidebar.slider("Hotspot Focus Threshold", min_value=0.05, max_value=0.40, value=0.15, step=0.05, help="Controls hotspot focus and eliminates diffuse background activations")

colormap_name = sidebar.selectbox("Color Map:", ["JET (Classic Hotspot)", "TURBO", "INFERNO", "HOT"])
cmap_dict = {
    "JET (Classic Hotspot)": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "HOT": cv2.COLORMAP_HOT
}
selected_cmap = cmap_dict[colormap_name]

if selected_img_bgr is not None:
    has_tumor, prob, confidence, cropped_bgr, input_tensor = process_and_predict(selected_img_bgr, model)

    if show_gradcam:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])
    else:
        col1, col2, col4 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("1. Original Scan")
        rgb_orig = cv2.cvtColor(selected_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(rgb_orig, use_container_width=True)

    with col2:
        st.subheader("2. Cropped Brain")
        rgb_cropped = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        st.image(rgb_cropped, use_container_width=True)

    if show_gradcam:
        with col3:
            st.subheader("3. Grad-CAM Localization")
            heatmap = make_gradcam_heatmap(input_tensor, model, cropped_img=cropped_bgr)
            cam_overlay, _ = overlay_gradcam(cropped_bgr, heatmap, alpha=cam_alpha, colormap=selected_cmap, threshold=cam_threshold)
            rgb_overlay = cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB)
            st.image(rgb_overlay, use_container_width=True)
            st.caption("🔴 Red/Yellow = High Neural Activation (Tumor Region)")

    with col4:
        st.subheader("Diagnostic Assessment")
        if has_tumor:
            st.markdown('<div class="result-card-yes">⚠️ TUMOR DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-card-no">✅ NO TUMOR DETECTED</div>', unsafe_allow_html=True)

        st.write("")
        st.metric(label="Model Confidence", value=f"{confidence * 100:.2f}%")
        st.metric(label="Tumor Probability", value=f"{prob:.4f}")
        st.progress(min(max(prob, 0.0), 1.0))
        
        st.markdown(f"""
        <div class="info-card">
            <b>🔬 Explainable AI (Multi-Layer Grad-CAM):</b><br>
            Collects gradients across all network layers (<code>conv0</code>, <code>bn0</code>, <code>activation</code>, <code>max_pool0</code>, <code>max_pool1</code>) to project a smooth, focused visual activation map directly over the tumor mass.
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👈 Please select a sample scan or upload an MRI image in the sidebar to run detection and view Grad-CAM.")
