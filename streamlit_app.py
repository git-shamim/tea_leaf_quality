import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import tempfile
import requests
from utils import preprocess_image

# Paths
MODEL_LOCAL_PATH = "models/tea_model.h5"
MODEL_GCS_URL = "https://storage.googleapis.com/tea-leaf-assets/models/tea_model.h5"

# Load model with fallback
@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_LOCAL_PATH):
            return tf.keras.models.load_model(MODEL_LOCAL_PATH)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                response = requests.get(MODEL_GCS_URL)
                tmp.write(response.content)
                return tf.keras.models.load_model(tmp.name)
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

model = load_model()
if model is None:
    st.stop()

# Define class names and explanations
class_names = ["T1 - Fresh (1–2 days)", "T2 - Mid-aged (3–4 days)", "T3 - Aged (5–7 days)", "T4 - Old (7+ days)"]
class_explanations = {
    "T1 - Fresh (1–2 days)": "🍃 Top quality: Ideal for making premium tea.",
    "T2 - Mid-aged (3–4 days)": "✅ Good quality: Suitable for commercial blends.",
    "T3 - Aged (5–7 days)": "⚠️ Moderate quality: Less flavor, used in lower-grade tea.",
    "T4 - Old (7+ days)": "❌ Not recommended: Overaged for tea processing."
}

# UI: Sidebar description
st.sidebar.title("🫖 Tea Quality Classifier")
st.sidebar.markdown("""
This app classifies tea leaves into one of four quality grades (T1–T4) based on images, using a deep learning model trained on the [TeaLeafAgeQuality Dataset](https://data.mendeley.com/datasets/7t964jmmy3/1).
""")

# UI: Main header
st.title("🍵 Predict Tea Leaf Quality from Image")

# File upload
uploaded_file = st.file_uploader("Upload a tea leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_array = preprocess_image(uploaded_file)

        # Predict
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]

        # Output
        st.subheader("🔍 Prediction")
        if confidence < 0.6:
            st.warning(f"⚠️ Confidence: {confidence:.2f}. This image may not be a valid tea leaf.")
        else:
            st.success(f"✅ Predicted Class: {predicted_class}")
            st.info(f"💡 {class_explanations[predicted_class]}")

        st.markdown(f"**Prediction Confidence:** {confidence:.2%}")

    except Exception as e:
        st.error(f"❌ Could not process the image. Error: {e}")
else:
    st.info("Please upload a tea leaf image to begin.")
