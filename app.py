import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Load the trained model
model = load_model('deepfake_detector.h5')  # Updated model path

# Image size for ResNet50
IMG_SIZE = (224, 224)

# Streamlit App
st.set_page_config(page_title="Deepfake Detector", layout="wide")
st.title("🧠 Deepfake Detection System")
st.markdown("### Upload an image to check for Deepfakes")

# Sidebar
with st.sidebar:
    st.markdown("## How it Works")
    st.markdown("1. Upload an image.\n2. Click 'Detect'.\n3. See the result instantly.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width = 400)

    if st.button("Detect"):
        try:
            processed_image = preprocess_image(uploaded_file)
            prediction = model.predict(processed_image)

            class_labels = ["Real", "Deepfake"]
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_label = class_labels[predicted_class_index]
            confidence = np.max(prediction) * 100

            if predicted_class_label == "Deepfake":
                st.error(f"🚨 **Detected as {predicted_class_label}** (Confidence: {confidence:.2f}%)")
            else:
                st.success(f"✅ **Detected as {predicted_class_label}** (Confidence: {confidence:.2f}%)")

        except Exception as e:
            st.error(f"Error: {str(e)}")