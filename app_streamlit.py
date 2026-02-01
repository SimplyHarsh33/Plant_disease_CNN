"""
Plant Disease Detector & Doctor 🌿
Streamlit UI Implementation - Phase 4 of Project Bible
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from disease_info import get_disease_info

# Configure Page
st.set_page_config(
    page_title="Plant Disease Doctor",
    page_icon="🌿",
    layout="centered"
)

# Custom CSS for glassmorphism
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: white;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #4ade80;
        text-align: center;
        padding: 20px;
    }
    .report-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('plant_disease_model.h5')
        return model
    except Exception as e:
        return None

@st.cache_data
def load_class_names():
    try:
        with open("class_indices.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    except:
        return []

# Application Header
st.markdown("<h1 class='main-header'>Plant Disease Detector & Doctor 🌿</h1>", unsafe_allow_html=True)
st.write("Upload a leaf image to detect diseases and get treatment recommendations.")

# Load Model
with st.spinner("Loading AI Model..."):
    model = load_model()
    class_names = load_class_names()

if model is None:
    st.error("Model not found! Please run 'Run Training' first.")
else:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # 1. Preprocess the image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Leaf', use_column_width=True)
                
                # Resize and scale to match training (224x224)
                img = image.resize((224, 224))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # 2. Make Prediction
                with st.spinner("Analyzing..."):
                    predictions = model.predict(img_array)
                    predicted_class_index = np.argmax(predictions)
                    confidence = np.max(predictions) * 100
                    
                    if len(class_names) > predicted_class_index:
                        result = class_names[predicted_class_index]
                    else:
                        result = "Unknown"

                # 3. Get Doctor Logic
                info = get_disease_info(result)

            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.stop()

        with col2:
            st.markdown(f"""
            <div class='report-card'>
                <h3 style='color: #4ade80;'>Analysis Results</h3>
                <p><b>Diagnosis:</b> {info['disease']}</p>
                <p><b>Confidence:</b> {confidence:.2f}%</p>
                <p><b>Status:</b> {'✅ Healthy' if 'healthy' in result.lower() else '⚠️ Disease Detected'}</p>
            </div>
            """, unsafe_allow_html=True)

        # 4. Display Treatment
        st.markdown("---")
        st.subheader("👨‍⚕️ Doctor's Prescription")
        
        tab1, tab2, tab3 = st.tabs(["Description", "Treatment", "Prevention"])
        
        with tab1:
            st.info(info['description'])
            st.write("**Symptoms:** " + info['symptoms'])
            
        with tab2:
            if isinstance(info['treatment'], list):
                for step in info['treatment']:
                    st.write(f"- {step}")
            else:
                st.write(info['treatment'])
                
        with tab3:
            st.success(info['prevention'])
