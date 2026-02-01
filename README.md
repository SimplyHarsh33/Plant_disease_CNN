# 🌿 Plant Disease Detection & Doctor AI

## 📋 Project Overview
**Objective**: Classify plant leaf diseases from images and recommend specific treatments.
**Impact**: Combines Computer Vision (CV) with real-world Agricultural utility to help farmers and gardeners save their crops.

## 🏗️ Architecture & Pipeline (Project Bible Implementation)

This system implements a comprehensive **Deep Learning Pipeline** combined with a **Knowledge Base Lookup**:

1.  **Input Layer**: User uploads an image via the **Streamlit Web UI**.
2.  **Preprocessing**:
    *   Images are resized to $224 \times 224$ pixels.
    *   Pixel values normalized to [0, 1] range.
3.  **Model Inference**:
    *   **Architecture**: **MobileNetV2** (CNN).
    *   **Technique**: **Transfer Learning** (Pretrained on ImageNet).
    *   The model extracts deep features and classifies into 38 disease categories (or 10 for current tomato subset).
4.  **Treatment Lookup**:
    *   The predicted label (e.g., `Tomato___Early_blight`) queries the `disease_info.py` database.
    *   Retrieves: Disease description, Symptoms, Treatment, and Prevention tips.
5.  **Output**: Results displayed in a responsive dashboard.

## 📊 Dataset: PlantVillage
We utilize the industry-standard **PlantVillage Dataset**.
*   **Current Training**: Tomato Disease Subset.
*   **Total Images**: ~14,500 images processed using Keras `ImageDataGenerator`.

## 🧠 Model: MobileNetV2
We use **MobileNetV2** as requested in the Project Design.
*   **Why?** It is lightweight, fast, and highly accurate for mobile/edge use cases.
*   **Training**: Fine-tuned on the plant dataset for 10 epochs.
*   **Accuracy**: High accuracy expected with Transfer Learning.

## 🛠️ Technology Stack
*   **Deep Learning**: TensorFlow / Keras
*   **Language**: Python
*   **Computer Vision**: OpenCV, Pillow
*   **Web Framework**: Streamlit
*   **Data Processing**: NumPy, Pandas

## 🚀 How to Run

1.  **Train the Model**
    Double-click `run_training.bat`
    *Or run:* `python train_mobilenet.py`

2.  **Start the App**
    Double-click `run_app.bat`
    *Or run:* `streamlit run app_streamlit.py`

3.  **Access UI**
    The app will automatically open in your browser (usually http://localhost:8501).

## 🩺 Supported Diseases
Includes comprehensive coverage for Tomato diseases:
*   Bacterial Spot
*   Early Blight
*   Late Blight
*   Leaf Mold
*   Septoria Leaf Spot
*   Spider Mites
*   Target Spot
*   Yellow Leaf Curl Virus
*   Mosaic Virus
*   Healthy

---
*Built according to Project Bible Phase 1-4*
