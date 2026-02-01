"""
Plant Disease Detection Web Application
Flask backend with trained ML model for plant disease classification
"""

import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import io
import pickle
from pathlib import Path

from disease_info import CLASS_NAMES as ALL_CLASS_NAMES, get_disease_info, DISEASE_INFO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variables
model = None
model_class_names = None
feature_mean = None
feature_std = None
IMG_SIZE = 128
MODEL_LOADED = False


def extract_features(img_array):
    """Extract color and texture features from an image."""
    # Color features
    r_mean = np.mean(img_array[:, :, 0])
    g_mean = np.mean(img_array[:, :, 1])
    b_mean = np.mean(img_array[:, :, 2])
    
    r_std = np.std(img_array[:, :, 0])
    g_std = np.std(img_array[:, :, 1])
    b_std = np.std(img_array[:, :, 2])
    
    # Color histograms (simplified)
    r_hist, _ = np.histogram(img_array[:, :, 0].flatten(), bins=8, range=(0, 1))
    g_hist, _ = np.histogram(img_array[:, :, 1].flatten(), bins=8, range=(0, 1))
    b_hist, _ = np.histogram(img_array[:, :, 2].flatten(), bins=8, range=(0, 1))
    
    # Normalize histograms
    total_pixels = img_array.shape[0] * img_array.shape[1]
    r_hist = r_hist / total_pixels
    g_hist = g_hist / total_pixels
    b_hist = b_hist / total_pixels
    
    # Texture features (gradient-based)
    gray = np.mean(img_array, axis=2)
    
    # Sobel-like gradients
    gx = np.abs(gray[:, :-1] - gray[:, 1:])
    gy = np.abs(gray[:-1, :] - gray[1:, :])
    
    gradient_mean = np.mean(gx) + np.mean(gy)
    gradient_std = np.std(gx) + np.std(gy)
    
    # Edge density
    edge_threshold = 0.1
    edge_density = np.sum(gx > edge_threshold) / gx.size + np.sum(gy > edge_threshold) / gy.size
    
    # Combine all features
    features = np.concatenate([
        [r_mean, g_mean, b_mean],
        [r_std, g_std, b_std],
        r_hist, g_hist, b_hist,
        [gradient_mean, gradient_std, edge_density],
        [g_mean / (r_mean + 0.001)],  # Green ratio
        [(r_mean + g_mean * 0.5) / (r_mean + g_mean + b_mean + 0.001)],  # Brown ratio
    ])
    
    return features


def load_trained_model():
    """Load the trained model from disk."""
    global model, model_class_names, feature_mean, feature_std, IMG_SIZE, MODEL_LOADED
    
    model_path = Path("model") / "plant_disease_model.pkl"
    
    if not model_path.exists():
        print("[!] Trained model not found. Please run train_model.py first.")
        print("[!] Using fallback color-based analysis...")
        MODEL_LOADED = False
        return False
    
    try:
        print("[*] Loading trained model...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        model_class_names = model_data['class_names']
        feature_mean = model_data['feature_mean']
        feature_std = model_data['feature_std']
        IMG_SIZE = model_data.get('img_size', 128)
        
        print(f"[*] Model loaded successfully!")
        print(f"[*] Classes: {model_class_names}")
        MODEL_LOADED = True
        return True
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        MODEL_LOADED = False
        return False


def preprocess_image(image_data):
    """Preprocess image for analysis."""
    img = Image.open(io.BytesIO(image_data))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    return img_array


def predict_disease(image_data):
    """Run prediction on the image using the trained model."""
    global model, model_class_names, feature_mean, feature_std, MODEL_LOADED
    
    # Preprocess image
    img_array = preprocess_image(image_data)
    
    # Extract features - MUST MATCH training script exactly
    # We duplicate the extraction logic here to ensure it's self-contained
    # In a real production app, this should be in a shared utility module
    
    # === 1. BASIC COLOR STATISTICS (12 features) ===
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    feats_color_stats = [
        np.mean(r), np.mean(g), np.mean(b),
        np.std(r), np.std(g), np.std(b),
        np.min(r), np.min(g), np.min(b),
        np.max(r), np.max(g), np.max(b)
    ]
    
    # === 2. COLOR HISTOGRAMS (48 features) ===
    bins = 16
    r_hist, _ = np.histogram(r, bins=bins, range=(0, 1), density=True)
    g_hist, _ = np.histogram(g, bins=bins, range=(0, 1), density=True)
    b_hist, _ = np.histogram(b, bins=bins, range=(0, 1), density=True)
    
    # === 3. TEXTURE / GRADIENTS (7 features) ===
    gray = 0.299*r + 0.587*g + 0.114*b
    gx = np.abs(gray[:, :-1] - gray[:, 1:])
    gy = np.abs(gray[:-1, :] - gray[1:, :])
    
    grad_mean = (np.mean(gx) + np.mean(gy)) / 2
    grad_std  = (np.std(gx) + np.std(gy)) / 2
    grad_max  = max(np.max(gx) if gx.size > 0 else 0, np.max(gy) if gy.size > 0 else 0)
    
    edges_low  = (np.sum(gx > 0.05) + np.sum(gy > 0.05)) / (gx.size + gy.size)
    edges_high = (np.sum(gx > 0.20) + np.sum(gy > 0.20)) / (gx.size + gy.size)
    
    feats_texture = [grad_mean, grad_std, grad_max, edges_low, edges_high]
    
    # === 4. MOMENTS / SHAPE (3 features) ===
    contrast = np.std(gray)
    mean_gray = np.mean(gray)
    skewness = np.mean((gray - mean_gray)**3)
    feats_shape = [contrast, skewness]
    
    # === 5. DOMAIN SPECIFIC (5 features) ===
    greenness = np.mean(g) / (np.mean(r) + np.mean(b) + 1e-6)
    brownness = (np.mean(r) + np.mean(g)) / (np.mean(b) + 1e-6)
    cmax = np.max(img_array, axis=2)
    cmin = np.min(img_array, axis=2)
    saturation = np.mean((cmax - cmin) / (cmax + 1e-6))
    feats_domain = [greenness, brownness, saturation]
    
    # Combine all
    features = np.concatenate([
        feats_color_stats,
        r_hist, g_hist, b_hist,
        feats_texture,
        feats_shape,
        feats_domain
    ])
    
    if MODEL_LOADED and model is not None:
        try:
            # Reshape for sklearn (1, n_features)
            features_reshaped = features.reshape(1, -1)
            
            # Predict
            # Sklearn pipeline handles scaling automatically
            probabilities = model.predict_proba(features_reshaped)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[::-1][:3]
            
            results = []
            for idx in top_indices:
                class_name = model_class_names[idx]
                confidence = float(probabilities[idx]) * 100
                
                # Filter out very low confidence predictions
                if confidence < 1: continue

                # Get disease info
                if class_name in DISEASE_INFO:
                    disease_info = get_disease_info(class_name)
                else:
                    parts = class_name.split('___')
                    plant = parts[0] if len(parts) > 0 else 'Unknown'
                    disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
                    
                    disease_info = {
                        'plant': plant,
                        'disease': disease,
                        'description': f'Detected {disease} on {plant} plant.',
                        'symptoms': 'See image for visual symptoms.',
                        'treatment': ['Consult an agricultural expert for treatment options.'],
                        'prevention': 'Regular monitoring and proper plant care.'
                    }
                
                results.append({
                    'class_name': class_name,
                    'confidence': round(confidence, 2),
                    'plant': disease_info['plant'],
                    'disease': disease_info['disease'],
                    'description': disease_info['description'],
                    'symptoms': disease_info['symptoms'],
                    'treatment': disease_info['treatment'],
                    'prevention': disease_info['prevention']
                })
            
            return results
            
        except Exception as e:
            print(f"[!] Prediction error: {e}")
            return fallback_predict(img_array)
    else:
        return fallback_predict(img_array)


def fallback_predict(img_array):
    """Fallback prediction using color analysis."""
    # Simple color-based heuristics
    r_mean = np.mean(img_array[:, :, 0])
    g_mean = np.mean(img_array[:, :, 1])
    b_mean = np.mean(img_array[:, :, 2])
    
    green_ratio = g_mean / (r_mean + g_mean + b_mean + 0.001)
    
    results = []
    
    if green_ratio > 0.38:
        # Likely healthy
        results.append({
            'class_name': 'Tomato___healthy',
            'confidence': 75.0,
            'plant': 'Tomato',
            'disease': 'Healthy',
            'description': 'Your plant appears healthy based on color analysis.',
            'symptoms': 'No visible disease symptoms.',
            'treatment': ['Continue regular care and maintenance.'],
            'prevention': 'Maintain good gardening practices.'
        })
    else:
        # Possible disease
        results.append({
            'class_name': 'Tomato___Early_blight',
            'confidence': 60.0,
            'plant': 'Tomato',
            'disease': 'Possible Disease Detected',
            'description': 'Some abnormality detected. Train the model for accurate predictions.',
            'symptoms': 'Color patterns suggest possible disease.',
            'treatment': ['Run train_model.py to train the ML model for accurate diagnosis.'],
            'prevention': 'Regular monitoring recommended.'
        })
    
    return results


# Routes
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return disease prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    file_ext = file.filename.rsplit('.', 1)[-1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, WEBP)'}), 400
    
    try:
        image_data = file.read()
        results = predict_disease(image_data)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'model_loaded': MODEL_LOADED
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED
    })


@app.route('/classes')
def get_classes():
    """Return list of supported disease classes."""
    if MODEL_LOADED and model_class_names:
        return jsonify({
            'classes': model_class_names,
            'count': len(model_class_names),
            'trained': True
        })
    return jsonify({
        'classes': ALL_CLASS_NAMES,
        'count': len(ALL_CLASS_NAMES),
        'trained': False
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("[*] Plant Disease Detection System")
    print("=" * 60)
    
    # Try to load trained model
    if load_trained_model():
        print("[*] Using trained ML model for predictions")
    else:
        print("[!] Run 'python train_model.py' to train the model")
        print("[*] Using fallback color analysis for now")
    
    print("\n[OK] Server starting...")
    print("[--> Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
