"""
Plant Disease Detection Model Training Script - Professional Version
Uses Scikit-Learn's high-performance ensemble models (Random Forest + Gradient Boosting)
for maximum accuracy and stability without deep learning DLL dependency issues.
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
import random
import warnings

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = 128
DATA_DIR = Path("Dataset")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "plant_disease_model.pkl"
MAX_IMAGES_PER_CLASS = None  # Cap images to prevent memory overflow, set None for all


def get_class_names(data_dir):
    """Get all class names from directory structure."""
    if not data_dir.exists():
        print(f"ERROR: Directory {data_dir} not found!")
        return []
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    return classes


def load_image(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Load and preprocess a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    except Exception as e:
        return None


def extract_features(img_array):
    """
    Extract comprehensive feature vector (Color + Texture + Stats).
    Vector size: 85 dimensions.
    """
    # === 1. BASIC COLOR STATISTICS (12 features) ===
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Mean, Std, Min, Max
    feats_color_stats = [
        np.mean(r), np.mean(g), np.mean(b),
        np.std(r), np.std(g), np.std(b),
        np.min(r), np.min(g), np.min(b),
        np.max(r), np.max(g), np.max(b)
    ]
    
    # === 2. COLOR HISTOGRAMS (48 features) ===
    # 16 bins per channel
    bins = 16
    r_hist, _ = np.histogram(r, bins=bins, range=(0, 1), density=True)
    g_hist, _ = np.histogram(g, bins=bins, range=(0, 1), density=True)
    b_hist, _ = np.histogram(b, bins=bins, range=(0, 1), density=True)
    
    # === 3. TEXTURE / GRADIENTS (7 features) ===
    # Grayscale conversion
    gray = 0.299*r + 0.587*g + 0.114*b
    
    # Sobel-like gradients
    gx = np.abs(gray[:, :-1] - gray[:, 1:])
    gy = np.abs(gray[:-1, :] - gray[1:, :])
    
    grad_mean = (np.mean(gx) + np.mean(gy)) / 2
    grad_std  = (np.std(gx) + np.std(gy)) / 2
    grad_max  = max(np.max(gx) if gx.size > 0 else 0, np.max(gy) if gy.size > 0 else 0)
    
    # Edge density
    edges_low  = (np.sum(gx > 0.05) + np.sum(gy > 0.05)) / (gx.size + gy.size)
    edges_high = (np.sum(gx > 0.20) + np.sum(gy > 0.20)) / (gx.size + gy.size)
    
    feats_texture = [grad_mean, grad_std, grad_max, edges_low, edges_high]
    
    # === 4. MOMENTS / SHAPE (3 features) ===
    contrast = np.std(gray)
    # Simple skewness/kurtosis approximations (avoid scipy dependency)
    mean_gray = np.mean(gray)
    skewness = np.mean((gray - mean_gray)**3)
    
    feats_shape = [contrast, skewness]
    
    # === 5. DOMAIN SPECIFIC (5 features) ===
    # Green ratio (health indicator)
    greenness = np.mean(g) / (np.mean(r) + np.mean(b) + 1e-6)
    # Brown ratio (disease indicator)
    brownness = (np.mean(r) + np.mean(g)) / (np.mean(b) + 1e-6)
    
    # HSV-like (Saturation estimate)
    cmax = np.max(img_array, axis=2)
    cmin = np.min(img_array, axis=2)
    saturation = np.mean((cmax - cmin) / (cmax + 1e-6))
    
    feats_domain = [greenness, brownness, saturation]
    
    # Combine all
    return np.concatenate([
        feats_color_stats,
        r_hist, g_hist, b_hist,
        feats_texture,
        feats_shape,
        feats_domain
    ])


def load_dataset(data_dir, max_per_class=None):
    """Load dataset with progress reporting."""
    print(f"[*] Scanning {data_dir}...")
    class_names = get_class_names(data_dir)
    
    if not class_names:
        raise ValueError("No classes found! Check Testing_Database folder.")
        
    print(f"[*] Found {len(class_names)} classes: {class_names}")
    
    X = []
    y = []
    
    for idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + \
                 list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.JPG"))
        
        # Limit count if needed
        if max_per_class and len(images) > max_per_class:
            images = random.sample(images, max_per_class)
            
        print(f"  > Processing {class_name}: {len(images)} images...")
        
        count = 0
        for img_path in images:
            img_arr = load_image(img_path)
            if img_arr is not None:
                feats = extract_features(img_arr)
                X.append(feats)
                y.append(idx)
                count += 1
                
                if count % 200 == 0:
                    print(f"    Loaded {count}...", end="\r")
        print(f"    Simple Completed: {count} images loaded.")

    return np.array(X), np.array(y), class_names


def train():
    print("\n" + "="*60)
    print("   HIGH-ACCURACY PLANT DISEASE MODEL TRAINING")
    print("   (Using Random Forest + Gradient Boosting Ensemble)")
    print("="*60)
    
    # 1. Load Data
    X, y, classes = load_dataset(DATA_DIR, MAX_IMAGES_PER_CLASS)
    print(f"\n[*] Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 2. Split Data
    print("[*] Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # 3. Create High-Performance Pipeline
    print("[*] Initializing Ensemble Model...")
    
    # Model 1: Random Forest (Robust, parallel processing)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    
    # Model 2: Histogram Gradient Boosting (Fast, high accuracy like LightGBM)
    hgb = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        random_state=42
    )
    
    # Model 3: MLP (Neural Network) standard sklearn implementation
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=42
    )
    
    # Ensemble Voting
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('hgb', hgb),
            ('mlp', mlp)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # Pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', ensemble)
    ])
    
    # 4. Train
    print("\n" + "="*60)
    print("TRAINING STARTED (This may take a minute)...")
    print("="*60)
    pipeline.fit(X_train, y_train)
    
    # 5. Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"[*] TEST ACCURACY: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # 6. Save Feature Statistics (for App preprocessing)
    # We need to save the scaler's mean/scale manually or save the whole pipeline
    # The app currently expects a dictionary. We will adapt the app later, 
    # but for compatibility, we save the raw stats too.
    
    scaler = pipeline.named_steps['scaler']
    
    model_data = {
        'model': pipeline,           # Saves the full pipeline including scaler
        'class_names': classes,
        'feature_mean': scaler.mean_,
        'feature_std': scaler.scale_,
        'img_size': IMG_SIZE,
        'accuracy': acc,
        'type': 'sklearn_ensemble'   # Marker for app to know how to use it
    }
    
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\n[*] Model saved to: {MODEL_PATH}")
    print("\n" + "="*60)
    print("DONE! You can now run 'python app.py'")
    print("="*60)

if __name__ == '__main__':
    train()
