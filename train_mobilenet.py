"""
Plant Disease Detection - MobileNetV2 Training Script
Phase 3 of Project Bible: Transfer Learning with MobileNetV2
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "Testing_Database"
MODEL_SAVE_PATH = "plant_disease_model.h5"

def train_model():
    print("\n" + "="*60)
    print("   MOBILE_NET_V2 TRAINING STARTED")
    print("="*60)
    
    # Check GPU
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found!")
        return

    # Phase 3 Step 2: Data Preprocessing & Augmentation
    print("\n[*] Setting up Data Generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Use 20% for validation
    )

    # Load Data
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    num_classes = train_generator.num_classes
    class_names = list(train_generator.class_indices.keys())
    print(f"[*] Found {num_classes} classes: {class_names}")

    # Phase 3 Step 3: Building the Model (Transfer Learning)
    print("\n[*] Building MobileNetV2 Model...")
    
    # 1. Load Base Model (Pretrained on ImageNet)
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # 2. Freeze the base model
    base_model.trainable = False

    # 3. Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Added dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x)

    # 4. Final Model
    model = Model(inputs=base_model.input, outputs=predictions)

    # 5. Compile
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # Phase 3 Step 4: Training
    print("\n[*] Starting Training...")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # Save Model
    print(f"\n[*] Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    
    # Save Class Names
    with open("class_indices.txt", "w") as f:
        f.write("\n".join(class_names))
        
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

if __name__ == '__main__':
    train_model()
