@echo off
echo ============================================================
echo    Plant Disease Detection - MobileNetV2 Training
echo ============================================================
echo.

cd /d "d:\Downloads-D\AI-plant_disease"

echo [1/2] Installing TensorFlow/Streamlit...
echo (This may take a few minutes. Please wait...)
pip install -r requirements.txt

echo.
echo [2/2] Starting MobileNetV2 Transfer Learning...
echo.

python train_mobilenet.py

echo.
pause
