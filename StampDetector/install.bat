@echo off
echo ============================================
echo StampDetector - Installation
echo ============================================
echo.

echo [1/3] Mise a jour de pip...
python -m pip install --upgrade pip

echo.
echo [2/3] Installation des dependances principales...
pip install PySide6 opencv-python numpy Pillow pytest PyInstaller

echo.
echo [3/3] Installation des backends de detection (optionnel)...
echo Tentative d'installation de YOLOv8...
pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo ============================================
echo Installation terminee!
echo Lancez: python app.py
echo ============================================
pause
