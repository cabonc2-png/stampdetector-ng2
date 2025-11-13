@echo off
REM Installation SAM pour Windows - Méthode alternative
REM Ce script utilise plusieurs méthodes pour installer SAM

echo ============================================================
echo   Installation SAM pour StampDetector (Windows)
echo ============================================================
echo.

echo [1/6] Mise a jour de pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERREUR: Impossible de mettre a jour pip
    pause
    exit /b 1
)
echo OK: pip a jour
echo.

echo [2/6] Installation PyTorch (methode 1 - officielle)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo AVERTISSEMENT: Methode 1 echouee, essai methode 2...
    echo.

    echo [2/6] Installation PyTorch (methode 2 - PyPI standard)...
    python -m pip install torch torchvision
    if errorlevel 1 (
        echo ERREUR: Impossible d'installer PyTorch
        echo.
        echo Solutions possibles:
        echo 1. Verifiez votre connexion Internet
        echo 2. Desactivez temporairement votre antivirus
        echo 3. Utilisez un VPN si le site PyTorch est bloque
        echo 4. Essayez l'installation manuelle (voir README_SAM.md)
        pause
        exit /b 1
    )
)
echo OK: PyTorch installe
echo.

echo [3/6] Installation Segment Anything...
python -m pip install segment-anything
if errorlevel 1 (
    echo ERREUR: Impossible d'installer Segment Anything
    pause
    exit /b 1
)
echo OK: Segment Anything installe
echo.

echo [4/6] Creation du dossier pour le modele...
if not exist "StampDetector\models\sam_checkpoints" mkdir "StampDetector\models\sam_checkpoints"
echo OK: Dossier cree
echo.

echo [5/6] Telechargement du modele SAM (~375 MB)...
echo Cela peut prendre plusieurs minutes selon votre connexion...
python -c "import urllib.request; import sys; url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'; path='StampDetector/models/sam_checkpoints/sam_vit_b_01ec64.pth'; print('Telechargement...'); urllib.request.urlretrieve(url, path); print('OK')"
if errorlevel 1 (
    echo ERREUR: Telechargement echoue
    echo.
    echo Telechargez manuellement depuis:
    echo https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    echo.
    echo Et placez-le dans:
    echo StampDetector\models\sam_checkpoints\sam_vit_b_01ec64.pth
    pause
    exit /b 1
)
echo OK: Modele telecharge
echo.

echo [6/6] Verification de l'installation...
python -c "import torch; from segment_anything import sam_model_registry; print('OK: Tous les modules importes avec succes')"
if errorlevel 1 (
    echo ERREUR: Verification echouee
    pause
    exit /b 1
)
echo.

echo ============================================================
echo   Installation reussie!
echo ============================================================
echo.
echo SAM est maintenant pret a l'emploi!
echo.
echo Prochaines etapes:
echo   1. Lancez StampDetector normalement
echo   2. SAM sera automatiquement utilise pour la detection
echo   3. Profitez d'une detection ultra-precise!
echo.
echo Note: SAM est plus lent qu'OpenCV (~2-5s par image)
echo       mais BEAUCOUP plus precis.
echo.
pause
