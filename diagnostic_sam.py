#!/usr/bin/env python3
"""
Script de diagnostic SAM
Vérifie pourquoi SAM n'est pas détecté
"""

import sys
from pathlib import Path

def print_check(name, status, details=""):
    """Affiche un résultat de vérification"""
    symbol = "✓" if status else "❌"
    print(f"{symbol} {name}")
    if details:
        print(f"  → {details}")

print("=" * 70)
print("  Diagnostic SAM pour StampDetector")
print("=" * 70)
print()

# 1. Vérifier PyTorch
print("[1/4] Vérification de PyTorch...")
try:
    import torch
    print_check("PyTorch installé", True, f"Version: {torch.__version__}")

    # Vérifier CUDA
    if torch.cuda.is_available():
        print_check("GPU CUDA disponible", True, f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print_check("GPU CUDA disponible", False, "Utilisation CPU (normal)")
except ImportError as e:
    print_check("PyTorch installé", False, f"Erreur: {e}")
    print("\n❌ PyTorch manquant!")
    print("   Installation: python -m pip install torch torchvision")
    sys.exit(1)

print()

# 2. Vérifier TorchVision
print("[2/4] Vérification de TorchVision...")
try:
    import torchvision
    print_check("TorchVision installé", True, f"Version: {torchvision.__version__}")
except ImportError as e:
    print_check("TorchVision installé", False, f"Erreur: {e}")
    print("\n⚠️ TorchVision manquant (optionnel pour SAM)")

print()

# 3. Vérifier Segment Anything
print("[3/4] Vérification de Segment Anything...")
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    print_check("Segment Anything installé", True)
    print_check("SamAutomaticMaskGenerator disponible", True)
except ImportError as e:
    print_check("Segment Anything installé", False, f"Erreur: {e}")
    print("\n❌ Segment Anything manquant!")
    print("   Installation: python -m pip install segment-anything")
    sys.exit(1)

print()

# 4. Vérifier le modèle
print("[4/4] Vérification du modèle SAM...")
model_path = Path("StampDetector/models/sam_checkpoints/sam_vit_b_01ec64.pth")

if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print_check("Modèle téléchargé", True, f"Taille: {size_mb:.1f} MB")

    # Vérifier que le fichier n'est pas corrompu (taille attendue ~375 MB)
    if size_mb < 300:
        print_check("Modèle valide", False, "Fichier trop petit, probablement corrompu")
        print("\n❌ Le modèle semble incomplet ou corrompu!")
        print("   Supprimez le fichier et retéléchargez:")
        print(f"   del {model_path}")
        print("   python install_sam.py")
        sys.exit(1)
    else:
        print_check("Modèle valide", True, "Taille correcte (~375 MB)")
else:
    print_check("Modèle téléchargé", False, f"Fichier non trouvé: {model_path}")
    print("\n❌ Modèle SAM manquant!")
    print("   Téléchargement:")
    print("   python -c \"import urllib.request; urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', 'StampDetector/models/sam_checkpoints/sam_vit_b_01ec64.pth')\"")
    sys.exit(1)

print()

# 5. Test de chargement
print("[5/5] Test de chargement du modèle...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device sélectionné: {device.upper()}")

    print("  Chargement du modèle (peut prendre 5-10 secondes)...")
    sam = sam_model_registry["vit_b"](checkpoint=str(model_path))
    sam.to(device=device)
    print_check("Modèle chargé", True, "SAM prêt à l'emploi")

    # Test du générateur
    print("  Création du générateur de masques...")
    mask_generator = SamAutomaticMaskGenerator(sam)
    print_check("Générateur créé", True)

except Exception as e:
    print_check("Modèle chargé", False, f"Erreur: {e}")
    print("\n❌ Erreur lors du chargement!")
    print(f"   Détails: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("  ✓ SAM est correctement installé et fonctionnel!")
print("=" * 70)
print()
print("Si StampDetector n'utilise toujours pas SAM:")
print("  1. Vérifiez que vous êtes dans le bon dossier")
print("  2. Relancez l'application: python main.py")
print("  3. Cherchez dans les logs: '🎯 Backend sélectionné: SAM (IA)'")
print()
print("Si vous voyez 'OpenCV (fallback)' à la place:")
print("  - Il y a peut-être un problème avec le code de détection")
print("  - Contactez le support avec ces informations de diagnostic")
print()
