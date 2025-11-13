#!/usr/bin/env python3
"""
Installation automatique de MobileSAM
Version légère et rapide de SAM (5-10x plus rapide)

Ce script installe:
- PyTorch + TorchVision (si pas déjà installés)
- MobileSAM (version optimisée)
- Télécharge le modèle MobileSAM (~40MB seulement!)
"""

import sys
import subprocess
import urllib.request
from pathlib import Path

def print_header(text):
    """Affiche un en-tête formaté"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_step(step_num, total, text):
    """Affiche une étape"""
    print(f"\n[{step_num}/{total}] {text}")

def install_package(package_name, pip_name=None):
    """Installe un package Python"""
    if pip_name is None:
        pip_name = package_name

    try:
        print(f"   Installation de {package_name}...")
        cmd = [sys.executable, "-m", "pip", "install", pip_name, "--upgrade"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"   ❌ Erreur: {result.stderr[:200]}")
            return False

        print(f"   ✓ {package_name} installé")
        return True
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def download_mobilesam_model():
    """Télécharge le modèle MobileSAM"""
    model_dir = Path("StampDetector/models/sam_checkpoints")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "mobile_sam.pt"

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ Modèle déjà téléchargé ({size_mb:.1f} MB)")
        return True

    model_url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"

    print(f"   Téléchargement du modèle MobileSAM (~40 MB)...")
    print(f"   URL: {model_url}")
    print(f"   Destination: {model_path}")

    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)

                if block_num % 10 == 0:
                    print(f"   Progression: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')

        urllib.request.urlretrieve(model_url, model_path, progress_hook)
        print(f"\n   ✓ Modèle téléchargé avec succès")
        return True

    except Exception as e:
        print(f"\n   ❌ Erreur lors du téléchargement: {e}")
        if model_path.exists():
            model_path.unlink()
        return False

def verify_installation():
    """Vérifie que MobileSAM fonctionne"""
    try:
        print("   Vérification de l'installation...")

        import torch
        print(f"   ✓ PyTorch {torch.__version__}")

        from mobile_sam import sam_model_registry
        print(f"   ✓ MobileSAM importé")

        model_path = Path("StampDetector/models/sam_checkpoints/mobile_sam.pt")
        if not model_path.exists():
            print(f"   ❌ Modèle non trouvé: {model_path}")
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_t"](checkpoint=str(model_path))
        sam.to(device=device)
        print(f"   ✓ Modèle chargé sur {device.upper()}")

        return True

    except Exception as e:
        print(f"   ❌ Erreur de vérification: {e}")
        return False

def main():
    """Installation principale"""
    print_header("Installation MobileSAM pour StampDetector")

    print("\nMobileSAM = Version légère et RAPIDE de SAM")
    print("  - 5-10x plus rapide que SAM")
    print("  - Précision quasi identique")
    print("  - Modèle léger: seulement 40 MB (vs 375 MB)")
    print("  - Performance: ~0.5-1s par image sur CPU")
    print("\nCe script va installer:")
    print("  - PyTorch (si pas déjà installé)")
    print("  - MobileSAM")
    print("  - Modèle MobileSAM (~40 MB)")
    print("\nDurée estimée: 1-3 minutes")

    response = input("\nContinuer? [O/n]: ").strip().lower()
    if response and response not in ['o', 'oui', 'y', 'yes']:
        print("Installation annulée.")
        return

    # Étape 1: Vérifier PyTorch
    print_step(1, 4, "Vérification de PyTorch")
    try:
        import torch
        print(f"   ✓ PyTorch déjà installé ({torch.__version__})")
    except ImportError:
        print("   PyTorch non trouvé, installation...")
        if not install_package("PyTorch", "torch torchvision"):
            print("\n❌ Impossible d'installer PyTorch")
            print("   Installez-le manuellement: python -m pip install torch torchvision")
            return

    # Étape 2: Installer MobileSAM
    print_step(2, 4, "Installation de MobileSAM")
    if not install_package("MobileSAM", "git+https://github.com/ChaoningZhang/MobileSAM.git"):
        print("\n❌ Installation échouée")
        print("\nEssayez manuellement:")
        print("   python -m pip install git+https://github.com/ChaoningZhang/MobileSAM.git")
        return

    # Étape 3: Télécharger le modèle
    print_step(3, 4, "Téléchargement du modèle MobileSAM")
    if not download_mobilesam_model():
        print("\n❌ Téléchargement échoué")
        return

    # Étape 4: Vérification
    print_step(4, 4, "Vérification de l'installation")
    if not verify_installation():
        print("\n❌ Vérification échouée")
        return

    # Succès!
    print_header("✓ Installation réussie!")
    print("\nMobileSAM est maintenant prêt à l'emploi!")
    print("\nPerformance attendue:")
    print("  - CPU: ~0.5-1s par image (vs 2-5s avec SAM)")
    print("  - GPU: ~0.1-0.2s par image")
    print("\nProchaines étapes:")
    print("  1. Lancez StampDetector normalement")
    print("  2. MobileSAM sera automatiquement utilisé (prioritaire)")
    print("  3. Profitez d'une détection rapide ET précise!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation interrompue par l'utilisateur.")
    except Exception as e:
        print(f"\n\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
