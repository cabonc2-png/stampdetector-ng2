#!/usr/bin/env python3
"""
Installation automatique de SAM (Segment Anything Model)
pour la détection avancée de timbres

Ce script installe:
- PyTorch + TorchVision (framework deep learning)
- Segment Anything (modèle de segmentation Meta)
- Télécharge le modèle SAM ViT-B (~375MB)
"""

import sys
import subprocess
import urllib.request
import os
from pathlib import Path
import platform

def print_header(text):
    """Affiche un en-tête formaté"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_step(step_num, total, text):
    """Affiche une étape"""
    print(f"\n[{step_num}/{total}] {text}")

def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ requis (version actuelle: {version.major}.{version.minor})")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_package(package_name, pip_name=None, show_output=False):
    """Installe un package Python"""
    if pip_name is None:
        pip_name = package_name

    try:
        print(f"   Installation de {package_name}...")

        # Construire la commande
        cmd = [sys.executable, "-m", "pip", "install"]

        # Ajouter les packages (peut contenir des arguments supplémentaires)
        if isinstance(pip_name, str):
            cmd.extend(pip_name.split())
        else:
            cmd.extend(pip_name)

        cmd.append("--upgrade")

        # Afficher la commande pour debug
        if show_output:
            print(f"   Commande: {' '.join(cmd)}")

        # Exécuter
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if not show_output else None,
            stderr=subprocess.PIPE if not show_output else None,
            text=True
        )

        if result.returncode != 0:
            print(f"   ❌ Erreur lors de l'installation de {package_name}")
            if result.stderr:
                print(f"   Détails: {result.stderr[:500]}")
            return False

        print(f"   ✓ {package_name} installé")
        return True

    except Exception as e:
        print(f"   ❌ Exception lors de l'installation de {package_name}: {e}")
        return False

def download_sam_model():
    """Télécharge le modèle SAM ViT-B"""
    model_dir = Path("StampDetector/models/sam_checkpoints")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "sam_vit_b_01ec64.pth"

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ Modèle déjà téléchargé ({size_mb:.1f} MB)")
        return True

    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    print(f"   Téléchargement du modèle SAM ViT-B (~375 MB)...")
    print(f"   URL: {model_url}")
    print(f"   Destination: {model_path}")

    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)

            # Afficher la progression tous les 5%
            if block_num % 50 == 0:
                print(f"   Progression: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')

        urllib.request.urlretrieve(model_url, model_path, progress_hook)
        print(f"\n   ✓ Modèle téléchargé avec succès")
        return True

    except Exception as e:
        print(f"\n   ❌ Erreur lors du téléchargement: {e}")
        if model_path.exists():
            model_path.unlink()  # Supprimer le fichier incomplet
        return False

def verify_installation():
    """Vérifie que SAM fonctionne"""
    try:
        print("   Vérification de l'installation...")

        # Test import torch
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")

        # Test import SAM
        from segment_anything import sam_model_registry
        print(f"   ✓ Segment Anything importé")

        # Test chargement du modèle
        model_path = Path("StampDetector/models/sam_checkpoints/sam_vit_b_01ec64.pth")
        if not model_path.exists():
            print(f"   ❌ Modèle non trouvé: {model_path}")
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_b"](checkpoint=str(model_path))
        sam.to(device=device)
        print(f"   ✓ Modèle chargé sur {device.upper()}")

        return True

    except Exception as e:
        print(f"   ❌ Erreur de vérification: {e}")
        return False

def main():
    """Installation principale"""
    print_header("Installation SAM pour StampDetector")

    print("\nCe script va installer:")
    print("  - PyTorch (framework deep learning)")
    print("  - TorchVision (utilitaires vision)")
    print("  - Segment Anything (modèle Meta)")
    print("  - Modèle SAM ViT-B (~375 MB)")
    print("\nDurée estimée: 3-5 minutes (selon votre connexion)")

    response = input("\nContinuer? [O/n]: ").strip().lower()
    if response and response not in ['o', 'oui', 'y', 'yes']:
        print("Installation annulée.")
        return

    # Étape 1: Vérifier Python
    print_step(1, 5, "Vérification de l'environnement")
    if not check_python_version():
        return

    # Détecter le système
    system = platform.system()
    print(f"✓ Système: {system}")

    # Étape 2: Installer PyTorch
    print_step(2, 5, "Installation de PyTorch")

    # Mise à jour de pip d'abord (souvent nécessaire sur Windows)
    print("   Mise à jour de pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Essayer plusieurs méthodes d'installation PyTorch
    torch_installed = False

    # Méthode 1: Index officiel PyTorch CPU (Windows)
    if system == "Windows" and not torch_installed:
        print("   Tentative 1/3: Index officiel PyTorch CPU...")
        torch_installed = install_package(
            "PyTorch + TorchVision",
            "torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )

    # Méthode 2: PyPI standard
    if not torch_installed:
        print("   Tentative 2/3: Installation depuis PyPI standard...")
        torch_installed = install_package("PyTorch + TorchVision", "torch torchvision")

    # Méthode 3: Installation séparée
    if not torch_installed:
        print("   Tentative 3/3: Installation séparée torch puis torchvision...")
        if install_package("PyTorch", "torch"):
            torch_installed = install_package("TorchVision", "torchvision")

    if not torch_installed:
        print("\n" + "="*70)
        print("❌ Impossible d'installer PyTorch")
        print("="*70)
        print("\nSolutions alternatives:")
        print("\n1. Installation manuelle PyTorch:")
        print("   python -m pip install torch torchvision")
        print("\n2. Utiliser le script Windows:")
        print("   install_sam_windows.bat")
        print("\n3. Vérifier:")
        print("   - Connexion Internet stable")
        print("   - Antivirus désactivé temporairement")
        print("   - Espace disque suffisant (~2 GB)")
        print("\n4. Consulter: https://pytorch.org/get-started/locally/")
        print("\nPuis relancez ce script.")
        input("\nAppuyez sur Entrée pour quitter...")
        return

    # Étape 3: Installer Segment Anything
    print_step(3, 5, "Installation de Segment Anything")
    if not install_package("Segment Anything", "segment-anything"):
        print("\n❌ Installation échouée")
        return

    # Étape 4: Télécharger le modèle
    print_step(4, 5, "Téléchargement du modèle SAM")
    if not download_sam_model():
        print("\n❌ Téléchargement échoué")
        return

    # Étape 5: Vérification
    print_step(5, 5, "Vérification de l'installation")
    if not verify_installation():
        print("\n❌ Vérification échouée")
        return

    # Succès!
    print_header("✓ Installation réussie!")
    print("\nSAM est maintenant prêt à l'emploi!")
    print("\nProchaines étapes:")
    print("  1. Lancez StampDetector normalement")
    print("  2. SAM sera automatiquement utilisé pour la détection")
    print("  3. Profitez d'une détection ultra-précise!\n")

    print("Note: SAM est plus lent qu'OpenCV (~2-5s par image)")
    print("      mais BEAUCOUP plus précis.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation interrompue par l'utilisateur.")
    except Exception as e:
        print(f"\n\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
