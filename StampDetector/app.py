"""
StampDetector - Application principale
Point d'entrée pour l'interface graphique de détection de timbres
"""
import sys
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# Configuration des chemins
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from ui.main_window import MainWindow
from utils.logger import setup_logger

def main():
    """Point d'entrée principal de l'application"""
    # Configuration du logging
    logger = setup_logger()
    logger.info("Démarrage de StampDetector...")
    
    # Création du dossier outputs si nécessaire
    outputs_dir = BASE_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # Configuration Qt pour le high DPI
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Création de l'application
    app = QApplication(sys.argv)
    app.setApplicationName("StampDetector")
    app.setOrganizationName("StampDetector")
    
    # Création et affichage de la fenêtre principale
    window = MainWindow()
    window.show()
    
    logger.info("Interface graphique chargée")
    
    # Lancement de la boucle événementielle
    sys.exit(app.exec())

if __name__ == "__main__":
    main()