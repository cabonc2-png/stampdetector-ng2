"""
Utilitaires pour scanner des documents
Compatible avec Windows Image Acquisition (WIA)
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from utils.logger import logger

# Vérifier si comtypes est installé
try:
    import comtypes.client
    WIA_AVAILABLE = True
except ImportError:
    WIA_AVAILABLE = False
    logger.warning("⚠️ comtypes non installé. Installation: pip install comtypes")

class ScannerManager:
    """Gestionnaire de scanner Windows via WIA"""

    def __init__(self):
        if not WIA_AVAILABLE:
            raise RuntimeError("comtypes requis pour utiliser le scanner")

        self.wia = None
        self.device = None
        self.device_name = None
        
    def list_scanners(self) -> List[str]:
        """Liste tous les scanners disponibles"""
        try:
            device_manager = comtypes.client.CreateObject("WIA.DeviceManager")
            devices = device_manager.DeviceInfos
            
            scanner_list = []
            for i in range(1, devices.Count + 1):
                device_info = devices.Item(i)
                if device_info.Type == 1:  # Scanner
                    scanner_list.append(device_info.Properties("Name").Value)
            
            logger.info(f"✓ {len(scanner_list)} scanner(s) trouvé(s)")
            return scanner_list
            
        except Exception as e:
            logger.error(f"Erreur énumération scanners: {e}")
            return []
    
    def connect_scanner(self, scanner_index: int = 0) -> bool:
        """
        Connecte au scanner

        Args:
            scanner_index: Index du scanner (0 = premier scanner)

        Returns:
            bool: True si connexion réussie
        """
        try:
            device_manager = comtypes.client.CreateObject("WIA.DeviceManager")
            devices = device_manager.DeviceInfos

            if devices.Count == 0:
                logger.error("Aucun scanner trouvé")
                return False

            # Prendre le scanner demandé
            device_info = devices.Item(scanner_index + 1)
            self.device = device_info.Connect()

            self.device_name = device_info.Properties("Name").Value
            logger.info(f"✓ Connecté au scanner: {self.device_name}")
            return True

        except Exception as e:
            logger.error(f"Erreur connexion scanner: {e}")
            return False

    def get_scanner_name(self) -> Optional[str]:
        """
        Retourne le nom du scanner connecté

        Returns:
            str: Nom du scanner ou None si non connecté
        """
        return self.device_name
    
    def scan_image(
        self,
        output_path: Path,
        dpi: int = 300,
        color_mode: str = "color",
        format: str = "PNG",
        brightness: int = 0,
        contrast: int = 0
    ) -> Optional[Path]:
        """
        Scanne une image avec paramètres de qualité optimisés

        Args:
            output_path: Chemin de sortie
            dpi: Résolution (150, 300, 600, 1200)
            color_mode: "color", "grayscale", "bw"
            format: "PNG", "JPG", "TIFF"
            brightness: Luminosité (-1000 à 1000, défaut: 0)
            contrast: Contraste (-1000 à 1000, défaut: 0)

        Returns:
            Path: Chemin du fichier scanné ou None
        """
        if not self.device:
            logger.error("Aucun scanner connecté")
            return None

        try:
            # Configuration du scan
            item = self.device.Items(1)

            # DPI (résolution) - CRITIQUE pour la qualité
            logger.info(f"Configuration résolution: {dpi}x{dpi} DPI")
            item.Properties("Horizontal Resolution").Value = dpi
            item.Properties("Vertical Resolution").Value = dpi

            # Configuration du mode couleur et profondeur de bits
            # IMPORTANT: Configurer directement les bits par pixel pour qualité maximale
            if color_mode == "color":
                # Mode couleur RGB - 24 bits (8 bits par canal)
                try:
                    item.Properties("Bits Per Pixel").Value = 24
                    logger.info("Mode couleur: RGB 24-bit (qualité maximale)")
                except:
                    # Fallback sur Current Intent si Bits Per Pixel non disponible
                    item.Properties("Current Intent").Value = 1  # Color intent
                    logger.warning("Utilisation de Current Intent (fallback)")
            elif color_mode == "grayscale":
                # Mode niveaux de gris - 8 bits
                try:
                    item.Properties("Bits Per Pixel").Value = 8
                    logger.info("Mode niveaux de gris: 8-bit")
                except:
                    item.Properties("Current Intent").Value = 2  # Grayscale intent
            else:  # bw
                # Mode noir et blanc - 1 bit
                try:
                    item.Properties("Bits Per Pixel").Value = 1
                    logger.info("Mode noir et blanc: 1-bit")
                except:
                    item.Properties("Current Intent").Value = 4  # B&W intent

            # Paramètres de qualité avancés (si supportés par le scanner)
            try:
                if brightness != 0:
                    item.Properties("Brightness").Value = brightness
                    logger.info(f"Luminosité: {brightness}")
            except:
                logger.debug("Brightness non supporté par ce scanner")

            try:
                if contrast != 0:
                    item.Properties("Contrast").Value = contrast
                    logger.info(f"Contraste: {contrast}")
            except:
                logger.debug("Contrast non supporté par ce scanner")

            # Canon LIDE 400: S'assurer que le mode de transfert est optimal
            logger.info(f"Démarrage du scan: {dpi}DPI, {color_mode}...")

            # Sélectionner le format de transfert approprié
            format_guids = {
                "PNG": "{B96B3CAE-0728-11D3-9D7B-0000F81EF32E}",  # PNG
                "JPG": "{B96B3CAF-0728-11D3-9D7B-0000F81EF32E}",  # JPEG
                "BMP": "{B96B3CAB-0728-11D3-9D7B-0000F81EF32E}",  # BMP
                "TIFF": "{B96B3CB1-0728-11D3-9D7B-0000F81EF32E}"  # TIFF
            }

            # Utiliser BMP pour transfert (sans perte) puis convertir
            # BMP évite la compression pendant le transfert WIA
            transfer_format = format_guids.get(format.upper(), format_guids["PNG"])
            image = item.Transfer(transfer_format)

            # Sauvegarder
            output_path.parent.mkdir(parents=True, exist_ok=True)

            final_path = output_path.with_suffix('.' + format.lower())
            image.SaveFile(str(final_path))

            logger.info(f"✓ Scan terminé: {final_path.name}")
            return final_path

        except Exception as e:
            logger.error(f"Erreur lors du scan: {e}", exc_info=True)
            return None
    
    def show_scanner_ui(self) -> Optional[Path]:
        """
        Affiche l'interface native du scanner
        
        Returns:
            Path: Chemin du fichier scanné ou None
        """
        try:
            common_dialog = comtypes.client.CreateObject("WIA.CommonDialog")
            image = common_dialog.ShowAcquireImage()
            
            if image:
                # Sauvegarder temporairement
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_path = Path(f"temp_scan_{timestamp}.png")
                image.SaveFile(str(temp_path))
                logger.info(f"✓ Scan terminé via UI: {temp_path.name}")
                return temp_path
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur UI scanner: {e}")
            return None

def is_scanner_available() -> bool:
    """Vérifie si un scanner est disponible"""
    if not WIA_AVAILABLE:
        return False
    
    try:
        scanner = ScannerManager()
        scanners = scanner.list_scanners()
        return len(scanners) > 0
    except:
        return False