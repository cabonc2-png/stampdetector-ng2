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
            
            scanner_name = device_info.Properties("Name").Value
            logger.info(f"✓ Connecté au scanner: {scanner_name}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur connexion scanner: {e}")
            return False
    
    def scan_image(
        self,
        output_path: Path,
        dpi: int = 300,
        color_mode: str = "color",
        format: str = "PNG"
    ) -> Optional[Path]:
        """
        Scanne une image
        
        Args:
            output_path: Chemin de sortie
            dpi: Résolution (150, 300, 600, 1200)
            color_mode: "color", "grayscale", "bw"
            format: "PNG", "JPG", "TIFF"
        
        Returns:
            Path: Chemin du fichier scanné ou None
        """
        if not self.device:
            logger.error("Aucun scanner connecté")
            return None
        
        try:
            # Configuration du scan
            item = self.device.Items(1)
            
            # DPI (résolution)
            item.Properties("Horizontal Resolution").Value = dpi
            item.Properties("Vertical Resolution").Value = dpi
            
            # Mode couleur
            color_modes = {
                "color": 1,      # RGB
                "grayscale": 2,  # Niveaux de gris
                "bw": 4          # Noir et blanc
            }
            item.Properties("Current Intent").Value = color_modes.get(color_mode, 1)
            
            logger.info(f"Démarrage du scan: {dpi}DPI, {color_mode}...")
            
            # Lancer le scan
            image = item.Transfer("{B96B3CAE-0728-11D3-9D7B-0000F81EF32E}")  # PNG format
            
            # Sauvegarder
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.upper() == "PNG":
                image.SaveFile(str(output_path.with_suffix('.png')))
            elif format.upper() == "JPG":
                image.SaveFile(str(output_path.with_suffix('.jpg')))
            elif format.upper() == "TIFF":
                image.SaveFile(str(output_path.with_suffix('.tiff')))
            
            logger.info(f"✓ Scan terminé: {output_path.name}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erreur lors du scan: {e}")
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