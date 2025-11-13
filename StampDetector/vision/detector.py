"""
Détecteur automatique avec fallback
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from utils.logger import logger
from models import yolo_detector, sam_detector, opencv_detector

class AutoDetector:
    """Détecteur automatique avec sélection du meilleur backend"""
    
    def __init__(self):
        self.backend = None
        self.detector = None
        self._select_backend()
    
    def _select_backend(self):
        logger.info("Sélection du backend de détection...")

        # Priorité 1: SAM (le plus précis)
        if sam_detector.is_available():
            try:
                self.detector = sam_detector.SAMDetector()
                if self.detector.load():
                    self.backend = "SAM"
                    logger.info("🎯 Backend sélectionné: SAM (IA)")
                    return
                else:
                    logger.warning("Échec du chargement de SAM, fallback vers OpenCV")
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de SAM: {e}")

        # Priorité 2: YOLO (si disponible)
        if yolo_detector.is_available():
            self.backend = "YOLO"
            logger.info("🎯 Backend: YOLOv8-seg (non chargé)")
            return

        # Fallback: OpenCV
        self.detector = opencv_detector.OpenCVDetector()
        self.detector.load()
        self.backend = "OpenCV"
        logger.info("🎯 Backend sélectionné: OpenCV (fallback)")
    
    def detect(self, image: np.ndarray, **kwargs) -> List[Tuple[Tuple[int, int, int, int], Optional[np.ndarray], Optional[np.ndarray]]]:
        if self.detector is None:
            raise RuntimeError("Aucun détecteur disponible")
        
        logger.info(f"Détection avec {self.backend}...")
        return self.detector.detect(image, **kwargs)
    
    def get_backend_name(self) -> str:
        return self.backend