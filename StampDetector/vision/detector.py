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
        
        if yolo_detector.is_available():
            self.backend = "YOLO"
            logger.info("🎯 Backend: YOLOv8-seg (non chargé)")
        elif sam_detector.is_available():
            self.backend = "SAM2"
            logger.info("🎯 Backend: SAM2 (non chargé)")
        else:
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