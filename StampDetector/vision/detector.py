"""
Détecteur automatique avec fallback
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from utils.logger import logger
from models import yolo_detector, sam_detector, mobile_sam_detector, opencv_detector

class AutoDetector:
    """Détecteur automatique avec sélection du meilleur backend"""
    
    def __init__(self):
        self.backend = None
        self.detector = None
        self._select_backend()
    
    def _select_backend(self):
        logger.info("Sélection du backend de détection...")

        # Priorité 1: MobileSAM (rapide ET précis - meilleur compromis)
        if mobile_sam_detector.is_available():
            try:
                self.detector = mobile_sam_detector.MobileSAMDetector()
                if self.detector.load():
                    self.backend = "MobileSAM"
                    logger.info("🎯 Backend sélectionné: MobileSAM (IA rapide)")
                    return
                else:
                    logger.warning("Échec du chargement de MobileSAM, essai SAM standard...")
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de MobileSAM: {e}")

        # Priorité 2: SAM standard (précis mais plus lent)
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

        # Priorité 3: YOLO (si disponible)
        if yolo_detector.is_available():
            self.backend = "YOLO"
            logger.info("🎯 Backend: YOLOv8-seg (non chargé)")
            return

        # Fallback: OpenCV
        self.detector = opencv_detector.OpenCVDetector()
        self.detector.load()
        self.backend = "OpenCV"
        logger.info("🎯 Backend sélectionné: OpenCV (fallback)")
    
    def detect(self, image: np.ndarray, min_area: int = 5000, max_area: Optional[int] = None, detect_blocks: bool = True) -> List[Tuple[Tuple[int, int, int, int], Optional[np.ndarray], Optional[np.ndarray]]]:
        if self.detector is None:
            raise RuntimeError("Aucun détecteur disponible")

        logger.info(f"Détection avec {self.backend}...")
        return self.detector.detect(image, min_area=min_area, max_area=max_area, detect_blocks=detect_blocks)
    
    def get_backend_name(self) -> str:
        return self.backend