"""
Détecteur OpenCV (fallback)
Utilise la détection de contours classique
Optimisé pour détecter aussi les grands blocs-feuillets
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from utils.logger import logger

class OpenCVDetector:
    """Détecteur utilisant OpenCV (contours)"""
    
    def __init__(self):
        logger.info("✓ OpenCV disponible (fallback)")
    
    def load(self) -> bool:
        return True
    
    def detect(self, image: np.ndarray, min_area: int = 5000, max_area: Optional[int] = None, detect_blocks: bool = True) -> List[Tuple[Tuple[int, int, int, int], Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Détecte les timbres par détection de contours
        Optimisé pour détecter aussi les grands blocs-feuillets

        Args:
            image: Image BGR
            min_area: Aire minimale en pixels²
            max_area: Aire maximale en pixels² (None = auto)
            detect_blocks: Si True, accepte les grands objets (blocs-feuillets)

        Returns:
            List[Tuple[bbox, mask, contour]]: Liste de détections
        """
        h, w = image.shape[:2]

        # Par défaut, accepter jusqu'à 80% de l'image (pour les blocs-feuillets)
        if max_area is None:
            if detect_blocks:
                max_area = int(h * w * 0.8)
            else:
                max_area = h * w // 2

        logger.info(f"Détection avec aire minimale: {min_area} px² (max: {max_area} px²)")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Détection adaptative pour les grands objets
        # Seuils équilibrés pour détecter tous les timbres
        edges = cv2.Canny(blurred, 30, 100)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filtrage par aire
            if area < min_area or area > max_area:
                continue

            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = max(w_box, h_box) / min(w_box, h_box)

            # Accepter un ratio plus large pour les blocs-feuillets
            max_ratio = 10 if detect_blocks else 4
            if aspect_ratio > max_ratio:
                continue

            # Filtrer les contours trop proches des bords (artefacts de scan)
            margin = 10
            if x < margin or y < margin or x + w_box > w - margin or y + h_box > h - margin:
                # Vérifier si c'est vraiment un timbre ou juste le bord du scan
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.3:  # Trop irrégulier, probablement un artefact
                        continue

            # Créer le masque pour ce contour
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            bbox = (x, y, x + w_box, y + h_box)
            detections.append((bbox, mask, cnt))

        logger.info(f"OpenCV: {len(detections)} timbre(s)/bloc(s) détecté(s)")
        return detections

def is_available() -> bool:
    """OpenCV est toujours disponible"""
    return True