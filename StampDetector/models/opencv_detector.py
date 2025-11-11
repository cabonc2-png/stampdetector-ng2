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
        Détecte les timbres par détection de contours améliorée
        Optimisé pour capturer TOUS les timbres sur une feuille

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

        # Égalisation d'histogramme pour améliorer le contraste
        # Crucial pour détecter les timbres peu contrastés
        gray = cv2.equalizeHist(gray)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Approche multi-échelle pour capturer différentes intensités de contours
        all_contours = []

        # Méthode 1: Canny avec seuils bas (pour contours faibles)
        edges_low = cv2.Canny(blurred, 20, 60)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_low = cv2.dilate(edges_low, kernel_small, iterations=1)
        contours_low, _ = cv2.findContours(dilated_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours_low)

        # Méthode 2: Canny avec seuils moyens (pour contours normaux)
        edges_mid = cv2.Canny(blurred, 40, 120)
        kernel_mid = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_mid = cv2.dilate(edges_mid, kernel_mid, iterations=1)
        contours_mid, _ = cv2.findContours(dilated_mid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours_mid)

        # Méthode 3: Seuillage adaptatif (pour timbres sur fond variable)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 5)
        kernel_thresh = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_thresh, iterations=2)
        contours_thresh, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours_thresh)

        logger.info(f"Contours détectés (bruts): {len(all_contours)}")

        # Filtrage et déduplication des contours
        detections = []
        seen_boxes = []

        for cnt in all_contours:
            area = cv2.contourArea(cnt)

            # Filtrage par aire
            if area < min_area or area > max_area:
                continue

            x, y, w_box, h_box = cv2.boundingRect(cnt)

            # Vérifier que le contour n'est pas trop petit en dimensions
            if w_box < 10 or h_box < 10:
                continue

            aspect_ratio = max(w_box, h_box) / min(w_box, h_box)

            # Ratio plus permissif pour les timbres rectangulaires
            max_ratio = 15 if detect_blocks else 5
            if aspect_ratio > max_ratio:
                continue

            # Vérifier la circularité pour éviter les formes bizarres
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Plus permissif : accepter même les formes irrégulières (timbres dentelés)
                if circularity < 0.15:
                    continue

            # Déduplication : vérifier si ce contour chevauche un contour déjà détecté
            bbox = (x, y, x + w_box, y + h_box)
            is_duplicate = False

            for seen_box in seen_boxes:
                # Calculer l'intersection over union (IoU)
                x1_inter = max(bbox[0], seen_box[0])
                y1_inter = max(bbox[1], seen_box[1])
                x2_inter = min(bbox[2], seen_box[2])
                y2_inter = min(bbox[3], seen_box[3])

                if x1_inter < x2_inter and y1_inter < y2_inter:
                    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                    box_area = w_box * h_box
                    seen_area = (seen_box[2] - seen_box[0]) * (seen_box[3] - seen_box[1])

                    # Si chevauchement > 70%, considérer comme doublon
                    iou = inter_area / min(box_area, seen_area)
                    if iou > 0.7:
                        is_duplicate = True
                        break

            if is_duplicate:
                continue

            # Créer le masque pour ce contour
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            detections.append((bbox, mask, cnt))
            seen_boxes.append(bbox)

        logger.info(f"✓ OpenCV: {len(detections)} timbre(s) détecté(s) après filtrage")
        return detections

def is_available() -> bool:
    """OpenCV est toujours disponible"""
    return True