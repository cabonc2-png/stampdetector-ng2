"""
Détecteur MobileSAM (version légère et rapide de SAM)
5-10x plus rapide que SAM avec une précision similaire
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from utils.logger import logger

class MobileSAMDetector:
    """Détecteur utilisant MobileSAM (version optimisée de SAM)"""

    def __init__(self):
        self.sam = None
        self.mask_generator = None
        self.device = None
        self.is_loaded = False

        logger.info("Initialisation du détecteur MobileSAM...")

    def load(self) -> bool:
        """Charge le modèle MobileSAM"""
        try:
            import torch
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

            # Chemin du modèle
            model_path = Path(__file__).parent / "sam_checkpoints" / "mobile_sam.pt"

            if not model_path.exists():
                logger.error(f"Modèle MobileSAM introuvable: {model_path}")
                logger.error("Lancez 'python install_mobilesam.py' pour l'installer")
                return False

            # Détection du device (GPU ou CPU)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Chargement du modèle MobileSAM
            logger.info(f"Chargement du modèle MobileSAM sur {self.device.upper()}...")
            self.sam = sam_model_registry["vit_t"](checkpoint=str(model_path))
            self.sam.to(device=self.device)

            # Configuration du générateur de masques automatique
            # Paramètres strictement optimisés pour timbres (réduit faux positifs)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=20,              # Réduit pour maximiser vitesse
                pred_iou_thresh=0.93,            # AUGMENTÉ : moins de faux positifs
                stability_score_thresh=0.96,      # AUGMENTÉ : masques plus stables
                crop_n_layers=0,                 # Désactivé : évite sur-détection
                crop_n_points_downscale_factor=2,
                min_mask_region_area=8000,       # AUGMENTÉ : ignore petits artefacts
            )

            self.is_loaded = True
            logger.info("✓ MobileSAM chargé avec succès (5-10x plus rapide que SAM)")

            if self.device == "cpu":
                logger.info("MobileSAM sur CPU : ~0.5-1s par image")
            else:
                logger.info("MobileSAM sur GPU : ~0.1-0.2s par image")

            return True

        except ImportError as e:
            logger.error(f"MobileSAM non installé: {e}")
            logger.error("Lancez 'python install_mobilesam.py' pour l'installer")
            return False
        except Exception as e:
            logger.error(f"Erreur chargement MobileSAM: {e}")
            return False

    def detect(self, image: np.ndarray, min_area: int = 5000, max_area: Optional[int] = None, detect_blocks: bool = True) -> List[Tuple[Tuple[int, int, int, int], Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Détecte les timbres avec MobileSAM

        Args:
            image: Image BGR
            min_area: Aire minimale en pixels²
            max_area: Aire maximale en pixels² (None = auto)
            detect_blocks: Si True, accepte les grands objets (blocs-feuillets)

        Returns:
            List[Tuple[bbox, mask, contour]]: Liste de détections
        """
        if not self.is_loaded:
            logger.error("MobileSAM n'est pas chargé. Appelez load() d'abord.")
            return []

        h, w = image.shape[:2]

        # Par défaut, accepter jusqu'à 80% de l'image (pour les blocs-feuillets)
        if max_area is None:
            if detect_blocks:
                max_area = int(h * w * 0.8)
            else:
                max_area = h * w // 2

        # Conversion BGR -> RGB (MobileSAM attend du RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logger.info("Génération des masques avec MobileSAM...")

        # Génération automatique des masques
        masks = self.mask_generator.generate(image_rgb)

        logger.info(f"MobileSAM a généré {len(masks)} masques candidats")

        # Filtrage et conversion au format compatible
        detections = []

        for mask_data in masks:
            # Extraire le masque binaire
            mask = mask_data['segmentation'].astype(np.uint8) * 255

            # Calculer l'aire
            area = mask_data['area']

            # Filtrage par aire
            if area < min_area or area > max_area:
                continue

            # Extraire la bounding box
            bbox = mask_data['bbox']  # Format [x, y, w, h]
            x, y, w_box, h_box = bbox

            # Convertir au format [x1, y1, x2, y2]
            bbox_converted = (int(x), int(y), int(x + w_box), int(y + h_box))

            # Calculer le ratio d'aspect
            aspect_ratio = max(w_box, h_box) / min(w_box, h_box)

            # Accepter un ratio plus large pour les blocs-feuillets
            max_ratio = 10 if detect_blocks else 4
            if aspect_ratio > max_ratio:
                continue

            # Extraire le contour depuis le masque (besoin pour calcul compacité)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            contour = max(contours, key=cv2.contourArea)

            # Filtrage par compacité (évite formes irrégulières/artefacts)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter * perimeter)
                # Timbres rectangulaires: compacité ~ 0.6-0.8
                # Artefacts irréguliers: compacité < 0.4
                if compactness < 0.35:  # Trop irrégulier
                    continue

            # Filtrer les masques trop proches des bords (artefacts de scan)
            margin = 10
            if x < margin or y < margin or x + w_box > w - margin or y + h_box > h - margin:
                # Vérifier la stabilité du masque
                stability_score = mask_data.get('stability_score', 0)
                if stability_score < 0.97:  # AUGMENTÉ : plus strict sur les bords
                    continue

            # Ajouter la détection
            detections.append((bbox_converted, mask, contour))

        # Trier par aire décroissante (les plus gros timbres d'abord)
        detections.sort(key=lambda d: cv2.contourArea(d[2]), reverse=True)

        logger.info(f"✓ MobileSAM: {len(detections)} timbre(s)/bloc(s) détecté(s) après filtrage")

        return detections

def is_available() -> bool:
    """Vérifie si MobileSAM est disponible"""
    try:
        import torch
        from mobile_sam import sam_model_registry

        model_path = Path(__file__).parent / "sam_checkpoints" / "mobile_sam.pt"
        return model_path.exists()
    except ImportError:
        return False
