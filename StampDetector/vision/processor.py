"""
Pipeline de traitement des images de timbres
Détection, découpe, ajout de marges, export
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from vision.detector import AutoDetector
from utils.image_utils import (
    get_image_dpi,
    mm_to_pixels,
    expand_bbox_with_margin,
    crop_stamp_with_transparency,
    save_stamp,
    rotate_and_crop,
    auto_crop_to_content
)
from utils.logger import logger

@dataclass
class ProcessingResult:
    """Résultat du traitement d'une image"""
    input_path: Path
    num_stamps: int
    output_files: List[Path]
    dpi: int
    dpi_was_default: bool
    backend_used: str
    detections: List

class StampProcessor:
    """Pipeline de traitement pour la détection et découpe de timbres"""
    
    def __init__(self, output_dir: Path):
        """
        Initialise le processeur
        
        Args:
            output_dir: Dossier de sortie pour les exports
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        self.detector = AutoDetector()
        logger.info(f"Processeur initialisé (backend: {self.detector.get_backend_name()})")
    
    def process_image(self, image_path: Path, margin_mm: float = 1.0, save_jpg: bool = False, progress_callback: Optional[Callable[[int, str], None]] = None) -> ProcessingResult:
        """
        Traite une image complète
        
        Args:
            image_path: Chemin vers l'image source
            margin_mm: Marge en millimètres
            save_jpg: Si True, génère aussi des JPG
            progress_callback: Callback(progress_pct, message)
        
        Returns:
            ProcessingResult: Résultat du traitement
        """
        logger.info(f"{'='*60}")
        logger.info(f"Traitement: {image_path.name}")
        logger.info(f"{'='*60}")
        
        # Chargement de l'image
        if progress_callback:
            progress_callback(10, "Chargement de l'image...")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        h, w = image.shape[:2]
        logger.info(f"Dimensions: {w}x{h}px")
        
        # Lecture du DPI
        if progress_callback:
            progress_callback(20, "Lecture du DPI...")
        
        dpi, dpi_was_default = get_image_dpi(image_path)
        margin_px = mm_to_pixels(margin_mm, dpi)
        logger.info(f"Marge: {margin_mm}mm = {margin_px}px @ {dpi}DPI")
        
        # Détection
        if progress_callback:
            progress_callback(30, f"Détection ({self.detector.get_backend_name()})...")
        
        detections = self.detector.detect(image)
        
        if not detections:
            logger.warning("⚠️ Aucun timbre détecté")
            return ProcessingResult(
                input_path=image_path,
                num_stamps=0,
                output_files=[],
                dpi=dpi,
                dpi_was_default=dpi_was_default,
                backend_used=self.detector.get_backend_name(),
                detections=[]
            )
        
        logger.info(f"✓ {len(detections)} timbre(s) détecté(s)")
        
        # Export des crops
        output_files = []
        base_name = image_path.stem
        
        for idx, detection in enumerate(detections, start=1):
            if progress_callback:
                progress = 30 + int((idx / len(detections)) * 60)
                progress_callback(progress, f"Export timbre {idx}/{len(detections)}...")
            
            try:
                # Extraire bbox, mask et contour
                if len(detection) == 3:
                    bbox, mask, contour = detection
                else:
                    bbox, mask = detection[:2]
                    contour = None
                
                # Redresser et recadrer si on a le contour
                if contour is not None:
                    rotated_image, bbox_adjusted = rotate_and_crop(image, mask, contour, margin_px)
                    crop = crop_stamp_with_transparency(rotated_image, None, bbox_adjusted)
                else:
                    # Méthode classique avec marge
                    bbox_with_margin = expand_bbox_with_margin(bbox, margin_px, image.shape[:2])
                    crop = crop_stamp_with_transparency(image, mask, bbox_with_margin)
                
                # Recadrage automatique au contenu
                crop = auto_crop_to_content(crop, mask)
                
                # Vérifier que le crop est valide avant de sauvegarder
                if crop is not None and crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
                    output_path = self.output_dir / f"{base_name}_stamp_{idx:03d}"
                    save_stamp(crop, output_path, save_jpg=save_jpg)
                    output_files.append(output_path.with_suffix('.png'))
                    if save_jpg:
                        output_files.append(output_path.with_suffix('.jpg'))
                else:
                    logger.warning(f"Timbre {idx} ignoré: crop invalide")
                    
            except Exception as e:
                logger.error(f"Erreur traitement timbre {idx}: {e}")
                continue
        
        if progress_callback:
            progress_callback(100, "Traitement terminé!")
        
        logger.info(f"✓ Traitement terminé: {len(output_files)} fichier(s) généré(s)")
        
        return ProcessingResult(
            input_path=image_path,
            num_stamps=len(detections),
            output_files=output_files,
            dpi=dpi,
            dpi_was_default=dpi_was_default,
            backend_used=self.detector.get_backend_name(),
            detections=detections
        )
    
    def draw_detections(self, image: np.ndarray, detections) -> np.ndarray:
        """
        Dessine les détections sur une image (pour preview)
        
        Args:
            image: Image BGR
            detections: Liste de détections
        
        Returns:
            np.ndarray: Image avec annotations
        """
        annotated = image.copy()
        
        for idx, detection in enumerate(detections, start=1):
            try:
                # Gérer les deux formats
                if len(detection) >= 3:
                    bbox, mask, contour = detection[:3]
                else:
                    bbox, mask = detection[:2]
                    contour = None
                
                x1, y1, x2, y2 = bbox
                
                if mask is not None:
                    overlay = annotated.copy()
                    overlay[mask > 0] = [0, 255, 0]
                    annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
                
                # Dessiner le rectangle minimum englobant si on a le contour
                if contour is not None:
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = box.astype(np.int32)
                    cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Numéro du timbre
                label = f"#{idx}"
                cv2.putText(annotated, label, (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            except Exception as e:
                logger.error(f"Erreur annotation timbre {idx}: {e}")
                continue
        
        return annotated