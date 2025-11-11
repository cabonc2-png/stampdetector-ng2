"""
Utilitaires pour la manipulation d'images
Conversion DPI, redimensionnement, calculs de marges
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from utils.logger import logger

DEFAULT_DPI = 300

def get_image_dpi(image_path: Path) -> Tuple[int, bool]:
    """
    Extrait le DPI d'une image
    
    Args:
        image_path: Chemin vers l'image
    
    Returns:
        Tuple[int, bool]: (DPI, is_default)
    """
    try:
        with Image.open(image_path) as img:
            dpi_info = img.info.get('dpi', None)
            
            if dpi_info:
                dpi = int(dpi_info[0]) if isinstance(dpi_info, tuple) else int(dpi_info)
                logger.info(f"DPI détecté: {dpi}")
                return dpi, False
            else:
                logger.warning(f"⚠️ DPI absent dans {image_path.name}, utilisation de {DEFAULT_DPI} DPI par défaut")
                return DEFAULT_DPI, True
                
    except Exception as e:
        logger.error(f"Erreur lecture DPI: {e}")
        return DEFAULT_DPI, True

def mm_to_pixels(mm: float, dpi: int) -> int:
    """
    Convertit des millimètres en pixels

    Args:
        mm: Valeur en millimètres
        dpi: DPI de l'image

    Returns:
        int: Valeur en pixels
    """
    px_per_mm = dpi / 25.4
    pixels = int(round(mm * px_per_mm))
    logger.debug(f"Conversion: {mm}mm @ {dpi}DPI = {pixels}px")
    return pixels

def calculate_min_stamp_area(dpi: int, min_width_mm: float = 15.0, min_height_mm: float = 15.0) -> int:
    """
    Calcule la surface minimale d'un timbre en pixels en fonction du DPI

    Les timbres les plus petits font généralement au moins 15mm x 15mm.
    Cette fonction garantit que le seuil de détection s'adapte à la résolution.

    Args:
        dpi: DPI de l'image
        min_width_mm: Largeur minimale d'un timbre en mm (défaut: 15mm)
        min_height_mm: Hauteur minimale d'un timbre en mm (défaut: 15mm)

    Returns:
        int: Surface minimale en pixels²
    """
    min_width_px = mm_to_pixels(min_width_mm, dpi)
    min_height_px = mm_to_pixels(min_height_mm, dpi)
    min_area = min_width_px * min_height_px

    logger.info(f"Seuil adaptatif @ {dpi}DPI: {min_width_mm}x{min_height_mm}mm = {min_width_px}x{min_height_px}px = {min_area}px²")
    return min_area

def expand_bbox_with_margin(bbox: Tuple[int, int, int, int], margin_px: int, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Agrandit une bounding box avec une marge
    
    Args:
        bbox: (x1, y1, x2, y2)
        margin_px: Marge en pixels
        image_shape: (height, width) de l'image
    
    Returns:
        Tuple[int, int, int, int]: BBox élargie
    """
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]
    
    x1_new = max(0, x1 - margin_px)
    y1_new = max(0, y1 - margin_px)
    x2_new = min(w, x2 + margin_px)
    y2_new = min(h, y2 + margin_px)
    
    return (x1_new, y1_new, x2_new, y2_new)

def crop_stamp_with_transparency(image: np.ndarray, mask: Optional[np.ndarray], bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Découpe un timbre avec transparence
    
    Args:
        image: Image source BGR
        mask: Masque de segmentation (None si non disponible)
        bbox: Bounding box (x1, y1, x2, y2)
    
    Returns:
        np.ndarray: Image BGRA avec transparence
    """
    x1, y1, x2, y2 = bbox
    
    # Vérifier les limites
    if x1 >= x2 or y1 >= y2:
        logger.error(f"BBox invalide: ({x1},{y1},{x2},{y2})")
        return np.zeros((1, 1, 4), dtype=np.uint8)
    
    crop = image[y1:y2, x1:x2].copy()
    
    if crop.size == 0:
        logger.error(f"Crop vide pour bbox ({x1},{y1},{x2},{y2})")
        return np.zeros((1, 1, 4), dtype=np.uint8)
    
    if crop.shape[2] == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    
    if mask is not None:
        mask_crop = mask[y1:y2, x1:x2]
        if mask_crop.shape[:2] == crop.shape[:2]:
            crop[:, :, 3] = mask_crop
    
    return crop

def get_rotation_angle(contour: np.ndarray) -> float:
    """
    Calcule l'angle de rotation pour redresser un contour
    
    Args:
        contour: Contour OpenCV
    
    Returns:
        float: Angle en degrés
    """
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    
    if rect[1][0] < rect[1][1]:
        angle = 90 + angle
    
    return angle

def rotate_and_crop(image: np.ndarray, mask: Optional[np.ndarray], contour: np.ndarray, margin_px: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Redresse et recadre une image selon un contour
    
    Args:
        image: Image source BGR
        mask: Masque de segmentation (optionnel)
        contour: Contour du timbre
        margin_px: Marge en pixels
    
    Returns:
        Tuple[np.ndarray, bbox]: Image redressée et bbox ajustée
    """
    h, w = image.shape[:2]
    
    angle = get_rotation_angle(contour)
    
    # Si l'angle est proche de 0, pas besoin de rotation
    if abs(angle) < 0.5:
        x, y, w_box, h_box = cv2.boundingRect(contour)
        bbox = (
            max(0, x - margin_px),
            max(0, y - margin_px),
            min(w, x + w_box + margin_px),
            min(h, y + h_box + margin_px)
        )
        return image, bbox
    
    center = tuple(np.array([w, h]) / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    abs_cos = abs(rot_matrix[0, 0])
    abs_sin = abs(rot_matrix[0, 1])
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)
    
    rot_matrix[0, 2] += bound_w / 2 - center[0]
    rot_matrix[1, 2] += bound_h / 2 - center[1]
    
    rotated_image = cv2.warpAffine(image, rot_matrix, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    
    rotated_mask = None
    if mask is not None:
        rotated_mask = cv2.warpAffine(mask, rot_matrix, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    
    contour_transformed = cv2.transform(contour.reshape(-1, 1, 2), rot_matrix).reshape(-1, 2)
    x, y, w_box, h_box = cv2.boundingRect(contour_transformed.astype(np.int32))
    
    bbox = (
        max(0, x - margin_px),
        max(0, y - margin_px),
        min(bound_w, x + w_box + margin_px),
        min(bound_h, y + h_box + margin_px)
    )
    
    logger.debug(f"Rotation de {angle:.1f}° appliquée")
    
    return rotated_image, bbox

def auto_crop_to_content(crop: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Recadre automatiquement l'image au contenu (supprime les bordures vides)
    
    Args:
        crop: Image BGRA
        mask: Masque optionnel
    
    Returns:
        np.ndarray: Image recadrée au plus juste
    """
    # Vérifier que l'image n'est pas vide
    if crop is None or crop.size == 0:
        logger.warning("Image vide détectée, pas de recadrage")
        return crop
    
    # Vérifier les dimensions minimales
    if crop.shape[0] < 1 or crop.shape[1] < 1:
        logger.warning(f"Dimensions invalides {crop.shape}, pas de recadrage")
        return crop
    
    if mask is not None and mask.shape[:2] == crop.shape[:2]:
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Vérifier que les dimensions sont valides
            if w > 0 and h > 0 and x >= 0 and y >= 0:
                x_end = min(crop.shape[1], x + w)
                y_end = min(crop.shape[0], y + h)
                if x_end > x and y_end > y:
                    return crop[y:y_end, x:x_end]
    
    # Détecter le contenu
    if len(crop.shape) == 3 and crop.shape[2] == 4:
        alpha = crop[:, :, 3]
        coords = cv2.findNonZero(alpha)
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        safety_margin = 2
        x = max(0, x - safety_margin)
        y = max(0, y - safety_margin)
        w = min(crop.shape[1] - x, w + 2 * safety_margin)
        h = min(crop.shape[0] - y, h + 2 * safety_margin)
        
        # Vérifier que les dimensions finales sont valides
        if w > 0 and h > 0 and x + w <= crop.shape[1] and y + h <= crop.shape[0]:
            return crop[y:y+h, x:x+w]
    
    # Si aucun recadrage n'est possible, retourner l'image originale
    logger.debug("Recadrage automatique impossible, conservation de l'image originale")
    return crop

def save_stamp(crop: np.ndarray, output_path: Path, save_jpg: bool = False) -> None:
    """
    Sauvegarde un crop de timbre en PNG et optionnellement JPG
    
    Args:
        crop: Image BGRA à sauvegarder
        output_path: Chemin de sortie (sans extension)
        save_jpg: Si True, génère aussi un JPG
    """
    try:
        # Vérifier que l'image n'est pas vide
        if crop is None or crop.size == 0:
            logger.error(f"Impossible de sauvegarder {output_path.name}: image vide")
            return
        
        # Vérifier les dimensions minimales
        if crop.shape[0] < 1 or crop.shape[1] < 1:
            logger.error(f"Impossible de sauvegarder {output_path.name}: dimensions invalides {crop.shape}")
            return
        
        # Sauvegarde PNG avec transparence
        png_path = output_path.with_suffix('.png')
        success = cv2.imwrite(str(png_path), crop)
        
        if success:
            logger.info(f"✓ PNG: {png_path.name}")
        else:
            logger.error(f"Échec sauvegarde PNG: {png_path.name}")
            return
        
        # Sauvegarde JPG optionnelle (fond blanc)
        if save_jpg:
            jpg_path = output_path.with_suffix('.jpg')
            
            if len(crop.shape) == 3 and crop.shape[2] == 4:
                bgr = crop[:, :, :3]
                alpha = crop[:, :, 3:4] / 255.0
                white_bg = np.ones_like(bgr) * 255
                jpg_image = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            else:
                jpg_image = crop
            
            success = cv2.imwrite(str(jpg_path), jpg_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                logger.info(f"✓ JPG: {jpg_path.name}")
            else:
                logger.error(f"Échec sauvegarde JPG: {jpg_path.name}")
            
    except Exception as e:
        logger.error(f"Erreur sauvegarde {output_path.name}: {e}")