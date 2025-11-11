"""
Détecteur YOLOv8-seg (optionnel)
"""
from utils.logger import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("✓ YOLOv8 disponible")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("⚠️ YOLOv8 non disponible (utilisation d'OpenCV)")

def is_available() -> bool:
    return YOLO_AVAILABLE