"""
Service gratuit amélioré de reconnaissance de timbres
"""
from utils.smart_recognition import recognize_stamp_online
from utils.logger import logger
from typing import Dict
import numpy as np

def search_stamp_online_free(image: np.ndarray, ocr_text: str = "") -> Dict:
    """
    Fonction principale pour recherche gratuite améliorée
    
    Args:
        image: Image du timbre
        ocr_text: Texte OCR extrait (ignoré car inutile)
    
    Returns:
        Dict: Résultats de reconnaissance
    """
    logger.info("🔍 Lancement de la reconnaissance visuelle intelligente...")
    
    try:
        # Utiliser la reconnaissance visuelle
        results = recognize_stamp_online(image)
        
        # Formater pour l'affichage
        formatted = {
            "identified": results.get("identified", False),
            "name": results.get("name", "Non identifié"),
            "country": results.get("country"),
            "confidence": results.get("confidence", 0.0),
            "suggestions": results.get("suggestions", []),
            "visual_info": results.get("visual_features", {})
        }
        
        return formatted
        
    except Exception as e:
        logger.error(f"Erreur reconnaissance en ligne: {e}")
        return {
            "identified": False,
            "name": "Erreur de reconnaissance",
            "confidence": 0.0,
            "suggestions": [str(e)]
        }