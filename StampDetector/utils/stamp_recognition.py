"""
Reconnaissance de timbres via OCR et recherche Yvert & Tellier
"""
import cv2
import numpy as np
import re
import os
from pathlib import Path
from typing import Optional, Dict, List
from utils.logger import logger

# Tesseract OCR - Configuration automatique pour Windows
try:
    import pytesseract
    
    # Configuration Tesseract pour Windows
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    tessdata_path = r'C:\Program Files\Tesseract-OCR\tessdata'
    
    # Vérifier si Tesseract est installé
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        os.environ['TESSDATA_PREFIX'] = tessdata_path
        logger.info(f"✓ Tesseract configuré: {tesseract_path}")
        
        # Vérifier si les données françaises sont disponibles
        fra_data = os.path.join(tessdata_path, 'fra.traineddata')
        if os.path.exists(fra_data):
            logger.info("✓ Données françaises disponibles")
            DEFAULT_LANG = 'fra'
        else:
            logger.warning("⚠️ fra.traineddata absent, utilisation de l'anglais")
            logger.info("Téléchargez depuis: https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata")
            DEFAULT_LANG = 'eng'
    else:
        logger.warning(f"⚠️ Tesseract introuvable à: {tesseract_path}")
        DEFAULT_LANG = 'eng'
    
    TESSERACT_AVAILABLE = True
    
except ImportError:
    TESSERACT_AVAILABLE = False
    DEFAULT_LANG = 'eng'
    logger.warning("⚠️ pytesseract non installé (pip install pytesseract)")

class StampRecognizer:
    """Reconnaissance automatique de timbres"""
    
    def __init__(self):
        """Initialise le reconnaisseur avec une base de données Yvert & Tellier"""
        self.yvert_database = self._load_yvert_database()
        self.ocr_lang = DEFAULT_LANG
    
    def _load_yvert_database(self) -> Dict:
        """
        Charge une base de données locale Yvert & Tellier
        À terme, cela pourrait être une vraie base SQL ou API
        
        Returns:
            Dict: Base de données de timbres
        """
        return {
            "marianne": {
                "patterns": ["marianne", "republique", "française", "rf"],
                "series": {
                    "1945": "Marianne de Gandon (n°719-762)",
                    "1951": "Marianne de Gandon surchargée (n°883-906)",
                    "1997": "Marianne du 14 juillet (n°3083-3101)",
                    "2005": "Marianne de Lamouche (n°3729-3752)",
                    "2008": "Marianne et l'Europe (n°4197-4230)"
                }
            },
            "semeuse": {
                "patterns": ["semeuse", "republique française"],
                "series": {
                    "1903": "Semeuse lignée (n°129-132)",
                    "1924": "Semeuse fond plein (n°189-197)",
                    "1927": "Semeuse sur fond blanc (n°277-279)"
                }
            },
            "paix": {
                "patterns": ["paix", "olive"],
                "series": {
                    "1932": "Type Paix (n°280-289)"
                }
            },
            "ceres": {
                "patterns": ["ceres", "république"],
                "series": {
                    "1849": "Cérès (n°3-10)"
                }
            },
            "napoleon": {
                "patterns": ["napoleon", "empire"],
                "series": {
                    "1853": "Napoléon III (n°11-17)"
                }
            }
        }
    
    def extract_text_from_stamp(self, image: np.ndarray) -> str:
        """
        Extrait le texte d'un timbre via OCR
        
        Args:
            image: Image du timbre (BGR ou BGRA)
        
        Returns:
            str: Texte détecté
        """
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract non disponible pour l'OCR")
            return ""
        
        try:
            # Prétraitement pour améliorer l'OCR
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Amélioration du contraste avec CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Binarisation adaptative
            binary = cv2.adaptiveThreshold(
                enhanced, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 
                2
            )
            
            # Débruitage
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            # OCR avec Tesseract
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(
                denoised,
                lang=self.ocr_lang,
                config=custom_config
            )
            
            text_clean = text.strip().lower()
            
            if text_clean:
                logger.debug(f"OCR ({self.ocr_lang}): '{text_clean[:50]}...'")
            else:
                logger.debug("OCR: Aucun texte détecté")
            
            return text_clean
            
        except Exception as e:
            logger.error(f"Erreur OCR: {e}")
            return ""
    
    def extract_value(self, text: str) -> Optional[str]:
        """
        Extrait la valeur faciale du texte OCR
        
        Args:
            text: Texte OCR
        
        Returns:
            str: Valeur faciale (ex: "0.50", "1F", "2.00€")
        """
        # Patterns de valeur faciale
        patterns = [
            (r'(\d+[.,]\d+)\s*€', '€'),      # 1.50€
            (r'(\d+)\s*€', '€'),             # 2€
            (r'(\d+[.,]\d+)\s*[fF]', 'F'),   # 0.50F
            (r'(\d+)\s*[fF]', 'F'),          # 5F
            (r'(\d+)\s*c(?:entimes?)?', 'c'), # 50c (centimes)
            (r'(\d+[.,]\d+)', ''),           # 1.50 (sans symbole)
            (r'(\d+)', ''),                  # 5 (nombre seul)
        ]
        
        for pattern, unit in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                return f"{value}{unit}" if unit else value
        
        return None
    
    def extract_year(self, text: str) -> Optional[str]:
        """
        Extrait l'année du texte OCR
        
        Args:
            text: Texte OCR
        
        Returns:
            str: Année (YYYY)
        """
        # Chercher une année (1800-2099)
        year_match = re.search(r'\b(1[89]\d{2}|20\d{2})\b', text)
        if year_match:
            return year_match.group(1)
        return None
    
    def identify_stamp(self, image: np.ndarray) -> Dict:
        """
        Identifie un timbre via OCR et recherche dans la base
        
        Args:
            image: Image du timbre
        
        Returns:
            Dict: Informations du timbre {identified, name, yvert_number, value, year, confidence, ocr_text}
        """
        result = {
            "identified": False,
            "name": "Inconnu",
            "yvert_number": None,
            "value": None,
            "year": None,
            "confidence": 0.0,
            "ocr_text": "",
            "series": None
        }
        
        # Extraction du texte via OCR
        text = self.extract_text_from_stamp(image)
        result["ocr_text"] = text[:200]  # Limiter à 200 caractères
        
        if not text:
            logger.debug("Aucun texte extrait, timbre non identifiable")
            return result
        
        # Extraction de la valeur faciale
        value = self.extract_value(text)
        if value:
            result["value"] = value
            logger.debug(f"Valeur faciale détectée: {value}")
        
        # Extraction de l'année
        year = self.extract_year(text)
        if year:
            result["year"] = year
            logger.debug(f"Année détectée: {year}")
        
        # Recherche dans la base Yvert & Tellier
        best_match = None
        best_score = 0
        matching_patterns = []
        
        for stamp_type, data in self.yvert_database.items():
            score = 0
            matched = []
            
            for pattern in data["patterns"]:
                if pattern in text:
                    score += 1
                    matched.append(pattern)
            
            if score > best_score:
                best_score = score
                best_match = stamp_type
                matching_patterns = matched
        
        # Si un match est trouvé
        if best_match and best_score > 0:
            result["identified"] = True
            result["name"] = best_match.capitalize()
            result["confidence"] = min(best_score / len(self.yvert_database[best_match]["patterns"]), 1.0)
            
            # Essayer de trouver la série correspondante
            if year:
                series_data = self.yvert_database[best_match]["series"]
                if year in series_data:
                    result["series"] = series_data[year]
                    result["yvert_number"] = series_data[year]
            
            logger.info(f"✓ Timbre identifié: {result['name']} (confiance: {result['confidence']:.0%}, patterns: {matching_patterns})")
        else:
            logger.debug("Aucune correspondance trouvée dans la base Yvert & Tellier")
        
        return result

def is_recognition_available() -> bool:
    """
    Vérifie si la reconnaissance est disponible
    
    Returns:
        bool: True si Tesseract est disponible
    """
    return TESSERACT_AVAILABLE

def get_tesseract_info() -> Dict:
    """
    Retourne les informations sur l'installation Tesseract
    
    Returns:
        Dict: Informations de configuration
    """
    info = {
        "available": TESSERACT_AVAILABLE,
        "tesseract_path": r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        "tessdata_path": r'C:\Program Files\Tesseract-OCR\tessdata',
        "default_lang": DEFAULT_LANG,
        "fra_available": False
    }
    
    if TESSERACT_AVAILABLE:
        tessdata_path = r'C:\Program Files\Tesseract-OCR\tessdata'
        fra_data = os.path.join(tessdata_path, 'fra.traineddata')
        info["fra_available"] = os.path.exists(fra_data)
    
    return info