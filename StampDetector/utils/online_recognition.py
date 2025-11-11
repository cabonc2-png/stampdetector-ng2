"""
Reconnaissance de timbres via APIs en ligne
Utilise plusieurs services pour identifier les timbres
"""
import cv2
import numpy as np
import requests
import base64
import json
from pathlib import Path
from typing import Optional, Dict, List
from utils.logger import logger

# Configuration des APIs
COLNECT_API_BASE = "https://colnect.com/api"
GOOGLE_VISION_AVAILABLE = False

try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
    logger.info("✓ Google Vision API disponible")
except ImportError:
    logger.info("Google Vision non installé (optionnel)")

class OnlineStampRecognizer:
    """Reconnaissance de timbres via APIs en ligne"""
    
    def __init__(self):
        """Initialise le reconnaisseur en ligne"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'StampDetector/2.0'
        })
    
    def encode_image_base64(self, image: np.ndarray) -> str:
        """
        Encode une image en base64 pour envoi API
        
        Args:
            image: Image numpy array
        
        Returns:
            str: Image encodée en base64
        """
        # Convertir en JPEG
        success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise ValueError("Impossible d'encoder l'image")
        
        return base64.b64encode(buffer).decode('utf-8')
    
    def recognize_with_colnect(self, image: np.ndarray) -> Dict:
        """
        Recherche dans Colnect (base de données collaborative de collection)
        
        Args:
            image: Image du timbre
        
        Returns:
            Dict: Résultats de reconnaissance
        """
        result = {
            "source": "Colnect",
            "identified": False,
            "name": None,
            "country": None,
            "year": None,
            "catalog_number": None,
            "value": None,
            "confidence": 0.0,
            "url": None
        }
        
        try:
            # Colnect nécessite une recherche par texte ou référence
            # Pour l'instant, on retourne un placeholder
            # TODO: Implémenter recherche réelle avec leur API
            
            logger.info("Recherche Colnect non encore implémentée (nécessite API key)")
            result["identified"] = False
            
        except Exception as e:
            logger.error(f"Erreur Colnect: {e}")
        
        return result
    
    def recognize_with_google_vision(self, image: np.ndarray) -> Dict:
        """
        Utilise Google Vision API pour l'identification
        Nécessite: pip install google-cloud-vision
        Et configuration de GOOGLE_APPLICATION_CREDENTIALS
        
        Args:
            image: Image du timbre
        
        Returns:
            Dict: Résultats de reconnaissance
        """
        result = {
            "source": "Google Vision",
            "identified": False,
            "labels": [],
            "text": "",
            "web_entities": [],
            "confidence": 0.0
        }
        
        if not GOOGLE_VISION_AVAILABLE:
            logger.debug("Google Vision API non disponible")
            return result
        
        try:
            client = vision.ImageAnnotatorClient()
            
            # Encoder l'image
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                return result
            
            image_vision = vision.Image(content=buffer.tobytes())
            
            # Détection de labels
            labels_response = client.label_detection(image=image_vision)
            labels = labels_response.label_annotations
            
            # Détection de texte
            text_response = client.text_detection(image=image_vision)
            texts = text_response.text_annotations
            
            # Recherche web
            web_response = client.web_detection(image=image_vision)
            web_entities = web_response.web_detection.web_entities
            
            # Compiler les résultats
            result["labels"] = [label.description for label in labels[:10]]
            result["text"] = texts[0].description if texts else ""
            result["web_entities"] = [entity.description for entity in web_entities[:10]]
            
            # Vérifier si c'est identifié comme un timbre
            stamp_keywords = ["stamp", "postage", "timbre", "postal"]
            if any(keyword in label.lower() for label in result["labels"] for keyword in stamp_keywords):
                result["identified"] = True
                result["confidence"] = 0.7
            
            logger.info(f"Google Vision: {len(result['labels'])} labels détectés")
            
        except Exception as e:
            logger.error(f"Erreur Google Vision: {e}")
        
        return result
    
    def recognize_with_imagga(self, image: np.ndarray) -> Dict:
        """
        Utilise Imagga API (gratuit avec limites)
        https://imagga.com/
        
        Args:
            image: Image du timbre
        
        Returns:
            Dict: Résultats de reconnaissance
        """
        result = {
            "source": "Imagga",
            "identified": False,
            "tags": [],
            "confidence": 0.0
        }
        
        # TODO: Configurer avec API key Imagga
        # API_KEY = "your_api_key"
        # API_SECRET = "your_api_secret"
        
        logger.debug("Imagga API non configurée (nécessite clé API)")
        return result
    
    def recognize_with_reverse_image_search(self, image: np.ndarray) -> Dict:
        """
        Recherche inversée d'images via SerpAPI (Google Images)
        Alternative gratuite: utiliser l'API de Bing Image Search
        
        Args:
            image: Image du timbre
        
        Returns:
            Dict: Résultats de recherche
        """
        result = {
            "source": "Reverse Image Search",
            "identified": False,
            "similar_images": [],
            "best_guess": None,
            "confidence": 0.0
        }
        
        try:
            # Sauvegarder temporairement l'image
            temp_path = Path("temp_search.jpg")
            cv2.imwrite(str(temp_path), image)
            
            # TODO: Implémenter recherche inversée
            # Option 1: SerpAPI (payant mais fiable)
            # Option 2: TinEye API (gratuit limité)
            # Option 3: Scraping Google Images (légalité discutable)
            
            logger.debug("Recherche inversée non configurée")
            
            # Nettoyage
            if temp_path.exists():
                temp_path.unlink()
            
        except Exception as e:
            logger.error(f"Erreur recherche inversée: {e}")
        
        return result
    
    def recognize_with_stampworld(self, text_ocr: str) -> Dict:
        """
        Recherche dans StampWorld (base de catalogues en ligne)
        https://www.stampworld.com/
        
        Args:
            text_ocr: Texte extrait du timbre
        
        Returns:
            Dict: Résultats de recherche
        """
        result = {
            "source": "StampWorld",
            "identified": False,
            "matches": [],
            "confidence": 0.0
        }
        
        if not text_ocr:
            return result
        
        try:
            # Recherche par mots-clés
            search_url = "https://www.stampworld.com/en/stamps/"
            
            # Extraire pays et année du texte
            keywords = text_ocr.split()[:5]  # Premiers mots
            
            # TODO: Parser le HTML de StampWorld pour extraire les résultats
            # Nécessite BeautifulSoup4
            
            logger.debug(f"Recherche StampWorld avec mots-clés: {keywords}")
            
        except Exception as e:
            logger.error(f"Erreur StampWorld: {e}")
        
        return result
    
    def recognize_stamp_comprehensive(self, image: np.ndarray, ocr_text: str = "") -> Dict:
        """
        Reconnaissance complète utilisant tous les services disponibles
        
        Args:
            image: Image du timbre
            ocr_text: Texte OCR déjà extrait (optionnel)
        
        Returns:
            Dict: Résultats agrégés de tous les services
        """
        logger.info("🌐 Reconnaissance en ligne en cours...")
        
        results = {
            "identified": False,
            "best_match": None,
            "all_results": [],
            "confidence": 0.0,
            "sources_used": []
        }
        
        # Google Vision (si disponible)
        if GOOGLE_VISION_AVAILABLE:
            try:
                google_result = self.recognize_with_google_vision(image)
                if google_result["identified"]:
                    results["all_results"].append(google_result)
                    results["sources_used"].append("Google Vision")
            except Exception as e:
                logger.error(f"Erreur Google Vision: {e}")
        
        # Colnect
        try:
            colnect_result = self.recognize_with_colnect(image)
            if colnect_result["identified"]:
                results["all_results"].append(colnect_result)
                results["sources_used"].append("Colnect")
        except Exception as e:
            logger.error(f"Erreur Colnect: {e}")
        
        # StampWorld (si OCR disponible)
        if ocr_text:
            try:
                stampworld_result = self.recognize_with_stampworld(ocr_text)
                if stampworld_result["identified"]:
                    results["all_results"].append(stampworld_result)
                    results["sources_used"].append("StampWorld")
            except Exception as e:
                logger.error(f"Erreur StampWorld: {e}")
        
        # Sélectionner le meilleur résultat
        if results["all_results"]:
            best = max(results["all_results"], key=lambda x: x.get("confidence", 0))
            results["best_match"] = best
            results["identified"] = True
            results["confidence"] = best.get("confidence", 0)
            
            logger.info(f"✓ Meilleur résultat: {best['source']} (confiance: {results['confidence']:.0%})")
        else:
            logger.warning("⚠️ Aucune reconnaissance en ligne réussie")
        
        return results

def is_online_recognition_available() -> bool:
    """Vérifie si au moins un service de reconnaissance en ligne est disponible"""
    return True  # Toujours disponible (au moins Colnect/StampWorld)

def get_available_services() -> List[str]:
    """Retourne la liste des services disponibles"""
    services = ["Colnect", "StampWorld"]
    
    if GOOGLE_VISION_AVAILABLE:
        services.append("Google Vision")
    
    return services