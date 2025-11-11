"""
Détecteur SAM2 (optionnel)
"""
from utils.logger import logger

SAM_AVAILABLE = False

def is_available() -> bool:
    return SAM_AVAILABLE