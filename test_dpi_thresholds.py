#!/usr/bin/env python3
"""
Script de test pour vérifier les seuils adaptatifs de détection
"""

def mm_to_pixels(mm: float, dpi: int) -> int:
    """Convertit des millimètres en pixels"""
    px_per_mm = dpi / 25.4
    pixels = int(round(mm * px_per_mm))
    return pixels

def calculate_min_stamp_area(dpi: int, min_width_mm: float = 15.0, min_height_mm: float = 15.0) -> int:
    """Calcule la surface minimale d'un timbre en pixels"""
    min_width_px = mm_to_pixels(min_width_mm, dpi)
    min_height_px = mm_to_pixels(min_height_mm, dpi)
    min_area = min_width_px * min_height_px
    return min_area

def test_dpi_thresholds():
    """Test des seuils adaptatifs pour différents DPI"""
    print("=" * 70)
    print("Test des seuils adaptatifs de détection par DPI")
    print("=" * 70)
    print()
    print("Taille minimale d'un timbre: 15mm x 15mm")
    print()

    test_dpis = [150, 300, 400, 500, 600, 800, 1200]

    print(f"{'DPI':<8} {'Surface (pixels²)':<20} {'Comparaison vs ancien seuil'}")
    print("-" * 70)

    OLD_THRESHOLD = 5000  # Ancien seuil fixe

    for dpi in test_dpis:
        min_area = calculate_min_stamp_area(dpi)
        ratio = min_area / OLD_THRESHOLD

        if ratio < 1:
            comparison = f"⚠️  {ratio:.1f}x PLUS BAS (risque de faux positifs)"
        elif ratio > 1:
            comparison = f"✓  {ratio:.1f}x PLUS HAUT (meilleur filtrage)"
        else:
            comparison = "=  Équivalent"

        print(f"{dpi:<8} {min_area:<20,} {comparison}")

    print()
    print("=" * 70)
    print("Conclusion:")
    print("  - À haute résolution (>500 DPI), le seuil adaptatif est BEAUCOUP")
    print("    plus élevé, ce qui évite les faux positifs ('timbres imaginaires')")
    print("  - L'ancien seuil fixe de 5000 px² était inadapté aux hauts DPI")
    print("=" * 70)

if __name__ == "__main__":
    test_dpi_thresholds()
