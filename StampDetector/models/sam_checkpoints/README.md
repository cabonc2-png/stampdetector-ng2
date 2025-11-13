# Modèles SAM

Ce dossier contient les checkpoints (poids) du modèle SAM.

## Installation

Lancez le script d'installation automatique :

```bash
python install_sam.py
```

Le modèle `sam_vit_b_01ec64.pth` (~375 MB) sera téléchargé ici automatiquement.

## Téléchargement manuel

Si le script d'installation ne fonctionne pas :

1. Téléchargez le modèle ViT-B :
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

2. Placez-le dans ce dossier :
   `StampDetector/models/sam_checkpoints/sam_vit_b_01ec64.pth`

## Modèles disponibles

SAM propose 3 modèles (du plus petit au plus gros) :

| Modèle | Taille | Précision | Vitesse | Recommandé |
|--------|--------|-----------|---------|------------|
| ViT-B  | 375 MB | ⭐⭐⭐⭐  | Rapide | ✅ **Oui** (par défaut) |
| ViT-L  | 1.2 GB | ⭐⭐⭐⭐⭐ | Moyen | Si GPU puissant |
| ViT-H  | 2.4 GB | ⭐⭐⭐⭐⭐ | Lent | Recherche uniquement |

**Nous utilisons ViT-B** : excellent compromis précision/vitesse pour la détection de timbres.
