# Installation Manuelle de SAM

Si le script automatique `install_sam.py` échoue, suivez ces instructions pas à pas.

## 📋 Prérequis

- Python 3.8+ installé
- Connexion Internet
- ~2 GB d'espace disque libre

## 🔧 Installation Manuelle (Windows)

### Étape 1: Ouvrir un terminal

1. Appuyez sur `Windows + R`
2. Tapez `cmd` et appuyez sur Entrée
3. Naviguez vers le dossier du projet :
   ```cmd
   cd C:\chemin\vers\stampdetector-ng2
   ```

### Étape 2: Mettre à jour pip

```cmd
python -m pip install --upgrade pip
```

### Étape 3: Installer PyTorch (IMPORTANT)

**Option A: Installation standard (recommandée)**
```cmd
python -m pip install torch torchvision
```

**Option B: Si l'option A échoue, essayez:**
```cmd
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Option C: Si tout échoue:**
1. Allez sur https://pytorch.org/get-started/locally/
2. Sélectionnez:
   - PyTorch Build: **Stable**
   - Your OS: **Windows**
   - Package: **Pip**
   - Language: **Python**
   - Compute Platform: **CPU**
3. Copiez la commande générée et exécutez-la

⏱️ **Cette étape peut prendre 5-10 minutes**

### Étape 4: Vérifier l'installation de PyTorch

```cmd
python -c "import torch; print('PyTorch OK:', torch.__version__)"
```

Vous devriez voir:
```
PyTorch OK: 2.x.x
```

### Étape 5: Installer Segment Anything

```cmd
python -m pip install segment-anything
```

### Étape 6: Créer le dossier pour le modèle

```cmd
mkdir StampDetector\models\sam_checkpoints
```

### Étape 7: Télécharger le modèle SAM

**Option A: Via Python (recommandé)**
```cmd
python -c "import urllib.request; urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', 'StampDetector/models/sam_checkpoints/sam_vit_b_01ec64.pth'); print('OK')"
```

**Option B: Téléchargement manuel**

1. Téléchargez le fichier depuis votre navigateur:
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

2. Placez le fichier téléchargé dans:
   ```
   StampDetector\models\sam_checkpoints\sam_vit_b_01ec64.pth
   ```

⏱️ **Taille: ~375 MB - Cela peut prendre quelques minutes**

### Étape 8: Vérifier l'installation complète

```cmd
python -c "import torch; from segment_anything import sam_model_registry; print('Installation OK!')"
```

Vous devriez voir:
```
Installation OK!
```

## 🐧 Installation Manuelle (Linux/Mac)

### Étapes complètes

```bash
# Mise à jour pip
python3 -m pip install --upgrade pip

# Installation PyTorch
python3 -m pip install torch torchvision

# Installation Segment Anything
python3 -m pip install segment-anything

# Création du dossier
mkdir -p StampDetector/models/sam_checkpoints

# Téléchargement du modèle
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O StampDetector/models/sam_checkpoints/sam_vit_b_01ec64.pth

# Vérification
python3 -c "import torch; from segment_anything import sam_model_registry; print('OK')"
```

## ✅ Test Final

Une fois l'installation terminée, lancez StampDetector:

```cmd
python main.py
```

Dans les logs, vous devriez voir:

```
Sélection du backend de détection...
Initialisation du détecteur SAM...
Chargement du modèle SAM sur CPU...
✓ SAM chargé avec succès
🎯 Backend sélectionné: SAM (IA)
```

✅ **Si vous voyez ça, SAM est opérationnel!**

## ❌ Résolution de Problèmes

### Erreur: "No module named 'torch'"

PyTorch n'est pas installé. Reprenez l'étape 3.

### Erreur: "No module named 'segment_anything'"

Segment Anything n'est pas installé:
```cmd
python -m pip install segment-anything
```

### Erreur: "FileNotFoundError: sam_vit_b_01ec64.pth"

Le modèle n'est pas téléchargé. Reprenez l'étape 7.

### Erreur: "SSL: CERTIFICATE_VERIFY_FAILED"

Problème de certificat SSL:
```cmd
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org torch torchvision
```

### Erreur: "Could not find a version that satisfies"

Votre version de Python est trop ancienne. PyTorch nécessite Python 3.8+.

Vérifiez:
```cmd
python --version
```

### L'installation prend trop de temps

PyTorch est gros (~2 GB). C'est normal. Patience! ☕

### Antivirus bloque le téléchargement

Désactivez temporairement votre antivirus pendant l'installation.

## 📞 Besoin d'Aide?

Si rien ne fonctionne:

1. Vérifiez que vous avez bien Python 3.8+
2. Vérifiez votre connexion Internet
3. Essayez de redémarrer votre PC
4. Consultez les logs d'erreur complets

## 🔄 Retour à OpenCV

Si SAM ne s'installe vraiment pas, l'application continuera d'utiliser OpenCV automatiquement (détection basique mais fonctionnelle).

---

Bon courage! 🚀
