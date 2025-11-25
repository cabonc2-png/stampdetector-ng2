"""
Fenêtre principale StampDetector Pro
Interface graphique avec scan intégré et reconnaissance visuelle (Google Vision + IA)
"""
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QProgressBar,
    QCheckBox, QGroupBox, QScrollArea, QMessageBox, QSpinBox,
    QTextEdit, QTabWidget, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np

from ui.widgets import DropZoneWidget, LogWidget
from vision.processor import StampProcessor
from utils.logger import logger  # ← IMPORTANT: logger doit être importé ICI

# Import conditionnel du scanner
try:
    from utils.scanner_utils import ScannerManager, is_scanner_available
    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False
    logger.warning("Module scanner non disponible")

# Import conditionnel de la reconnaissance locale (OCR)
try:
    from utils.stamp_recognition import StampRecognizer, is_recognition_available
    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False
    logger.warning("Module reconnaissance locale non disponible")

# Import conditionnel de la reconnaissance visuelle intelligente
try:
    from utils.smart_recognition import recognize_stamp_online
    ONLINE_RECOGNITION_AVAILABLE = True
    logger.info("✓ Reconnaissance visuelle intelligente disponible")
except ImportError:
    ONLINE_RECOGNITION_AVAILABLE = False
    logger.warning("Module reconnaissance visuelle non disponible")

# Import Google Vision API (priorité maximale si configuré)
try:
    from utils.google_vision_api import is_google_vision_available
    GOOGLE_VISION_AVAILABLE = is_google_vision_available()
    if GOOGLE_VISION_AVAILABLE:
        logger.info("✓ Google Vision API configurée et prête ⭐")
    else:
        logger.info("Google Vision disponible mais credentials non configurés")
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    logger.info("Google Vision non installé (optionnel)")

class ProcessingThread(QThread):
    """Thread pour le traitement des images sans bloquer l'UI"""
    
    progress_updated = Signal(int, str)
    processing_complete = Signal(object)
    error_occurred = Signal(str)
    
    def __init__(self, processor, image_path, margin_mm, save_jpg, detection_mode="stamps", recognize_stamps=False, use_online=False):
        super().__init__()
        self.processor = processor
        self.image_path = image_path
        self.margin_mm = margin_mm
        self.save_jpg = save_jpg
        self.detection_mode = detection_mode
        self.recognize_stamps = recognize_stamps
        self.use_online = use_online
    
    def run(self):
        """Exécute le traitement dans un thread séparé"""
        try:
            # Traitement de l'image (détection + découpe)
            result = self.processor.process_image(
                self.image_path,
                margin_mm=self.margin_mm,
                save_jpg=self.save_jpg,
                detection_mode=self.detection_mode,
                progress_callback=self.progress_updated.emit
            )
            
            # Reconnaissance si demandée
            if self.recognize_stamps and (RECOGNITION_AVAILABLE or ONLINE_RECOGNITION_AVAILABLE):
                self.progress_updated.emit(95, "🔍 Reconnaissance des timbres en cours...")
                
                recognizer_local = StampRecognizer() if RECOGNITION_AVAILABLE else None
                recognized_stamps = []
                
                for idx, output_file in enumerate(result.output_files):
                    if output_file.suffix == '.png':
                        try:
                            stamp_img = cv2.imread(str(output_file))
                            if stamp_img is not None:
                                self.progress_updated.emit(95 + idx, f"Analyse du timbre {idx+1}...")
                                
                                # Reconnaissance locale (OCR Tesseract) - optionnelle
                                info_local = {}
                                if recognizer_local and not self.use_online:
                                    try:
                                        info_local = recognizer_local.identify_stamp(stamp_img)
                                        logger.debug(f"Timbre {idx+1}: Reconnaissance locale terminée")
                                    except Exception as e:
                                        logger.error(f"Erreur OCR timbre {idx+1}: {e}")
                                
                                # Reconnaissance visuelle en ligne (prioritaire si activée)
                                info_online = {}
                                if self.use_online and ONLINE_RECOGNITION_AVAILABLE:
                                    try:
                                        # Utiliser directement la fonction recognize_stamp_online
                                        info_online = recognize_stamp_online(stamp_img)
                                        logger.info(f"Timbre {idx+1}: Reconnaissance visuelle terminée")
                                    except Exception as e:
                                        logger.error(f"Erreur reconnaissance en ligne timbre {idx+1}: {e}")
                                
                                # Fusionner les résultats (priorité à l'en ligne)
                                if info_online and info_online.get('identified'):
                                    combined_info = {
                                        'identified': True,
                                        'name': info_online.get('name', 'Inconnu'),
                                        'country': info_online.get('country'),
                                        'confidence': info_online.get('confidence', 0.0),
                                        'suggestions': info_online.get('suggestions', []),
                                        'visual_features': info_online.get('visual_info', {}),
                                        'recognition_source': ['🌐 Reconnaissance visuelle intelligente']
                                    }
                                    
                                    # Ajouter infos locales si pertinentes
                                    if info_local.get('value'):
                                        combined_info['value'] = info_local['value']
                                        combined_info['recognition_source'].append('OCR Local (valeur)')
                                    if info_local.get('year'):
                                        combined_info['year'] = info_local['year']
                                        combined_info['recognition_source'].append('OCR Local (année)')
                                    
                                elif info_local and info_local.get('identified'):
                                    # Fallback sur reconnaissance locale
                                    combined_info = {
                                        **info_local,
                                        'recognition_source': ['📝 OCR Local']
                                    }
                                else:
                                    # Aucune reconnaissance réussie
                                    combined_info = {
                                        'identified': False,
                                        'name': 'Non identifié',
                                        'confidence': 0.0,
                                        'suggestions': info_online.get('suggestions', []) if info_online else [],
                                        'visual_features': info_online.get('visual_info', {}) if info_online else {},
                                        'recognition_source': ['❌ Aucune reconnaissance réussie']
                                    }
                                
                                recognized_stamps.append({
                                    'file': output_file,
                                    'info': combined_info
                                })
                        
                        except Exception as e:
                            logger.error(f"Erreur reconnaissance timbre {idx+1}: {e}")
                            continue
                
                result.recognized_stamps = recognized_stamps
                logger.info(f"✓ Reconnaissance terminée pour {len(recognized_stamps)} timbre(s)")
            
            self.processing_complete.emit(result)
            
        except Exception as e:
            logger.error(f"Erreur traitement: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

class ScanThread(QThread):
    """Thread pour le scan"""

    scan_complete = Signal(object)
    scan_error = Signal(str)
    scan_status = Signal(str)

    def __init__(self, dpi, scanner_index=0, color_mode="color"):
        super().__init__()
        self.dpi = dpi
        self.scanner_index = scanner_index
        self.color_mode = color_mode

    def run(self):
        """Exécute le scan dans un thread séparé"""
        try:
            self.scan_status.emit("Connexion au scanner...")
            scanner = ScannerManager()

            if not scanner.connect_scanner(scanner_index=self.scanner_index):
                self.scan_error.emit("Impossible de se connecter au scanner")
                return

            scanner_name = scanner.get_scanner_name()
            self.scan_status.emit(f"Scan avec {scanner_name} ({self.dpi} DPI)...")

            # Créer le dossier scans
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"scans/scan_{timestamp}.png")
            output_path.parent.mkdir(exist_ok=True)

            scanned_file = scanner.scan_image(
                output_path,
                dpi=self.dpi,
                color_mode=self.color_mode,
                format="PNG"
            )

            if scanned_file and scanned_file.exists():
                self.scan_complete.emit(scanned_file)
            else:
                self.scan_error.emit("Échec du scan")

        except Exception as e:
            self.scan_error.emit(str(e))

class MainWindow(QMainWindow):
    """Fenêtre principale de StampDetector Pro"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StampDetector Pro - Détection & Découpage de Timbres")
        self.setMinimumSize(1000, 800)
        
        self.current_image_path = None
        self.current_result = None
        self.processor = None
        self.processing_thread = None
        self.scan_thread = None
        
        self._init_ui()
        
        # Initialisation du processeur dans un timer
        QTimer.singleShot(100, self._init_processor)
    
    def _init_ui(self):
        """Initialise l'interface utilisateur"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Style sombre moderne
        self.setStyleSheet("""
            QMainWindow, QWidget { 
                background-color: #1e1e1e; 
                color: #d4d4d4; 
            }
            QGroupBox {
                border: 1px solid #3c3c3c;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { 
                background-color: #1177bb; 
            }
            QPushButton:pressed { 
                background-color: #0d5689; 
            }
            QPushButton:disabled { 
                background-color: #3c3c3c; 
                color: #666; 
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2b2b2b;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 5px;
                color: #d4d4d4;
            }
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 5px;
                text-align: center;
                background-color: #2b2b2b;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #2b2b2b;
                color: #d4d4d4;
                padding: 8px 20px;
                border: 1px solid #3c3c3c;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #0e639c;
                color: white;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #2b2b2b;
                border: 2px solid #3c3c3c;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #0e639c;
                border: 2px solid #0e639c;
                border-radius: 3px;
            }
        """)
        
        # Titre avec version
        title_layout = QHBoxLayout()
        title = QLabel("🔍 StampDetector Pro")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #0078d4;")
        title_layout.addWidget(title)
        
        version_label = QLabel("v2.4 - Mode Carnet")
        version_label.setStyleSheet("font-size: 12px; color: #666;")
        title_layout.addWidget(version_label)
        title_layout.addStretch()
        
        main_layout.addLayout(title_layout)
        
        # Section Scanner (si disponible)
        if SCANNER_AVAILABLE and is_scanner_available():
            scan_group = QGroupBox("📷 Scanner")
            scan_layout = QVBoxLayout()

            # Ligne 1: Sélection du scanner
            scanner_select_layout = QHBoxLayout()
            scanner_select_layout.addWidget(QLabel("Scanner:"))

            self.scanner_combo = QComboBox()
            self.scanner_combo.setMinimumWidth(250)
            self.scanner_combo.setStyleSheet("""
                QComboBox {
                    background-color: #2b2b2b;
                    border: 1px solid #3c3c3c;
                    border-radius: 3px;
                    padding: 5px;
                    color: #d4d4d4;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox::down-arrow {
                    image: url(none);
                    border-left: 4px solid transparent;
                    border-right: 4px solid transparent;
                    border-top: 6px solid #d4d4d4;
                    margin-right: 5px;
                }
                QComboBox:hover {
                    border: 1px solid #0e639c;
                }
                QComboBox QAbstractItemView {
                    background-color: #2b2b2b;
                    color: #d4d4d4;
                    selection-background-color: #0e639c;
                    border: 1px solid #3c3c3c;
                }
            """)
            scanner_select_layout.addWidget(self.scanner_combo)

            refresh_btn = QPushButton("🔄")
            refresh_btn.setMaximumWidth(40)
            refresh_btn.setToolTip("Actualiser la liste des scanners")
            refresh_btn.clicked.connect(self._refresh_scanners)
            scanner_select_layout.addWidget(refresh_btn)

            scanner_select_layout.addStretch()
            scan_layout.addLayout(scanner_select_layout)

            # Ligne 2: Contrôles de scan
            scan_controls_layout = QHBoxLayout()

            self.scan_btn = QPushButton("🖨️ Lancer un scan")
            self.scan_btn.setMinimumWidth(150)
            self.scan_btn.clicked.connect(self._on_scan_clicked)
            scan_controls_layout.addWidget(self.scan_btn)

            scan_controls_layout.addWidget(QLabel("Résolution:"))
            self.scan_dpi = QSpinBox()
            self.scan_dpi.setRange(150, 1200)
            self.scan_dpi.setValue(300)
            self.scan_dpi.setSuffix(" DPI")
            self.scan_dpi.setMinimumWidth(100)
            scan_controls_layout.addWidget(self.scan_dpi)

            scan_controls_layout.addStretch()

            self.scan_status_label = QLabel("Scanner prêt")
            self.scan_status_label.setStyleSheet("color: #4ec9b0; padding: 5px;")
            scan_controls_layout.addWidget(self.scan_status_label)

            scan_layout.addLayout(scan_controls_layout)

            scan_group.setLayout(scan_layout)
            main_layout.addWidget(scan_group)

            # Remplir la liste des scanners
            QTimer.singleShot(200, self._refresh_scanners)
        
        # Zone de drop
        self.drop_zone = DropZoneWidget()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        main_layout.addWidget(self.drop_zone)
        
        # Paramètres de traitement
        params_group = QGroupBox("⚙️ Paramètres de découpage")
        params_layout = QHBoxLayout()

        # Mode de détection
        params_layout.addWidget(QLabel("Mode:"))
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.addItem("🔍 Timbres individuels", "stamps")
        self.detection_mode_combo.addItem("📚 Carnets/Blocs", "booklets")
        self.detection_mode_combo.setMinimumWidth(180)
        self.detection_mode_combo.setToolTip("Mode de détection:\n"
                                              "• Timbres individuels : détecte tous les timbres\n"
                                              "• Carnets/Blocs : détecte uniquement les grands carnets et blocs")
        self.detection_mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #2b2b2b;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 5px;
                color: #d4d4d4;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(none);
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #d4d4d4;
                margin-right: 5px;
            }
            QComboBox:hover {
                border: 1px solid #0e639c;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: #d4d4d4;
                selection-background-color: #0e639c;
                border: 1px solid #3c3c3c;
            }
        """)
        params_layout.addWidget(self.detection_mode_combo)

        params_layout.addSpacing(20)

        params_layout.addWidget(QLabel("Marge:"))
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 10.0)
        self.margin_spin.setValue(1.0)
        self.margin_spin.setSingleStep(0.1)
        self.margin_spin.setSuffix(" mm")
        self.margin_spin.setMinimumWidth(100)
        self.margin_spin.setToolTip("Marge ajoutée autour de chaque timbre détecté")
        params_layout.addWidget(self.margin_spin)

        params_layout.addStretch()

        # Bouton traiter
        self.process_btn = QPushButton("✂️ Détecter et Découper les Timbres")
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumWidth(220)
        self.process_btn.clicked.connect(self._on_process_clicked)
        params_layout.addWidget(self.process_btn)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        main_layout.addWidget(self.progress_bar)
        
        # Aperçu des détections
        preview_group = QGroupBox("👁️ Aperçu des détections")
        preview_layout = QVBoxLayout()

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setMinimumHeight(350)

        self.preview_label = QLabel("Glissez une image ou lancez un scan pour commencer")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("color: #666; font-size: 14px; padding: 40px;")

        self.preview_scroll.setWidget(self.preview_label)
        preview_layout.addWidget(self.preview_scroll)

        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)
        
        # Logs
        logs_group = QGroupBox("📋 Logs")
        logs_layout = QVBoxLayout()
        
        self.log_widget = LogWidget()
        logs_layout.addWidget(self.log_widget)
        
        logs_group.setLayout(logs_layout)
        main_layout.addWidget(logs_group)
        
        # Barre de statut
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Prêt")
        self.status_label.setStyleSheet("color: #4ec9b0; padding: 5px; font-size: 13px;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        # Info backend
        self.backend_label = QLabel("")
        self.backend_label.setStyleSheet("color: #888; padding: 5px; font-size: 11px;")
        status_layout.addWidget(self.backend_label)
        
        main_layout.addLayout(status_layout)
    
    def _init_processor(self):
        """Initialise le processeur de traitement"""
        try:
            output_dir = Path(__file__).parent.parent / "outputs"
            self.processor = StampProcessor(output_dir)
            
            backend = self.processor.detector.get_backend_name()
            self.log_widget.append_log(f"Moteur de détection: {backend}", "SUCCESS")
            self.backend_label.setText(f"Détection: {backend}")

            self.log_widget.append_log("✓ Mode: Découpage de timbres (PNG avec transparence)", "INFO")

            self.status_label.setText(f"✓ Prêt à découper")
            
        except Exception as e:
            self.log_widget.append_log(f"Erreur initialisation: {e}", "ERROR")
            self.status_label.setText("❌ Erreur d'initialisation")
            QMessageBox.critical(self, "Erreur", f"Impossible d'initialiser:\n{e}")
    
    def _refresh_scanners(self):
        """Actualise la liste des scanners disponibles"""
        if not SCANNER_AVAILABLE:
            return

        try:
            scanner_manager = ScannerManager()
            scanners = scanner_manager.list_scanners()

            self.scanner_combo.clear()

            if scanners:
                self.scanner_combo.addItems(scanners)
                self.log_widget.append_log(f"✓ {len(scanners)} scanner(s) trouvé(s)", "SUCCESS")
                self.scan_status_label.setText(f"✓ {len(scanners)} scanner(s) disponible(s)")
            else:
                self.scanner_combo.addItem("Aucun scanner trouvé")
                self.scan_btn.setEnabled(False)
                self.log_widget.append_log("⚠️ Aucun scanner trouvé", "WARNING")
                self.scan_status_label.setText("⚠️ Aucun scanner")

        except Exception as e:
            self.log_widget.append_log(f"Erreur énumération scanners: {e}", "ERROR")
            self.scanner_combo.addItem("Erreur détection")
            self.scan_btn.setEnabled(False)

    def _on_scan_clicked(self):
        """Lance un scan"""
        if not SCANNER_AVAILABLE:
            QMessageBox.warning(self, "Scanner", "Module scanner non disponible")
            return

        # Vérifier qu'un scanner est sélectionné
        if self.scanner_combo.count() == 0 or self.scanner_combo.currentText() == "Aucun scanner trouvé":
            QMessageBox.warning(self, "Scanner", "Aucun scanner disponible")
            return

        try:
            self.scan_btn.setEnabled(False)
            scanner_index = self.scanner_combo.currentIndex()
            scanner_name = self.scanner_combo.currentText()

            self.log_widget.append_log(f"🖨️ Scan avec {scanner_name}...", "INFO")

            # Lancer le scan dans un thread
            self.scan_thread = ScanThread(
                dpi=self.scan_dpi.value(),
                scanner_index=scanner_index,
                color_mode="color"
            )

            self.scan_thread.scan_status.connect(self._on_scan_status)
            self.scan_thread.scan_complete.connect(self._on_scan_complete)
            self.scan_thread.scan_error.connect(self._on_scan_error)

            self.scan_thread.start()

        except Exception as e:
            self.log_widget.append_log(f"Erreur scan: {e}", "ERROR")
            self.scan_btn.setEnabled(True)
    
    def _on_scan_status(self, message):
        """Met à jour le statut du scan"""
        self.log_widget.append_log(message, "INFO")
        if SCANNER_AVAILABLE and hasattr(self, 'scan_status_label'):
            self.scan_status_label.setText(message)
    
    def _on_scan_complete(self, scanned_file):
        """Gère la fin du scan"""
        self.scan_btn.setEnabled(True)
        self.log_widget.append_log(f"✓ Scan réussi: {scanned_file.name}", "SUCCESS")
        
        if SCANNER_AVAILABLE and hasattr(self, 'scan_status_label'):
            self.scan_status_label.setText("✓ Scan terminé")
        
        self.current_image_path = scanned_file
        self._show_preview(scanned_file)
        self.process_btn.setEnabled(True)
        self.status_label.setText(f"📄 {scanned_file.name}")
    
    def _on_scan_error(self, error_msg):
        """Gère les erreurs de scan"""
        self.scan_btn.setEnabled(True)
        self.log_widget.append_log(f"❌ Erreur scan: {error_msg}", "ERROR")
        
        if SCANNER_AVAILABLE and hasattr(self, 'scan_status_label'):
            self.scan_status_label.setText("❌ Échec du scan")
        
        QMessageBox.critical(self, "Erreur de scan", f"Le scan a échoué:\n{error_msg}")
    
    def _on_files_dropped(self, files):
        """Gère le drop de fichiers"""
        if not files:
            return
        
        file_path = Path(files[0])
        self.current_image_path = file_path
        
        self.log_widget.append_log(f"📁 Image chargée: {file_path.name}", "INFO")
        self.status_label.setText(f"📄 {file_path.name}")
        self.process_btn.setEnabled(True)
        
        self._show_preview(file_path)
    
    def _show_preview(self, image_path):
        """Affiche un aperçu de l'image"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return
            
            # Redimensionner pour l'aperçu
            h, w = image.shape[:2]
            max_width = 700
            if w > max_width:
                scale = max_width / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
            
            # Conversion pour Qt
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_image)
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setScaledContents(False)
            
        except Exception as e:
            logger.error(f"Erreur aperçu: {e}")
    
    def _on_process_clicked(self):
        """Lance le traitement de l'image"""
        if not self.current_image_path or not self.processor:
            return

        # Désactiver l'UI
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("⏳ Détection et découpage en cours...")

        # Récupérer le mode de détection sélectionné
        detection_mode = self.detection_mode_combo.currentData()  # "stamps" ou "booklets"

        # Lancer le traitement dans un thread (PNG uniquement, pas de reconnaissance)
        self.processing_thread = ProcessingThread(
            self.processor,
            self.current_image_path,
            self.margin_spin.value(),
            save_jpg=False,  # Toujours PNG uniquement
            detection_mode=detection_mode,
            recognize_stamps=False,  # Pas de reconnaissance
            use_online=False
        )

        self.processing_thread.progress_updated.connect(self._on_progress_updated)
        self.processing_thread.processing_complete.connect(self._on_processing_complete)
        self.processing_thread.error_occurred.connect(self._on_error_occurred)

        self.processing_thread.start()
    
    def _on_progress_updated(self, progress, message):
        """Met à jour la barre de progression"""
        self.progress_bar.setValue(progress)
        self.log_widget.append_log(message, "INFO")
    
    def _on_processing_complete(self, result):
        """Gère la fin du traitement"""
        self.current_result = result

        # Réactiver l'UI
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Afficher les résultats
        if result.dpi_was_default:
            self.log_widget.append_log(f"⚠️ DPI par défaut utilisé ({result.dpi})", "WARNING")

        if result.num_stamps == 0:
            self.log_widget.append_log("⚠️ Aucun timbre détecté", "WARNING")
            self.status_label.setText("⚠️ Aucun timbre détecté")
            QMessageBox.warning(
                self,
                "Aucun timbre détecté",
                "Aucun timbre n'a été détecté sur cette image.\n\n"
                "Conseils:\n"
                "- Vérifiez que les timbres sont bien visibles\n"
                "- Augmentez la résolution du scan (300 DPI minimum)\n"
                "- Assurez-vous que les timbres ont un bon contraste"
            )
            return

        self.log_widget.append_log(f"✓ {result.num_stamps} timbre(s) détecté(s)", "SUCCESS")
        self.log_widget.append_log(f"✓ {len(result.output_files)} fichier(s) PNG générés", "SUCCESS")

        # Afficher le dossier de sortie
        output_folder = result.output_files[0].parent if result.output_files else None
        if output_folder:
            self.log_widget.append_log(f"📁 Dossier: {output_folder}", "INFO")

        self.status_label.setText(f"✓ {result.num_stamps} timbres découpés → {len(result.output_files)} fichiers PNG")

        # Afficher l'aperçu avec détections
        if result.detections:
            self._show_detections_preview(result)
    
    def _show_detections_preview(self, result):
        """Affiche l'aperçu avec les détections"""
        try:
            image = cv2.imread(str(result.input_path))
            annotated = self.processor.draw_detections(image, result.detections)
            
            # Redimensionner
            h, w = annotated.shape[:2]
            max_width = 700
            if w > max_width:
                scale = max_width / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                annotated = cv2.resize(annotated, (new_w, new_h))
            
            # Conversion pour Qt
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_image)
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setScaledContents(False)
            
        except Exception as e:
            logger.error(f"Erreur aperçu détections: {e}")

    def _on_error_occurred(self, error_msg):
        """Gère les erreurs de traitement"""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Erreur")
        
        self.log_widget.append_log(f"Erreur: {error_msg}", "ERROR")
        
        QMessageBox.critical(self, "Erreur de traitement", f"Une erreur s'est produite:\n{error_msg}")