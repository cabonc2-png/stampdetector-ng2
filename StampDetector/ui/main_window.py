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
    QTextEdit, QTabWidget
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
    
    def __init__(self, processor, image_path, margin_mm, save_jpg, recognize_stamps=False, use_online=False):
        super().__init__()
        self.processor = processor
        self.image_path = image_path
        self.margin_mm = margin_mm
        self.save_jpg = save_jpg
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
    
    def __init__(self, dpi, color_mode="color"):
        super().__init__()
        self.dpi = dpi
        self.color_mode = color_mode
    
    def run(self):
        """Exécute le scan dans un thread séparé"""
        try:
            self.scan_status.emit("Connexion au scanner...")
            scanner = ScannerManager()
            
            if not scanner.connect_scanner():
                self.scan_error.emit("Impossible de se connecter au scanner")
                return
            
            self.scan_status.emit(f"Scan en cours ({self.dpi} DPI)...")
            
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
        self.setWindowTitle("StampDetector Pro - Détection, Scan & Reconnaissance Visuelle")
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
        
        version_label = QLabel("v2.2 - Smart Vision")
        version_label.setStyleSheet("font-size: 12px; color: #666;")
        title_layout.addWidget(version_label)
        title_layout.addStretch()
        
        main_layout.addLayout(title_layout)
        
        # Section Scanner (si disponible)
        if SCANNER_AVAILABLE and is_scanner_available():
            scan_group = QGroupBox("📷 Scanner")
            scan_layout = QHBoxLayout()
            
            self.scan_btn = QPushButton("🖨️ Lancer un scan")
            self.scan_btn.setMinimumWidth(150)
            self.scan_btn.clicked.connect(self._on_scan_clicked)
            scan_layout.addWidget(self.scan_btn)
            
            scan_layout.addWidget(QLabel("Résolution:"))
            self.scan_dpi = QSpinBox()
            self.scan_dpi.setRange(150, 1200)
            self.scan_dpi.setValue(300)
            self.scan_dpi.setSuffix(" DPI")
            self.scan_dpi.setMinimumWidth(100)
            scan_layout.addWidget(self.scan_dpi)
            
            scan_layout.addStretch()
            
            self.scan_status_label = QLabel("Scanner prêt")
            self.scan_status_label.setStyleSheet("color: #4ec9b0; padding: 5px;")
            scan_layout.addWidget(self.scan_status_label)
            
            scan_group.setLayout(scan_layout)
            main_layout.addWidget(scan_group)
        
        # Zone de drop
        self.drop_zone = DropZoneWidget()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        main_layout.addWidget(self.drop_zone)
        
        # Paramètres de traitement
        params_group = QGroupBox("⚙️ Paramètres de traitement")
        params_layout = QHBoxLayout()
        
        params_layout.addWidget(QLabel("Marge:"))
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 10.0)
        self.margin_spin.setValue(1.0)
        self.margin_spin.setSingleStep(0.1)
        self.margin_spin.setSuffix(" mm")
        self.margin_spin.setMinimumWidth(100)
        params_layout.addWidget(self.margin_spin)
        
        self.jpg_checkbox = QCheckBox("Exporter en JPG")
        self.jpg_checkbox.setChecked(False)
        params_layout.addWidget(self.jpg_checkbox)
        
        # Options de reconnaissance
        if RECOGNITION_AVAILABLE or ONLINE_RECOGNITION_AVAILABLE:
            self.recognize_checkbox = QCheckBox("🔍 Identifier les timbres")
            self.recognize_checkbox.setChecked(True)
            self.recognize_checkbox.setStyleSheet("color: #ffd700;")
            params_layout.addWidget(self.recognize_checkbox)
            
            if ONLINE_RECOGNITION_AVAILABLE:
                self.online_checkbox = QCheckBox("🌐 Reconnaissance visuelle")
                self.online_checkbox.setChecked(True)
                self.online_checkbox.setStyleSheet("color: #00ff7f;")
                self.online_checkbox.setToolTip("Analyse intelligente des couleurs, motifs et caractéristiques visuelles")
                params_layout.addWidget(self.online_checkbox)
            else:
                self.online_checkbox = None
        else:
            self.recognize_checkbox = None
            self.online_checkbox = None
        
        params_layout.addStretch()
        
        # Bouton traiter
        self.process_btn = QPushButton("🚀 Détecter et Découper")
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumWidth(180)
        self.process_btn.clicked.connect(self._on_process_clicked)
        params_layout.addWidget(self.process_btn)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        main_layout.addWidget(self.progress_bar)
        
        # Onglets (Preview + Résultats)
        self.tabs = QTabWidget()
        
        # Onglet Preview
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setMinimumHeight(300)
        
        self.preview_label = QLabel("Aucune détection")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("color: #666; font-size: 14px;")
        
        self.preview_scroll.setWidget(self.preview_label)
        preview_layout.addWidget(self.preview_scroll)
        
        self.tabs.addTab(preview_tab, "👁️ Aperçu")
        
        # Onglet Résultats (reconnaissance)
        if RECOGNITION_AVAILABLE or ONLINE_RECOGNITION_AVAILABLE:
            results_tab = QWidget()
            results_layout = QVBoxLayout(results_tab)
            
            self.results_text = QTextEdit()
            self.results_text.setReadOnly(True)
            self.results_text.setStyleSheet("""
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    border: 1px solid #3c3c3c;
                    border-radius: 5px;
                    padding: 10px;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 12px;
                }
            """)
            results_layout.addWidget(self.results_text)
            
            self.tabs.addTab(results_tab, "🏷️ Identification")
        
        main_layout.addWidget(self.tabs)
        
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
            self.log_widget.append_log(f"Backend de détection: {backend}", "SUCCESS")
            self.backend_label.setText(f"Backend: {backend}")
            
            # Afficher les services de reconnaissance disponibles
            recognition_services = []
            if RECOGNITION_AVAILABLE:
                recognition_services.append("OCR Local")
            if ONLINE_RECOGNITION_AVAILABLE:
                recognition_services.append("Reconnaissance visuelle IA")
            
            if recognition_services:
                services_str = ", ".join(recognition_services)
                self.log_widget.append_log(f"Services reconnaissance: {services_str}", "INFO")
            
            self.status_label.setText(f"✓ Prêt")
            
        except Exception as e:
            self.log_widget.append_log(f"Erreur initialisation: {e}", "ERROR")
            self.status_label.setText("❌ Erreur d'initialisation")
            QMessageBox.critical(self, "Erreur", f"Impossible d'initialiser:\n{e}")
    
    def _on_scan_clicked(self):
        """Lance un scan"""
        if not SCANNER_AVAILABLE:
            QMessageBox.warning(self, "Scanner", "Module scanner non disponible")
            return
        
        try:
            self.scan_btn.setEnabled(False)
            self.log_widget.append_log("🖨️ Démarrage du scanner...", "INFO")
            
            # Lancer le scan dans un thread
            self.scan_thread = ScanThread(
                dpi=self.scan_dpi.value(),
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
        self.status_label.setText("⏳ Traitement en cours...")
        
        # Options de reconnaissance
        recognize = self.recognize_checkbox.isChecked() if self.recognize_checkbox else False
        use_online = self.online_checkbox.isChecked() if self.online_checkbox else False
        
        # Lancer le traitement dans un thread
        self.processing_thread = ProcessingThread(
            self.processor,
            self.current_image_path,
            self.margin_spin.value(),
            self.jpg_checkbox.isChecked(),
            recognize_stamps=recognize,
            use_online=use_online
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
        
        self.log_widget.append_log(f"✓ {result.num_stamps} timbre(s) détecté(s)", "SUCCESS")
        self.log_widget.append_log(f"✓ {len(result.output_files)} fichier(s) générés", "SUCCESS")
        
        self.status_label.setText(f"✓ {result.num_stamps} timbres → {len(result.output_files)} fichiers")
        
        # Afficher l'aperçu avec détections
        if result.detections:
            self._show_detections_preview(result)
        
        # Afficher les résultats de reconnaissance
        if hasattr(result, 'recognized_stamps') and result.recognized_stamps:
            self._show_recognition_results(result.recognized_stamps)
            self.tabs.setCurrentIndex(1)  # Passer à l'onglet Identification
    
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
    
    def _show_recognition_results(self, recognized_stamps):
        """Affiche les résultats de reconnaissance visuelle"""
        if not (RECOGNITION_AVAILABLE or ONLINE_RECOGNITION_AVAILABLE):
            return
        
        html = """
        <div style='font-family: Segoe UI, Arial, sans-serif;'>
            <h2 style='color: #0078d4; border-bottom: 2px solid #0078d4; padding-bottom: 10px;'>
                🏷️ Résultats de reconnaissance visuelle
            </h2>
        """
        
        for idx, stamp_data in enumerate(recognized_stamps, start=1):
            info = stamp_data['info']
            file_path = stamp_data['file']
# Couleur de la carte selon statut
            if info.get('identified', False):
                border_color = "#00ff7f"  # Vert néon si identifié
                bg_color = "#1a3a2e"
            else:
                border_color = "#888"
                bg_color = "#2b2b2b"
            
            html += f"""
            <div style='margin: 15px 0; padding: 15px; background-color: {bg_color}; 
                        border-left: 4px solid {border_color}; border-radius: 5px;'>
                <h3 style='color: #00ff7f; margin: 0 0 10px 0;'>📮 Timbre #{idx:03d}</h3>
                <p style='margin: 5px 0; color: #999; font-size: 11px;'>
                    <b>Fichier:</b> {file_path.name}
                </p>
            """
            
            # Sources de reconnaissance
            if info.get('recognition_source'):
                sources = " • ".join(info['recognition_source'])
                html += f"<p style='margin: 5px 0; color: #00bfff; font-size: 11px;'>{sources}</p>"
            
            # Résultats principaux
            if info.get('identified', False):
                confidence_color = "#4ec9b0" if info.get('confidence', 0) > 0.5 else "#dcdcaa"
                html += f"<p style='margin: 8px 0;'><b>🏷️ Nom:</b> <span style='color: {confidence_color};'>{info.get('name', 'Inconnu')}</span></p>"
                html += f"<p style='margin: 5px 0;'><b>📊 Confiance:</b> <span style='color: {confidence_color};'>{info.get('confidence', 0):.0%}</span></p>"
                
                if info.get('country'):
                    html += f"<p style='margin: 5px 0;'><b>🌍 Pays:</b> <span style='color: #ffd700;'>{info['country']}</span></p>"
                
                if info.get('value'):
                    html += f"<p style='margin: 5px 0;'><b>💰 Valeur:</b> <span style='color: #ffd700;'>{info['value']}</span></p>"
                
                if info.get('year'):
                    html += f"<p style='margin: 5px 0;'><b>📅 Année:</b> {info['year']}</p>"
            else:
                html += "<p style='margin: 8px 0; color: #f48771;'><b>❌ Statut:</b> Non identifié</p>"
            
            # Caractéristiques visuelles
            if info.get('visual_features'):
                vf = info['visual_features']
                html += "<hr style='border: 1px dashed #3c3c3c; margin: 10px 0;'>"
                html += "<p style='margin: 5px 0; color: #888; font-size: 11px;'><b>🎨 Analyse visuelle:</b></p>"
                
                if vf.get('dominant_colors'):
                    colors_str = ", ".join(vf['dominant_colors'][:3])
                    html += f"<p style='margin: 5px 0 5px 20px; font-size: 10px;'>Couleurs: {colors_str}</p>"
                
                if vf.get('has_portrait'):
                    html += "<p style='margin: 5px 0 5px 20px; font-size: 10px;'>✓ Portrait détecté</p>"
                
                if vf.get('estimated_country'):
                    html += f"<p style='margin: 5px 0 5px 20px; font-size: 10px;'>Pays estimé: {vf['estimated_country']}</p>"
            
            # Suggestions
            if info.get('suggestions'):
                html += "<hr style='border: 1px dashed #3c3c3c; margin: 10px 0;'>"
                html += "<p style='margin: 5px 0; color: #00bfff; font-size: 11px;'><b>💡 Suggestions:</b></p>"
                for suggestion in info['suggestions'][:3]:  # Limiter à 3 suggestions
                    html += f"<p style='margin: 3px 0 3px 20px; color: #888; font-size: 10px;'>• {suggestion}</p>"
            
            html += "</div>"
        
        # Résumé final
        identified_count = sum(1 for s in recognized_stamps if s['info'].get('identified', False))
        total_count = len(recognized_stamps)
        success_rate = (identified_count / total_count * 100) if total_count > 0 else 0
        
        html += "<hr style='border: 1px solid #3c3c3c; margin-top: 20px;'>"
        html += f"""
        <div style='text-align: center; margin-top: 15px; padding: 15px; background-color: #2b2b2b; border-radius: 5px;'>
            <p style='color: #4ec9b0; margin: 0; font-size: 14px;'>
                <b>📊 Résumé:</b> {identified_count}/{total_count} timbre(s) identifié(s) ({success_rate:.0f}%)
            </p>
        """
        
        if identified_count < total_count:
            html += """
            <p style='color: #888; margin: 10px 0 0 0; font-size: 11px;'>
                💡 Pour améliorer la reconnaissance, essayez avec des images de meilleure qualité<br>
                ou configurez Google Vision API pour des résultats précis
            </p>
            """
        
        html += "</div>"
        html += "</div>"
        
        self.results_text.setHtml(html)
    
    def _on_error_occurred(self, error_msg):
        """Gère les erreurs de traitement"""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Erreur")
        
        self.log_widget.append_log(f"Erreur: {error_msg}", "ERROR")
        
        QMessageBox.critical(self, "Erreur de traitement", f"Une erreur s'est produite:\n{error_msg}")