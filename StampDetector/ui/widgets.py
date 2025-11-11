"""
Composants UI personnalisés
"""
from PySide6.QtWidgets import QLabel, QTextEdit
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent

class DropZoneWidget(QLabel):
    files_dropped = Signal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 200)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #555;
                border-radius: 10px;
                background-color: #2b2b2b;
                color: #aaa;
                font-size: 14px;
                padding: 20px;
            }
        """)
        self.setText("🖼️ Glissez-déposez vos images ici\n\nFormats: PNG, JPG, JPEG, TIFF")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                files.append(file_path)
        
        if files:
            self.files_dropped.emit(files)

class LogWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(150)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
    
    def append_log(self, message: str, level: str = "INFO"):
        colors = {"INFO": "#4ec9b0", "WARNING": "#dcdcaa", "ERROR": "#f48771", "SUCCESS": "#4ec9b0"}
        color = colors.get(level, "#d4d4d4")
        self.append(f'<span style="color: {color};">[{level}]</span> {message}')