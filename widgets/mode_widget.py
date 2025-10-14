from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ModeWidget(QWidget):
    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.SubWindow | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 180); color: white; padding: 10px;")

        self.mode_label = QLabel(f"Mode: {mode}")

        # Font setting
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.mode_label.setFont(font)

        layout = QVBoxLayout()
        layout.addWidget(self.mode_label)
        self.setLayout(layout)

    def update_mode(self, mode: str):
        self.mode_label.setText(f"Mode: {mode}")
