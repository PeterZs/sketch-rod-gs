import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class InfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.SubWindow | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 180); color: white; padding: 10px;")

        self.gs_fps_label = QLabel("GS FPS       : --")
        self.anim_fps_label = QLabel("Animation FPS: --")
        self.pos_label = QLabel("Pos: --")
        self.up_label = QLabel("Up: --")
        self.to_label = QLabel("To: --")

        layout = QVBoxLayout()
        layout.addWidget(self.gs_fps_label)
        layout.addWidget(self.anim_fps_label)
        layout.addWidget(self.pos_label)
        layout.addWidget(self.up_label)
        layout.addWidget(self.to_label)
        self.setLayout(layout)
        self.setFixedWidth(250)

    def update_gs_fps(self, fps: float):
        self.gs_fps_label.setText(f"GS FPS       : {fps:.1f}")

    def update_anim_fps(self, fps: float):
        self.anim_fps_label.setText(f"Animation FPS: {fps:.1f}")

    def set_camera_info(self, pos: np.ndarray, up: np.ndarray, to: np.ndarray):
        self.pos_label.setText("Pos: " + ", ".join(f"{v:.3f}" for v in pos))
        self.up_label.setText("Up: " + ", ".join(f"{v:.3f}" for v in up))
        self.to_label.setText("To: " + ", ".join(f"{v:.3f}" for v in to))

    def get_camera_info(self):
        def parse(text):
            try:
                return [float(v) for v in text.split(",")]
            except (ValueError, OverflowError):
                return None

        return (parse(self.pos_label.text()), parse(self.lookat_label.text()), parse(self.up_label.text()))
