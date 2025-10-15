from enum import Enum, auto

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QCursor, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication


def get_svg_cursor(svg_path: str, logical_size: int = 64, hotspot_ratio=(0.1, 0.1)) -> QCursor:
    dpr = QApplication.primaryScreen().devicePixelRatio()

    pixel_size = int(round(logical_size * dpr))

    pm = QPixmap(QSize(pixel_size, pixel_size))
    pm.fill(Qt.transparent)

    renderer = QSvgRenderer(svg_path)
    painter = QPainter(pm)
    renderer.render(painter, QRectF(0, 0, pixel_size, pixel_size))
    painter.end()

    pm.setDevicePixelRatio(dpr)

    hot_x = int(logical_size * hotspot_ratio[0])
    hot_y = int(logical_size * hotspot_ratio[1])

    return QCursor(pm, hot_x, hot_y)


class CursorMode(Enum):
    ARROW = auto()
    PEN = auto()
    PINCH_OPEN = auto()
    PINCH_CLOSE = auto()


class CursorManager:

    def __init__(self):
        self.cursors = {}
        self.current_cursor_name = None

    def register_cursor(self, name: str, svg_path: str, logical_size: int = 32, hotspot_ratio=(0.2, 0.1)):
        try:
            cursor = get_svg_cursor(svg_path, logical_size, hotspot_ratio)
            self.cursors[name] = cursor
        except Exception as e:
            print(f"Warning: Failed to load cursor '{name}': {e}")

    def get_cursor(self, name: str) -> QCursor:
        return self.cursors.get(name, QCursor(Qt.ArrowCursor))

    def set_cursor(self, widget, name: str):
        cursor = self.get_cursor(name)
        widget.setCursor(cursor)
        self.current_cursor_name = name

    def register_default_cursors(self):
        # You can set arbitrary svg file.
        cursor_configs = [
            (CursorMode.ARROW, "resources/arrow_cursor.svg", 50, (0.25, 0.05)),
            (CursorMode.PEN, "resources/pen.svg", 50, (0.05, 0.95)),
            (CursorMode.PINCH_OPEN, "resources/arrow_cursor.svg", 50, (0.25, 0.05)),
            (CursorMode.PINCH_CLOSE, "resources/arrow_cursor.svg", 50, (0.25, 0.05)),
        ]

        for name, path, size, hotspot in cursor_configs:
            self.register_cursor(name, path, size, hotspot)
