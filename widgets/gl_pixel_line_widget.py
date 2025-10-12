from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_LINE_STRIP,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor,
    glEnable,
    glEnd,
    glLineWidth,
    glVertex2f,
)
from PySide6.QtCore import Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from utils.rope_utils import pix2ndc


class GLPixelLineWidget(QOpenGLWidget):
    def __init__(
        self,
        line: list[tuple[int, int]],
        color_list: list[tuple[float, float, float]],
        point_size: float = 2.0,
        mouse_tracking=False,
        parent=None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_AlwaysStackOnTop)
        self.setAutoFillBackground(False)

        self.mouse_tracking = mouse_tracking

        self.line = line
        self.color_list = color_list

        self.point_size = point_size

    def initializeGL(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.setMouseTracking(self.mouse_tracking)

    def paintGL(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLineWidth(self.point_size)

        def draw_path(path, color):
            if len(path) < 2:
                return
            glBegin(GL_LINE_STRIP)
            for pt, c in zip(path, color):
                glColor(c[0], c[1], c[2])
                pt_ndc = pix2ndc(pt[0], pt[1], self.width(), self.height())
                glVertex2f(pt_ndc[0], pt_ndc[1])
            glEnd()

        draw_path(self.line, self.color_list)

    def update_layer(self, line, color_list):
        self.line = line
        self.color_list = color_list
        self.update()
