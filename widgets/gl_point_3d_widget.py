import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT_SMOOTH,
    GL_POINTS,
    GL_PROJECTION,
    GL_SRC_ALPHA,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glLoadMatrixf,
    glMatrixMode,
    glPointSize,
    glVertex3f,
    glViewport,
)
from PySide6.QtCore import Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class GLPoint3DWidget(QOpenGLWidget):
    points: np.ndarray
    color: tuple[float, float, float]
    opacity: float
    view_mat: np.ndarray
    proj_mat: np.ndarray

    def __init__(
        self,
        points: np.ndarray,
        opengl_view_mat: np.ndarray,
        opengl_proj_mat: np.ndarray,
        color=(1.0, 0.0, 0.0),
        opacity=1.0,
        point_size=5.0,
        mouse_tracking=False,
        parent=None,
    ):
        super().__init__(parent)

        self.points = points.astype(np.float32) if points.size > 0 else np.array([], dtype=np.float32)
        self.color = color
        self.opacity = opacity
        self.view_mat = opengl_view_mat
        self.proj_mat = opengl_proj_mat
        self.point_size = point_size

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_AlwaysStackOnTop)
        self.setAutoFillBackground(False)
        self.setMouseTracking(mouse_tracking)

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)

    def updateCameraMatrices(self, opengl_view_mat: np.ndarray, opengl_proj_mat: np.ndarray):
        self.view_mat = opengl_view_mat
        self.proj_mat = opengl_proj_mat

    def updatePoints(self, points: np.ndarray):
        self.points = points.astype(np.float32) if points.size > 0 else np.array([], dtype=np.float32)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        if self.points.size == 0:
            return

        # Set up matrices
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(self.proj_mat.cpu().numpy().T.flatten().astype(np.float32))
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(self.view_mat.cpu().numpy().T.flatten().astype(np.float32))

        # Render points using primitive commands
        glPointSize(self.point_size)
        glColor4f(*self.color, self.opacity)
        glBegin(GL_POINTS)
        for point in self.points:
            glVertex3f(point[0], point[1], point[2])
        glEnd()
