import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINE_SMOOTH,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_SRC_ALPHA,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glLineWidth,
    glLoadIdentity,
    glLoadMatrixf,
    glMatrixMode,
    glVertex3f,
    glViewport,
)
from PySide6.QtCore import Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class GLAxisWidget(QOpenGLWidget):
    def __init__(self, opengl_view_mat: np.ndarray, opengl_proj_mat: np.ndarray, mouse_tracking=False, parent=None):
        super().__init__(parent)

        self.view_mat = opengl_view_mat
        self.proj_mat = opengl_proj_mat

        self.mouse_tracking = mouse_tracking

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_AlwaysStackOnTop)
        self.setAutoFillBackground(False)

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)  # Background with alpha 0.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)  # Depth test isn't needed.
        glEnable(GL_LINE_SMOOTH)
        glLineWidth(2.0)

        self.setMouseTracking(self.mouse_tracking)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glLoadMatrixf(self.proj_mat.cpu().numpy().T.flatten().astype(np.float32))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLoadMatrixf(self.view_mat.cpu().numpy().T.flatten().astype(np.float32))

        glBegin(GL_LINES)

        # x-axis(red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)

        # y-axis(green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)

        # z-axis(blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)

        glEnd()

    def updateCameraMatrices(self, opengl_view_mat: np.ndarray, opengl_proj_mat: np.ndarray):
        self.view_mat = opengl_view_mat
        self.proj_mat = opengl_proj_mat
