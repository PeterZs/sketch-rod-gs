import numpy as np
from OpenGL.arrays import vbo
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FLOAT,
    GL_LINE_SMOOTH,
    GL_LINE_STRIP,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT_SMOOTH,
    GL_POINTS,
    GL_PROJECTION,
    GL_SRC_ALPHA,
    GL_VERTEX_ARRAY,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor4f,
    glDisable,
    glDisableClientState,
    glDrawArrays,
    glEnable,
    glEnableClientState,
    glLineWidth,
    glLoadMatrixf,
    glMatrixMode,
    glPointSize,
    glVertexPointer,
    glViewport,
)
from PySide6.QtCore import Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class GLPolyline3DWidget(QOpenGLWidget):
    polyline: np.ndarray
    color: tuple[float, float, float]
    opacity: float
    view_mat: np.ndarray
    proj_mat: np.ndarray
    hover_point_id: int

    def __init__(
        self,
        polyline: np.ndarray,
        opengl_view_mat: np.ndarray,
        opengl_proj_mat: np.ndarray,
        color=(1.0, 0.0, 0.0),
        opacity=0.55,
        line_width=2.0,
        point_size=5.0,
        hover_point_size=8.0,
        mouse_tracking=False,
        parent=None,
    ):
        super().__init__(parent)

        self.polyline = polyline.astype(np.float32) if polyline.size > 0 else np.array([], dtype=np.float32)
        self.color = color
        self.opacity = opacity
        self.line_width = line_width
        self.point_size = point_size
        self.hover_point_size = hover_point_size
        self.view_mat = opengl_view_mat
        self.proj_mat = opengl_proj_mat
        self.hover_point_id = None

        # VBO for efficient rendering
        self.vertex_vbo = None
        self.polyline_dirty = True

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_AlwaysStackOnTop)
        self.setAutoFillBackground(False)
        self.setMouseTracking(mouse_tracking)

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)  # Background has alpha 0.0
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)  # Don't need depth test
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)

        # Initialize VBOs
        self._update_vbos()

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)

    def updateCameraMatrices(self, opengl_view_mat: np.ndarray, opengl_proj_mat: np.ndarray):
        self.view_mat = opengl_view_mat
        self.proj_mat = opengl_proj_mat

    def updatePolyline(self, polyline: np.array):
        self.polyline = polyline.astype(np.float32) if polyline.size > 0 else np.array([], dtype=np.float32)
        self.polyline_dirty = True

    def update_hover_point_id(self, point_id: int):
        self.hover_point_id = point_id

    def _update_vbos(self):
        if self.polyline.size == 0:
            return

        # Clean up existing VBO
        if self.vertex_vbo is not None:
            self.vertex_vbo.delete()

        # Create VBO for points and lines
        if len(self.polyline.shape) == 2 and self.polyline.shape[0] > 0:
            self.vertex_vbo = vbo.VBO(self.polyline)

        self.polyline_dirty = False

    def paintGL(self):
        if self.polyline.size == 0:
            glClear(GL_COLOR_BUFFER_BIT)
            return

        if self.polyline_dirty:
            self._update_vbos()

        glClear(GL_COLOR_BUFFER_BIT)

        # Set up matrices efficiently
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(self.proj_mat.cpu().numpy().T.flatten().astype(np.float32))
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(self.view_mat.cpu().numpy().T.flatten().astype(np.float32))

        if self.vertex_vbo is None or len(self.polyline.shape) != 2 or self.polyline.shape[0] == 0:
            return

        # Enable vertex arrays
        glEnableClientState(GL_VERTEX_ARRAY)

        try:
            self.vertex_vbo.bind()
            glVertexPointer(3, GL_FLOAT, 0, self.vertex_vbo)

            # Render line strip efficiently
            if self.polyline.shape[0] > 1:
                glLineWidth(self.line_width)
                glColor4f(*self.color, self.opacity)
                glDrawArrays(GL_LINE_STRIP, 0, self.polyline.shape[0])

            # Render regular points efficiently
            glPointSize(self.point_size)
            glColor4f(*self.color, self.opacity)

            if self.hover_point_id is not None and 0 <= self.hover_point_id < self.polyline.shape[0]:
                # Draw all points except hover point
                if self.hover_point_id > 0:
                    glDrawArrays(GL_POINTS, 0, self.hover_point_id)
                if self.hover_point_id < self.polyline.shape[0] - 1:
                    glDrawArrays(GL_POINTS, self.hover_point_id + 1, self.polyline.shape[0] - self.hover_point_id - 1)

                # Draw hover point with different color and size
                glPointSize(self.hover_point_size)
                glColor4f(0.0, 0.0, 1.0, self.opacity)
                glDrawArrays(GL_POINTS, self.hover_point_id, 1)
            else:
                # Draw all points
                glDrawArrays(GL_POINTS, 0, self.polyline.shape[0])

            self.vertex_vbo.unbind()

        except Exception as e:
            print(f"Error in polyline rendering: {e}")

        finally:
            glDisableClientState(GL_VERTEX_ARRAY)

    def __del__(self):
        # Clean up VBO when widget is destroyed
        if hasattr(self, "vertex_vbo") and self.vertex_vbo is not None:
            self.vertex_vbo.delete()
