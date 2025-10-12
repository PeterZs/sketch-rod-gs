from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_LINEAR,
    GL_QUADS,
    GL_RGB,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glClear,
    glColor3f,
    glEnable,
    glEnd,
    glGenTextures,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
)
from PySide6.QtGui import QImage
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class GLImageWidget(QOpenGLWidget):
    def __init__(self, image, mouse_tracking=False, parent=None):
        super().__init__(parent)
        self.image = image
        self.texture_id = None
        self.mouse_tracking = mouse_tracking

    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)
        self.setMouseTracking(self.mouse_tracking)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor3f(1.0, 1.0, 1.0)

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        img = QImage(self.image.data, self.image.shape[1], self.image.shape[0], QImage.Format_RGB888)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width(), img.height(), 0, GL_RGB, GL_UNSIGNED_BYTE, img.bits())

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(-1, -1)
        glTexCoord2f(1, 0)
        glVertex2f(1, -1)
        glTexCoord2f(1, 1)
        glVertex2f(1, 1)
        glTexCoord2f(0, 1)
        glVertex2f(-1, 1)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)

    def update_image(self, image):
        self.image = image
        self.update()
