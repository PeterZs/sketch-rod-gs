from __future__ import annotations  # For self-reference in class

import sys
from enum import Enum, auto
from typing import Final

# isort: off
import torch  # noqa: F401 (Needed from cpp "moving_primitives" module)
import matplotlib.pyplot as plt
import moving_primitives
import numpy as np
import rotating_primitives
from create_polyline import cerate_line
from gs_renderer import GaussianModel, init, render_no_grad
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QKeyEvent,
    QMouseEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QApplication, QMainWindow
from qasync import QEventLoop
from utils.camera import Camera
from utils.fps_counter import Fps_Counter
from utils.move_and_hover import get_closest_polyline_idx, get_goal_pos
from utils.polyline import PolyLine
from widgets.cursor import CursorManager, CursorMode
from widgets.gl_axis_widget import GLAxisWidget
from widgets.gl_image_widget import GLImageWidget
from widgets.gl_pixel_line_widget import GLPixelLineWidget
from widgets.gl_polyline_3d_widget import GLPolyline3DWidget
from widgets.info_widget import InfoWidget
from widgets.mode_widget import ModeWidget

# isort: on

# Window size
W, H = 960, 540

# 3D polyline widget params
LINE_OPACITY = 1.0
LINE_WIDTH = 4.0
POINT_SIZE = 8.0
HOVER_POINT_SIZE = 10.0

# 3D polyline radius
TUBE_RADIUS = 0.05


class Mode(Enum):
    VIEWER = auto()
    GUIDE_EDITOR = auto()
    PLAY = auto()


class MainWindow(QMainWindow):
    SAMPLING_RATE: Final = 5

    camera: Camera
    pc: GaussianModel
    gaussian_points: np.ndarray
    fps_counter: Fps_Counter
    moving_point_id: int | None

    timer: QTimer
    anim_timer: QTimer

    # Widgets
    viewer: GLImageWidget
    polyline_layer: GLPolyline3DWidget
    pixel_line_layer: GLPixelLineWidget
    axis_layer: GLAxisWidget
    info_layer: InfoWidget
    mode_layer: ModeWidget

    def setup_widgets(self, width: int, height: int):
        self.setWindowTitle("SketchRodGS")
        self.setFixedSize(width, height)

        mouse_tracking = True
        self.setMouseTracking(mouse_tracking)

        # Initialize CursorManager
        self.cursor_manager = CursorManager()
        self.cursor_manager.register_default_cursors()
        # Setting initial cursor
        self.cursor_manager.set_cursor(self, CursorMode.ARROW)

        self.viewer = GLImageWidget(self.image, mouse_tracking=mouse_tracking, parent=self)
        self.viewer.setGeometry(0, 0, width, height)

        self.polyline_layer = GLPolyline3DWidget(
            self.polyline.get_polyline,
            self.camera.world2camera_mat_opengl,
            self.camera.camera2ndc_mat_opengl,
            opacity=LINE_OPACITY,
            line_width=LINE_WIDTH,
            point_size=POINT_SIZE,
            hover_point_size=HOVER_POINT_SIZE,
            mouse_tracking=mouse_tracking,
            parent=self,
        )
        self.polyline_layer.setGeometry(0, 0, width, height)
        self.polyline_layer.raise_()  # Pushing out to the front

        self.pixel_line_layer = GLPixelLineWidget([], [], point_size=3.0, mouse_tracking=mouse_tracking, parent=self)
        self.pixel_line_layer.setGeometry(0, 0, width, height)
        self.pixel_line_layer.raise_()  # Pushing out to the front

        self.axis_layer = GLAxisWidget(
            self.camera.world2camera_mat_opengl,
            self.camera.camera2ndc_mat_opengl,
            mouse_tracking=mouse_tracking,
            parent=self,
        )
        self.axis_layer.setGeometry(0, 0, width, height)
        self.axis_layer.raise_()  # Pushing out to the front
        self.axis_layer.hide()

        self.info_layer = InfoWidget(self)
        self.info_layer.move(width - 270, 10)
        self.info_layer.setFixedSize(250, 200)
        self.info_layer.raise_()
        self.info_layer.hide()

        self.mode_layer = ModeWidget(self.mode.name, self)
        self.mode_layer.move(0, 0)
        self.mode_layer.setFixedSize(280, 70)
        self.mode_layer.raise_()

    def __init__(
        self,
        ply_path: str,
        sh_deg: int,
        width: int,
        height: int,
        pos: list[float],
        up: list[float],
        to: list[float],
        random_color: bool = False,
    ):
        super().__init__()

        # Loading Gaussian model
        self.camera, self.pc = init(width, height, sh_deg, ply_path, pos, up, to, random_color)

        self.gaussian_points = self.pc._xyz.cpu().numpy().astype(np.float32)
        print("num of visible points: ", len(self.gaussian_points))

        # Rendering initial GS scene
        self.render()

        self.gs_fps_counter = Fps_Counter()
        self.anim_fps_counter = Fps_Counter()

        self.current_path = []
        self.color_list = []
        self.last_mouse_pos = None

        # Polyline data
        self.polyline = PolyLine()

        self.moving_point_id = None

        # Setting up mode for widget
        self.modes = list(Mode)
        self.current_mode_idx = 0
        self.mode = self.modes[self.current_mode_idx]

        # Setting up Widgets
        self.setup_widgets(width=width, height=height)

        self.during_restore_anim = False
        self.during_move_anim = False
        self.gs_fps_counter.start()
        QTimer.singleShot(0, self.update_screen)

    def update_screen(self):
        self.render()
        self.viewer.update_image(self.image)
        self.polyline_layer.update()
        self.axis_layer.update()
        self.pixel_line_layer.update_layer(self.current_path, self.color_list)

        self.gs_fps_counter.end()
        self.info_layer.update_gs_fps(fps=self.gs_fps_counter.get_fps())

        if self.during_restore_anim is True:
            self.anim_fps_counter.start()
            QTimer.singleShot(0, self.step_restore_polyline_animation)
        elif self.during_move_anim is True:
            self.gs_fps_counter.start()
        else:
            self.gs_fps_counter.start()
            QTimer.singleShot(0, self.update_screen)

    def render(self) -> np.ndarray:
        image, most_contribute_id, out_depth_of_most_contribute, pixel_coords = render_no_grad(self.camera, self.pc)
        self.image_for_saving = image[::-1, :, :]
        image = (image * 255).astype(np.uint8).copy()
        self.image = image
        self.most_contribute_id = most_contribute_id
        self.out_depth_of_most_contribute = out_depth_of_most_contribute
        self.pixel_coords = pixel_coords

    ################
    # UI fuctions
    ################
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        mod = event.modifiers()

        if key == Qt.Key_Escape:
            sys.exit()
        elif key == Qt.Key_M:
            if not self.is_dragging():
                self.current_mode_idx = (self.current_mode_idx + 1) % len(self.modes)
                self.mode = self.modes[self.current_mode_idx]
                self.mode_layer.update_mode(self.mode.name)
                # Changing cursor
                if self.mode == Mode.VIEWER:
                    self.cursor_manager.set_cursor(self, CursorMode.ARROW)
                elif self.mode == Mode.GUIDE_EDITOR:
                    self.cursor_manager.set_cursor(self, CursorMode.PEN)
                elif self.mode == Mode.PLAY:
                    self.cursor_manager.set_cursor(self, CursorMode.PINCH_OPEN)
        elif key == Qt.Key_I:
            if self.info_layer.isVisible():
                self.info_layer.hide()
                self.axis_layer.hide()
            else:
                self.info_layer.show()
                self.axis_layer.show()
        elif key == Qt.Key_W:
            if mod & Qt.ShiftModifier:
                self.camera.rotate(0.02, np.array([1.0, 0.0, 0.0]))
            else:
                self.camera.move_parallel(0.0, -0.1, 0.0)
        elif key == Qt.Key_S:
            if mod & Qt.ShiftModifier:
                self.camera.rotate(-0.02, np.array([1.0, 0.0, 0.0]))
            else:
                self.camera.move_parallel(0.0, 0.1, 0.0)
        elif key == Qt.Key_A:
            if mod & Qt.ShiftModifier:
                self.camera.rotate(0.02, np.array([0.0, 1.0, 0.0]))
            else:
                self.camera.move_parallel(0.1, 0.0, 0.0)
        elif key == Qt.Key_D:
            if mod & Qt.ShiftModifier:
                self.camera.rotate(-0.02, np.array([0.0, 1.0, 0.0]))
            else:
                self.camera.move_parallel(-0.1, 0.0, 0.0)
        elif key == Qt.Key_Right:
            self.camera.rotate(0.03, np.array([0.0, 0.0, 0.2]))
        elif key == Qt.Key_Left:
            self.camera.rotate(-0.03, np.array([0.0, 0.0, 0.2]))
        elif key == Qt.Key_P:
            self.save_screen("screen_shot.png")
        elif key == Qt.Key_Backspace:
            if self.mode == Mode.GUIDE_EDITOR:
                self.polyline.reset_polyline()
                self.polyline_layer.updatePolyline(self.polyline.get_polyline)

        self.polyline_layer.updateCameraMatrices(
            self.camera.world2camera_mat_opengl, self.camera.camera2ndc_mat_opengl
        )
        self.axis_layer.updateCameraMatrices(self.camera.world2camera_mat_opengl, self.camera.camera2ndc_mat_opengl)

        self.info_layer.set_camera_info(self.camera.base_pos, self.camera.base_up, self.camera.base_to)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.camera.move_parallel(0.0, 0.0, -0.2, retarget=False)
        elif delta < 0:
            self.camera.move_parallel(0.0, 0.0, 0.2, retarget=False)

        self.polyline_layer.updateCameraMatrices(
            self.camera.world2camera_mat_opengl, self.camera.camera2ndc_mat_opengl
        )
        self.axis_layer.updateCameraMatrices(self.camera.world2camera_mat_opengl, self.camera.camera2ndc_mat_opengl)

        self.info_layer.set_camera_info(self.camera.base_pos, self.camera.base_up, self.camera.base_to)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            pos = event.position()
            pix_x, pix_y = int(pos.x()), int(pos.y())

            if self.mode == Mode.GUIDE_EDITOR:
                self.current_path = [(pix_x, pix_y)]
                self.color_list = [(1.0, 0.0, 0.0)]
            elif self.mode == Mode.PLAY:
                # Changing cursor
                self.cursor_manager.set_cursor(self, CursorMode.PINCH_CLOSE)

            self.last_mouse_pos = (pix_x, pix_y)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        pix_x, pix_y = int(pos.x()), int(pos.y())

        if event.buttons() & Qt.LeftButton:
            if self.mode == Mode.VIEWER:
                diff_x = (pix_x - self.last_mouse_pos[0]) * 0.01
                diff_y = (pix_y - self.last_mouse_pos[1]) * 0.01
                if abs(diff_x) < abs(diff_y):
                    diff_x = 0.0
                else:
                    diff_y = 0.0
                self.camera.move_orbit(-diff_x, -diff_y)
                self.polyline_layer.updateCameraMatrices(
                    self.camera.world2camera_mat_opengl, self.camera.camera2ndc_mat_opengl
                )
                self.axis_layer.updateCameraMatrices(
                    self.camera.world2camera_mat_opengl, self.camera.camera2ndc_mat_opengl
                )
                self.info_layer.set_camera_info(self.camera.base_pos, self.camera.base_up, self.camera.base_to)

            if self.mode == Mode.GUIDE_EDITOR:
                if abs(pix_x - self.last_mouse_pos[0]) + abs(pix_y - self.last_mouse_pos[1]) < self.SAMPLING_RATE:
                    return
                self.current_path.append((pix_x, pix_y))
                self.color_list.append((1.0, 0.0, 0.0))

            if self.mode == Mode.PLAY and self.polyline.is_polyline_exist():
                if self.moving_point_id is not None:
                    self.during_move_anim = True
                    self.step_move_polyline_animation(pix_x, pix_y, self.moving_point_id)
                    self.update_screen()

            self.last_mouse_pos = (pix_x, pix_y)

        if not event.buttons():
            if self.mode == Mode.PLAY and self.polyline.is_polyline_exist():
                self.moving_point_id = get_closest_polyline_idx(
                    pix_x, pix_y, W, H, self.camera, self.polyline.get_polyline
                )
                self.polyline_layer.update_hover_point_id(self.moving_point_id)
            else:
                # Resetting hover point
                self.polyline_layer.update_hover_point_id(None)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self.mode == Mode.GUIDE_EDITOR:
                self.cerate_line()
                self.current_path = []

                # Changing the mode back to viewer
                self.current_mode_idx = self.modes.index(Mode.PLAY)
                self.mode = self.modes[self.current_mode_idx]
                self.mode_layer.update_mode(self.mode.name)
                # Changing cursor
                self.cursor_manager.set_cursor(self, CursorMode.ARROW)

            if self.mode == Mode.PLAY:
                # Changing cursor
                self.cursor_manager.set_cursor(self, CursorMode.PINCH_OPEN)
                if self.polyline.is_polyline_exist():
                    self.moving_point_id = None
                    self.during_move_anim = False
                    self.during_restore_anim = True
                    self.start_restore_animation()

            self.last_mouse_pos = None

    def is_dragging(self) -> bool:
        return self.last_mouse_pos is not None

    def save_screen(self, fname: str):
        plt.imsave(fname, self.image_for_saving)

    ###########################
    # Creating line fuctions
    ###########################
    def cerate_line(self):
        color_list, primitive_binding_id, primitive_binding_time, polyline = cerate_line(
            W,
            H,
            self.current_path,
            self.gaussian_points,
            self.most_contribute_id,
            self.out_depth_of_most_contribute,
            self.image,
            tube_radius=TUBE_RADIUS,
            debug=False,
        )
        self.color_list = color_list
        self.primitive_binding_id = primitive_binding_id
        self.primitive_binding_time = primitive_binding_time
        self.polyline.set_polyline(polyline=polyline)
        self.polyline_layer.updatePolyline(self.polyline.get_polyline)

    ###########################################
    # Polyline moving and animation fuctions
    ###########################################
    def step_polyline_animation(self, pull_point_id: int | None = None, pos_goal: np.ndarray | None = None):
        # Update polyline
        self.polyline.simulate(time_step=10.5, pull_point_id=pull_point_id, pos_goal=pos_goal)

        # Moving primitives along with polyline
        self.pc._dx = moving_primitives.compute_primitive_displacement(
            self.polyline.get_polyline,
            self.polyline.get_rest_polyline,
            self.primitive_binding_id,
            self.primitive_binding_time,
        ).to("cuda")

        # Computing affine matrix for rotation
        self.pc._affine = rotating_primitives.compute_rotation_affine_batch(
            self.polyline.get_polyline,
            self.polyline.get_rest_polyline,
            torch.tensor(self.primitive_binding_id, device="cuda"),
            torch.tensor(self.primitive_binding_time, device="cuda"),
        )

    def step_move_polyline_animation(self, pixel_x: int, pixel_y: int, point_id: int):
        if point_id is None:
            return None
        pos_goal = get_goal_pos(point_id, pixel_x, pixel_y, W, H, self.camera, self.polyline.get_polyline)
        self.step_polyline_animation(point_id, pos_goal)
        self.polyline_layer.updatePolyline(self.polyline.get_polyline)

    def step_restore_polyline_animation(self):
        self.step_polyline_animation()

        self.anim_fps_counter.end()
        self.info_layer.update_anim_fps(fps=self.anim_fps_counter.get_fps())

        if self.polyline.is_anim_finished():
            print("Animation finished.")
            # Showing up polyline again
            self.polyline_layer.updatePolyline(self.polyline.get_polyline)
            self.during_restore_anim = False
            self.gs_fps_counter.start()
            QTimer.singleShot(0, self.update_screen)
        else:
            self.gs_fps_counter.start()
            QTimer.singleShot(0, self.update_screen)

    def start_restore_animation(self):
        # Hiding polyline
        self.polyline_layer.updatePolyline(np.array([[]]))
        # Starting restore animation
        self.anim_fps_counter.start()
        QTimer.singleShot(0, self.step_restore_polyline_animation)


if __name__ == "__main__":

    random_color = False

    SH_DEG = 3
    if random_color:
        SH_DEG = 0

    # ply_path = "gs_data/rope_sample_2.ply"
    # ply_path = "gs_data/rope_sample_2_without_depth.ply"
    ply_path = "gs_data/looping_toy_30000.ply"  # multi
    # ply_path = "gs_data/bicycle_with_chain_30000.ply"
    # ply_path = "gs_data/dinosaur_2_30000.ply" # multi
    # ply_path = "gs_data/street_light_1_30000.ply"
    # ply_path = "gs_data/chain_30000.ply" # multi
    # ply_path = "gs_data/desk_lamp_1_30000.ply"
    # ply_path = "gs_data/desk_lamp_2_30000.ply"
    # ply_path = "gs_data/koraidon_30000.ply" # multi
    # ply_path = "gs_data/wire_art_1_30000.ply"
    # ply_path = "gs_data/wire_art_2_2_30000.ply"
    # ply_path = "gs_data/cone_bar_30000.ply"
    # ply_path = "gs_data/swing_2_30000.ply" # multi
    # ply_path = "gs_data/whale_30000.ply"

    # For test
    # pos = [10.0, 0.0, 0.0]
    # up = [0, 1, 0]
    # to = [0, 0, 1]

    # Default
    pos = [-0.085, -0.372, -2.796]
    up = [-0.168, -0.938, 0.304]
    to = [0.509, 1.933, 4.632]  # Be careful! This is "to" not "lookat".

    # For teaser (bad)
    # pos = [0.761, -2.650, 2.870]
    # up = [-0.296, 0.156, -0.942]
    # to = [-0.505, 1.751, 3.999]  # Be careful! This is "to" not "lookat".

    # For teaser (looping toy)
    # pos = [-1.004, 1.030, 1.219]
    # up = [0.236, -0.842, -0.485]
    # to = [1.549, -1.153, 6.252] # Be careful! This is "to" not "lookat".

    # For bicycle with chain
    # pos = [-0.133, 1.918, -3.661]
    # up = [0.002, -0.994, -0.110]
    # to = [0.334, 0.925, 5.315] # Be careful! This is "to" not "lookat".

    # For comparison (zoom) (coat stand)
    # pos = [0.543, 1.983, -0.142]
    # up = [-0.172, -0.954, 0.247]
    # to = [1.028, 3.051, 4.323] # Be careful! This is "to" not "lookat".

    # For desk lamp 1
    # pos = [3.868, -0.980, -0.045]
    # up = [-0.452, -0.714, -0.534]
    # to = [-3.139, 1.912, 2.012] # Be careful! This is "to" not "lookat".

    # For cone bar
    # pos = [-1.638, 2.405, -3.894]
    # up = [0.206, -0.957, -0.203]
    # to = [0.909, 1.224, 4.267] # Be careful! This is "to" not "lookat".

    # For koraidon
    # pos = [-3.883, -0.176, 0.182]
    # up = [0.483, -0.535, -0.693]
    # to = [3.184, 1.320, 3.956] # Be careful! This is "to" not "lookat".

    # For chain
    # pos = [-0.895, 4.080, -1.011]
    # up = [0.005, -0.79, -0.602]
    # to = [-0.072, -1.785, 6.780] # Be careful! This is "to" not "lookat".

    # For street light 1
    # pos = [-1.270, -0.970, -1.754]
    # up = [-0.050, -0.998, 0.026]
    # to = [-0.727, -0.686, 10.123] # Be careful! This is "to" not "lookat".

    # For dinasour 2
    # pos = [-2.256, -1.581, 3.097]
    # up = [0.205, -0.383, -0.901]
    # to = [1.246, 3.688, 1.653] # Be careful! This is "to" not "lookat".

    # For swing 2
    # pos = [3.601, 0.378, 0.011]
    # up = [0.001, -0.968, -0.251]
    # to = [-3.677, -0.248, 2.402] # Be careful! This is "to" not "lookat".

    # For wire art 2
    # pos = [-1.231, -0.021, 1.300]
    # up = [0.029, -0.203, -0.979]
    # to = [-1.443, 5.375, 0.176] # Be careful! This is "to" not "lookat".

    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    window = MainWindow(ply_path, SH_DEG, W, H, pos, up, to, random_color=random_color)
    window.show()

    with loop:  # ここが asyncio loop の起動点になる
        loop.run_forever()
