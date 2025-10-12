import math
from typing import Final

import numpy as np
import scipy.spatial.transform
import torch

"""
This Camera class holds both camera vectors and camera matrices.
Please be careful, you have to modify both vectors and matrices updating logic
when you modify camera operation methods.
Internal calculation must prioritize vectors rather matrices because of parameter consistency.
"""


class Camera:
    OPENGL: Final = "opengl"
    COLMAP: Final = "colmap"
    BLENDER: Final = "blender"

    def __init__(
        self, width: int, height: int, pos: np.ndarray, up: np.ndarray, to: np.ndarray, device: torch.device
    ) -> None:
        self.device = device
        self.__get_camera(width, height, pos, up, to)

    def __get_camera(self, width: int, height: int, pos: np.ndarray, up: np.ndarray, to: np.ndarray):
        world2camera_mat_colmap = self.__get_world2camera_mat_from_camera_vecs(pos, up, to, mode=self.COLMAP)
        world2camera_mat_opengl = self.__get_world2camera_mat_from_camera_vecs(pos, up, to, mode=self.OPENGL)

        intrinsic_mat = torch.tensor(
            [
                [711.1111, 0.0000, 256.0000, 0.0000],
                [0.0000, 711.1111, 256.0000, 0.0000],
                [0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ],
            device=self.device,
        )

        focal_x, focal_y = intrinsic_mat[0, 0], intrinsic_mat[1, 1]
        fov_x = self.__focal2fov(focal_x, width)
        fov_y = self.__focal2fov(focal_y, height)
        znear = 0.2
        zfar = 1000

        camera2ndc_mat_colmap = self.__get_camera2ndc_matrix(
            znear=znear, zfar=zfar, fov_x=fov_x, fov_y=fov_y, mode=self.COLMAP
        )
        camera2ndc_mat_opengl = self.__get_camera2ndc_matrix(
            znear=znear, zfar=zfar, fov_x=fov_x, fov_y=fov_y, mode=self.OPENGL
        )

        self.intrinsic_mat = intrinsic_mat
        self.world2camera_mat_colmap = world2camera_mat_colmap
        self.world2camera_mat_opengl = world2camera_mat_opengl
        self.camera2ndc_mat_colmap = camera2ndc_mat_colmap
        self.camera2ndc_mat_opengl = camera2ndc_mat_opengl
        self.world2ndc_mat_colmap = camera2ndc_mat_colmap @ world2camera_mat_colmap
        self.world2ndc_mat_opengl = camera2ndc_mat_opengl @ world2camera_mat_opengl

        self.height = height
        self.width = width
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.znear = znear
        self.zfar = zfar
        self.base_pos = pos
        self.base_up = up
        self.base_to = to
        self.focal_x = focal_x
        self.focal_y = focal_y

    def __get_world2camera_mat_from_camera_vecs(
        self, pos: np.ndarray, up: np.ndarray, to: np.ndarray, mode: str
    ) -> torch.Tensor:
        if mode == self.OPENGL:
            zaxis = (pos - to) / np.linalg.norm(pos - to)
            xaxis = np.cross(up, zaxis) / np.linalg.norm(np.cross(up, zaxis))
            yaxis = np.cross(zaxis, xaxis)
        elif mode == self.COLMAP:
            zaxis = (to - pos) / np.linalg.norm(to - pos)
            xaxis = np.cross(zaxis, up) / np.linalg.norm(np.cross(zaxis, up))
            yaxis = np.cross(xaxis, zaxis)
        elif mode == self.BLENDER:
            zaxis = (pos - to) / np.linalg.norm(pos - to)
            xaxis = np.cross(up, zaxis) / np.linalg.norm(np.cross(up, zaxis))
            yaxis = np.cross(zaxis, xaxis)
        else:
            zaxis = (to - pos) / np.linalg.norm(to - pos)
            xaxis = np.cross(zaxis, up) / np.linalg.norm(np.cross(zaxis, up))
            yaxis = np.cross(xaxis, zaxis)

        tx = -np.dot(xaxis, pos)
        ty = -np.dot(yaxis, pos)
        tz = -np.dot(zaxis, pos)

        view_mat = torch.tensor(
            [
                [xaxis[0], xaxis[1], xaxis[2], tx],
                [yaxis[0], yaxis[1], yaxis[2], ty],
                [zaxis[0], zaxis[1], zaxis[2], tz],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        return view_mat

    def get_camera_vecs_from_world2camera_mat(self, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mode == self.OPENGL:
            view_mat = self.world2camera_mat_opengl
        else:
            view_mat = self.world2camera_mat_colmap
        inv_view_mat = torch.inverse(view_mat).cpu()
        pos = np.array([inv_view_mat[0][3], inv_view_mat[1][3], inv_view_mat[2][3]], dtype=np.float32)
        up = np.array([inv_view_mat[0][1], inv_view_mat[1][1], inv_view_mat[2][1]], dtype=np.float32)
        lookat = np.array([inv_view_mat[0][2], inv_view_mat[1][2], inv_view_mat[2][2]], dtype=np.float32)

        return pos, up, lookat

    def __focal2fov(self, focal: float, pixels: float):
        return 2 * math.atan(pixels / (2 * focal))

    def __get_camera2ndc_matrix(
        self, znear: float, zfar: float, fov_x: float, fov_y: float, mode: str
    ) -> torch.Tensor:
        tan_half_fov_x = math.tan(fov_x / 2)
        tan_half_fov_y = math.tan(fov_y / 2)

        top = tan_half_fov_y * znear
        bottom = -top
        right = tan_half_fov_x * znear
        left = -right

        P = torch.zeros(4, 4, dtype=torch.float32, device=self.device)

        if mode == self.OPENGL:
            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[2, 2] = -(zfar + znear) / (zfar - znear)
            P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
            P[3, 2] = -1.0
        else:
            z_sign = 1.0
            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            P[3, 2] = z_sign

        return P

    def __set_world2camera_from_vecs(self, pos: np.ndarray, up: np.ndarray, to: np.ndarray):
        # Updating camera vectors
        self.base_pos = pos
        self.base_up = up
        self.base_to = to
        # Updating camera matrices
        self.world2camera_mat_colmap = self.__get_world2camera_mat_from_camera_vecs(pos, up, to, mode=self.COLMAP)
        self.world2camera_mat_opengl = self.__get_world2camera_mat_from_camera_vecs(pos, up, to, mode=self.OPENGL)
        self.__update_world2ndc_mat()

    def __update_camera2ndc_matrix(self):
        camera2ndc_mat_colmap = self.__get_camera2ndc_matrix(
            znear=self.znear, zfar=self.zfar, fov_x=self.fov_x, fov_y=self.fov_y, mode=self.COLMAP
        )
        camera2ndc_mat_opengl = self.__get_camera2ndc_matrix(
            znear=self.znear, zfar=self.zfar, fov_x=self.fov_x, fov_y=self.fov_y, mode=self.OPENGL
        )
        self.camera2ndc_mat_colmap = camera2ndc_mat_colmap
        self.camera2ndc_mat_opengl = camera2ndc_mat_opengl
        self.__update_world2ndc_mat()

    def __update_world2ndc_mat(self):
        self.world2ndc_mat_colmap = self.camera2ndc_mat_colmap @ self.world2camera_mat_colmap
        self.world2ndc_mat_opengl = self.camera2ndc_mat_opengl @ self.world2camera_mat_opengl

    ######################
    # Camera operations
    ######################
    def move_parallel(self, dx: float, dy: float, dz: float, retarget=True):
        # Updating camera matrices
        t = torch.tensor(
            [[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]], dtype=torch.float32, device=self.device
        )
        self.world2camera_mat_colmap = t @ self.world2camera_mat_colmap

        # Updating camera vectors (base_pos, base_to)
        pos, _, lookat = self.get_camera_vecs_from_world2camera_mat(mode=self.COLMAP)
        self.base_pos = pos
        if retarget:  # Updating base_to param
            r = np.linalg.norm(self.base_to - self.base_pos)
            self.base_to = pos + r * lookat / np.linalg.norm(lookat)

        # Updating opengl camera matrices
        self.world2camera_mat_opengl = self.__get_world2camera_mat_from_camera_vecs(
            self.base_pos, self.base_up, self.base_to, mode=self.OPENGL
        )

        # Updating world2ndc matrices
        self.__update_world2ndc_mat()

    def rotate(self, angle: float, axis: np.ndarray):
        """Rotate the camera by a given angle around a specified axis (x, y, z).

        Parameters:
            view_matrix (numpy.ndarray): The 4x4 view matrix representing the camera's transformation.
            angle (float): The angle in radians to rotate the camera.
            axis (numpy.ndarray): The axis around which to rotate (should be normalized).

        Returns:
            numpy.ndarray: The new view matrix after applying the rotation.
        """
        # Normalize the axis
        axis = axis / np.linalg.norm(axis)

        # Create the rotation matrix using Rodrigues' rotation formula
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        one_minus_cos = 1 - cos_a

        x, y, z = axis
        rotation_matrix = torch.tensor(
            [
                [
                    cos_a + x * x * one_minus_cos,
                    x * y * one_minus_cos - z * sin_a,
                    x * z * one_minus_cos + y * sin_a,
                    0,
                ],
                [
                    y * x * one_minus_cos + z * sin_a,
                    cos_a + y * y * one_minus_cos,
                    y * z * one_minus_cos - x * sin_a,
                    0,
                ],
                [
                    z * x * one_minus_cos - y * sin_a,
                    z * y * one_minus_cos + x * sin_a,
                    cos_a + z * z * one_minus_cos,
                    0,
                ],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Apply the rotation to the view matrix
        self.world2camera_mat_colmap = rotation_matrix @ self.world2camera_mat_colmap

        # Updating camera vectors (base_up, base_to)
        r = np.linalg.norm(self.base_to - self.base_pos)
        _, up, lookat = self.get_camera_vecs_from_world2camera_mat(mode=self.COLMAP)
        self.base_up = up
        self.base_to = self.base_pos + r * lookat / np.linalg.norm(lookat)

        # Updating opengl camera matrices
        self.world2camera_mat_opengl = self.__get_world2camera_mat_from_camera_vecs(
            self.base_pos, self.base_up, self.base_to, mode=self.OPENGL
        )

        # Updating world2ndc matrices
        self.__update_world2ndc_mat()

    def move_orbit(self, dtheta: float, dphi: float):
        offset = self.base_pos - self.base_to

        q = scipy.spatial.transform.Rotation.align_vectors([[0, 1, 0]], [self.base_up])[0]  # from up → y-axis
        q_inv = q.inv()

        offset_rot = q.apply(offset)

        r = np.linalg.norm(offset_rot)
        x, y, z = offset_rot
        spherical_theta = np.arctan2(x, z)
        spherical_phi = np.arccos(y / r)

        spherical_theta += dtheta
        spherical_phi += dphi
        spherical_phi = np.clip(spherical_phi, 1e-4, np.pi - 1e-4)  # makeSafe()

        new_x = r * np.sin(spherical_phi) * np.sin(spherical_theta)
        new_y = r * np.cos(spherical_phi)
        new_z = r * np.sin(spherical_phi) * np.cos(spherical_theta)
        new_offset_rot = np.array([new_x, new_y, new_z])

        new_offset = q_inv.apply(new_offset_rot)

        new_pos = self.base_to + new_offset

        self.__set_world2camera_from_vecs(new_pos, self.base_up, self.base_to)
