#
# Portions of this code are adapted from:
# "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
# Copyright (C) 2023, Inria / GRAPHDECO research group (https://team.inria.fr/graphdeco)
# Used under the terms of the original LICENSE.md (non-commercial research use).
#
# Modifications by Haato Watanabe, 2025.
# Additional code © 2025 Haato Watanabe.
#

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from diff_gaussian_rasterization_for_sketchrodgs import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from plyfile import PlyData
from utils.camera import Camera
from utils.general_utils import build_scaling_rotation, inverse_sigmoid, strip_symmetric
from utils.sh_utils import RGB2SH, eval_sh

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("this viewer needs CUDA environment")
    exit()


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            actual_covariance = self._affine.transpose(1, 2) @ actual_covariance @ self._affine
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self._dx = torch.empty(0)
        self._affine = torch.empty(0)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz + self._dx

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_affine(self):
        return self._affine

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def load_ply(self, path, random_color: bool):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if random_color:
            fused_color = RGB2SH(torch.tensor(np.random.rand(xyz.shape[0], 3)).float().cuda())
            features = torch.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0
        else:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        if random_color:
            self._features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
            self._features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        else:
            self._features_dc = (
                torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
            )
            self._features_rest = (
                torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
            )
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")

        self._dx = torch.zeros_like(self._xyz, dtype=torch.float, device="cuda")
        self._affine = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0]), dtype=torch.float, device="cuda").repeat(
            self._xyz.shape[0], 1
        )

        # Only visible Gaussians
        visible_id, _ = torch.where(0 < self.get_opacity)
        self._xyz = self._xyz[visible_id, :]
        self._features_dc = self._features_dc[visible_id, :, :]
        self._features_rest = self._features_rest[visible_id, :, :]
        self._opacity = self._opacity[visible_id]
        self._scaling = self._scaling[visible_id, :]
        self._rotation = self._rotation[visible_id, :]

        self._dx = self._dx[visible_id, :]
        self._affine = self._affine[visible_id, :]

        self.active_sh_degree = self.max_sh_degree

        if random_color:
            self._opacity[:] = 1.0


def init(
    width: int,
    height: int,
    sh_degree: int,
    model_path: str,
    pos: list[float],
    up: list[float],
    to: list[float],
    random_color: bool,
) -> tuple[Camera, GaussianModel]:
    camera = Camera(width, height, np.array(pos), np.array(up), np.array(to), device=device)

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(model_path, random_color)

    return camera, gaussians


bg_color = [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


def render(camera: Camera, pc: GaussianModel, scaling_modifier=1.0, override_color=None) -> torch.Tensor:
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    viewpoint_camera = MiniCam(
        camera.width,
        camera.height,
        camera.fov_y,
        camera.fov_x,
        camera.znear,
        camera.zfar,
        camera.world2camera_mat_colmap.T,
        camera.world2ndc_mat_colmap.T,
    )

    pipe = {"debug": False, "compute_cov3D_python": False, "convert_SHs_python": False}

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe["debug"],
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe["compute_cov3D_python"]:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        affines = pc.get_affine

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe["convert_SHs_python"]:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, most_contribute_id, out_depth_of_most_contribute, pixel_coords = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        affines=affines,
        cov3D_precomp=cov3D_precomp,
    )

    return rendered_image, most_contribute_id, out_depth_of_most_contribute, pixel_coords


def render_no_grad(camera: Camera, pc: GaussianModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        image, most_contribute_id, out_depth_of_most_contribute, pixel_coords = render(camera, pc)

    image = image.permute(1, 2, 0)
    image = image.clamp(max=1.0, min=0.0)
    image = image.cpu().numpy()
    most_contribute_id = most_contribute_id.cpu().numpy()
    out_depth_of_most_contribute = out_depth_of_most_contribute.cpu().numpy()
    pixel_coords = pixel_coords.cpu().numpy()

    most_contribute_id = most_contribute_id[::-1, :]
    out_depth_of_most_contribute = out_depth_of_most_contribute[::-1, :]
    pixel_coords = pixel_coords[::-1, :]

    return image, most_contribute_id, out_depth_of_most_contribute, pixel_coords


# For debug
if __name__ == "__main__":
    sh_deg = 0
    camera, pc = init(sh_deg, "gs_data/sample_1.ply")
    image, most_contribute_id, out_depth_of_most_contribute, pixel_coords = render_no_grad(camera, pc)

    print("most_contribute_id.shape: \n", most_contribute_id.shape)
    print("out_depth_of_most_contribute.shape: \n", out_depth_of_most_contribute.shape)
    print("pixel_coords.shape: \n", pixel_coords.shape)

    plt.imsave("2d_gabor_cuda.png", image)
