import numpy as np

from utils.camera import Camera

THRESHOLD = 10


"""
Converting polyline (N, 3) to pixel space (N, 3).
Output: [x_pixel, y_pixel, z_ndc]
"""
def project_polyline_to_screen(W: int, H: int, camera: Camera, polyline: np.ndarray) -> np.ndarray:
    width = W
    height = H

    proj = camera.camera2ndc_mat_opengl.cpu().numpy()
    view = camera.world2camera_mat_opengl.cpu().numpy()

    mvp = proj @ view  # (4,4) x (4,4) -> (4,4)

    polyline_homo = np.concatenate([polyline, np.ones((len(polyline), 1), dtype=np.float32)], axis=1)  # (N,4)
    clip_coords = (mvp @ polyline_homo.T).T  # (N,4)

    # Clip space -> NDC
    ndc = clip_coords[:, :3] / clip_coords[:, 3:4]  # (N,3)

    # NDC -> Window (pixel) coordinates
    x_win = (ndc[:, 0] + 1) * 0.5 * width
    y_win = (1 - ndc[:, 1]) * 0.5 * height
    z_win = ndc[:, 2]

    return np.stack([x_win, y_win, z_win], axis=1)


def get_closest_polyline_idx(
    pixel_x: int, pixel_y: int, W: int, H: int, camera: Camera, polyline: np.ndarray
) -> int | None:
    if polyline.shape[0] <= 0:
        return None
    screen_coords = project_polyline_to_screen(W, H, camera, polyline)

    # Searching nearest point
    mouse_pos = np.array([pixel_x, pixel_y])
    screen_xy = screen_coords[:, :2]
    dists = np.linalg.norm(screen_xy - mouse_pos, axis=1)
    closest_idx = np.argmin(dists)
    if dists[closest_idx] < THRESHOLD:  # Choise if the point is in THRESHOLD px.
        print(f"Closest point index: {closest_idx}, distance: {dists[closest_idx]}")
        return closest_idx
    else:
        return None


def get_goal_pos(
    closest_idx: int,
    pixel_x: int,
    pixel_y: int,
    W: int,
    H: int,
    camera: Camera,
    polyline: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    assert 0 <= closest_idx
    point = polyline[closest_idx]

    # Camera vecs
    t_pos, t_up, t_lookat = camera.get_camera_vecs_from_world2camera_mat(camera.BLENDER)
    x_vec = np.cross(t_lookat, t_up)
    y_vec = t_up
    camera_pos = t_pos
    camera_dir = t_lookat
    x_vec /= np.linalg.norm(x_vec)
    y_vec /= np.linalg.norm(y_vec)
    camera_dir /= np.linalg.norm(camera_dir)

    # Ray on pixel coords
    ndc_x = (pixel_x + 0.5) / W * 2.0 - 1.0
    ndc_y = -((pixel_y + 0.5) / H * 2.0 - 1.0)
    ray_x = ndc_x * W / (2.0 * camera.focal_x.cpu().numpy())
    ray_y = ndc_y * H / (2.0 * camera.focal_y.cpu().numpy())
    ray_z = 1.0

    # Ray on world coords
    ray_direction = ray_x * x_vec + ray_y * y_vec + ray_z * camera_dir
    ray_direction /= np.linalg.norm(ray_direction)

    # Calcurating goal position on vertex plane
    n = np.cross(x_vec, y_vec)
    n_norm = np.linalg.norm(n)
    assert eps <= n_norm
    n /= n_norm
    denom = float(n.dot(ray_direction))
    num = float(n.dot(point - camera_pos))
    assert eps <= abs(denom) and eps <= abs(num)
    t = num / denom
    pos_goal = camera_pos + t * ray_direction

    return pos_goal
