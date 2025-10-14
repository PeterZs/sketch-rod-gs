import numpy as np


def pix2ndc(pix_x, pix_y, W, H) -> np.ndarray:
    x = (pix_x / W) * 2.0 - 1.0
    y = 1.0 - (pix_y / H) * 2.0
    return np.array([x, y])
