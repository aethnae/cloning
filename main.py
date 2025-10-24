import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util
from typing import Callable
from cloneGrayscale import clone_grayscale
from cloneMixed import clone_mixed

def clone_naive(f: np.ndarray, g: np.ndarray, target_y: int, target_x: int) -> np.ndarray:
    N, M = g.shape
    res = f.copy()
    res[target_y : target_y + N, target_x : target_x + M] = g
    return res

def process_RGB(cloning: Callable, f: np.ndarray, g: np.ndarray, target_y: int, target_x: int) -> np.ndarray:
    f_float = util.img_as_float(f)
    g_float = util.img_as_float(g)
    channels = []

    for channel in range(3):
        f_channel = f_float[:, :, channel]
        g_channel = g_float[:, :, channel]
        h_channel = cloning(f_channel, g_channel, target_y, target_x)
        channels.append(h_channel)

    img_f = np.stack(channels, axis=-1)
    img_i = util.img_as_ubyte(img_f)
    return img_i

if __name__ == "__main__":
    pass