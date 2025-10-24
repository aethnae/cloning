import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util
from skimage.transform import resize
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
    try:
        bear = io.imread("Resources/Images/bear.jpg")
        water = io.imread("Resources/Images/water.jpg")
        plane = io.imread("Resources/Images/plane.jpg")
        bird = io.imread("Resources/Images/bird.jpg")
        print("Read Images.")
        print(f"Size Bear:   {bear.shape}")
        print(f"Size Water: {water.shape}")
        print(f"Size Airplane: {plane.shape}")
        print(f"Size Bird:  {bird.shape}")
    except FileNotFoundError as e:
        print(f"ERROR: Source path not found.")
        print(f"({e})")
        exit()

    pos_bear_x, pos_bear_y = 30, 120
    res_bear_naive = process_RGB(clone_naive, water, bear, pos_bear_y, pos_bear_x)
    res_bear_laplace = process_RGB(clone_grayscale, water, bear, pos_bear_y, pos_bear_x)
    res_bear_mixed = process_RGB(clone_mixed, water, bear, pos_bear_y, pos_bear_x)

    plane_resized_float = resize(plane,(plane.shape[0] // 2, plane.shape[1] // 2), anti_aliasing=True)
    plane_small = util.img_as_ubyte(plane_resized_float)
    print(f"Neue Flugzeug-Maße: {plane_small.shape}")

    pos_plane_x, pos_plane_y = 400, 60
    res_plane_naive = process_RGB(clone_naive, bird, plane_small, pos_plane_y, pos_plane_x)
    res_plane_laplace = process_RGB(clone_grayscale, bird, plane_small, pos_plane_y, pos_plane_x)
    res_plane_mixed = process_RGB(clone_mixed, bird, plane_small, pos_plane_y, pos_plane_x)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(res_bear_naive)
    axes[0, 0].set_title("Bear: Naive approach")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(res_bear_laplace)
    axes[0, 1].set_title("Bear: Laplace approach")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(res_bear_mixed)
    axes[0, 2].set_title("Bear: Mixed gradients")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(res_plane_naive)
    axes[1, 0].set_title("Airplane: Naive approach")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(res_plane_laplace)
    axes[1, 1].set_title("Airplane: Laplace approach")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(res_plane_mixed)
    axes[1, 2].set_title("Airplane: Mixed gradients")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()