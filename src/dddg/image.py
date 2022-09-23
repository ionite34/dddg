from __future__ import annotations

from functools import cache

import cv2
import webcolors
import numpy as np
from PIL import ImageOps, Image


@cache
def _color_pool(within: frozenset[str]):
    return [*filter(
        lambda x: x[1] in within,
        webcolors.CSS3_HEX_TO_NAMES.items()
    )]


def scale_abs(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """Convert relative percentage to absolute pixel values."""
    width = int(img.shape[1] * x)
    height = int(img.shape[0] * y)
    return np.array([width, height], dtype=np.int32)


def sample_color(
        img: np.ndarray,
        x: int,
        y: int,
        bounds: tuple[int, int] = (1, 1),
) -> np.ndarray:
    """Sample a color from an image at a given position."""
    x, y = int(x), int(y)
    x1, y1 = x - bounds[0], y - bounds[1]
    x2, y2 = x + bounds[0], y + bounds[1]
    return img[y1:y2, x1:x2].mean(axis=0).mean(axis=0)


def closest_color(arr, pool: set[str] | None = None):
    """
    Get the closest color name to a given RGB value.

    Args:
        arr: RGB array
        pool: Set of color names to search within.

    Returns:
        The closest color name.
    """
    min_colours = {}

    if pool is None:
        colors = webcolors.CSS3_HEX_TO_NAMES.items()
    else:
        colors = _color_pool(frozenset(pool))

    for key, name in colors:
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - arr[0]) ** 2
        gd = (g_c - arr[1]) ** 2
        bd = (b_c - arr[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def _find_duck_bounds(img: np.ndarray) -> np.ndarray:
    """
    Find the duck bounds in an image.
    [[top_left, top_right, bottom_left, bottom_right, size], ...]

    Args:
        img: Image to search.

    Returns:
        2D Array of duck bounds.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # binarize the image
    ret, bw = cv2.threshold(gray, 128, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # find connected components
    connectivity = 4
    result = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
    nb_components, output, stats, centroids = result
    return stats


def find_duck(img: np.ndarray) -> np.ndarray:
    """
    Finds a duck within an image.

    Args:
        img: Image to search.

    Returns:
        A view of the duck within the image.
    """
    # Get bound stats
    stats = _find_duck_bounds(img)
    found = stats[(stats[:, -1] > 1200) & (stats[:, -1] < 2000)]
    try:
        stat = found[0]
        y1 = stat[1]
        y2 = stat[1] + stat[3]
        x1 = stat[0]
        x2 = stat[0] + stat[2]
        # Return a cropped view of the image
        return img[y1:y2, x1:x2]
    except IndexError as e:
        raise ValueError("Could not find duck bounds in image.") from e


def pad_image(img: Image | np.ndarray, to_size=60) -> Image:
    """Pads an image to a given size."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    desired_size = to_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padded = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padded, fill=(255, 255, 255))
