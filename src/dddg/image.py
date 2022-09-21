from __future__ import annotations

from functools import cache

import cv2
import webcolors
import numpy as np


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
