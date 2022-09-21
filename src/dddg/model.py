from functools import cached_property

import numpy as np

from dddg.config import Image
from dddg.image import scale_abs, sample_color, closest_color


class Ducky:
    def __init__(self, image: np.ndarray):
        self.image = image

    @cached_property
    def n(self) -> int:
        """Number of ducks in the tile."""

        for i in (3, 2):
            pos = scale_abs(self.image, *Image[f"C{i}"].value)
            color = sample_color(self.image, *pos)
            named_color = closest_color(color)
            if named_color != "white":
                return i

        return 1

    @cached_property
    def color(self):
        """Color of the duck."""
        pos = scale_abs(self.image, *Image[f"C{self.n}"].value)
        color = sample_color(self.image, *pos)
        return closest_color(color)
