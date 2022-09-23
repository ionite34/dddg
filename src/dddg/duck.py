from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from dddg.config import Image
from dddg.image import scale_abs, sample_color, closest_color, find_duck
from dddg.inference.infer import Result

COLORS = {'darkred', 'darkblue', 'yellow'}


class Ducky:
    def __init__(self, image: np.ndarray, frame: int):
        self.image = image
        self.frame = frame
        self.infer_result: Result | None = None

    @cached_property
    def n(self) -> int:
        """Number of ducks in the tile."""

        for i in (3, 2):
            bounds: list[int] = Image[f"C{i}"].value  # type: ignore
            pos = scale_abs(self.image, *bounds)
            color = sample_color(self.image, *pos)
            named_color = closest_color(color)
            if named_color != "white":
                return i

        return 1

    @cached_property
    def color(self):
        """Color of the duck."""
        bounds: list[int] = Image[f"C{self.n}"].value  # type: ignore
        pos = scale_abs(self.image, *bounds)
        color = sample_color(self.image, *pos)
        return closest_color(color, COLORS)

    @cached_property
    def single_view(self) -> np.ndarray:
        """Get a view of a single duck within the tile."""
        return find_duck(self.image)

    @property
    def infer_hat(self) -> str:
        """Hat of the duck."""
        if self.infer_result is None:
            raise ValueError(f"Inference result not set for {self}")
        return self.infer_result["prediction"]["hat"]

    @property
    def infer_acc(self) -> str:
        """Accessory of the duck."""
        if self.infer_result is None:
            raise ValueError(f"Inference result not set for {self}")
        return self.infer_result["prediction"]["acc"]


@dataclass(frozen=True, eq=True)
class FrozenDucky:
    """A duck with known attributes."""

    n: int
    color: str
    hat: str
    accessory: str
