from functools import cached_property
from itertools import combinations
from collections.abc import Iterable

import numpy as np

from dddg import loader
from dddg.duck import Ducky
from dddg.inference.infer import DuckyModel


def is_school(ducks: Iterable[Ducky]) -> bool:
    """Check if three ducks are in a valid school."""
    # 3 cards where each feature is either all the same or all different
    return (
        len(set(d.color for d in ducks)) in (1, 3)
        and len(set(d.n for d in ducks)) in (1, 3)
        and len(set(d.infer_hat for d in ducks)) in (1, 3)
        and len(set(d.infer_acc for d in ducks)) in (1, 3)
    )


class Solver:
    def __init__(self, full_image: np.ndarray):
        self.full_image = full_image
        self.model = DuckyModel()

    @cached_property
    def ducks(self) -> list[Ducky]:
        images = loader.split_tiles(self.full_image, 3, 4)
        return [Ducky(image, i) for i, image in enumerate(images)]

    def run_inference(self):
        images = [duck.single_view for duck in self.ducks]
        results = self.model.infer_batch(images)
        for duck, result in zip(self.ducks, results):
            duck.infer_result = result

    def solve(self):
        results = set()
        # Check all duck combinations to see if they form a school
        for ducks in combinations(self.ducks, 3):
            if is_school(ducks):
                results.add(frozenset(d.frame for d in ducks))
        return results

    def solve_str(self) -> str:
        return "\n".join(" ".join(map(str, sorted(r))) for r in self.solve())
