from pathlib import Path
from enum import Enum

import tomli

CFG_DIR = Path(__file__).parent

_f = tomli.loads((CFG_DIR / "default.toml").read_text())


class Load(Enum):
    TILES_X: int = _f["load"]["tiles_x"]
    TILES_Y: int = _f["load"]["tiles_y"]


class Image(Enum):
    C1 = _f["image"]["c1"]
    C2 = _f["image"]["c2"]
    C3 = _f["image"]["c3"]
