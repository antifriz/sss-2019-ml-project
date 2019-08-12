from typing import NamedTuple





class BoundingBox(NamedTuple):
    x0: float  # (x0, y0) ------------- |
    y0: float  # |                      |
    x1: float  # |                      |
    y1: float  # | ---------------(x1, y1)


def _box_from_bounding_rect(self, rect) -> BoundingBox:
    return BoundingBox(
        x0=rect[0],
        x1=rect[0] + rect[2],
        y0=rect[1],
        y1=rect[1] + rect[3]
    )


class CharacterWithBoundingBox(NamedTuple):
    character: str
    bounding_box: BoundingBox
