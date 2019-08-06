from typing import NamedTuple


class BoundingBox(NamedTuple):
    left: float
    top: float
    right: float
    bottom: float


class CharacterWithBoundingBox(NamedTuple):
    character: str
    bounding_box: BoundingBox
