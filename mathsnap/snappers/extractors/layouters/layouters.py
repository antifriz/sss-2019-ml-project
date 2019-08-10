from typing import Sequence, NamedTuple

from mathsnap.snappers.extractors.geometry import CharacterWithBoundingBox
from mathsnap.math_expression import MathExpression, math_expression_from_latex


class LayouterResult(NamedTuple):
    problem: MathExpression


class Layouter:
    def process(self, characters_with_bounding_boxes: Sequence[CharacterWithBoundingBox]) -> LayouterResult:
        raise NotImplementedError()


class DummyLayouter(Layouter):
    def process(self, characters_with_bounding_boxes: Sequence[CharacterWithBoundingBox]) -> LayouterResult:
        return LayouterResult(
            problem=math_expression_from_latex("5 + 6")
        )