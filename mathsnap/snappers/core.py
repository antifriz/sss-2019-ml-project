from typing import NamedTuple

import numpy as np

from mathsnap.math_expression import math_expression_from_latex, MathExpression
from mathsnap.snappers.extractors.core import Extractor
from mathsnap.snappers.solvers.core import Solver
from mathsnap.utils import decompress_image


class SnapperResult(NamedTuple):
    problem: MathExpression
    solution: MathExpression


class Snapper:
    def process(self, image_buffer: str) -> SnapperResult:
        image = decompress_image(image_buffer)
        return self._process(image)

    def _process(self, image: np.ndarray) -> SnapperResult:

        raise NotImplementedError()


class DummySnapper(Snapper):
    def _process(self, image: np.ndarray) -> SnapperResult:
        return SnapperResult(
            problem=math_expression_from_latex('4 + 2'),
            solution=math_expression_from_latex('$$ 6 $$'),
        )


class ExtractorSolverSnapper(Snapper):
    def __init__(self, extractor: Extractor, solver: Solver):
        self._extractor = extractor
        self._solver = solver

    def _process(self, image: np.ndarray) -> SnapperResult:
        extractor_result = self._extractor.process(image)
        solver_result = self._solver.process(extractor_result.problem)
        return SnapperResult(
            problem=extractor_result.problem,
            solution=solver_result.solution,
        )
