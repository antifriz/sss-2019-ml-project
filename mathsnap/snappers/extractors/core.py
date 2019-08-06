from typing import NamedTuple

import numpy as np

from mathsnap.math_expression import math_expression_from_latex, MathExpression
from mathsnap.snappers.extractors.layouters.layouters import Layouter
from mathsnap.snappers.extractors.ocrs.core import OCR


class ExtractorResult(NamedTuple):
    problem: MathExpression


class Extractor:
    def process(self, image: np.ndarray) -> ExtractorResult:
        raise NotImplementedError()


class DummyExtractor(Extractor):
    def process(self, image: np.ndarray) -> ExtractorResult:
        return ExtractorResult(
            problem=math_expression_from_latex('5 + 5')
        )


class OCRLayouterExtractor(Extractor):
    def __init__(self, ocr: OCR, layouter: Layouter):
        self._ocr = ocr
        self._layouter = layouter

    def process(self, image: np.ndarray) -> ExtractorResult:
        ocr_result = self._ocr.process(image)
        layouter_result = self._layouter.process(ocr_result.characters_with_bounding_boxes)
        return ExtractorResult(
            problem=layouter_result.problem
        )
