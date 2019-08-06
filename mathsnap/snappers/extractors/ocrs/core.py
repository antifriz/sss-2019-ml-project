from typing import NamedTuple, Sequence

import numpy as np

from mathsnap.snappers.extractors.geometry import CharacterWithBoundingBox, BoundingBox
from mathsnap.snappers.extractors.ocrs.classifiers.core import Classifier
from mathsnap.snappers.extractors.ocrs.detectors.core import Detector


class OCRResult(NamedTuple):
    characters_with_bounding_boxes: Sequence[CharacterWithBoundingBox]


class OCR:
    def process(self, image: np.ndarray) -> OCRResult:
        raise NotImplementedError()


class DummyOCR(OCR):
    def process(self, image: np.ndarray) -> OCRResult:
        return OCRResult(
            characters_with_bounding_boxes=[
                CharacterWithBoundingBox(
                    character='2',
                    bounding_box=BoundingBox(
                        left=1,
                        top=2,
                        right=3,
                        bottom=4,
                    )
                )
            ]
        )


class DetectorClassifierOCR(OCR):
    def __init__(self, detector: Detector, classifier: Classifier):
        self._detector = detector
        self._classifier = classifier

    def process(self, image: np.ndarray) -> OCRResult:
        return OCRResult(
            characters_with_bounding_boxes=[
                CharacterWithBoundingBox(
                    character=self._classifier.process(detection.image).label,
                    bounding_box=detection.bounding_box
                )
                for detection in self._detector.process(image).detections
            ],
        )
