from typing import NamedTuple, Sequence, Dict

import numpy as np

from mathsnap.snappers.extractors.geometry import CharacterWithBoundingBox, BoundingBox
from mathsnap.snappers.extractors.ocrs.classifiers.core import Classifier, ClassifierResult
from mathsnap.snappers.extractors.ocrs.detectors.core import Detector


class OCRResult(NamedTuple):
    characters_with_bounding_boxes: Sequence[CharacterWithBoundingBox]
    images: Dict[str, str]


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
                        x0=1,
                        y0=2,
                        x1=3,
                        y1=4,
                    )
                )
            ],
            images={},
        )


class DetectorClassifierOCR(OCR):
    def __init__(self, detector: Detector, classifier: Classifier):
        self._detector = detector
        self._classifier = classifier

    def process(self, image: np.ndarray) -> OCRResult:
        detection_result = self._detector.process(image)

        characters = []
        c_images = []
        for detection in detection_result.detections:
            c_result = self._classifier.process(detection.image)
            c_images.append(c_result.image)
            characters.append(
                CharacterWithBoundingBox(
                    character=c_result.label,
                    bounding_box=detection.bounding_box
                ))

        images = detection_result.images
        images["classifier"] = c_images

        return OCRResult(
            characters_with_bounding_boxes=characters,
            images=images
        )
