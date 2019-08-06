from typing import NamedTuple, Sequence

import numpy as np

from mathsnap.snappers.extractors.geometry import BoundingBox


class Detection(NamedTuple):
    image: np.ndarray
    bounding_box: BoundingBox


class DetectorResult(NamedTuple):
    detections: Sequence[Detection]


class Detector:
    def process(self, image: np.ndarray) -> DetectorResult:
        raise NotImplementedError()


class DummyDetector(Detector):
    def process(self, image: np.ndarray) -> DetectorResult:
        return DetectorResult(
            detections=[
                Detection(
                    image=np.zeros((640, 480, 3), dtype='uint8'),
                    bounding_box=BoundingBox(
                        left=1,
                        top=2,
                        right=3,
                        bottom=4,
                    )
                ),
            ]
        )
