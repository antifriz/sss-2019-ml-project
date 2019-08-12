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


class GreedyDetector(Detector):

    def process(self, image: np.ndarray) -> DetectorResult:
        boundingBoxes= []

        _detections = [
            Detection(
                image=image[x:x + w, y:y + h],
                bounding_box=BoundingBox(left=x, top=y, right=x + w, bottom=y + h)
            )
            for x, y, w, h in boundingBoxes]

        result = DetectorResult(detections=_detections)
        return result
