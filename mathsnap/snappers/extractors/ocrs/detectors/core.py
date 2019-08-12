from typing import NamedTuple, Sequence
import cv2
import numpy as np

from mathsnap.snappers.extractors.geometry import BoundingBox, _box_from_bounding_rect


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
                        x0=1,
                        y0=2,
                        x1=3,
                        y1=4,
                    )
                ),
            ]
        )


class GreedyDetector(Detector):

    def box_detection(self, img: np.ndarray) -> [BoundingBox]:
        magic_threshold = 130

        kernel = np.ones((20, 20), 'uint8')
        ret, thresh = cv2.threshold(img, magic_threshold, 255, cv2.THRESH_BINARY)

        # Morph transformation
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours = cv2.findContours(thresh, 1, 2)[0][:-1]  # Remove last one because it's the whole image border.

        return [_box_from_bounding_rect(cv2.boundingRect(c)) for c in contours]

    def process(self, image: np.ndarray) -> DetectorResult:
        bounding_boxes = self.box_detection(image)

        _detections = [
            Detection(
                image=image[b.x0:b.x1, b.y0:b.y1],
                bounding_box=b
            )
            for b in bounding_boxes]

        result = DetectorResult(detections=_detections)
        return result
