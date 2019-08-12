from typing import NamedTuple, Sequence
import cv2
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

    def box_detection(img: np.ndarray):
        kernel = np.ones((20, 20), np.uint8)
        ret, thresh = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)

        # Morph transformation
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours = cv2.findContours(thresh, 1, 2)[0][:-1]  # Remove last one because it's the whole image border.

        return [cv2.boundingRect(c) for c in contours]

    def process(self, image: np.ndarray) -> DetectorResult:
        bounding_boxes = self.box_detection(image)

        _detections = [
            Detection(
                image=image[x:x + w, y:y + h],
                bounding_box=BoundingBox(left=x, top=y, right=x + w, bottom=y + h)
            )
            for x, y, w, h in bounding_boxes]

        result = DetectorResult(detections=_detections)
        return result
