from typing import NamedTuple, Sequence, Dict
import cv2
import numpy as np

from mathsnap.snappers.extractors.geometry import BoundingBox, _box_from_bounding_rect
from mathsnap.utils import convert_to_datauri


class Detection(NamedTuple):
    image: np.ndarray
    bounding_box: BoundingBox


class DetectorResult(NamedTuple):
    detections: Sequence[Detection]
    images: Dict[str, str]


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
            ],
            images={},
        )


def _make_detection_image(img, detected_boxes: [BoundingBox]):
    for box in detected_boxes:
        cv2.rectangle(img, (box.x0, box.y0), (box.x1, box.y1), (0, 255, 0), 2)
    return img


class GreedyDetector(Detector):

    def box_detection(self, img: np.ndarray) -> [BoundingBox]:
        # denoising
        dst = cv2.fastNlMeansDenoising(img, h=6)

        # binarize
        magic_threshold = 130
        ret, thresh = cv2.threshold(dst, magic_threshold, 255, cv2.THRESH_BINARY)

        # Morph transformation
        kernel = np.ones((20, 20), 'uint8')
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours = cv2.findContours(thresh, 1, 2)[0][:-1]  # Remove last one because it's the whole image border.

        return [
            _box_from_bounding_rect(cv2.boundingRect(c))
            for c in contours
        ]

    def process(self, image: np.ndarray) -> DetectorResult:
        bounding_boxes = self.box_detection(image)

        detections = [
            Detection(
                image=image[b.y0:b.y1, b.x0:b.x1],
                bounding_box=b
            )
            for b in bounding_boxes]

        return DetectorResult(
            detections=detections,
            images={
                "detections": convert_to_datauri(_make_detection_image(image, bounding_boxes)),
            }
        )
