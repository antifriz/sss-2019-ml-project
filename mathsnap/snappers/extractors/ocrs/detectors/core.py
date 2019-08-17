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
    img_with_boxes = img.copy()
    for box in detected_boxes:
        padding=2
        cv2.rectangle(img_with_boxes, (box.x0-padding, box.y0-padding), (box.x1+padding, box.y1+padding), (0, 255, 0), 2)
    return img_with_boxes

# TODO: Improve Detecter


class GreedyDetector(Detector):
    def overlap(self, box_i: BoundingBox, box_j: BoundingBox) -> int:
        x_o = max(min(box_i.x1, box_j.x1) - max(box_i.x0, box_j.x0), 0)
        y_o = max(min(box_i.y1, box_j.y1) - max(box_i.y0, box_j.y0), 0)
        o = x_o * y_o
        return o

    def nms(self, boxes: [BoundingBox]) -> [BoundingBox]:
        boxes = sorted(boxes, key=lambda x: (x.x1 - x.x0) * (x.y1 - x.y0), reverse=True)
        new_boxes = []
        for i, box_i in enumerate(boxes):
            for box_j in boxes[0:i]:
                if self.overlap(box_i, box_j) > 0:
                    break
            else:
                new_boxes.append(box_i)
        return new_boxes

    def remove_small_boxes(self, boxes: [BoundingBox]) -> [BoundingBox]:
        _b = []
        for x in boxes:
            # print((x.x1 - x.x0) * (x.y1 - x.y0))
            if (x.x1 - x.x0) * (x.y1 - x.y0) > 150:
                _b.append(x)
        return _b

    def box_detection(self, img: np.ndarray) -> [BoundingBox]:
        # denoising
        dst = cv2.fastNlMeansDenoising(img, h=6)

        # binarize
        magic_threshold = 130
        ret, thresh = cv2.threshold(dst, magic_threshold, 255, cv2.THRESH_BINARY)

        # thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, , 20)
        # Morph transformation
        kernel = np.ones((20, 20), 'uint8')
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


        # Boxes
        contours = cv2.findContours(thresh, 1, 2)[0][:-1]  # Remove last one because it's the whole image border.

        return self.nms (
            self.remove_small_boxes(
                [
                    _box_from_bounding_rect(cv2.boundingRect(c))
                    for c in contours
                ]
            )
        )

    def process(self, image: np.ndarray) -> DetectorResult:
        image = cv2.resize(image, (1024, int(image.shape[0]/image.shape[1]*1024)) )

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
