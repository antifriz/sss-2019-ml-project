import cv2
import numpy as np


def decompress_image(buffer: str) -> np.ndarray:
    return cv2.imdecode(
        np.fromstring(buffer, 'uint8'),
        cv2.IMREAD_UNCHANGED,
    )
