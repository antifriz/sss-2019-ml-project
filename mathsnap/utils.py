import base64

import cv2
import numpy as np


def decompress_image(buffer: str) -> np.ndarray:
    return cv2.imdecode(
        np.fromstring(buffer, 'uint8'),
        cv2.IMREAD_GRAYSCALE
    )


def convert_to_datauri(img):
    _, buffer = cv2.imencode(".jpeg", img)
    b64 = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{b64}"
