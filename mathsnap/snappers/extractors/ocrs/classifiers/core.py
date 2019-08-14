from typing import NamedTuple
from keras.models import load_model
import numpy as np
import cv2
from mathsnap.utils import convert_to_datauri



class ClassifierResult(NamedTuple):
    label: str
    image: np.ndarray


class Classifier:
    def process(self, image: np.ndarray) -> ClassifierResult:
        raise NotImplementedError()


class DummyClassifier(Classifier):
    def process(self, image: np.ndarray) -> ClassifierResult:
        return ClassifierResult(
            label='2'
        )


# TODO : Improve resizing
# TODO : get prediction
# TODO : Signs

class KerasClassifier(Classifier):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def process(self, image: np.ndarray) -> ClassifierResult:
        model = load_model(self.file_name)

        img = cv2.resize(image, (20, 20))
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

        return_img = convert_to_datauri(img)

        img = img[np.newaxis, :, :, np.newaxis]

        string = "0123456789-+"
        prediction = string[list(model.predict(img, 1)[0]).index(1)]

        return ClassifierResult(
            label=prediction,
            image=return_img
        )

