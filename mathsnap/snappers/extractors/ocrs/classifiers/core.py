from typing import NamedTuple
from keras.models import load_model
import numpy as np
import cv2



class ClassifierResult(NamedTuple):
    label: str


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

class KerasClassifier(Classifier):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def process(self, image: np.ndarray) -> ClassifierResult:
        model = load_model(self.file_name)

        img = cv2.resize(image, (28, 28))
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 0)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        prediction = list(model.predict(img, 1)[0]).index(1);

        return ClassifierResult(label=str(prediction))

