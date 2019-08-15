from typing import NamedTuple
from keras.models import load_model
import numpy as np
import cv2
from mathsnap.utils import convert_to_datauri
import tensorflow as tf


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


# TODO: Equalize amount of samples for each symbol - np.repeat

class KerasClassifier(Classifier):
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.model = load_model(self.file_name)
        self.graph = tf.get_default_graph()

    def process(self, image: np.ndarray) -> ClassifierResult:
        img = cv2.resize(image, (20, 20)) # TODO: keep ratio!!!
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

        return_img = convert_to_datauri(img)

        img = img[np.newaxis, :, :, np.newaxis]

        string = "0123456789+-/*"

        with self.graph.as_default():
            predictions = self.model.predict(img, 1)[0]

        print(predictions)

        index = np.argmax(predictions)
        prediction = string[index]

        return ClassifierResult(
            label=prediction,
            image=return_img
        )

