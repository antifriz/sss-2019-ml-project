from typing import NamedTuple
from keras.models import load_model
import numpy as np
import cv2
from mathsnap.utils import convert_to_datauri
import tensorflow as tf
from scipy import ndimage



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
    def __init__(self, file_name: str, classes_char: str):
        self.file_name = file_name
        self.model = load_model(self.file_name)
        self.graph = tf.get_default_graph()
        self.classes = classes_char

    def resize_padding(self, image: np.ndarray) -> np.ndarray:
        n = 20
        a = image.shape[0] / image.shape[1]
        h = min(n, int(round(n * a)))
        w = min(n, int(round(n / a)))

        # improvement
        image = cv2.resize(image, (4*w, 4*h))
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 20)

        kernel = np.ones((3, 3), 'uint8')
        image = cv2.dilate(image, kernel)

        # arranging
        output = cv2.resize(image, (w, h))
        c = ndimage.measurements.center_of_mass(255 - output)
        c_y = h - int(round(c[0]))
        c_x = w - int(round(c[1]))

        # binarize
        output = cv2.threshold(output, 200, 255, cv2.THRESH_BINARY)[1]

        # padding
        return cv2.copyMakeBorder(output, 14-c_y, 14-(h-c_y), 14-c_x, 14-(w-c_x), cv2.BORDER_CONSTANT, value=0)

    def process(self, img: np.ndarray) -> ClassifierResult:
        img = self.resize_padding(img)

        with self.graph.as_default():
            predictions = self.model.predict(
                (img / 255)[np.newaxis, :, :, np.newaxis], 1
            )[0]

        return ClassifierResult(
            label=self.classes[np.argmax(predictions)],
            image=convert_to_datauri(img)
        )

