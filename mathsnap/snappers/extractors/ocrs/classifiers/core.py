from typing import NamedTuple
from keras.models import load_model

import numpy as np



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


# TODO : Define the image size
class KerasClassifier(Classifier):
    def __init__(self, file_name: str):
        self.model = load_model(file_name)

    def process(self, image: np.ndarray) -> ClassifierResult:
        return ClassifierResult(label=self.model.perdict(image))

