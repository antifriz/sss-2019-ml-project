from mathsnap.snappers.core import Snapper, ExtractorSolverSnapper, DummySnapper
from mathsnap.snappers.extractors.core import OCRLayouterExtractor
from mathsnap.snappers.extractors.layouters.layouters import BasicLayouter
from mathsnap.snappers.extractors.ocrs.classifiers.core import KerasClassifier
from mathsnap.snappers.extractors.ocrs.core import DetectorClassifierOCR
from mathsnap.snappers.extractors.ocrs.detectors.core import GreedyDetector
from mathsnap.snappers.solvers.core import DummySolver


def get_snapper() -> Snapper:
    return ExtractorSolverSnapper(
        extractor=OCRLayouterExtractor(
            ocr=DetectorClassifierOCR(
                detector=GreedyDetector(),
                classifier=KerasClassifier("res/keras_model.h5"),
            ),
            layouter=BasicLayouter(),
        ),
        solver=DummySolver(),
    )
