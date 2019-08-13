from mathsnap.snappers.core import Snapper, ExtractorSolverSnapper, DummySnapper
from mathsnap.snappers.extractors.core import OCRLayouterExtractor
from mathsnap.snappers.extractors.layouters.layouters import DummyLayouter
from mathsnap.snappers.extractors.ocrs.classifiers.core import DummyClassifier
from mathsnap.snappers.extractors.ocrs.core import DetectorClassifierOCR
from mathsnap.snappers.extractors.ocrs.detectors.core import GreedyDetector
from mathsnap.snappers.solvers.core import DummySolver


def get_snapper() -> Snapper:
    return ExtractorSolverSnapper(
        extractor=OCRLayouterExtractor(
            ocr=DetectorClassifierOCR(
                detector=GreedyDetector(),
                classifier=DummyClassifier(),
            ),
            layouter=DummyLayouter(),
        ),
        solver=DummySolver(),
    )
