# Author(s): Cedric Donié (cedricdonie@gmail.com)

from json import load
from sktime.datasets import load_basic_motions
from sktime_dl.classification import InceptionTimeClassifier
import numpy as np


def test_validation_has_fewer_classes_than_test():
    """
    Expected to pass even unexpectedly.
    See https://github.com/sktime/sktime-dl/issues/131
    """
    X, _ = load_basic_motions(return_X_y=True, split="train")
    X_val, _ = load_basic_motions(return_X_y=True, split="test")

    y = np.ones(len(X))
    y[0] = 0
    y[1] = 2

    y_val = np.ones_like(y)
    y_val[0] = 0

    clf = InceptionTimeClassifier(nb_epochs=2)

    clf.fit(X, y, validation_data=(X_val, y_val))
