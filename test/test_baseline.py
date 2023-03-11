# Author(s): Cedric Donié (cedricdonie@gmail.com)

import sktime.datasets
import pandas as pd
import numpy as np
from src.models.baseline import GaussianProcessClassifier
import src.models.train_model
import src.models.predict_model
import torch


def test_gaussian_process_classifier():
    X, y = sktime.datasets.load_basic_motions(return_X_y=True, split="train")
    Xt = X[["dim_0"]]

    clf = GaussianProcessClassifier(nb_epochs=10)
    clf.fit(Xt, y)

    X_val, y_val = sktime.datasets.load_basic_motions(return_X_y=True, split="test")
    X_val = X_val[["dim_0"]]
    pred_proba = clf.predict_proba(X_val)
    pred = clf.predict(X_val)
    score = clf.score(X_val, y_val)

    assert score > 0.55


def test_gaussian_process_saving_and_loading_is_equivalent(tmp_path):
    X, y = sktime.datasets.load_basic_motions(return_X_y=True, split="train")
    Xt = X[["dim_1"]]
    X_val, y_val = sktime.datasets.load_basic_motions(return_X_y=True, split="test")
    X_val = X_val[["dim_0"]]

    clf = GaussianProcessClassifier(nb_epochs=10)
    clf.fit(Xt, y)
    pred_proba = clf.predict_proba(X_val)
    src.models.train_model.save_classifier(clf, tmp_path / "model.pkl")

    clf2 = src.models.predict_model.load_classifier(tmp_path / "model.pkl")
    pred_proba2 = clf2.predict_proba(X_val)

    assert clf != clf2
    np.testing.assert_equal(pred_proba, pred_proba2)


def test_reshape():
    # Format datapoint d<n>, wavelet level w<n>, feature f<n>
    actual = np.array(
        [
            ["d0-w0-f0", "d0-w0-f1", "d0-w0-f2"],
            ["d0-w1-f0", "d0-w1-f1", "d0-w1-f2"],
            ["d0-w2-f0", "d0-w2-f1", "d0-w2-f2"],
            ["d1-w0-f0", "d1-w0-f1", "d1-w0-f2"],
            ["d1-w1-f0", "d1-w1-f1", "d1-w1-f2"],
            ["d1-w2-f0", "d1-w2-f1", "d1-w2-f2"],
        ]
    )

    desired = np.array(
        [
            [
                "d0-w0-f0",
                "d0-w0-f1",
                "d0-w0-f2",
                "d0-w1-f0",
                "d0-w1-f1",
                "d0-w1-f2",
                "d0-w2-f0",
                "d0-w2-f1",
                "d0-w2-f2",
            ],
            [
                "d1-w0-f0",
                "d1-w0-f1",
                "d1-w0-f2",
                "d1-w1-f0",
                "d1-w1-f1",
                "d1-w1-f2",
                "d1-w2-f0",
                "d1-w2-f1",
                "d1-w2-f2",
            ],
        ]
    )

    reshaped = actual.reshape((desired.shape[0], -1))

    np.testing.assert_equal(reshaped, desired)
