# Test for model training/loading reproducability
# Author(s): Cedric Donie (cedricdonie@gmail.com)

from mimetypes import init
import pandas
from src.models.train_model import init_classifier, train, save_classifier
from src.models.predict_model import load_classifier
import numpy as np
from sktime.datasets import load_basic_motions


def test_inceptiontime_saving_and_loading_is_equivalent(tmp_path):
    """
    Saving the inceptiontme model is a little bit hacky.
    I need to make sure that the saved model is roughly equivalent to the trained model.
    """

    # Load tiny dataset
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    print(X_train.head())
    print(X_test.head())
    save_filename = tmp_path / "test.h5"

    classifier_in_memory = init_classifier("inceptiontime")
    classifier_in_memory.nb_epochs = 2  # Training all epochs takes too long.
    classifier_in_memory = train(classifier_in_memory, X_train, y_train)

    save_classifier(classifier_in_memory, filename=tmp_path / "test.h5")
    classifier_from_file = load_classifier(filename=save_filename)

    pred_proba_in_memory = classifier_in_memory.predict_proba(X_test)
    pred_in_memory = classifier_in_memory.predict(X_test)

    pred_proba_from_file = classifier_from_file.predict_proba(X_test)
    pred_from_file = classifier_from_file.predict(X_test)

    assert classifier_in_memory.classes_ is not None
    assert classifier_from_file.classes_ is not None
    np.testing.assert_equal(
        classifier_from_file.classes_, classifier_in_memory.classes_
    )
    assert classifier_from_file.classes_.dtype == classifier_in_memory.classes_.dtype
    assert np.all(np.isclose(pred_proba_in_memory, pred_proba_from_file))
    assert np.all(pred_in_memory == pred_from_file)
