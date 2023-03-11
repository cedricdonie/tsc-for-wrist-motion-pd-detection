# Generate predictions from an already trained model
# Author(s): Cedric Donie (cedricdonie@gmail.com)

import argparse
import pickle
from pathlib import Path
import sys
from time import time

import h5py
import keras.models
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sktime_dl.classification import InceptionTimeClassifier
from src.data.utils import data_split
from src.models.baseline import GaussianProcessClassifier
from src.models import phenotype_from_model_name


parser = argparse.ArgumentParser()
parser.add_argument(
    "model_filename", help="Filename of the model to predict.", type=Path
)
parser.add_argument(
    "dataset_filename",
    help="Filename of the data on which to apply the model.",
    type=Path,
)
parser.add_argument(
    "output_filename", help="Filename to put the predictions", type=Path
)
parser.add_argument(
    "--sample-every-nth-row",
    default=1,
    metavar="n",
    type=int,
    help="Only use every nth row of the input dataframe. Can be used to save time.",
)


def load_inceptiontime(filename):
    # InceptionTime
    classifier = InceptionTimeClassifier()
    classifier.label_encoder = LabelEncoder()
    # classifier.onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
    classifier._is_fitted = True
    classifier.model = keras.models.load_model(filename)
    with h5py.File(filename) as hf:
        classifier.label_encoder = pickle.loads(
            hf.attrs[
                "sktime_dl.classification.InceptionTimeClassifier.label_encoder"
            ].tobytes()
        )
        classifier.classes_ = hf.attrs[
            "sktime_dl.classification.InceptionTimeClassifier.classes_"
        ].astype("U")
    return classifier


def load_classifier(filename):
    filename = Path(filename)
    if filename.suffix == ".pkl":
        with open(filename, "rb") as f:
            classifier = pickle.load(f)
    if filename.suffix == ".h5":
        classifier = load_inceptiontime(filename)
    if not hasattr(classifier, "predict_proba"):
        # This is expected for Rocket/MiniRocket. Patch for same method names.
        classifier.predict_proba = classifier.decision_function
    return classifier


if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading classifier from file...")
    t_load_clf_start = time()
    classifier = load_classifier(args.model_filename)
    print("Loaded classifier. Took", time() - t_load_clf_start, "seconds.")

    print("Loading dataset split...")
    t_load_data_start = time()
    X, y = data_split(
        args.dataset_filename,
        multivariate=not isinstance(classifier, GaussianProcessClassifier),
        phenotype=phenotype_from_model_name(args.model_filename),
    )
    print("Loaded dataset. Took", time() - t_load_data_start, "seconds.")
    print("Dataset shape:", X.shape)
    print("Dataset sequence length:", X.iloc[0, 0].size)

    ## Preconditions
    assert X.shape[0] == y.shape[0]

    print("Predicting probabilities...")
    t_predict_start = time()
    probabilities = classifier.predict_proba(X)
    print("Predicted probabilities. Took", time() - t_predict_start, "seconds.")

    print("Predicting classes...")
    t_predict_class_start = time()
    class_predictions = classifier.predict(X)
    print("Predicted classes. Took", time() - t_predict_class_start, "seconds.")

    args.output_filename.parent.mkdir(exist_ok=True)
    print("Saving predictions")

    # Postconditions
    assert probabilities.shape[0] == X.shape[0]
    assert (
        classifier.classes_.size == 2
        and probabilities.ndim == 1
        or probabilities.shape[1] == classifier.classes_.size
    )
    assert class_predictions.shape == y.shape

    t_save_start = time()
    np.savez_compressed(
        args.output_filename,
        probabilities=probabilities,
        class_predictions=class_predictions,
        groundtruth=y,
        classes=classifier.classes_,
    )
    print("Saved predictions. Took", time() - t_save_start, "seconds.")
