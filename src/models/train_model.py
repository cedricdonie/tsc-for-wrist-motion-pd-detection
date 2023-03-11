# Train various time series classification models and save them to file.
# Author(s): Cedric Donie (cedricdonie@gmail.com)

import argparse
import pickle
import sys
import tempfile
import time
from cloudpathlib import AnyPath as Path, CloudPath

import h5py
import keras.callbacks
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import MiniRocketMultivariate, Rocket
from sktime_dl.classification import InceptionTimeClassifier
from src.data.global_constants import _CONSTS
from src.data.utils import data_split
from src.models.baseline import GaussianProcessClassifier
from tensorflow_addons.callbacks import TimeStopping

BEST_PARAMS = {
    "tremor": {"depth": 3, "kernel_size": 20, "nb_filters": 2},
    "dyskinesia": {"depth": 3, "kernel_size": 17, "nb_filters": 4},
    "bradykinesia": {"depth": 7, "kernel_size": 183, "nb_filters": 64},
}

parser = argparse.ArgumentParser()
parser.add_argument("input_data_file")
parser.add_argument("output_model_file", type=Path)
parser.add_argument("phenotype", choices=["tremor", "dyskinesia", "bradykinesia"])
parser.add_argument("--classifier", default="inceptiontime")
parser.add_argument(
    "--sample-every-nth-row",
    default=1,
    metavar="n",
    type=int,
    help="Only use every nth row of the input dataframe. Can be used to save time.",
)
parser.add_argument("--validation-data-file", default=None)
parser.add_argument("--epochs", type=int, default=1500)
parser.add_argument(
    "--use-best-params",
    action="store_true",
    help="Use the best params from the hyperparameter search instead of defaults for InceptionTime",
    default=False,
)
parser.add_argument("--random-state", type=int, default=_CONSTS.seed)
parser.add_argument(
    "--no-checkpointing",
    action="store_false",
    default=True,
    dest="checkpointing",
    help="Don't checkpoints every few epochs (only for InceptionTime)",
)


def save_dl_classifier(filename, classifier: InceptionTimeClassifier, model=None):
    if model == None:
        model = classifier.model
    if isinstance(filename, CloudPath):
        cloud_filename = filename
        _, filename = tempfile.mkstemp()
    else:
        cloud_filename = None
    model.save(filename, save_format="h5")
    # We need to save the label encoder to predict later.
    # Storing it in the same same Keras model.
    with h5py.File(filename, mode="r+") as hf:
        hf.attrs.create(
            "sktime_dl.classification.InceptionTimeClassifier.label_encoder",
            np.void(pickle.dumps(classifier.label_encoder)),
        )
        classes = classifier.classes_
        assert classes is not None
        if not np.issubdtype(classes.dtype, np.bool):
            # Categoricals not supported. Use int instead of string.
            classes = classes.astype(np.int8)
        hf.attrs.create(
            "sktime_dl.classification.InceptionTimeClassifier.classes_", classes
        )
    if cloud_filename is not None:
        cloud_filename.upload_from(filename)


class CheckPointingCallback(keras.callbacks.Callback):
    def __init__(self, classifier, final_filename, save_every_n_epochs=10):
        self.classifier = classifier
        self.final_filename = final_filename
        self.save_every_n_epochs = save_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_every_n_epochs != 0:
            return

        filename = self.final_filename.with_suffix("").with_suffix(
            f".epoch-{epoch:04d}.model.h5"
        )
        save_dl_classifier(filename, classifier, model=self.model)
        print("Saved checkpoint:", filename)


class EarlyStoppingEnhanced(keras.callbacks.EarlyStopping):
    """
    Stop only after a minimum number of epochs.

    Source: https://stackoverflow.com/a/46294390
    """

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_epoch=0,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
        )
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


class TerminateOnBaseline(keras.callbacks.Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline"""

    def __init__(self, monitor="acc", baseline=0.9, mode="max", min_consecutive=1):
        assert mode in ("max", "min")
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
        self.mode = mode
        self.min_consecutive = min_consecutive
        self.consecutive_baseline_count = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        qty = logs.get(self.monitor)
        if qty is not None:
            if (self.mode == "max" and qty >= self.baseline) or (
                self.mode == "min" and qty <= self.baseline
            ):
                self.consecutive_baseline_count += 1
            else:
                self.consecutive_baseline_count = 0
            if self.consecutive_baseline_count >= self.min_consecutive:
                print("Epoch %d: Reached baseline, terminating training" % (epoch))
                self.model.stop_training = True


def init_classifier(
    classifier_name, nb_epochs=1500, additional_params={}, random_state=_CONSTS.seed
):
    if classifier_name == "rocket":
        return make_pipeline(
            Rocket(random_state=random_state),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
        )
    if classifier_name == "minirocket":
        return make_pipeline(
            MiniRocketMultivariate(random_state=random_state),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
        )
    if classifier_name == "inceptiontime":
        return InceptionTimeClassifier(
            verbose=2,
            random_state=random_state,
            nb_epochs=nb_epochs,
            **additional_params,
        )
    if classifier_name == "gp":
        return GaussianProcessClassifier(verbose=2, nb_epochs=100)

    raise NotImplemented(f"Classifier {classifier_name} is not implemented yet!")


def train(classifier, X, y, X_val=None, y_val=None):
    if X_val is None and y_val is None:
        # sklearn Pipeline does not accept validation data.
        classifier.fit(X, y)
    else:
        classifier.fit(X, y, validation_X=X_val, validation_y=y_val)
    return classifier


def score(classifier, X, y):
    return classifier.score(X, y)


def save_classifier(classifier, filename):
    filename.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(classifier, InceptionTimeClassifier):
        # InceptionTime allows this
        save_dl_classifier(filename, classifier)
    else:
        print(filename)
        with filename.open("wb") as f:
            pickle.dump(classifier, f)


if __name__ == "__main__":
    print(sys.argv)
    args = parser.parse_args()

    print("Splitting data")
    X, y = data_split(
        args.input_data_file,
        phenotype=args.phenotype,
        multivariate=args.classifier != "gp",
    )
    X_val = None
    y_val = None

    # Sample every nth row only.
    X = X.iloc[:: args.sample_every_nth_row, :]
    y = y.iloc[:: args.sample_every_nth_row]

    if args.validation_data_file is not None:
        X_val, y_val = data_split(args.validation_data_file, phenotype=args.phenotype)

    print("Making classifier")
    t_mc_start = time.time()
    additional_params = {}
    if args.use_best_params:
        additional_params = BEST_PARAMS[args.phenotype]
    classifier = init_classifier(
        args.classifier,
        nb_epochs=args.epochs,
        additional_params=additional_params,
        random_state=args.random_state,
    )
    if args.classifier == "inceptiontime" and args.checkpointing:
        classifier.callbacks = [
            CheckPointingCallback(classifier, args.output_model_file)
        ]
    print("Made classifier. Took", time.time() - t_mc_start, "seconds.")
    if hasattr(classifier, "random_state"):
        print("Random state", classifier.random_state, "set for classifier.")
    else:
        print("Random state unknown.")

    print("Starting to train.")
    t0 = time.time()
    classifier = train(classifier, X, y, X_val=X_val, y_val=y_val)
    print("Training completed. Took", time.time() - t0, "seconds.")
    save_classifier(classifier, args.output_model_file)
    print("Classifier saved")
