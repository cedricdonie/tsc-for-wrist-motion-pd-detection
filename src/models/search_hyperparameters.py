# Hyperparameter Tuning
# Author(s): Cedric Donié (cedricdonie@gmail.com)
#
# Experiments with scikit-learn's hyperparameter search API

import argparse
import sys
import time
from cloudpathlib import AnyPath as Path
import sklearn

import tensorflow as tf

# https://github.com/tensorflow/tensorflow/issues/36508
_physical_devices = tf.config.list_physical_devices("GPU")
for _device in _physical_devices:
    tf.config.experimental.set_memory_growth(_device, enable=True)

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


import numpy as np
import pandas as pd
import src.data.loading
import src.data.utils
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import parallel_backend
from sktime_dl.classification import InceptionTimeClassifier
from src.data.global_constants import _CONSTS
from src.models.model_selection import StratifiedGroupKFold

parser = argparse.ArgumentParser()
parser.add_argument("input_dataset_path", type=Path)
parser.add_argument(
    "output_results_file",
    type=Path,
    help="CSV file to store the output models with parameters. Otherwise, print to stdout.",
    nargs="?",
)
parser.add_argument(
    "--iterations",
    type=int,
    metavar="n",
    help="Number of random search iterations to use.",
    default=59,
)
parser.add_argument(
    "--crossval-splits",
    type=int,
    metavar="k",
    help="Number of folds to use for k-fold cross-validation.",
    default=5,
)
parser.add_argument(
    "--jobs",
    type=int,
    metavar="n",
    default="-1",
    help="Number of jobs to run in parallel."
    "n=-1 means using one job per GPU available.",
)
parser.add_argument(
    "--sample-every-nth-row",
    default=1,
    metavar="n",
    type=int,
    help="Only use every nth row of the input dataframe. Can be used to save time.",
)

param_grid = {
    "nb_filters": 2 ** np.arange(1, 7),
    "kernel_size": list(range(8, 256)),
    "depth": np.arange(1, 12),
}


if __name__ == "__main__":
    print("Args:", sys.argv)
    args = parser.parse_args()

    classifier = InceptionTimeClassifier(
        verbose=True, random_state=_CONSTS.seed, nb_epochs=50  # 600
    )

    cv = StratifiedGroupKFold(
        shuffle=True, random_state=_CONSTS.seed, n_splits=args.crossval_splits
    )

    n_gpus = len(tf.config.list_physical_devices("GPU"))
    print("Using", n_gpus, "GPUs.")
    n_jobs = n_gpus if n_gpus else -1

    with sklearn.utils.parallel_backend("threading"):
        rs = RandomizedSearchCV(
            classifier,
            param_distributions=param_grid,
            random_state=_CONSTS.seed,
            verbose=True,
            cv=cv,
            n_iter=args.iterations,
            refit=False,
            n_jobs=n_jobs,
            pre_dispatch="n_jobs",
        )

        print("Parameter grid:")
        print(param_grid)

        print("Random search parameters:")
        print(rs.get_params())

        print("Loading data")
        X, y, groups = src.data.utils.data_split(
            args.input_dataset_path, return_subjects=True
        )
        X = X.iloc[:: args.sample_every_nth_row, :]
        y = y.iloc[:: args.sample_every_nth_row]
        groups = groups.iloc[:: args.sample_every_nth_row]
        print("Loaded data.")

        print("Starting to search...")

        t0 = time.time()
        rs.fit(X, y, groups)
        t1 = time.time()
        print("Completed random search. Saving results...")

        results_df = pd.DataFrame(rs.cv_results_).sort_values(by="rank_test_score")

    print("Results:")
    print("---")
    print(results_df.to_csv())
    print("---")
    if args.output_results_file:
        args.output_results_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_results_file.open("w", newline="") as f:
            results_df.to_csv(f)
        print("Saved results to", args.output_results_file)
    print("Best parameters are", rs.best_params_, "with score", rs.best_score_)
