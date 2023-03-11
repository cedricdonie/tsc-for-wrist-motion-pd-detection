# Author(s): Cedric Donié (cedricdonie@gmail.com)

import argparse
import json
import sys
from keras.utils.layer_utils import count_params
import numpy as np
import sklearn.model_selection
import src.data.loading
import src.data.utils
from cloudpathlib import AnyPath as Path
from sktime_dl.classification import InceptionTimeClassifier
from src.data.global_constants import _CONSTS, _PATHS
from src.models.model_selection import StratifiedGroupKFold
from src.visualization.metrics import NpEncoder
import src.visualization.metrics
import src.models.train_model
import tensorflow_addons as tfa

parser = argparse.ArgumentParser(
    epilog="""Example:
python src/models/crossval_score.py \\
    data/processed/sktime/GENEActiv \\
    _train-val_windowsz-30.0_overlap-0.0.data.pkl \\
    test.json \\
    0 \\
    1 \\
    --vm-count 4 \\
    --sample-every-nth-row 10 \\
    --epochs 10""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument("input_parameter_config_path", type=Path)
parser.add_argument("vm_number", help="Which VM number to run on.", type=int)
parser.add_argument("job_number", help="Which job to run on the given VM.", type=int)
parser.add_argument(
    "output_results_file",
    type=Path,
    help="JSON file to store the output models with parameters. Otherwise, print to stdout.",
    nargs="?",
)
parser.add_argument(
    "--vm-count",
    metavar="N",
    help="The number of VMs/instances to split training among",
    default=1,
    type=int,
)
parser.add_argument(
    "--crossval-splits",
    type=int,
    metavar="k",
    help="Number of folds to use for k-fold cross-validation.",
    default=5,
)
parser.add_argument(
    "--epochs",
    type=int,
    metavar="n",
    help="Train each candidate for n epochs.",
    default=600,
)
parser.add_argument(
    "--sample-every-nth-row",
    default=1,
    metavar="n",
    type=int,
    help="Only use every nth row of the input dataframe. Can be used to save time.",
)
parser.add_argument(
    "--phenotype", choices=["tremor", "dyskinesia", "bradykinesia"], default="tremor"
)


def dataset_path_from_window_length(window_length):
    """
    >>> dataset_path_from_window_length(22.333)
    'data/processed/sktime/GENEActiv_train-val_windowsz-22.33_overlap-11.17.data.pkl'
    """
    return f"data/processed/sktime/GENEActiv_train-val_windowsz-{window_length:1.2f}_overlap-{window_length * 0.5:1.2f}.data.pkl"


METRICS_MULTICLASS = {
    "mAP": src.visualization.metrics.mean_average_precision_scorer,
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}

METRICS_BINARY = [
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
]

if __name__ == "__main__":
    args = parser.parse_args()
    assert args.vm_number <= args.vm_count
    assert args.vm_number >= 0
    assert args.job_number >= 0
    assert args.input_parameter_config_path.suffix == ".json"
    print(args)

    with args.input_parameter_config_path.open() as f:
        params_overall = json.load(f)

    params_per_vm = np.array_split(params_overall, args.vm_count)
    params_for_job = params_per_vm[args.vm_number][args.job_number]
    # TODO: Don't always assume that we are running on cloud. Try/except?
    dataset_path = (
        _PATHS.gcp_bucket_uri
        + "/"
        + dataset_path_from_window_length(params_for_job["window_length"])
    )

    print(
        "Running job number",
        args.job_number,
        "on VM number",
        args.vm_number,
        "with params",
        params_for_job,
        "and dataset",
        dataset_path,
        "...",
    )

    if args.output_results_file and args.output_results_file.is_file():
        print(
            "Skipping job",
            args.job_number,
            "on VM",
            args.vm_number,
            "since",
            args.output_results_file,
            "already exists.",
        )
        if args.output_results_file.stat().st_size < 1000:
            print(
                "WARNING: Exsting file",
                args.output_results_file,
                "is quite small, perhaps it is out of date?",
            )
        sys.exit(0)

    print("Loading data...")
    X, y, groups = src.data.utils.data_split(
        dataset_path, return_subjects=True, phenotype=args.phenotype
    )
    X = X.iloc[:: args.sample_every_nth_row, :]
    y = y.iloc[:: args.sample_every_nth_row]
    # Will cause error in mAP calculation if left as string
    y = y.astype(np.int64)
    groups = groups.iloc[:: args.sample_every_nth_row]

    params_inceptiontime = dict(params_for_job)
    del params_inceptiontime["window_length"]
    classifier = InceptionTimeClassifier(
        verbose=2,
        random_state=_CONSTS.seed,
        nb_epochs=args.epochs,
        **params_inceptiontime,
    )
    cv = StratifiedGroupKFold(
        shuffle=True, random_state=_CONSTS.seed, n_splits=args.crossval_splits
    )

    metrics = (
        METRICS_BINARY
        if args.phenotype in ("dyskinesia", "bradykinesia")
        else METRICS_MULTICLASS
    )

    print("Starting to cross-validate...")
    cv_results = sklearn.model_selection.cross_validate(
        classifier,
        X,
        y,
        groups=groups,
        cv=cv,
        verbose=2,
        return_estimator=True,
        scoring=metrics,
        error_score="raise",
    )

    models = cv_results["estimator"]
    trainable_count = [
        count_params(est.model.trainable_weights) for est in cv_results["estimator"]
    ]
    non_trainable_count = [
        count_params(est.model.non_trainable_weights) for est in cv_results["estimator"]
    ]
    history = [dict(est.history.history) for est in cv_results["estimator"]]
    del cv_results["estimator"]
    cv_results["params"] = params_for_job
    cv_results["nb_trainable_params"] = trainable_count
    cv_results["nb_non_trainable_params"] = non_trainable_count
    cv_results["history"] = history
    print("Cross-validated. Saving results...")
    print("Results:")
    print("---")
    print(json.dumps(cv_results, cls=NpEncoder))
    print("---")

    if args.output_results_file:
        args.output_results_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_results_file.open("w") as f:
            json.dump(cv_results, f, cls=NpEncoder)
        print(
            "Saved job number",
            args.job_number,
            "on VM number",
            args.vm_number,
            "to",
            args.output_results_file,
        )
