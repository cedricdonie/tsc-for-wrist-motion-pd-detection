# Author(s): Cedric Donié (cedricdonie@gmail.com)

import argparse
import re
from statistics import mean
from cloudpathlib import GSPath
from pathlib import Path
import pandas as pd
import json
from src.data.global_constants import _PATHS

parser = argparse.ArgumentParser()
parser.add_argument("phenotype", choices=["tremor", "dyskinesia", "bradykinesia"])
parser.add_argument("results_output", type=Path)
parser.add_argument("mean_results_output", type=Path)
parser.add_argument(
    "--subdirectory",
    type=str,
    default="",
    help="Subdirectory to search instead of the top-level phenotype directory."
    "E.g. for experiments.",
)


def parse_cv_run_filename(filename):
    """
    >>> parse_cv_run_filename("tmp/models/inceptiontime/hyperparameter_tuning/tremor/GENEActiv_train-val.v-0_j-8.json")
    (0, 8)
    """
    result = re.search(r"\.v-(\d+)_j-(\d+)\.json", str(filename))
    return int(result.group(1)), int(result.group(2))


def load_results(phenotype, subdirectory=""):
    rows = []
    directory = (
        GSPath(_PATHS.gcp_bucket_uri)
        / "models/inceptiontime/hyperparameter_tuning"
        / phenotype
        / subdirectory
    )
    for path in directory.iterdir():
        if path.suffix != ".json":
            # Globbing is buggy, see https://github.com/drivendataorg/cloudpathlib/issues/210
            continue
        print(path)
        v, j = parse_cv_run_filename(path)
        d = json.loads(path.read_text())
        if not "nb_trainable_params" in d:
            # Remove e.g., for reports/data/dyskinesia_GENEActiv_samewindowsize_meanresults.csv
            continue
        for i, _ in enumerate(d["fit_time"]):
            row = {"vm_number": v, "job_number": j}
            row["fold_number"] = i
            row["reduced_window_range"] = path.parent.name == "tremor"
            row.update(d["params"])
            row.update({k: v[i] for k, v in d.items() if isinstance(v, list)})
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["vm_number", "job_number", "fold_number"], ascending=False
    ).reset_index(drop=True)
    df["model_number"] = df.groupby(["vm_number", "job_number"]).ngroup()
    return df


if __name__ == "__main__":
    args = parser.parse_args()

    results = load_results(args.phenotype, args.subdirectory)
    results.to_csv(args.results_output)

    metric_name = "test_mAP" if args.phenotype == "tremor" else "test_average_precision"

    mean_results = (
        results.groupby(["vm_number", "job_number", "reduced_window_range"])
        .agg(pd.Series.mean, skipna=False)
        .sort_values(by=metric_name, ascending=False)
    )
    std_results = (
        results.groupby(["vm_number", "job_number", "reduced_window_range"])
        .agg(pd.Series.std, skipna=False)
        .sort_values(by=metric_name, ascending=False)
    )
    if args.phenotype == "tremor":
        mean_results["test_mAP_std"] = std_results["test_mAP"]
    else:
        mean_results["test_average_precision_std"] = std_results[
            "test_average_precision"
        ]
    mean_results["test_accuracy_std"] = std_results["test_accuracy"]
    mean_results["test_balanced_accuracy_std"] = std_results["test_balanced_accuracy"]
    mean_results.to_csv(args.mean_results_output)
