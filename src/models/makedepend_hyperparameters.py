# Author(s): Cedric Donié (cedricdonie@gmail.com)

import argparse
import json
from cloudpathlib import AnyPath as Path
from src.models.crossval_score import dataset_path_from_window_length

parser = argparse.ArgumentParser()
parser.add_argument("input_hyperparameter_config_path", type=Path)
parser.add_argument(
    "--filename-template",
    type=str,
    default="data/processed/sktime/GENEActiv_train-val_windowsz-{}_overlap-{}.data.pkl",
)

if __name__ == "__main__":
    args = parser.parse_args()

    with args.input_hyperparameter_config_path.open() as f:
        j = json.load(f)

    window_lengths = [i["window_length"] for i in j]
    filenames = [dataset_path_from_window_length(wl) for wl in window_lengths]

    print(" ".join(filenames))
