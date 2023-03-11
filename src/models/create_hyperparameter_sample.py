# Author(s): Cedric Donié (cedricdonie@gmail.com)

import argparse
import json

import numpy as np
import scipy.stats
from cloudpathlib import AnyPath as Path
from sklearn.model_selection import ParameterSampler
from src.data.global_constants import _CONSTS
from src.visualization.metrics import NpEncoder

parser = argparse.ArgumentParser()
parser.add_argument("output_filename", type=Path)
parser.add_argument(
    "--iterations",
    type=int,
    metavar="n",
    help="Number of random search iterations to use.",
    default=60,
)

param_grid = {
    "nb_filters": 2 ** np.arange(1, 7),
    "kernel_size": list(range(8, 256)),
    "depth": np.arange(1, 12),
    "window_length": scipy.stats.uniform(loc=3, scale=27),
}

if __name__ == "__main__":
    args = parser.parse_args()
    param_combinations = list(
        ParameterSampler(param_grid, args.iterations, random_state=_CONSTS.seed + 1)
    )
    assert len(param_combinations) == args.iterations
    with args.output_filename.open("w") as f:
        json.dump(param_combinations, f, cls=NpEncoder, indent=4)
