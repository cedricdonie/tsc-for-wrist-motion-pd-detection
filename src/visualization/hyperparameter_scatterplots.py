# Plot the impact of a single hyperparameter.
# Author(s): Cedric Donie (cedricdonie@gmail.com)

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

plt.style.use(["science", "src/visualization/main.mplstyle"])

HYPERPARAMETERS = ("depth", "kernel_size", "nb_filters", "window_length")

parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path, help="CSV file with results.")
parser.add_argument("mean_results", type=Path, help="CSV file with mean results.")
for h in HYPERPARAMETERS:
    parser.add_argument(
        "--" + h.replace("_", "-"),
        default=None,
        type=Path,
        help=f"Path to the {h} plot.",
    )

if __name__ == "__main__":
    print(sys.argv)
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    mean_results = pd.read_csv(args.mean_results)

    for x in ("depth", "kernel_size", "nb_filters", "window_length"):
        if vars(args)[x] is None:
            continue
        fig, ax = plt.subplots()
        sns.scatterplot(data=results.dropna(), x=x, y="test_mAP", ax=ax)
        sns.regplot(data=mean_results.dropna(), x=x, y="test_mAP", ax=ax, color="C1")
        rho, p = scipy.stats.spearmanr(
            results.dropna()[x], results.dropna()["test_mAP"]
        )
        ax.text(
            0.1,
            0.9,
            rf"$\rho = {rho:.2f}$",
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="left",
        )
        # mean_results.plot.scatter(x=x, y="test_mAP", ax=ax, color="red", label="CV mean")
        if x == "window_length":
            ax.set_xlabel("Window length / s")
        ax.set_ylabel("mAP")
        ax.get_figure().legend(
            labels=[
                "_nolegend_",
                "all cross-validation folds",
                "cross-validation mean",
            ],
            loc="upper center",
            ncol=2,
        )
        ax.get_figure().set_size_inches(
            (fig.get_size_inches()[0], fig.get_size_inches()[1] - 0.5)
        )
        ax.get_figure().savefig(vars(args)[x])
