# Author(s): Cedric Donié (cedricdonie@gmail.com)

import argparse
import json
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "metrics", nargs="+", help="JSON metrics files for each model", type=Path
)
parser.add_argument("--metric-names", nargs="+", type=str)
parser.add_argument("--output", type=Path)
parser.add_argument("--legend-name", default="name")


def metrics_to_dataframe(metrics, metric_names, name_key="name"):
    rows = []
    for metric, name in zip(metrics, metric_names):
        d = json.loads(metric.read_text())
        row_overall = d["overall"]
        row_overall["Class"] = "overall"
        row_overall[name_key] = name
        rows.append(row_overall)
        for k, v in d["classwise"].items():
            row_cw = v
            row_cw["Class"] = k
            row_cw[name_key] = name
            rows.append(row_cw)
    return pd.DataFrame(rows)


def label_bars(ax):
    """
    Add label to all the bars in `ax`

    See https://stackoverflow.com/a/49820775
    """
    for p in ax.patches:
        width = p.get_width()  # get bar length
        ax.text(
            width + 0.001,  # set the text at 1 unit right of the bar
            p.get_y() + p.get_height() / 2,  # get Y coordinate + X coordinate / 2
            "{:1.2f}".format(width),  # set variable to display, 2 decimals
            ha="left",
            va="center",
        )


if __name__ == "__main__":
    args = parser.parse_args()
    metric_names = (
        args.metric_names
        if args.metric_names
        else list(range(1, len(args.metrics) + 1))
    )
    assert len(metric_names) == len(args.metrics)

    df = metrics_to_dataframe(args.metrics, metric_names, name_key=args.legend_name)
    mdf = df[df["Class"] == "overall"].melt(
        id_vars=[args.legend_name, "Class"],
        value_vars=["mAP", "accuracy", "balanced accuracy"],
        var_name="Metric",
        value_name="Score",
    )
    ax = sns.barplot(data=mdf, hue=args.legend_name, x="Score", y="Metric")
    label_bars(ax)
    ax.set_xlim([0, 1])

    fig = ax.get_figure()
    if args.output is None:
        fig.show()
    else:
        fig.savefig(args.output, bbox_inches="tight")
