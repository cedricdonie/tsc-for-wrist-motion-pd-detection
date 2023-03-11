# Visualize InceptionTime training based on console output of sktime-dl
# Author(s): Cedric Donie (cedricdonie@gmail.com)

import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas
import re

parser = argparse.ArgumentParser()
parser.add_argument("input_log_filename", type=Path)
parser.add_argument("output_plot_filename", type=Path)

plt.style.use(
    [
        "science",
        "src/visualization/main.mplstyle",
        "src/visualization/smallfigures.mplstyle",
    ]
)


def parse_line_for_step(line):
    """
    >>> parse_line_for_step(" 25/43 [================>.............] - ETA: 18s - loss: 1.1468 - accuracy: 0.5288 ")
    {'Step': 25, 'target_steps': 43, 'ETA': 18.0, 'Loss': 1.1468, 'Accuracy': 0.5288}
    >>> parse_line_for_step("26/26 - 0s - loss: 0.8527 - accuracy: 0.6183 - val_loss: 1.3137 - val_accuracy: 0.4884")
    {'Step': 26, 'target_steps': 26, 'Loss': 0.8527, 'Accuracy': 0.6183}
    >>> parse_line_for_step("35/35 - 4s - loss: 1.3926e-05 - accuracy: 1.0000 - val_loss: 5.4181 - val_accuracy: 0.5752")
    {'Step': 35, 'target_steps': 35, 'Loss': 1.3926e-05, 'Accuracy': 1.0}
    """
    result = re.search(
        r"\s*(?P<Step>\d+)/(?P<target_steps>\d+) \[[=>.]+\] - ETA: (?P<ETA>\d+\.\d+|\d+)s - loss: (?P<Loss>\d+\.\d+) - accuracy: (?P<Accuracy>\d+.\d+)",
        line,
    )
    if result is None:
        result = re.search(
            r"(?P<Step>\d+)/(?P<target_steps>\d+) - \d+s - loss: (?P<Loss>\d+.\d+e-?\d+|\d+\.\d+) - accuracy: (?P<Accuracy>\d+.\d+)",
            line,
        )
    if result is None:
        return None
    dictionary = result.groupdict()
    for k in ("Step", "target_steps"):
        dictionary[k] = int(dictionary[k])
    for k in ("ETA", "Loss", "Accuracy"):
        try:
            dictionary[k] = float(dictionary[k])
        except KeyError:
            pass
    return dictionary


def parse_line_for_epoch(line):
    """
    >>> parse_line_for_epoch("Epoch 23/50")
    {'Epoch': 23, 'target_epochs': 50}
    >>> parse_line_for_epoch("Training completed. Took 44.6703827381134 seconds.")
    """
    result = re.match(r"Epoch (\d+)/(\d+)", line)
    if result is None:
        return None
    return {"Epoch": int(result.group(1)), "target_epochs": int(result.group(2))}


def dataframe(filename):
    rows = []
    with open(filename) as f:
        epoch = 0
        for line in f:
            row = {}
            per_step = parse_line_for_step(line)
            per_epoch = parse_line_for_epoch(line)
            if per_step is not None:
                row.update(per_step)
            if per_epoch is not None:
                row.update(per_epoch)
            rows.append(row)
    df = pandas.DataFrame(rows)
    df = df.dropna(how="all")
    if not "Epoch" in df.columns:
        df["Epoch"] = 0
        df["target_epochs"] = 0
    df = df.fillna(method="ffill")
    if "Step" in df.columns:
        df["Progress"] = df["Epoch"] + df["Step"] / df["target_steps"]
    else:
        df["Progress"] = df["Epoch"]
    return df


def plot(df):
    ax = df.plot(x="Progress", y="Accuracy")
    x_max = df.iloc[0]["target_epochs"] + 1
    ax.set_xlim([0, x_max])
    ax.set_ylim([0, 1])
    ax2 = df.plot(x="Progress", y="Loss", secondary_y=True, ax=ax)
    ax.spines["left"].set_color("C0")
    ax2.spines["left"].set_color("C0")
    ax.yaxis.label.set_color("C0")
    ax.tick_params(axis="y", colors="C0")
    ax2.spines["right"].set_color("C1")
    ax2.yaxis.label.set_color("C1")
    ax2.tick_params(axis="y", colors="C1")
    # ax.legend().set_visible(False)
    ax2.set_ylim([0, 0.7])
    ax.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")
    return ax


if __name__ == "__main__":
    args = parser.parse_args()

    df = dataframe(args.input_log_filename)
    ax = plot(df)
    ax.get_figure().savefig(args.output_plot_filename)
