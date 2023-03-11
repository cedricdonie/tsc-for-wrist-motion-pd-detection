# Author(s): Cedric Donié (cedricdonie@gmail.com)

import sys
from numpy import DataSource
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import re
import json


plt.style.use(
    [
        "science",
        "src/visualization/main.mplstyle",
        "src/visualization/midfigures.mplstyle",
    ]
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "input_model_path",
    type=Path,
    help="The name of the model.h5 file, of which the checkpoints should be evaluated.",
)
parser.add_argument(
    "eval_dataset_name",
    type=str,
    help='E.g., "GENEActiv_val_windowsz-5.0_overlap-4.0.metrics.json"',
)
parser.add_argument("output_plot_filename", type=Path)
parser.add_argument(
    "--makedeps", action="store_true", help="Just print dependencies and exit."
)


def parse_line_for_step(line):
    """
    >>> parse_line_for_step(" 25/43 [================>.............] - ETA: 18s - loss: 1.1468 - accuracy: 0.5288 ")
    {'Step': 25, 'target_steps': 43, 'ETA': 18.0, 'train loss': 1.1468, 'train accuracy': 0.5288}
    >>> parse_line_for_step("26/26 - 0s - loss: 0.8527 - accuracy: 0.6183 - val_loss: 1.3137 - val_accuracy: 0.4884")
    {'Step': 26, 'target_steps': 26, 'train loss': 0.8527, 'train accuracy': 0.6183}
    >>> parse_line_for_step("35/35 - 4s - loss: 1.3926e-05 - accuracy: 1.0000 - val_loss: 5.4181 - val_accuracy: 0.5752")
    {'Step': 35, 'target_steps': 35, 'train loss': 1.3926e-05, 'train accuracy': 1.0}
    """
    result = re.search(
        r"\s*(?P<Step>\d+)/(?P<target_steps>\d+) \[[=>.]+\] - ETA: (?P<ETA>\d+\.\d+|\d+)s - loss: (?P<train_loss>\d+\.\d+) - accuracy: (?P<train_accuracy>\d+.\d+)",
        line,
    )
    if result is None:
        result = re.search(
            r"(?P<Step>\d+)/(?P<target_steps>\d+) - \d+s - loss: (?P<train_loss>\d+.\d+e-?\d+|\d+\.\d+) - accuracy: (?P<train_accuracy>\d+.\d+)",
            line,
        )
    if result is None:
        return None
    dictionary = result.groupdict()
    for k in ("Step", "target_steps"):
        dictionary[k] = int(dictionary[k])
    for k in ("ETA", "train loss", "train accuracy"):
        try:
            dictionary[k] = float(dictionary[k.replace(" ", "_")])
        except KeyError:
            pass
    del dictionary["train_accuracy"]
    del dictionary["train_loss"]
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


def parse_epoch(filename):
    """
    >>> parse_epoch("tremor_GENEActiv_train_windowsz-5.0_overlap-2.5.epoch-2829.model.h5")
    2829
    >>> parse_epoch("bla/bla/dyskinesia_GENEActiv_train_windowsz-50.2_overlap-2.5.epoch-0034.model.h5/bla.metrics.json")
    34
    """
    filename = str(filename)
    PATTERN = r"\.epoch-(\d+)\."
    return int(re.search(PATTERN, filename).group(1))


def predictions_to_dataframe(model_path, dataset_name):
    rows = []
    model_name_glob = model_path.name.replace(".model.h5", ".epoch-*.model.h5")
    glob_pattern = f"{model_name_glob}.pred/{dataset_name}.metrics.json"
    metrics_json_list = list(model_path.parent.glob(glob_pattern))
    metrics_json_list.append(
        model_path.parent
        / (model_path.name + ".pred")
        / (dataset_name + ".metrics.json")
    )  # Final model
    for f in metrics_json_list:
        try:
            epoch = parse_epoch(f)
        except AttributeError:
            epoch = 1500
        if "overall" in json.loads(f.read_text()):
            # Multiclass (tremor, dykinesia)
            d = json.loads(f.read_text())["overall"]
        else:
            d = json.loads(f.read_text())
        d.update({"Epoch": epoch})
        rows.append(d)
    df = pd.DataFrame(rows).sort_values(by="Epoch")
    return df


def log_to_dataframe(filename):
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
    df = pd.DataFrame(rows)
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


def plot_training(df):
    ax = df.plot(x="Progress", y="train accuracy")
    x_max = df.iloc[0]["target_epochs"] + 1
    ax.set_xlim([0, x_max])
    ax.set_ylim([0, 1])
    ax2 = df.plot(x="Progress", y="train loss", secondary_y=True, ax=ax)
    ax.spines["left"].set_color("black")
    ax2.spines["left"].set_color("black")
    # ax.yaxis.label.set_color("C0")
    # ax.tick_params(axis="y", colors="C0")
    ax2.spines["right"].set_color("C1")
    ax2.yaxis.label.set_color("C1")
    ax2.tick_params(axis="y", colors="C1")
    ax2.set_ylim([0, 0.7])
    ax.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")
    return ax, ax2


def plot(df, ax=None, plot_metrics=True, **plot_kwargs):
    metric = "mAP" if "mAP" in df.columns else "AP"
    y = "accuracy"
    if plot_metrics:
        y = y[y, metric]
    df.plot(x="Epoch", y=y, linewidth=0.9, alpha=0.5, ax=ax, **plot_kwargs)
    df[f"EWMA {metric}"] = df[metric].ewm(halflife=5).mean()
    df["EWMA accuracy"] = df["accuracy"].ewm(halflife=5).mean()
    ax.set_prop_cycle(None)  # Use same colors again
    y = "EWMA accuracy"
    if plot_metrics:
        y = [y, f"EWMA {metric}"]
    df.plot(x="Epoch", y=y, linewidth=1.6, ax=ax, **plot_kwargs)
    ax.set_ylabel("Score")
    if plot_metrics and metric == "AP":
        ax.set_ylim([0, 1])
    return ax


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.input_model_path.name.endswith(".model.h5")
    if args.makedeps:
        model_name_glob = args.input_model_path.name.replace(
            ".model.h5", ".epoch-*.model.h5"
        )
        deps = [
            p.parent / (p.name + ".pred") / (args.eval_dataset_name + ".metrics.json")
            for p in args.input_model_path.parent.glob(model_name_glob)
        ]
        print(" ".join(str(p) for p in deps))
        sys.exit(0)

    train_df = log_to_dataframe(
        args.input_model_path.parent / (args.input_model_path.name + ".txt")
    )
    df = predictions_to_dataframe(args.input_model_path, args.eval_dataset_name)

    ax, ax2 = plot_training(train_df)
    ax.plot(
        df["Epoch"],
        df["accuracy"],
        label="val. accuracy",
        linewidth=0.6,
        alpha=0.5,
        color="C3",
    )
    ax.plot(
        df["Epoch"],
        df["accuracy"].ewm(halflife=5).mean(),
        label="EWMA val. accuracy",
        color="C3",
    )
    ax.set_ylim([0.3, 1])
    # plot(df, ax=ax, plot_metrics=False, color="C3")
    args.output_plot_filename.parent.mkdir(exist_ok=True, parents=True)
    ax.get_legend().remove()
    bbox_to_anchor = (
        (0.9, 0.5) if "dyskinesia" in str(args.output_plot_filename) else (0.9, 1)
    )
    ax.set_xlabel("Epoch")
    ax.get_figure().legend(bbox_to_anchor=bbox_to_anchor)
    ax.get_figure().savefig(args.output_plot_filename, bbox_inches="tight")
