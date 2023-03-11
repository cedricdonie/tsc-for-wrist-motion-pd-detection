# Author(s): Cedric Donié (cedricdonie@gmail.com)

import argparse
import json
from pathlib import Path
import sys
from click import style
from pyparsing import col
import seaborn as sns
import pandas as pd
from src.models import phenotype_from_model_name
import deepsig
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


plt.style.use(
    [
        "science",
        "ieee",
        "src/visualization/main.mplstyle",
        "src/visualization/framelegend.mplstyle",
        "vibrant",
    ]
)

parser = argparse.ArgumentParser()
parser.add_argument("metric_files", type=Path, nargs="+")
parser.add_argument("--output-figure", type=Path, default=None)
parser.add_argument("--comparisons", metavar="n", default=30)
parser.add_argument("--output-csv")


TAU = 0.2
MODEL_NAMES = {
    "gp": "wavelet MLP",
    "rocket": "ROCKET",
    "inceptiontime": "default InceptionTime",
    "inceptiontimetuned": "tuned InceptionTime",
}
MIN_POWER = 0.8


def ttest_rel_greater(scores_a, scores_b):
    """
    Needed to monkey-patch for older scipy version.
    See https://stackoverflow.com/a/64756011
    """
    p_twosided = scipy.stats.ttest_rel(scores_a, scores_b).pvalue
    if np.mean(scores_a) > np.mean(scores_b):
        return p_twosided / 2.0
    return 1.0 - p_twosided / 2


def power_analysis(scores):
    # Monkey-patch since `alternative="greater"` not supported by default significance test
    # of `scipy.stats.ttest_rel` in this version.
    return deepsig.bootstrap_power_analysis(scores, significance_test=ttest_rel_greater)


def load_df(paths):
    rows = []
    for path in paths:
        phenotype = phenotype_from_model_name(path.parent)
        if "_GENEActiv" in str(path.parent):
            device = "GENEActiv"
        elif "_Shimmer-Wrist" in str(path.parent):
            device = "Shimmer"
        else:
            raise NotImplementedError(f"Unknown device in {path.parent}")

        random_state = int(path.parent.parent.name)
        model = path.parent.parent.parent.name
        row = pd.json_normalize(json.loads(path.read_text()))
        row["random_state"] = random_state
        row["model"] = model
        row["phenotype"] = phenotype
        row["device"] = device
        rows.append(row)
    return pd.concat(rows)


def aso(
    df, model_b, metric_name, model_a="inceptiontime", num_comparisons=1, alpha=0.05
):
    a = df.loc[df["model"] == model_a, metric_name].dropna()
    b = df.loc[df["model"] == model_b, metric_name].dropna()
    power_a = power_analysis(a)
    power_b = power_analysis(b)
    assert power_a > MIN_POWER, f"Power ({power_a}) of {model_a} is too low!"
    assert power_b > MIN_POWER, f"Power ({power_b}) of {model_b} is too low!"
    min_eps = deepsig.aso(
        a, b, confidence_level=(1 - alpha), num_comparisons=num_comparisons, num_jobs=1
    )
    return min_eps, power_analysis(a), power_analysis(b)


def interpret_aso(df, model_a, metric_name):
    for model in df["model"].unique():
        if model == model_a or model == "inceptiontimetuned":
            continue
        min_eps, _, _ = aso(
            df,
            metric_name=metric_name,
            model_a=model_a,
            model_b=model,
            num_comparisons=10,
            alpha=alpha,
        )
        if min_eps < TAU:
            print(
                MODEL_NAMES[model_a],
                "over",
                MODEL_NAMES[model],
                ": stochastically dominant\t",
                min_eps,
            )
        else:
            print(
                MODEL_NAMES[model_a],
                "over",
                MODEL_NAMES[model],
                ": NOT stochastically dominant\t",
                min_eps,
            )


if __name__ == "__main__":
    args = parser.parse_args()

    df = load_df(args.metric_files)
    ID_VARS = ["random_state", "model", "phenotype"]
    metric_name = "overall.mAP" if "overall.mAP" in df.columns else "AP"
    accuracy_name = (
        "overall.balanced accuracy"
        if "overall.accuracy" in df.columns
        else "balanced accuracy"
    )
    if "overall.balanced accuracy" in df.columns and "balanced accuracy" in df.columns:
        # Replace the overall with the phenotype balanced accuracy.
        df.loc[
            df["overall.balanced accuracy"].isnull(), "overall.balanced accuracy"
        ] = df.loc[
            df["overall.balanced accuracy"].isnull(),
            "balanced accuracy",
        ]
    if "overall.mAP" in df.columns and "AP" in df.columns:
        # Replace the overall with the phenotype balanced accuracy.
        df.loc[df["overall.mAP"].isnull(), "overall.mAP"] = df.loc[
            df["overall.mAP"].isnull(),
            "AP",
        ]
    dfm = df[[metric_name, accuracy_name] + ID_VARS].melt(
        id_vars=["random_state", "model", "phenotype"],
        var_name="Metric",
        value_name="Score",
    )
    dfm["Metric"] = dfm["Metric"].str.replace("overall.", "")
    dfm["model"] = dfm["model"].replace(MODEL_NAMES)
    dfm.columns = dfm.columns.str.title()
    alpha = 0.05

    if len(dfm["Phenotype"].unique()) > 1:
        fig, axs = plt.subplots(nrows=3, sharex=True)
        s = fig.get_size_inches()
        fig.set_size_inches(s[0], s[1] * 2)

    if args.output_figure is not None:
        for phenotype, ax in zip(("tremor", "bradykinesia", "dyskinesia"), axs):
            dfm_phenotype = dfm[dfm["Phenotype"] == phenotype]
            if len(dfm_phenotype) == 0:
                continue
            sns.boxplot(
                x="Model",
                y="Score",
                data=dfm_phenotype,
                hue="Metric",
                ax=ax,
                saturation=1,
            )
            ax.get_legend().set_visible(False)
            ax.set_xlabel(None)
            ax.set_ylabel(phenotype.title() + " " + "Score")
            if phenotype != "tremor":  # Binary classification
                expected_ap = (
                    df[df["phenotype"] == phenotype]
                    .groupby("model")
                    .mean()["expected AP"]
                )
                ax.hlines(
                    expected_ap,
                    np.arange(len(expected_ap)) - 0.4,
                    np.arange(len(expected_ap)),
                    linestyles="dashed",
                    color="C3",
                    label="random classifier",
                )
                ax.hlines(
                    np.ones_like(expected_ap) * 0.5,
                    np.arange(len(expected_ap)),
                    np.arange(len(expected_ap)) + 0.4,
                    linestyles="dashed",
                    color="C3",
                    label="random classifier",
                )
                ax.set_ylim([0, 0.85])
            else:
                ax.axhline(
                    0.25, color="C3", linestyle="dashed", label="random classifier"
                )
                ax.set_ylim([0.2, 0.6])
            ax.xaxis.set_tick_params(which="minor", bottom=False)
            # ax.xaxis.set_ticks_position("bottom")
            for i in range(3):
                ax.axvline(i + 0.5, color="lightgray")
        ax.set_xlabel("Model")
        for item in ax.get_xticklabels():
            item.set_rotation(15)
        ax.set_xlim([-0.5, 3.5])

        handles, labels = ax.get_legend_handles_labels()
        labels[0] = "(m)AP"
        fig.legend(
            handles=handles[:3],
            labels=labels[:3],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=3,
            frameon=False,
            columnspacing=0.75,
        )

        fig.savefig(args.output_figure)
        ax.get_figure().show()

    if args.output_csv is not None:
        overall_df = df.loc[:, ~df.columns.str.contains("classwise")].copy()
        overall_df["model"] = overall_df["model"].replace(MODEL_NAMES)
        overall_df.columns = overall_df.columns.str.replace("overall.", "")
        overall_df.columns = overall_df.columns.str.replace(" ", "")
        overall_df.columns = overall_df.columns.str.replace("_", "")
        overall_df.to_csv(args.output_csv, index=False)

    if args.output_figure is None and args.output_csv is None:
        interpret_aso(df, "inceptiontime", metric_name)
        interpret_aso(df, "inceptiontimetuned", metric_name)
