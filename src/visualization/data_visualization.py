# Visualization/Plotting functions for MJFF Levadopa Data Analysis
# Author(s): Neha Das (neha.das@tum.de)

import numpy as np
import pandas
import os, sys
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd
from src.data.global_constants import _PATHS, _LISTS, SUBJECTS
from src.data.loading import *
from pathlib import Path

import argparse

plt.style.use(
    [
        "science",
        "muted",
        "src/visualization/main.mplstyle",
        "src/visualization/smallfigures.mplstyle",
    ]
)

parser = argparse.ArgumentParser()
parser.add_argument("--figure", type=Path)
parser.add_argument("--full-data", type=Path)
parser.add_argument("--trainval-vs-test", type=Path)
parser.add_argument("--train-vs-val", type=Path)


def plot_task_tbl_time_stats(
    is_shimmer=False, secondary_col="score", location=None, patient_ids=None, ax=None
):
    """
    This function plots the time spent stats for the phenotypes (tremor, bradykinesia, dyskinesia)
    in the shimmer table - tbl_task_sc_2.csv grouped by values in the secondary column

    Args:
        secondary_col - The secondary column
    """

    colors = [f"C{i}".replace("C4", "C8") for i in range(9)]

    if is_shimmer:
        table_file_path = _PATHS.shimmer_task_tbl_path
    else:
        table_file_path = _PATHS.smartdevice_task_tbl_path

    tbl = pandas.read_csv(
        get_table_path(table_file_path=table_file_path), delimiter=",", low_memory=False
    )
    tbl["score"] = tbl["score"].str.lower()
    tbl["score"] = tbl["score"].replace("notapplicable", "N/A")
    tbl = tbl[
        tbl.body_segment
        == tbl.subject_id.apply(get_device_position, device="GENEActiv")
    ]
    if location is not None:
        assert location in ("BOS", "NYC")
        tbl = tbl[tbl.subject_id.str.endswith(location)]
    if patient_ids is not None:
        tbl = tbl[tbl.subject_id.isin(patient_ids)]
    tbl["duration"] = (tbl.timestamp_end - tbl.timestamp_start) / 60 / 60
    gb = tbl.groupby(["phenotype", "score"])
    sums = gb["duration"].sum()
    patient_count = gb["subject_id"].unique().size
    ax = sums.unstack().plot.barh(
        stacked="true", figsize=set_size(15), ax=ax, color=colors
    )
    ax.set_xlabel("overall duration /h")
    ax.set_xlim([0, 60])
    save_figure(
        "phenotypedurationoverall" + ("shimmer" if is_shimmer else "other"), ax=ax
    )
    sums = sums.to_frame()
    sums["proportion"] = sums["duration"].div(sums["duration"].sum(level=0), level=0)
    return sums, ax


def plot_raw_shimmer_data_for_symptom(
    symptom="tremor",
    area="RightUpperLimb",
    plot_first_x_rows=5,
    df_all=None,
    df_all_subject_id=None,
):
    tbl = read_table(_PATHS.shimmer_task_tbl_path)
    sensor_list = ["back", "leftAnkle", "rightAnkle", "leftWrist", "rightWrist"]
    axis_list = ["X", "Y", "Z"]
    sub_tbl = tbl[
        np.logical_and(tbl["phenotype"] == symptom, tbl["body_segment"] == area)
    ]

    count_rows = 0
    for j, [index, row] in enumerate(sub_tbl.iterrows()):
        if count_rows < plot_first_x_rows:
            # print(dict(row))
            subject_id = row["subject_id"].split("_")[0]
            time_start = row["timestamp_start"]
            time_end = row["timestamp_end"]
            score = row["score"]
            if df_all is None or df_all_subject_id is None:
                # picks and shows data only from the first available subject
                df_all = get_raw_shimmer_data_for_patient(
                    patient_id=subject_id, display_frames=False
                )
                df_all_subject_id = subject_id
            elif df_all_subject_id != subject_id:
                continue

            count_rows += 1
            corr_raw_data = df_all[
                np.logical_and(
                    df_all["timestamp"] >= time_start, df_all["timestamp"] < time_end
                )
            ]
            # count = len(corr_raw_data)
            # print(f"Number of datapoints {symptom}-{area}-{score} - {count}")

            fig, ax = plt.subplots(
                1, len(sensor_list), figsize=(5 * len(sensor_list), 5)
            )
            for i, sensor in enumerate(sensor_list):
                for axis in axis_list:
                    sensor_axis = f"{sensor}_{axis}"
                    ax[i].plot(
                        np.arange(len(corr_raw_data[sensor_axis])) * 0.02,
                        list(corr_raw_data[sensor_axis]),
                    )
                ax[i].set_title(f"{sensor}", fontsize=20)
            time_passed = time_end - time_start
            dt_start = datetime.utcfromtimestamp(time_start).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
            plt.suptitle(
                f"SUB:{subject_id} {symptom}-{area}-{score}\n Time Start: {dt_start}; Time Passed: {round(time_passed, 2)} s",
                fontsize=22,
                y=1.05,
            )

            # print(list(corr_raw_data[sensor_axis]))
    plt.show()

    return df_all, df_all_subject_id


def plot_raw_shimmer_data_chunks(
    plot_first_x_chunks=10, df_all=None, df_all_subject_id="3", task_code=None
):
    tbl = read_table(_PATHS.shimmer_task_tbl_path)
    sensor_list = ["back", "leftAnkle", "rightAnkle", "leftWrist", "rightWrist"]
    axis_list = ["X", "Y", "Z"]

    act_df = read_table(_PATHS.action_dict_path)

    subtbl = tbl[tbl["subject_id"] == f"{df_all_subject_id}_BOS"]
    act_txt = ""
    if task_code is not None:
        subtbl = subtbl[subtbl["task_code"] == task_code]
        act_txt += "_" + task_code

    fig, ax = plt.subplots(
        plot_first_x_chunks,
        len(sensor_list) + 1,
        figsize=(5 * (len(sensor_list) + 1), 5 * plot_first_x_chunks),
    )

    count_rows = 0
    for j, [index, row] in enumerate(subtbl.iterrows()):
        if count_rows < plot_first_x_chunks and j % 12 == 0:
            # print(dict(row))
            subject_id = row["subject_id"].split("_")[0]
            time_start = row["timestamp_start"]
            time_end = row["timestamp_end"]
            task_code = row["task_code"]
            if df_all is None:
                df_all = get_raw_shimmer_data_for_patient(
                    patient_id=subject_id, display_frames=False
                )
            elif df_all_subject_id != subject_id:
                continue

            corr_raw_data = df_all[
                np.logical_and(
                    df_all["timestamp"] >= time_start, df_all["timestamp"] < time_end
                )
            ]
            # count = len(corr_raw_data)
            # print(f"Number of datapoints {symptom}-{area}-{score} - {count}")

            for i, sensor in enumerate(sensor_list):
                for axis in axis_list:
                    sensor_axis = f"{sensor}_{axis}"
                    ax[count_rows, i].plot(
                        np.arange(len(corr_raw_data[sensor_axis])) * 0.02,
                        list(corr_raw_data[sensor_axis]),
                    )
                ax[count_rows, i].set_title(f"{sensor}", fontsize=20)
            time_passed = time_end - time_start
            dt_start = datetime.utcfromtimestamp(time_start).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
            # import pdb; pdb.set_trace()
            symptom_list = list(subtbl["phenotype"][j : j + 12])
            area_list = list(subtbl["body_segment"][j : j + 12])
            score_list = list(subtbl["score"][j : j + 12])
            symptom_area_score = ""
            for i in range(12):
                symptom_area_score += (
                    f"\n{symptom_list[i]}_{area_list[i]}_{score_list[i]}"
                )
            activity = list(act_df[act_df["task_code"] == task_code]["description"])[0]
            text = f"SUB:{subject_id}{symptom_area_score}\n Time Start: {dt_start}\n Time Passed: {round(time_passed, 2)} s\nActivity: {activity}"

            left, width = 0.25, 0.5
            bottom, height = 0.25, 0.5
            right = left + width
            top = bottom + height
            ax[count_rows, -1].text(
                0.5 * (left + right),
                0.5 * (bottom + top),
                text,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                color="red",
                transform=ax[count_rows, -1].transAxes,
            )

            count_rows += 1
            # print(list(corr_raw_data[sensor_axis]))
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig(f"results/raw_data_chunks_pat_{df_all_subject_id}{act_txt}")
    plt.show()

    return df_all, df_all_subject_id


if __name__ == "__main__":
    args = parser.parse_args()

    if args.figure:
        fig, ax = plt.subplots()
        plot_task_tbl_time_stats(ax=ax)
        ax.set_xlabel("Duration / h")
        ax.set_ylabel("Phenotype")
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_tick_params(which="minor", left=False)
        ax.legend().remove()
        ax.get_figure().set_size_inches(
            (
                ax.get_figure().get_size_inches()[0],
                ax.get_figure().get_size_inches()[1] * 0.45,
            )
        )
        ax.get_figure().legend(
            ncol=4, loc="lower center", bbox_to_anchor=[0.5, -0.47], title="Score"
        )
        ax.get_figure().savefig(args.figure)

    if args.full_data:
        df, _ = plot_task_tbl_time_stats()
        df.columns = df.columns.str.capitalize()

    if args.trainval_vs_test:
        df_trainval, _ = plot_task_tbl_time_stats(
            patient_ids=SUBJECTS["train"] + SUBJECTS["val"]
        )
        df_trainval.columns = df_trainval.columns.str.capitalize()
        df_test, _ = plot_task_tbl_time_stats(patient_ids=SUBJECTS["test"])
        df_test.columns = df_test.columns.str.capitalize()
        df = pd.concat((df_trainval, df_test), axis=1, keys=["Train-Validate", "Test"])

    if args.train_vs_val:
        df_trainval, _ = plot_task_tbl_time_stats(patient_ids=SUBJECTS["train"])
        df_trainval.columns = df_trainval.columns.str.capitalize()
        df_test, _ = plot_task_tbl_time_stats(patient_ids=SUBJECTS["val"])
        df_test.columns = df_test.columns.str.capitalize()
        df = pd.concat((df_trainval, df_test), axis=1, keys=["Train", "Validate"])

    if args.trainval_vs_test or args.train_vs_val or args.full_data:
        df.columns = df.columns.map(lambda x: tuple("{" + i.title() + "}" for i in x))
        df.index = df.index.map(lambda x: tuple(i.lower() for i in x))
        df.index.names = ["{" + i.title() + "}" for i in df.index.names]
        df = df.fillna(0)

    if args.full_data:
        table = df

    if args.trainval_vs_test or args.train_vs_val:
        table = df.swaplevel(i=0, j=1, axis="columns").sort_index(axis=1)

    if args.trainval_vs_test or args.train_vs_val or args.full_data:
        tex = table.to_latex(
            multicolumn=True,
            multirow=True,
            escape=False,
            multicolumn_format="c",
            column_format="llSSSS",
        )
        tex = (
            tex.replace(r"\cline{", r"\cmidrule{")
            .replace("Duration", "Duration / h")
            .replace("notapplicable", "N/A")
        )
        lines = tex.split("\n")
        lines.insert(3, r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
        tex = "\n".join(lines)

    if args.trainval_vs_test:
        args.trainval_vs_test.write_text(tex)
    elif args.train_vs_val:
        args.train_vs_val.write_text(tex)
    elif args.full_data:
        args.full_data.write_text(tex)
