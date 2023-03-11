# Scripts to generate datasets for sktime from the MJFF L-DOPA data
# Author(s): Neha Das (neha.das@tum.de), Cedric Donie (cedricdonie@gmail.com)

import argparse
import numpy as np
import matplotlib.pyplot as plt
from attrdict import AttrDict
from src.data.utils import DatasetAttributes
import re
from pathlib import Path
import pandas as pd
from src.data.global_constants import _PATHS, _LISTS, SUBJECTS
from src.data.loading import *
from src.visualization.data_visualization import *
import sklearn.utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "raw_data_directory",
    type=Path,
    help="Directory of the MJFF L-DOPA downloaded dataset.",
)
parser.add_argument(
    "output_filename", type=Path, help="Filename to save the trained model"
)


SAMPLING_RATE_HZ = 50


def create_filename_stem(pat_ids=None, window_sec=5, overlap_sec=1):
    if pat_ids is not None:
        filename_stem = f"smartdevice_data_windowsz-{window_sec}_overlap-{overlap_sec}_pat-{'-'.join(pat_ids)}"
    else:
        filename_stem = f"smartdevice_data_windowsz-{window_sec}_overlap-{overlap_sec}"
    return filename_stem


def parse_filename(filename):
    """
    >>> parse_filename("GENEActiv_train_windowsz-5.0_overlap-1.0")
    DatasetAttributes(device='GENEActiv', type='train', window_sec=5.0, overlap_sec=1.0, pat_ids=None)
    >>> parse_filename("GENEActiv_train_windowsz-5.0_overlap-1.0_pat-10_BOS-6_NYC")
    DatasetAttributes(device='GENEActiv', type='train', window_sec=5.0, overlap_sec=1.0, pat_ids=['10_BOS', '6_NYC'])
    >>> parse_filename("GENEActiv_train_windowsz-5.0_overlap-1.0_pat-10_BOS-6_NYC.data.pkl")
    DatasetAttributes(device='GENEActiv', type='train', window_sec=5.0, overlap_sec=1.0, pat_ids=['10_BOS', '6_NYC'])
    >>> parse_filename("data/processed/sktime/Shimmer-Wrist_train_windowsz-30.0_overlap-15.0.data.pkl")
    DatasetAttributes(device='Shimmer_Wrist', type='train', window_sec=30.0, overlap_sec=15.0, pat_ids=None)
    """
    filename = str(filename)
    if filename.endswith(".data.pkl"):
        filename = filename[: -len(".data.pkl")]
    float_pattern = r"(\d+\.\d+)"
    pattern_no_patients = (
        r"(GENEActiv|Pebble|Shimmer-LeftWrist|Shimmer-RightWrist|Shimmer-Wrist)_(train-val|test-val|test|val|train|all)_windowsz-"
        + float_pattern
        + r"_overlap-"
        + float_pattern
    )
    pattern_with_patients = pattern_no_patients + r"_pat-(.+)?"
    result = re.search(pattern_with_patients, str(filename))
    if result:
        pat_ids = result.group(5).split("-")
    else:
        pat_ids = None
        result = re.search(pattern_no_patients, str(filename))
    # Underscore should separate filename components
    device = result.group(1).replace("-", "_")
    type = result.group(2)
    window_sec = float(result.group(3))
    overlap_sec = float(result.group(4))
    return DatasetAttributes(
        device=device,
        type=type,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        pat_ids=pat_ids,
    )


def load_task_table():
    tbl = read_table(_PATHS.shimmer_task_tbl_path)
    tbl = tbl[
        tbl["body_segment"].str.contains("Upper") & (tbl["phenotype"] == "tremor")
    ]
    tbl = tbl.drop_duplicates(subset=["timestamp_start", "timestamp_end"])
    tbl["duration"] = tbl.timestamp_end - tbl.timestamp_start
    return tbl


def get_sktime_dataframe(device, pat_ids=None, window_sec=5, overlap_sec=1):
    """
    Creates data points for each patient from raw shimmer data and returns a
    DataFrame. Each data point is a row that represents sequential data that
    is 'window_sec' seconds long containing:
        1. subject_id: pat_ids
        2. task_code: a string literal specifying the patient activity during
           the period
        3. phenotype: the score for the indicated <phenotype> associated with
           the <body_segment>
        4. X/Y/Z: a pandas.Series of acceleration measurements in the
           direction <X/Y/Z> at <sensorlocation>

    Args:
        device: Name of the sensor (Pebble, Phone, GENEActiv, Shimmer, ...) to
                use
        pat_ids: Subject ids
        window_sec: The length of the sequences to be created in seconds
        overlap_sec: Amount of time by which subsequent sequences overlap

    Returns:
        A DataFrame in sktime format according to
        https://www.sktime.org/en/stable/examples/loading_data.html
    """

    window_step_sec = window_sec - overlap_sec
    assert (
        window_step_sec > 1e-5
    ), f"The window step size {window_step_sec} is too small (or negative)."

    task_tbl = read_table(table_file_path=_PATHS.smartdevice_task_tbl_path)

    rows = []

    if pat_ids is None:
        pat_ids = np.unique(task_tbl["subject_id"].to_numpy())

    for pat_id in pat_ids:
        assert pat_id in task_tbl["subject_id"].unique()
        print(f"Patient ID: {pat_id}")

        if device.startswith("Shimmer") and pat_id not in _LISTS.shimmer_patients:
            print(f"No shimmer data will be found for {pat_id}. Skipping.")
            continue

        raw_data = get_device_data_for_patient(device=device, patient_id=pat_id)
        timestamp_min = raw_data["timestamp"].min()
        timestamp_max = raw_data["timestamp"].max()
        tbl_sub_pat = task_tbl[task_tbl["subject_id"] == pat_id]

        all_times = np.stack(
            [tbl_sub_pat["timestamp_start"], tbl_sub_pat["timestamp_end"]], axis=-1
        )
        times = np.unique(all_times, axis=0)
        print("Number of datapoints:", len(times))

        for time_start, time_end in times:
            if time_end - time_start < window_sec:
                continue
            # print(dict(row))

            tbl_sub_pat_sub_time = tbl_sub_pat[
                np.logical_and(
                    tbl_sub_pat["timestamp_start"] == time_start,
                    tbl_sub_pat["timestamp_end"] == time_end,
                )
            ]

            Y = AttrDict()
            Y["subject_id"] = pat_id
            if "task_code" not in Y:
                val = tbl_sub_pat_sub_time["task_code"].to_numpy()[0]
                Y["task_code"] = val

            for _, [_, row] in enumerate(tbl_sub_pat_sub_time.iterrows()):
                # print(row)
                score = row["score"]
                Y[f'{row["phenotype"]}_{row["body_segment"]}'] = score

            # Handle exceptions in the data
            if not (pat_id == "4_BOS" and device == "Pebble"):
                for phenotype in ("tremor", "dyskinesia", "bradykinesia"):
                    key = (
                        phenotype
                        + "_"
                        + get_device_position(subject_id=pat_id, device=device)
                    )
                    Y[phenotype] = Y[key]

            for time_start_sub in np.arange(
                time_start, time_end - window_sec, window_step_sec
            ):
                # Make sure we actually have enough left to cut.
                if (
                    timestamp_min > time_start_sub
                    or timestamp_max < time_start_sub + window_sec
                ):
                    print(
                        "Skipping a timestamp for since it does not fit in the window."
                    )
                    continue

                assert timestamp_min <= time_start_sub
                assert timestamp_max >= time_start_sub + window_sec

                corr_raw_data = raw_data[
                    np.logical_and(
                        raw_data["timestamp"] >= time_start_sub,
                        raw_data["timestamp"] < time_start_sub + window_sec,
                    )
                ]
                data_dict = AttrDict()
                for key in corr_raw_data.keys():
                    data_dict[key] = corr_raw_data[key].to_numpy()
                data_dict.update(Y)

                skip_row = False
                for key, value in data_dict.items():
                    if key in {
                        "X",
                        "Y",
                        "Z",
                        "Magnitude",
                        "back_X",
                        "back_Y",
                        "back_Z",
                        "back_Magnitude",
                        "leftAnkle_Magnitude",
                        "leftAnkle_X",
                        "leftAnkle_Y",
                        "leftAnkle_Z",
                        "rightAnkle_Magnitude",
                        "rightAnkle_X",
                        "rightAnkle_Y",
                        "rightAnkle_Z",
                        "leftWrist_Magnitude",
                        "leftWrist_X",
                        "leftWrist_Y",
                        "leftWrist_Z",
                        "rightWrist_Magnitude",
                        "rightWrist_X",
                        "rightWrist_Y",
                        "rightWrist_Z",
                        "GENEActiv_Magnitude",
                        "GENEActiv_X",
                        "GENEActiv_Y",
                        "GENEActiv_Z",
                        "Phone_Magnitude",
                        "Phone_X",
                        "Phone_Y",
                        "Phone_Z",
                        "Pebble_Magnitude",
                        "Pebble_X",
                        "Pebble_Y",
                        "Pebble_Z",
                    }:
                        data_dict[key] = pd.Series(
                            value, index=data_dict["timestamp"], dtype=np.float64
                        )
                        na_proportion = (
                            data_dict[key]
                            .isna()
                            .value_counts(normalize="True")
                            .get(True, default=0)
                        )
                        if 0 < na_proportion <= 0.1:
                            print("Forward-filling NaN with proportion", na_proportion)
                            data_dict[key] = data_dict[key].fillna(method="ffil")
                        if na_proportion > 0.1:
                            print(
                                "NaN proportion",
                                na_proportion,
                                "too high. Skipping",
                                key,
                                "for",
                                pat_id,
                                ".",
                            )
                            skip_row = True  # HACK
                        else:
                            sklearn.utils.assert_all_finite(data_dict[key])
                            sklearn.utils.assert_all_finite(data_dict[key].index)
                if not skip_row:
                    rows.append(data_dict)

    df = pd.DataFrame(rows)

    # Pad all series to same length
    len_X = df["X"].apply(len)
    if len_X.unique().size > 1:
        max_length = len_X.max()
        n_pads = max_length - len_X.min()
        print(
            "Need to pad to get all timeseries to the same length. Padding with up to",
            n_pads,
            "repeats to reach length",
            max_length,
            "uniformly.",
        )
        assert max_length > 10
        assert n_pads < 5
        for index, row in df[len_X < max_length].iterrows():
            for column in df.columns:
                if not isinstance(df.loc[index, column], pd.Series):
                    continue
                # Append last value as often as necessary
                for i in range(n_pads):
                    last = df.loc[index, column].iloc[-1]
                    index_value = (i + 2) * df.loc[index, column].index[-1] - (
                        i + 1
                    ) * df.loc[index, column].index[-2]
                    series = pd.Series(last, index=[index_value])
                    new = df.loc[index, column].append(series)
                    # Us `.at` to avoid https://github.com/pandas-dev/pandas/issues/37593
                    df.at[index, column] = new
    assert df["X"].apply(len).unique().size == 1

    dtypes = dict(subject_id="category", task_code="category")
    dtypes.update(
        {
            k: "category"
            for k in df.columns
            if k.startswith("bradykinesia")
            or k.startswith("dyskinesia")
            or k.startswith("tremor")
        }
    )

    return df.astype(dtypes)


def get_shimmer_wrist_sktime_dataframe(pat_ids=None, window_sec=5, overlap_sec=1):
    print("Left wrist")
    left = get_sktime_dataframe(
        "Shimmer_LeftWrist",
        pat_ids=pat_ids,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
    )
    print("Right wrist")
    right = get_sktime_dataframe(
        "Shimmer_RightWrist",
        pat_ids=pat_ids,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
    )
    left["Device"] = "Shimmer_LeftWrist"
    right["Device"] = "Shimmer_RightWrist"
    return pd.concat([left, right])


def create_sktime_dataframe(
    output_filename, device, pat_ids=None, window_sec=5, overlap_sec=1
):
    output_filename.parent.mkdir(exist_ok=True)

    if device == "Shimmer_Wrist":
        df = get_shimmer_wrist_sktime_dataframe(
            pat_ids=pat_ids, window_sec=window_sec, overlap_sec=overlap_sec
        )
    else:
        df = get_sktime_dataframe(
            device, pat_ids=pat_ids, window_sec=window_sec, overlap_sec=overlap_sec
        )
    df.to_pickle(output_filename, protocol=4)


if __name__ == "__main__":
    args = parser.parse_args()

    attributes = parse_filename(args.output_filename)
    if attributes.pat_ids is None:
        if attributes.type == "train-val":
            pat_ids = SUBJECTS["train"] + SUBJECTS["val"]
        else:
            pat_ids = SUBJECTS[attributes.type]
    else:
        pat_ids = attributes.pat_ids
    create_sktime_dataframe(
        args.output_filename,
        device=attributes.device,
        window_sec=attributes.window_sec,
        overlap_sec=attributes.overlap_sec,
        pat_ids=pat_ids,
    )
