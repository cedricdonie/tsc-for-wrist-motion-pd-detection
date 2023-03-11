# Data Loading scripts for MJFF Levadopa Data Analysis
# Author(s): Neha Das (neha.das@tum.de), Cedric Donié (cedricdonie@gmail.com)

import os
import numpy as np
import pandas
import functools

from src.data.global_constants import _LISTS, _PATHS


def save_figure(name, ax=None, fig=None):
    return  # Uncomment to export figures
    assert not (ax is None and fig is None), "Either fig or ax required!"
    if fig is None:
        fig = ax.get_figure()
    fig.savefig(f"results/{name}.pgf", bbox_inches="tight")


def set_size(width_cm, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_cm = width_cm * fraction
    # Convert from pt to inches
    inches_per_cm = 1 / 2.54

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_cm * inches_per_cm
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def print_ids():
    print("\npatient_id list: ")
    print(_LISTS.shimmer_patients)
    print("\nday_id list: ")
    print(_LISTS.day_list)
    print("\nsensor_id list: ")
    print(_LISTS.shimmer_locations)


def get_full_patient_imu_path(patient_id, day_id, sensor_id, is_loc_boston=True):
    if "NY" in patient_id:
        is_loc_boston = False
    patient_id = patient_id.strip("_BOSNYC")
    if is_loc_boston:
        patient_id_string = f"patient{patient_id}"
    else:
        patient_id_string = f"patient{patient_id}_NY"
    return os.path.join(
        _PATHS.root,
        "LDOPA_DATA",
        sensor_id,
        patient_id_string,
        f"rawData_Day{day_id}.txt",
    )


def get_table_path(table_file_path):
    return os.path.join(_PATHS.root, "TABLES", table_file_path)


def read_raw_data(
    patient_id, day_id, sensor_id, is_loc_boston=True, delimiter="\t", low_memory=False
):
    filename = get_full_patient_imu_path(
        patient_id=patient_id,
        day_id=day_id,
        sensor_id=sensor_id,
        is_loc_boston=is_loc_boston,
    )
    return pandas.read_csv(filename, delimiter=delimiter, low_memory=low_memory)


def read_table(table_file_path, delimiter=",", **pandas_kwargs):
    filename = get_table_path(table_file_path=table_file_path)
    return pandas.read_csv(
        filename, delimiter=delimiter, low_memory=False, **pandas_kwargs
    )


def get_raw_shimmer_data_for_patient(patient_id, low_memory=False):
    """
    The Labelled data (in tbl_task_sc_2.csv) corresponds to day 1 and day 4 of the raw data.

    Args:
        patient_id - patient or subject ID
        display_frames - whether to display the dataframes in a formatted manner or not
    """
    df1b = read_raw_data(
        patient_id=patient_id,
        day_id="1",
        sensor_id="Shimmer_Back",
        low_memory=low_memory,
    )
    df4b = read_raw_data(
        patient_id=patient_id,
        day_id="4",
        sensor_id="Shimmer_Back",
        low_memory=low_memory,
    )
    dfb = pandas.concat([df1b, df4b])

    df1la = read_raw_data(
        patient_id=patient_id,
        day_id="1",
        sensor_id="Shimmer_LeftAnkle",
        low_memory=low_memory,
    )
    df4la = read_raw_data(
        patient_id=patient_id,
        day_id="4",
        sensor_id="Shimmer_LeftAnkle",
        low_memory=low_memory,
    )
    dfla = pandas.concat([df1la, df4la])
    dfla_cols = dfla.columns.difference(dfb.columns)

    df1ra = read_raw_data(
        patient_id=patient_id,
        day_id="1",
        sensor_id="Shimmer_RightAnkle",
        low_memory=low_memory,
    )
    df4ra = read_raw_data(
        patient_id=patient_id,
        day_id="4",
        sensor_id="Shimmer_RightAnkle",
        low_memory=low_memory,
    )
    dfra = pandas.concat([df1ra, df4ra])
    dfra_cols = dfra.columns.difference(dfb.columns)

    df1lw = read_raw_data(
        patient_id=patient_id,
        day_id="1",
        sensor_id="Shimmer_LeftWrist",
        low_memory=low_memory,
    )
    df4lw = read_raw_data(
        patient_id=patient_id,
        day_id="4",
        sensor_id="Shimmer_LeftWrist",
        low_memory=low_memory,
    )
    dflw = pandas.concat([df1lw, df4lw])
    dflw_cols = dflw.columns.difference(dfb.columns)

    df1rw = read_raw_data(
        patient_id=patient_id,
        day_id="1",
        sensor_id="Shimmer_RightWrist",
        low_memory=low_memory,
    )
    df4rw = read_raw_data(
        patient_id=patient_id,
        day_id="4",
        sensor_id="Shimmer_RightWrist",
        low_memory=low_memory,
    )
    dfrw = pandas.concat([df1rw, df4rw])
    dfrw_cols = dfrw.columns.difference(dfb.columns)

    df = pandas.concat(
        [dfb, dfla[dfla_cols], dfra[dfra_cols], dflw[dflw_cols], dfrw[dfrw_cols]],
        axis=1,
    )

    return df


def get_raw_wearable_data_for_patient(patient_id):
    """
    The Labelled data (in tbl_task_sc_2.csv) corresponds to day 1 and day 4 of the raw data.

    Args:
        patient_id - patient or subject ID
        display_frames - whether to display the dataframes in a formatted manner or not
    """
    try:
        df1b = read_raw_data(patient_id=patient_id, day_id="1", sensor_id="GENEActiv")
        df4b = read_raw_data(patient_id=patient_id, day_id="4", sensor_id="GENEActiv")
        dfb = pandas.concat([df1b, df4b])
        dfb.set_index("timestamp", inplace=True)
    except FileNotFoundError:
        print("Skipping GENEActiv for patient", patient_id)
        dfb = pandas.DataFrame()

    try:
        df1la = read_raw_data(patient_id=patient_id, day_id="1", sensor_id="Pebble")
        df4la = read_raw_data(patient_id=patient_id, day_id="4", sensor_id="Pebble")
        dfla_full = pandas.concat([df1la, df4la])
        dfla_full.set_index("timestamp", inplace=True)
        dfla_cols = dfla_full.columns.difference(dfb.columns)
        dfla = dfla_full[dfla_cols]
    except FileNotFoundError:
        print("Skipping Pebble for patient", patient_id)
        dfla = None

    try:
        df1ra = read_raw_data(patient_id=patient_id, day_id="1", sensor_id="Phone")
        df4ra = read_raw_data(patient_id=patient_id, day_id="4", sensor_id="Phone")
        dfra_full = pandas.concat([df1ra, df4ra])
        dfra_full.set_index("timestamp", inplace=True)
        dfra_cols = dfra_full.columns.difference(dfb.columns)
        dfra = dfra_full[dfra_cols]
    except FileNotFoundError:
        print("Skipping Phone for patient", patient_id)
        dfra = None

    df = dfb
    if dfla is not None:
        df = df.join(dfla, how="outer")
    if dfra is not None:
        df = df.join(dfra, how="outer")
    return df.reset_index()


def get_device_data_for_patient(
    device, patient_id, no_device_name=True
) -> pandas.DataFrame:
    try:
        df1 = read_raw_data(patient_id=patient_id, day_id="1", sensor_id=device)
    except FileNotFoundError:
        print(f"Device {device} not found for patient {patient_id} on day 1. Skipping.")
        df1 = None
    try:
        df2 = read_raw_data(patient_id=patient_id, day_id="4", sensor_id=device)
    except FileNotFoundError:
        print(f"Device {device} not found for patient {patient_id} on day 2. Skipping.")
        df2 = None
    df = pandas.concat([df1, df2])
    if no_device_name:
        if device == "Shimmer_LeftWrist":
            device_column_suffix = "leftWrist"
        elif device == "Shimmer_RightWrist":
            device_column_suffix = "rightWrist"
        else:
            device_column_suffix = device
        df.columns = df.columns.str.replace(device_column_suffix + "_", "")
    return df


@functools.lru_cache(maxsize=128)
def get_device_position(subject_id, device):
    """
    >>> get_device_position("10_BOS", "GENEActiv")
    'RightUpperLimb'
    >>> get_device_position("<Patient name does not matter>", "Shimmer_LeftWrist")
    'LeftUpperLimb'
    >>> get_device_position("6_NYC", "Shimmer_RightWrist")
    'RightUpperLimb'
    """
    # Trival case
    if device == "Shimmer_LeftWrist":
        return "LeftUpperLimb"
    if device == "Shimmer_RightWrist":
        return "RightUpperLimb"

    # Handle exceptions in the data
    if subject_id == "4_BOS" and device == "GENEActiv":
        # See discussion on dataset website: https://www.synapse.org/#!Synapse:syn20681023/discussion/threadId=9167
        return "LeftUpperLimb"

    details_tbl = read_table(_PATHS.sensor_grp_1_details_path)
    assert device in details_tbl.device.unique()
    position = details_tbl[
        (details_tbl.subject_id == subject_id) & (details_tbl.device == device)
    ]["device_position"].unique()
    assert (
        position.size == 1
    ), f"There are multiple positions for device {device} on patient {subject_id}!"
    return position[0]
