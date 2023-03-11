# Author(s): Neha Das (neha.das@tum.de), Cedric Donié (cedricdonie@gmail.com)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

from src.data.loading import *
from src.visualization.data_visualization import *
from src.data.global_constants import _PATHS, _CONSTS


plt.style.use(["science", "ieee", "src/visualization/main.mplstyle", "vibrant"])


parser = argparse.ArgumentParser()
parser.add_argument("output_figure", type=Path)


TREMORS_OF_PATIENTS = {"14473_5": 4, "32042_11": 3, "49025_17": 0, "1_1": 0}


def read_table_data(identity):
    tbl = read_table(_PATHS.smartdevice_task_tbl_path)
    tbl.set_index("Unnamed: 0", inplace=True)
    tbl.index.name = None
    tbl_st = tbl.loc[identity]
    raw_data = read_raw_data(tbl_st["subject_id"], tbl_st["session"], "GENEActiv")
    raw_data = raw_data.set_index("timestamp")
    raw_data.index -= raw_data.index[0]
    raw_data.index.name = "Time / s"
    return raw_data


def plot_over_time(identity, ax=None, **kwargs):
    raw_data = read_table_data(identity)
    ax = raw_data.iloc[15000:20000].plot(
        y="GENEActiv_Magnitude",
        logy=False,
        linewidth=0.2,
        ax=ax,
        **kwargs,
        legend=False,
    )
    ax.set_ylim([8.5, 11.5])
    handles, labels = ax.get_legend_handles_labels()
    # copy the handles
    handles = [copy.copy(ha) for ha in handles]
    # set the linewidths to the copies
    [ha.set_linewidth(0) for ha in handles]
    # put the copies into the legend
    ax.legend(handles=handles, labels=labels)
    ax.set_ylabel(r"Acceleration / \si{m\per\s\squared}")
    return ax, raw_data


if __name__ == "__main__":
    args = parser.parse_args()

    COLORS = ["C4", "C3", "C1", "C1"]
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for ax, pat_id, color in zip(axs.flatten(), TREMORS_OF_PATIENTS.keys(), COLORS):
        plot_over_time(
            pat_id,
            label=f"tremor severity {TREMORS_OF_PATIENTS[pat_id]}",
            ax=ax,
            color=color,
        )

    bbox = axs[1, 0].legend().get_bbox_to_anchor()
    # axs[1, 0].legend().set_bbox_to_anchor((bbox.x0, bbox.y0))

    w, _ = fig.get_size_inches()
    fig.set_size_inches((w, 3.1))

    fig.savefig(args.output_figure)
