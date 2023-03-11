# Utility scripts and type definitions.
# Author(s): Cedric Donié (cedricdonie@gmail.com)

from collections import namedtuple
import pandas
from src.data.global_constants import _PATHS, SUBJECTS
from src.data.loading import read_table
import numpy as np


def data_split(filename, phenotype="tremor", return_subjects=False, multivariate=True):
    df = pandas.read_pickle(filename)

    if phenotype == "bradykinesia":
        print("Filtering out NotApplicable. Length before:", len(df), end="")
        df = df[df["bradykinesia"] != "NotApplicable"]
        print(". Length after:", len(df))

    X = df[["X", "Y", "Z"]] if multivariate else df[["Magnitude"]]
    y = df[phenotype]

    if phenotype in ("bradykinesia", "dyskinesia"):
        y = y == "Yes"

    if not return_subjects:
        return X, y

    subjects = df["subject_id"]

    return X, y, subjects


DatasetAttributes = namedtuple(
    "DatasetAttributes", ["device", "type", "window_sec", "overlap_sec", "pat_ids"]
)
