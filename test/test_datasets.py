# Author(s): Cedric Donie (cedricdonie@gmail.com)

from glob import glob
import pytest
import pandas as pd
import numpy as np


@pytest.mark.parametrize(
    "dataset_path", glob("data/processed/sktime/*_windowsz-*_overlap-*.data.pkl")
)
def test_dataset_equal_length(dataset_path):
    """
    Tests that all pd.Series in the files have the same length.
    This is necessary for sktime classifiers.
    """
    try:
        df = pd.read_pickle(dataset_path)
    except AttributeError:
        pytest.skip("Dataframe was from another pickle version. Skipped.")
    df["lenX"] = df["X"].apply(len)
    df["lenY"] = df["Y"].apply(len)
    df["lenZ"] = df["Z"].apply(len)
    df["lenMagnitude"] = df["Magnitude"].apply(len)
    print(
        df["lenX"].unique(),
        df["lenY"].unique(),
        df["lenZ"].unique(),
        df["lenMagnitude"].unique(),
    )
    assert (df["lenX"] == df["lenY"]).all()
    assert (df["lenX"] == df["lenZ"]).all()
    assert (df["lenMagnitude"] == df["lenMagnitude"]).all()
    assert df["lenX"].unique().size == 1
    assert df["lenY"].unique().size == 1
    assert df["lenZ"].unique().size == 1
