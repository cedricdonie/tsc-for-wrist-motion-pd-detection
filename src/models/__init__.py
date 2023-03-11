import pathlib


def phenotype_from_model_name(model_path: pathlib.Path):
    """
    >>> phenotype_from_model_name(pathlib.Path("models/gp/bradykinesia_GENEActiv_train_windowsz-30.0_overlap-15.0.model.pkl"))
    'bradykinesia'
    >>> phenotype_from_model_name(pathlib.Path("models/gp/GENEActiv_train_windowsz-30.0_overlap-15.0.model.pkl"))
    Traceback (most recent call last):
    ...
    NotImplementedError: ...
    """
    if model_path.stem.startswith("bradykinesia"):
        return "bradykinesia"
    if model_path.stem.startswith("tremor"):
        return "tremor"
    if model_path.stem.startswith("dyskinesia"):
        return "dyskinesia"
    raise NotImplementedError(f"Could not determine phenotype from '{model_path}'.")
