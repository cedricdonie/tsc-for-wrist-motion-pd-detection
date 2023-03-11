# Calculate metrics from predictions and groundtruth
# Author(s): Cedric Donie (cedricdonie@gmail.com)
#
# Plotting functions with `ax=None`as default are inspired by
# https://towardsdatascience.com/creating-custom-plotting-functions-with-matplotlib-1f4b8eba6aa1

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.preprocessing

parser = argparse.ArgumentParser()
parser.add_argument("input_filename", type=Path)
parser.add_argument("--output-path", type=Path, default=None)
parser.add_argument("--require-real-probabilities", action="store_true")


class NpEncoder(json.JSONEncoder):
    """https://stackoverflow.com/a/57915246"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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


def plot_pr_curve(groundtruth, pred_proba, ax=None):
    assert groundtruth.ndim == 1, "Expected 1d array with various classes, not one-hot."
    if ax is None:
        ax = plt.gca()
    classes = np.unique(groundtruth)
    is_multilabel_classification = classes.size > 2
    if not is_multilabel_classification:
        raise NotImplementedError(
            "PR curve plotting currently only implemented for multiclass."
        )
    for cls in classes:
        cls_prob = pred_proba[:, cls]
        pr, re, _ = sklearn.metrics.precision_recall_curve(groundtruth == cls, cls_prob)
        ax.plot(re, pr, label=f"Ground truth: {cls}")
    ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    return ax


def plot_confusion_matrix(groundtruth, pred_class, ax=None, rename=False):
    # if len(np.unique(groundtruth) == 2):
    #    groundtruth = ["yes" if i else "no" for i in groundtruth]
    #    pred_class = ["yes" if i else "no" for i in pred_class]
    ax = (
        sklearn.metrics.ConfusionMatrixDisplay(
            sklearn.metrics.confusion_matrix(groundtruth, pred_class, normalize="true")
        )
        .plot(ax=ax, colorbar=False)
        .ax_
    )
    ax.get_images()[0].set_clim([0, 1])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    if rename:
        a = ax.get_xticks().tolist()
        a[0] = "no"
        a[1] = "yes"
        ax.set_xticklabels(["no", "yes"])
        a = ax.get_yticks().tolist()
        a[0] = "no"
        a[1] = "yes"
        ax.set_yticklabels(["no", "yes"])


def mean_average_precision_score(groundtruth, pred_proba):
    if pred_proba.shape[1] < 5:
        print(
            "WARNING: one or several tremor severities missing in pred_proba! Shape:",
            pred_proba.shape,
        )
    gt_classes = np.sort(groundtruth.unique())
    pred_proba_in_gt = pred_proba
    gt_onehot = sklearn.preprocessing.label_binarize(groundtruth, classes=gt_classes)
    if gt_classes.size < pred_proba.shape[1]:
        pred_proba_in_gt = pred_proba[:, gt_classes]
    if gt_classes.size == 2:
        assert gt_classes[0] == 0
        assert gt_classes[1] == 1
        print("This is a binary case for tremor!")
        gt_onehot = np.hstack((gt_onehot, 1 - gt_onehot))
    return sklearn.metrics.average_precision_score(gt_onehot, pred_proba_in_gt)


mean_average_precision_scorer = sklearn.metrics.make_scorer(
    mean_average_precision_score, greater_is_better=True, needs_proba=True
)


plt.style.use(["science", "src/visualization/main.mplstyle"])


def binary_scores(y_true, y_pred, y_prob):
    assert np.unique(y_true).size == 2
    assert np.unique(y_pred).size <= 2
    assert y_prob.ndim == 1
    d = {}

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    p = tp + fn
    n = tn + fp
    d["TN"] = tn
    d["FP"] = fp
    d["FN"] = fn
    d["TP"] = tp
    d["AP"] = sklearn.metrics.average_precision_score(y_true, y_prob)
    d["recall"] = sklearn.metrics.recall_score(y_true, y_pred)
    d["precision"] = sklearn.metrics.precision_score(y_true, y_pred)
    d["specificity"] = tn / (tn + fp)
    d["accuracy"] = sklearn.metrics.accuracy_score(y_true, y_pred)
    d["expected AP"] = p / (p + n)

    assert (tp + tn) / (p + n) == d["accuracy"]
    return d


if __name__ == "__main__":
    args = parser.parse_args()

    pred = np.load(args.input_filename, allow_pickle=True)
    pred_class = pred["class_predictions"].astype(np.int8)
    pred_proba = pred["probabilities"]
    groundtruth = pred["groundtruth"].astype(np.int8)

    # Preconditions
    if args.require_real_probabilities:
        np.testing.assert_almost_equal(
            pred_proba.sum(axis=1),
            1,
            decimal=4,
            err_msg="Predicted probabilites should add up to approx. 1 (second Kolmogorov axiom).",
        )

    # HACK since there is no tremor severity 4 in the training data.
    if np.unique(groundtruth).size == 5 and args.input_filename.parent.name.startswith(
        "tremor_"
    ):
        mask = groundtruth != 4
        pred_class = pred_class[mask]
        pred_proba = pred_proba[mask, :]
        groundtruth = groundtruth[mask]

    gt_classes = np.unique(groundtruth)

    if gt_classes.size > 2:
        metrics_dict = {"overall": {}}
        pred_proba_in_gt = pred_proba
        metrics_dict["classwise"] = {}
        # Ignore prediction probabilities corresponding to labels not in ground truth
        pred_proba_in_gt = pred_proba[:, tuple(gt_classes)]

        # Calculate per-class metrics
        for cls in gt_classes:
            cls_key = str(cls)
            metrics_dict["classwise"][cls_key] = {}
            cls_prob = pred_proba[:, cls]
            metrics_dict["classwise"][cls_key] = binary_scores(
                y_true=groundtruth == cls, y_pred=pred_class == cls, y_prob=cls_prob
            )
        # Calculate overall metrics
        metrics_dict["overall"]["mAP"] = sklearn.metrics.average_precision_score(
            sklearn.preprocessing.label_binarize(groundtruth, classes=gt_classes),
            pred_proba_in_gt,
        )
        metrics_dict["overall"]["accuracy"] = sklearn.metrics.accuracy_score(
            groundtruth, pred_class
        )
        metrics_dict["overall"]["expected mAP"] = np.average(
            [i["expected AP"] for i in metrics_dict["classwise"].values()]
        )
        metrics_dict["overall"][
            "balanced accuracy"
        ] = sklearn.metrics.balanced_accuracy_score(groundtruth, pred_class)
        metrics_dict["overall"][
            "adjusted balanced accuracy"
        ] = sklearn.metrics.balanced_accuracy_score(
            groundtruth, pred_class, adjusted=True
        )

        # Postconditions
        np.testing.assert_almost_equal(
            metrics_dict["overall"]["mAP"],
            np.average([i["AP"] for i in metrics_dict["classwise"].values()]),
        )
        np.testing.assert_almost_equal(
            metrics_dict["overall"]["balanced accuracy"],
            np.average([i["recall"] for i in metrics_dict["classwise"].values()]),
        )

    else:
        pred_proba_1d = pred_proba
        if pred_proba.ndim == 2:
            pred_proba_1d = pred_proba[:, 1]
        metrics_dict = binary_scores(groundtruth, pred_class, pred_proba_1d)
        metrics_dict["auroc"] = sklearn.metrics.roc_auc_score(
            groundtruth, pred_proba_1d
        )
        metrics_dict["balanced accuracy"] = sklearn.metrics.balanced_accuracy_score(
            groundtruth, pred_class
        )
        metrics_dict[
            "adjusted balanced accuracy"
        ] = sklearn.metrics.balanced_accuracy_score(
            groundtruth, pred_class, adjusted=True
        )

    print(json.dumps(metrics_dict, cls=NpEncoder, indent=4))

    if args.output_path is not None:
        # Make figures
        args.output_path.mkdir(exist_ok=True)
        fig, ax = plt.subplots()
        try:
            plot_pr_curve(groundtruth, pred_proba, ax=ax)
        except NotImplementedError:
            pass
        fig.savefig(args.output_path / "classwisepr.pgf")
        fig, ax = plt.subplots()
        plot_confusion_matrix(
            groundtruth, pred_class, ax=ax, rename="tremor" not in str(args.output_path)
        )
        SIZE = 4.2
        factor = 1.0 if "tremor" in str(args.output_path) else 0.65
        fig.set_size_inches((SIZE * 0.46 * factor, 5.92 * SIZE * factor))
        #        plt.colorbar(ax.get_images()[0], fraction=1, shrink=0.7)
        fig.savefig(args.output_path / "confusionmatrix.pgf")
