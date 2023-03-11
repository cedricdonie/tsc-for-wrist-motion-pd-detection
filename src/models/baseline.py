# Baseline models for comparison
# Author(s): Cedric Donie (cedricdonie@gmail.com)

import numpy as np
import torch
from sktime_dl.classification._classifier import BaseDeepClassifier
from sktime_dl.utils import check_and_clean_data, check_is_fitted
from src.features.get_features import get_features_dynamic_gauss
from src.models.nn_mlp import MLPClassModel
from torchsample.modules import ModuleTrainer


class GaussianProcessClassifier(BaseDeepClassifier):
    def build_model(self, input_shape, nb_classes, hidden_dims=[128, 128]):
        mlpclassmodel = MLPClassModel(
            input_dim=input_shape, output_dim=nb_classes, hidden_dims=hidden_dims
        )
        trainer = ModuleTrainer(mlpclassmodel)
        loss = torch.nn.CrossEntropyLoss(torch.ones(nb_classes))
        trainer.compile(loss=loss, optimizer="adam")
        return trainer

    def __init__(self, nb_epochs=100, verbose=False):
        self.nb_epochs = nb_epochs
        self.verbose = verbose
        super().__init__()

    def fit(self, X, y, input_checks=True, validation_X=None, validation_y=None):
        """
        Fit the classifier on the training set (X, y)

        Parameters
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        y : array-like, shape = [n_instances]
            The training data class labels.
        input_checks : boolean
            whether to check the X and y parameters
        validation_X : [Not implemented] a nested pd.Dataframe, or array-like of shape =
        (n_instances, series_length, n_dimensions)
            The validation samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
            Unless strictly defined by the user via callbacks (such as
            EarlyStopping), the presence or state of the validation
            data does not alter training in any way. Predictions at each epoch
            are stored in the model's fit history.
        validation_y : [Not implemented] array-like, shape = [n_instances]
            The validation class labels.

        Returns
        -------
        self : object
        """
        assert (
            validation_X is None
        ), "Passing validation data is currently not implemented."
        assert (
            validation_y is None
        ), "Passing validation data is currently not implemented."

        X = check_and_clean_data(X, y, input_checks=input_checks)
        X = X.squeeze()
        y_onehot = self.convert_y(y)

        X_features = get_features_dynamic_gauss(X)

        self.input_shape = X_features.shape[1]

        self.model = self.build_model(self.input_shape, self.nb_classes)

        X_features_tensor = torch.from_numpy(X_features.astype(np.float32))
        y_tensor = torch.from_numpy(y_onehot.astype(np.float32))

        self.model.fit(
            X_features_tensor,
            y_tensor,
            num_epoch=self.nb_epochs,
            shuffle=True,
            verbose=self.verbose,
        )
        self._is_fitted = True

    def predict_proba(self, X, input_checks=True, **kwargs):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        input_checks: boolean
            whether to check the X parameter

        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        check_is_fitted(self)

        X = check_and_clean_data(X, input_checks=input_checks)
        X = X.squeeze()
        Xf = get_features_dynamic_gauss(X)
        Xf_tensor = torch.from_numpy(Xf.astype(np.float32))

        with torch.no_grad():
            likelihoods = self.model.predict(Xf_tensor)
            probs = torch.nn.functional.softmax(likelihoods, dim=1).numpy()

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])

        return probs
