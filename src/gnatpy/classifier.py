"""
Scikit-learn classifiers based on DIRAC and CRANE
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.metrics import euclidean_distances

from gnatpy import dirac_functions as dirac
from gnatpy import crane_functions as crane


class DiracClassifier(ClassifierMixin, BaseEstimator):
    """
    a classifier based on DIRAC rank templates

    Attributes
    ----------
    x_ : ndarray, shape (n_samples, n_features)
        the input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        the labels passed during :meth:`fit`.

    rank_templates_ : ndarray, shape (n_classes, (n_features_in_*(n_features_in_-1)/2))
        the rank templates for each class

    classes_ : ndarray, shape (n_classes,)
        the classes seen at :meth:`fit`.

    n_features_in_ : int
        number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        names of features seen during :term:`fit`. defined only when `x`
        has feature names that are all strings.
    """

    _parameter_constraints = {}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the classifier

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Retuns self.rns
        -------
        self : object
            Returns self.
        """

        # Validate the input data
        X, y = validate_data(self, X, y)
        # Check that the targets are valid
        check_classification_targets(y)

        # Store the classes
        self.classes_ = np.unique(y)

        # Store the training data
        self.X_ = X
        self.y_ = y

        # Create the rank templates for the different classes
        self.rank_templates_ = np.empty(
            (
                self.classes_.shape[0],
                (self.n_features_in_ * (self.n_features_in_ - 1) // 2),
            )
        )
        for idx, c in enumerate(self.classes_):
            c_X = X[y == c, :]
            self.rank_templates_[idx, :] = dirac._rank_template(c_X)

        # Return the classifier
        return self

    def predict(self, X):
        """
        Predict the classes of X by matching them against the rank
        templates seen during fit

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of class with the
            best rank template match
        """
        # Check that the model is fit
        check_is_fitted(self)

        # Validate the input
        X = validate_data(self, X, reset=False)

        # Find which class template each sample is closest to
        # First, find the rank array (which is the DIRAC rank vector for each row)
        rank_array = dirac._rank_array(X)
        # Let c = number of classes, v = length of dirac rank vectors
        # and s = number of samples
        # Rank array is currently a SxV, and we want an SxCxV,
        # so use expand dims and repeat C times along the expanded dim
        rank_array = np.repeat(np.expand_dims(rank_array, 1), self.classes_.shape[0], 1)
        # The rank template array is a CxV array, so is now broadcastable with the rank_array
        classif_array = np.equal(rank_array, self.rank_templates_)
        # This array is now a SxCxV array of 0 and 1, and we want to sum over the V dimension
        classif_array = np.sum(classif_array, 2)
        # Leaving an SxC array, which we want to argmax along the c dimension
        classif_array = np.argmax(classif_array, 1)
        # Now a 1D array of S, just need to find the classes
        return self.classes_[classif_array]


class CraneClassifier(ClassifierMixin, BaseEstimator):
    """
    a classifier based on CRANE rank centroids

    Parameters
    ----------
    ties_method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        Method used to assign ranks to tied elements, see `SciPy Docs for details <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html>`_

    Attributes
    ----------
    x_ : ndarray, shape (n_samples, n_features)
        the input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        the labels passed during :meth:`fit`.

    rank_centroids : ndarray, shape (n_classes, n_features)
        the rank templates for each class

    classes_ : ndarray, shape (n_classes,)
        the classes seen at :meth:`fit`.

    n_features_in_ : int
        number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        names of features seen during :term:`fit`. defined only when `x`
        has feature names that are all strings.
    """

    _parameter_constraints = {"ties_method": [str]}

    def __init__(self, ties_method="average"):
        self.ties_method = ties_method

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the classifier

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Retuns self.rns
        -------
        self : object
            Returns self.
        """
        # validate the data
        X, y = validate_data(self, X, y)
        # Check that the targets are valid
        check_classification_targets(y)

        # Store the classes
        self.classes_ = np.unique(y)

        # Store the training data
        self.X_ = X
        self.y_ = y

        # Create the rank centroids
        self.rank_centroids_ = np.empty((self.classes_.shape[0], self.n_features_in_))
        for idx, c in enumerate(self.classes_):
            c_X = X[y == c, :]
            self.rank_centroids_[idx, :] = crane._rank_centroid(
                c_X, method=self.ties_method
            )

        # Return the classifier
        return self

    def predict(self, X):
        """
        Predict the classes of X by finding the class
        with the closest rank centroid

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of class with the
            best rank template match
        """
        # Check that the model is fit
        check_is_fitted(self)

        # Validate the input
        X = validate_data(self, X, reset=False)

        # Compute the rank array for the input X
        rank_array = crane._rank_array(X, "average")

        # Compute the euclidean distances between the samples
        # in the rank array
        # Euclidean distances returns array with shape n_samples_X, n_samples_Y
        # In this case, the n_samples_Y corresponds to the classes seen during fitting
        # So we take the argmin along that axis, and get the class index for
        # each sample, and we use that to select the class labels from the
        # classes array
        return self.classes_[
            np.argmin(euclidean_distances(rank_array, self.rank_centroids_), axis=1)
        ]
