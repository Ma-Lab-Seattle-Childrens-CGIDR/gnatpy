"""Functions for computing Centroid Rank Entropy (CRANE)"""

# Imports
# Standard Library Imports
from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple, Union

# External Imports
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import gaussian_kde, rankdata

# Local imports
from gnatpy._bootstrap_pvalue import (
    _bootstrap_rank_entropy_p_value,
)
from gnatpy.gnatpy_types import Array1D, Array2D

# region Main Fuctions


def crane_gene_set_classification(
    expression_data: Array2D | pd.DataFrame,
    sample_group1,
    sample_group2,
    gene_network,
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str | float | Callable[[gaussian_kde], float]]] = None,
    iterations: int = 10_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=1,
) -> Tuple[float, float]:
    """Calculate the classification rate using CRANE rank centroid distances for a given network and its significance

    Parameters
    ----------
    expression_data : np.ndarray | pd.DataFrame
        Gene expression data, either a numpy array or a pandas
        DataFrame, with rows representing different samples, and columns
        representing different genes
    sample_group1
        Which samples belong to group1. If expression_data is a numpy
        array, this should be a something able to index the rows of the
        array. If expression_data is a pandas dataframe, this should be
        something that can index rows of a dataframe inside a .loc (see
        pandas documentation for details)
    sample_group2
        Which samples belong to group2, see sample_group1 information
        for more details.
    gene_network
        Which genes belong to the gene network. If expression_data is a
        numpy array, this should be something able to index the columns
        of the array. If expression_data is a pandas dataframe, this
        should be something be anything that can index columns of a
        dataframe inside a .loc (see pandas documentation for details)
    kernel_density_estimate : bool
        Whether to use a kernel density estimate for calculating the
        p-value. If True, will use a Gaussian Kernel Density Estimate,
        if False will use an empirical CDF
    bw_method : Optional[Union[str|float|Callable[[gaussian_kde], float]]]
        Bandwidth method, see `scipy.stats.gaussian_kde <https://docs.sc
        ipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.h
        tml>`_ for details
    iterations : int
        Number of iterations to perform during bootstrapping the null
        distribution
    replace : bool
        Whether to sample with replacement when randomly sampling from
        the sample groups during bootstrapping
    seed : int
        Seed to use for the random number generation during
        bootstrapping
    processes : int
        Number of processes to use during the bootstrapping, default 1

    Returns
    -------
    Tuple[float,float]
        Tuple of the classification rate, and the significance level
        found via bootstrapping
    """
    return _bootstrap_rank_entropy_p_value(
        samples_array=expression_data,
        sample_groups=[sample_group1, sample_group2],
        gene_network=gene_network,
        rank_entropy_fun=_crane_classification_rate,  # type: ignore
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


def crane_gene_set_entropy(
    expression_data: Array2D | pd.DataFrame,
    sample_group1,
    sample_group2,
    gene_network,
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str | float | Callable[[gaussian_kde], float]]] = None,
    iterations: int = 1_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=-1,
) -> Tuple[float, float]:
    """Calculate the difference in centroid rank entropy, and it's significance

    Parameters
    ----------
    expression_data : np.ndarray | pd.DataFrame
        Gene expression data, either a numpy array or a pandas
        DataFrame, with rows representing different samples, and columns
        representing different genes
    sample_group1
        Which samples belong to group1. If expression_data is a numpy
        array, this should be a something able to index the rows of the
        array. If expression_data is a pandas dataframe, this should be
        something that can index rows of a dataframe inside a .loc (see
        pandas documentation for details)
    sample_group2
        Which samples belong to group2, see sample_group1 information
        for more details.
    gene_network
        Which genes belong to the gene network. If expression_data is a
        numpy array, this should be something able to index the columns
        of the array. If expression_data is a pandas dataframe, this
        should be something be anything that can index columns of a
        dataframe inside a .loc (see pandas documentation for details)
    kernel_density_estimate : bool
        Whether to use a kernel density estimate for calculating the
        p-value. If True, will use a Gaussian Kernel Density Estimate,
        if False will use an empirical CDF
    bw_method : Optional[Union[str|float|Callable[[gaussian_kde], float]]]
        Bandwidth method, see `scipy.stats.gaussian_kde <https://docs.sc
        ipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.h
        tml>`_ for details
    iterations : int
        Number of iterations to perform during bootstrapping the null
        distribution
    replace : bool
        Whether to sample with replacement when randomly sampling from
        the sample groups during bootstrapping
    seed : int
        Seed to use for the random number generation during
        bootstrapping
    processes : int, optional
        Number of processes to use during the bootstrapping, defaults to
        all available processes

    Returns
    -------
    Tuple[float,float]
        Tuple of the difference in centroid rank entropy, and the
        significance level found via bootstrapping
    """
    return _bootstrap_rank_entropy_p_value(
        samples_array=expression_data,
        sample_groups=[sample_group1, sample_group2],
        gene_network=gene_network,
        rank_entropy_fun=_crane_differential_entropy,  # type: ignore
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


# endregion Main Functions


# region Rank Centroid Functions


def _rank_array(
    in_array: Array2D,
    method: Literal[
        "average",
        "min",
        "max",
        "dense",
        "ordinal",
    ] = "average",
) -> Array2D:
    """
    For each row in array, perform ranking and then rank normalization
    """
    ranks = rankdata(in_array, method=method, axis=1, nan_policy="omit")
    # Perform rank normalization, w = 1- (r(i) - 1)/|r|
    # This is equivalent to Borda normalization in the case where we have
    # full rank lists
    return 1 - (ranks - 1) / ranks.shape[1]


def _rank_centroid(
    in_array: Array2D,
    method: Literal[
        "average",
        "min",
        "max",
        "dense",
        "ordinal",
    ] = "average",
) -> NDArray[int]:
    return _rank_array(in_array=in_array, method=method).mean(axis=0).reshape(1, -1)


def _rank_grouping_score(in_array: Array2D) -> Array1D:
    ranked_array = _rank_array(in_array)
    centroid = ranked_array.mean(axis=0)
    return np.sqrt(np.square(np.subtract(ranked_array, centroid)).sum(axis=1)).mean()


def _crane_differential_entropy(
    a: Array2D,
    b: Array2D,
) -> float:
    return np.abs(_rank_grouping_score(a) - _rank_grouping_score(b))


# endregion Rank Centroid Functions

# region Classification rate functions


def _crane_classification_rate(a: Array2D, b: Array2D) -> float:
    # Compute the rank arrays
    rank_array_a = _rank_array(a)
    rank_array_b = _rank_array(b)

    # Compute the rank centroids
    centroid_a = rank_array_a.mean(axis=0).reshape(1, -1)
    centroid_b = rank_array_b.mean(axis=0).reshape(1, -1)

    # Calculate distances from the rank arrays to the centroids
    centroid_distance_a_array_a = np.sqrt(
        np.square(np.subtract(rank_array_a, centroid_a)).sum(axis=1)
    )
    centroid_distance_b_array_a = np.sqrt(
        np.square(np.subtract(rank_array_a, centroid_b)).sum(axis=1)
    )

    centroid_distance_a_array_b = np.sqrt(
        np.square(np.subtract(rank_array_b, centroid_a)).sum(axis=1)
    )
    centroid_distance_b_array_b = np.sqrt(
        np.square(np.subtract(rank_array_b, centroid_b)).sum(axis=1)
    )

    # Calculate the rank centroid distance difference
    dist_diff_a = centroid_distance_a_array_a - centroid_distance_b_array_a
    dist_diff_b = centroid_distance_a_array_b - centroid_distance_b_array_b

    # Calculate the accuracy
    total_samples = a.shape[0] + b.shape[0]
    correct_samples = (dist_diff_a < 0.0).sum() + (dist_diff_b >= 0.0).sum()

    return correct_samples / total_samples

    pass


# endregion Classification rate functions
