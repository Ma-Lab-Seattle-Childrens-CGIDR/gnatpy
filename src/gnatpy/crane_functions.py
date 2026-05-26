"""Functions for computing Centroid Rank Entropy (CRANE)"""

# Imports
# Standard Library Imports
from __future__ import annotations

from typing import Callable, Hashable, Iterable, Literal, Optional, Tuple, Union

# External Imports
import numpy as np
import pandas as pd
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
        rank_fun=_rank_array,
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
        rank_fun=_rank_array,
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
    rank_array: Array2D,
) -> Array1D:
    return rank_array.mean(axis=0).reshape(1, -1)


def _rank_grouping_score(
    rank_array: Array2D, centroid: Optional[Array1D] = None
) -> float:
    if centroid is None:
        centroid = _rank_centroid(rank_array)
    return _centroid_distances(rank_array, centroid).mean()


def _centroid_distances(
    rank_array: Array2D, rank_centroid: Optional[Array1D] = None
) -> Array1D:
    if rank_centroid is None:
        rank_centroid = _rank_centroid(rank_array)
    return np.sqrt(
        np.square(np.subtract(rank_array, rank_centroid)).sum(axis=1)
    ).reshape(1, -1)


def _crane_differential_entropy(
    rank_array_a: Array2D,
    rank_array_b: Array2D,
) -> float:
    return np.abs(
        _rank_grouping_score(rank_array_a) - _rank_grouping_score(rank_array_b)
    )


def _gene_contributions(
    in_array: Array2D,
    method: Literal[
        "average",
        "min",
        "max",
        "dense",
        "ordinal",
    ] = "average",
) -> Array1D:
    # Get the rank array, and the rank centroid
    rank_array = _rank_array(in_array, method=method)
    # Use the standard deviation as the contribution
    return np.std(rank_array, axis=0)


# endregion Rank Centroid Functions

# region Classification rate functions


def _crane_classification_rate(rank_array_a: Array2D, rank_array_b: Array2D) -> float:
    # Compute the rank centroids
    centroid_a = _rank_centroid(rank_array_a)
    centroid_b = _rank_centroid(rank_array_b)

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
    total_samples = rank_array_a.shape[0] + rank_array_b.shape[0]
    correct_samples = (dist_diff_a < 0.0).sum() + (dist_diff_b >= 0.0).sum()

    return correct_samples / total_samples


# endregion Classification rate functions


# Multiway CRANE:
# 1.) Find the centroid for each group
# 2.) Find the centroid across all samples (grand mean)
# 3.) Calculate between group sum of squared differences
# 4.) Calculate the within group sum of squared differences
# 5.) Divide values from 3,4 by their degrees of freedom (N-k for between group, and n-1 for within group)
# 6.) Take the ratio of the between group value and the within group value
def crane_multiway_classification(
    expression_data: Union[Array2D, pd.DataFrame],
    sample_groups: Union[Iterable[Array1D], Iterable[Iterable[Hashable]]],
    gene_network: Union[Array1D, Iterable[Hashable]],
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str, float, Callable[[gaussian_kde], float]]] = None,
    iterations: int = 1_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes: int = -1,
) -> Tuple[float, float]:
    """
    Calculate the CRANE multiway rank classification, an extension of
    CRANE classification rate to more than 2 groups

    Parameters
    ----------
    expression_data : Array2D or pd.DataFrame
        Gene expression data, either a numpy array or a pandas
        dataframe, with rows representing different samples, and
        columns representing different genes
    sample_groups : Iterable of Array1D or Iterable of Iterable of Hashable
        The sample groups to compare, can be an iterable of numpy arrays with
        integer indices, or an iterable of iterables of values used to index a pandas
        DataFrame (if the expression data is a DataFrame)
    gene_network : Array1D or Iterable of Hashable
        Which genes belong to the gene network, can be a numpy array with
        integer indices, or an iterable of values used to index a pandas
        DataFrame (if the expression data is a DataFrame)
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
    Tuple of (float,float)
        Tuple of the multiway DIRAC statistic, and the significance level
        found via bootstrapping
    """
    return _bootstrap_rank_entropy_p_value(
        samples_array=expression_data,
        sample_groups=sample_groups,
        gene_network=gene_network,
        rank_entropy_fun=_crane_multiway,
        rank_fun=_rank_array,
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


def _crane_multiway(*rank_arrays: Array2D) -> float:
    # Combine the arrays and find the rank array
    combined_rank_array = np.vstack(rank_arrays)
    # Find the combined mean
    combined_centroid = _rank_centroid(combined_rank_array)

    # Find the centroids of each group
    start_idx = 0
    # Track the within group and between group centroid distances
    between_group_distance_sum = 0.0
    within_group_distance_sum = 0.0
    for row_idx, a in enumerate(rank_arrays):
        ra = combined_rank_array[start_idx : (start_idx + a.shape[0])]
        start_idx += a.shape[0]
        # Get the centroid of the group
        centroid = _rank_centroid(ra)
        # Get the within group distances
        within_group_distance_sum += _centroid_distances(ra, centroid).sum()
        # Get the between group distance
        between_group_distance_sum += (
            np.sqrt(np.square(np.subtract(centroid, combined_centroid)).sum())
            * ra.shape[0]  # weight by the number of samples in this group
        )
    return between_group_distance_sum / within_group_distance_sum
