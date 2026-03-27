"""Functions for computing differential rank conservation (DIRAC)"""

# Imports
# Standard Library Imports
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

# External Imports
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

# Local Imports
from gnatpy._bootstrap_pvalue import (
    _bootstrap_rank_entropy_p_value,
)
from gnatpy.gnatpy_types import Array1D, Array2D

# region Main Functions


def dirac_gene_set_classification(
    expression_data: Union[Array2D, pd.DataFrame],
    sample_group1,
    sample_group2,
    gene_network,
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str, float, Callable[[gaussian_kde], float]]] = None,
    iterations: int = 10_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=1,
) -> Tuple[float, float]:
    """Calculate the classification rate using DIRAC rank difference scores for a given network and its significance

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
        sample_group1=sample_group1,
        sample_group2=sample_group2,
        gene_network=gene_network,
        rank_entropy_fun=_dirac_classification_rate,
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


def dirac_gene_set_entropy(
    expression_data: Union[Array2D, pd.DataFrame],
    sample_group1,
    sample_group2,
    gene_network,
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str, float, Callable[[gaussian_kde], float]]] = None,
    iterations: int = 1_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=1,
) -> Tuple[float, float]:
    """Calculate the difference in rank conservation indices, and its significance

    Parameters
    ----------
    expression_data : np.ndarray or pd.DataFrame
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
    bw_method : str or float or Callable[[gaussian_kde], float], optional
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
    tuple of float,float
        Tuple of the difference in rank conservation index, and the
        significance level found via bootstrapping
    """
    return _bootstrap_rank_entropy_p_value(
        samples_array=expression_data,
        sample_group1=sample_group1,
        sample_group2=sample_group2,
        gene_network=gene_network,
        rank_entropy_fun=_dirac_differential_entropy,
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


# endregion Main Functions


# region Rank Vector


def _rank_vector(in_vector: Array1D) -> Array1D:
    rank_array = np.repeat(in_vector.reshape(1, -1), len(in_vector), axis=0)
    diff_array = rank_array - rank_array.T
    return (diff_array[np.triu_indices(len(in_vector), k=1)] > 0).astype(int)


def _rank_array(in_array: Array2D) -> Array2D:
    return np.apply_along_axis(_rank_vector, axis=1, arr=in_array)


def _rank_template(in_array: Array2D) -> Array1D:
    return (
        np.greater(_rank_array(in_array).mean(axis=0), 0.5).astype(int).reshape(1, -1)
    )


def _rank_matching_scores(in_array: Array2D) -> Array1D:
    rank_array = _rank_array(in_array)
    rank_template = np.greater(rank_array.mean(axis=0), 0.5).astype(int).reshape(1, -1)
    return np.equal(rank_array, rank_template).mean(axis=1)


def _rank_conservation_index(in_array: NDArray[int]) -> float:
    return _rank_matching_scores(in_array).mean()


def _dirac_differential_entropy(a: Array2D, b: Array2D) -> float:
    return np.abs(_rank_conservation_index(a) - _rank_conservation_index(b))


# endregion Rank Vector

# region classification


def _dirac_classification_rate(a: Array2D, b: Array2D) -> float:
    # Find the rank Templates
    rank_array_a = _rank_array(a)
    rank_array_b = _rank_array(b)

    rank_template_a = (rank_array_a.mean(axis=0) > 0.5).astype(int).reshape(1, -1)
    rank_template_b = (rank_array_b.mean(axis=0) > 0.5).astype(int).reshape(1, -1)

    # Compute the Rank matching score for each array, for each phenotype
    rank_matching_score_array_a_phenotype_a = (
        np.equal(rank_array_a, rank_template_a)
    ).mean(axis=1)
    rank_matching_score_array_a_phenotype_b = (
        np.equal(rank_array_a, rank_template_b)
    ).mean(axis=1)

    rank_matching_score_array_b_phenotype_a = (
        np.equal(rank_array_b, rank_template_a)
    ).mean(axis=1)
    rank_matching_score_array_b_phenotype_b = (
        np.equal(rank_array_b, rank_template_b)
    ).mean(axis=1)

    # Calculate Rank Difference Scores
    rank_difference_a = (
        rank_matching_score_array_a_phenotype_a
        - rank_matching_score_array_a_phenotype_b
    )
    rank_difference_b = (
        rank_matching_score_array_b_phenotype_a
        - rank_matching_score_array_b_phenotype_b
    )

    # Calculate the accuracy
    total_samples = a.shape[0] + b.shape[0]
    correct_samples = (rank_difference_a > 0.0).sum() + (rank_difference_b <= 0.0).sum()

    return correct_samples / total_samples


# endregion classification


# NOTE: Multiway DIRAC:
# 1.) Find rank templates for each group, and all samples combined
# 2.) Find weighted sum (weighted by sample count) of matches from group templates to overall templates
# 3.) Find sum of matches from each sample to its own groups template
# 4.) The statistic is then the ratio of these two, between group mismatches / within group mismatches
