"""Functions for computing differential rank conservation (DIRAC)"""

# Imports
# Standard Library Imports
from __future__ import annotations

from typing import Callable, Hashable, Iterable, Optional, Tuple, Union

# External Imports
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy import special

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
    sample_group1, sample_group2
        Which samples belong to each group. If expression_data is a numpy
        array, this should be a something able to index the rows of the
        array. If expression_data is a pandas dataframe, this should be
        something that can index rows of a dataframe inside a .loc (see
        pandas documentation for details)
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
        rank_entropy_fun=_dirac_classification_rate,  # type: ignore
        rank_fun=_rank_array,
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
    sample_group1, sample_group2
        Which samples belong to each group. If expression_data is a numpy
        array, this should be a something able to index the rows of the
        array. If expression_data is a pandas dataframe, this should be
        something that can index rows of a dataframe inside a .loc (see
        pandas documentation for details)
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
        sample_groups=[sample_group1, sample_group2],
        gene_network=gene_network,
        rank_entropy_fun=_dirac_differential_entropy,  # type:ignore
        rank_fun=_rank_array,
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


def _rank_template(rank_array: Array2D) -> Array1D:
    return np.greater(rank_array.mean(axis=0), 0.5).astype(int).reshape(1, -1)


def _rank_matching_scores(
    rank_array: Array2D, rank_template: Optional[Array1D] = None
) -> Array1D:
    if rank_template is None:
        rank_template = _rank_template(rank_array)
    return np.equal(rank_array, rank_template).mean(axis=1)


def _rank_mismatching_scores(
    rank_array: Array2D, rank_template: Optional[Array1D] = None
) -> Array1D:
    if rank_template is None:
        rank_template = _rank_template(rank_array)
    return np.not_equal(rank_array, rank_template).mean(axis=1)


def _rank_conservation_index(
    rank_array: Array2D, rank_template: Optional[Array1D] = None
) -> float:
    return _rank_matching_scores(rank_array, rank_template).mean()


def _rank_entropy(
    rank_array: Array2D, rank_template: Optional[Array1D] = None
) -> float:
    return _rank_mismatching_scores(rank_array, rank_template).mean()


def _gene_contributions(in_array: Array2D) -> Array1D:
    # Get the rank array, and rank template
    rank_array = _rank_array(in_array)
    rank_template = _rank_template(rank_array)
    # Find the proportion of mismatches for each comparison
    mismatch_prop = np.not_equal(rank_array, rank_template).mean(axis=1)

    # Set up a numpy array of masks for which comparisons each gene is involved in
    num_genes = in_array.shape[1]
    comparison_mask_array = np.zeros(
        (num_genes, special.comb(num_genes, 2, exact=True)), dtype=bool
    )
    counter = 0
    for g1 in range(num_genes):
        for g2 in range(g1 + 1, num_genes):
            comparison_mask_array[g1, counter] = True
            comparison_mask_array[g2, counter] = True
            counter += 1
    # Find the contribution of each gene
    res_array = np.zeros((num_genes,))
    for idx in range(num_genes):
        res_array[idx] = mismatch_prop[comparison_mask_array[idx]].mean()
    # Each gene comparison is double counted, so divide by 2 so that the
    # sum of the gene contributions adds up to the rank entropy (calculated
    # via mismatches)
    return res_array / 2.0


def _dirac_differential_entropy(rank_array_a: Array2D, rank_array_b: Array2D) -> float:
    return np.abs(
        _rank_conservation_index(rank_array_a) - _rank_conservation_index(rank_array_b)
    )


# endregion Rank Vector

# region classification


def _dirac_classification_rate(rank_array_a: Array2D, rank_array_b: Array2D) -> float:
    # Find the rank Templates
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
    total_samples = rank_array_a.shape[0] + rank_array_b.shape[0]
    correct_samples = (rank_difference_a > 0.0).sum() + (rank_difference_b <= 0.0).sum()

    return correct_samples / total_samples


# endregion classification

# region DIRAC multiway


def dirac_multiway_classification(
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
    Calculate the DIRAC multiway rank classification, an extension of
    DIRAC classification rate to more than 2 groups

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
        rank_entropy_fun=_dirac_multiway,
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


def _dirac_multiway(*arrays: Array2D) -> float:
    # Start by finding the mismatch score between a shared template and
    # all individual samples,
    # and then also find this for each group

    # Combine all the samples into a single array
    combined_samples = np.vstack(arrays)
    # Find the rank array for all of the samples
    combined_rank_array = _rank_array(combined_samples)
    # Find the mismatch score between this combined rank array and all the individual samples
    combined_mismatch = _rank_mismatching_scores(
        combined_rank_array, _rank_template(combined_rank_array)
    ).sum()

    # Now find the gruopwise mismatch scores
    groupwise_mismatch = 0.0
    cur_idx = 0
    for a in arrays:
        # Get the rank array from the combined_rank_array to avoid recalculating
        rank_array = combined_rank_array[cur_idx : (cur_idx + a.shape[0])]
        cur_idx += a.shape[0]
        # Find the mismatch score
        groupwise_mismatch += _rank_mismatching_scores(
            rank_array, _rank_template(rank_array)
        ).sum()

    # The statistic is the ratio between these two values
    return combined_mismatch / groupwise_mismatch


# endregion DIRAC multiway


# NOTE: Multiway DIRAC:
# 1.) Find rank templates for each group, and all samples combined
# 2.) Find weighted sum (weighted by sample count) of matches from group templates to overall templates
# 3.) Find sum of matches from each sample to its own groups template
# 4.) The statistic is then the ratio of these two, between group mismatches / within group mismatches
