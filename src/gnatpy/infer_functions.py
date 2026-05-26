"""Functions for computing the Information Entropy of Ranks"""

# Imports
# Standard Library Imports
from __future__ import annotations

from typing import cast, Callable, Optional, Tuple, Union

# Enternal Imports
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy import special


# Local Imports
from gnatpy.dirac_functions import _rank_array
from gnatpy._bootstrap_pvalue import (
    _bootstrap_rank_entropy_p_value,
)
from gnatpy.gnatpy_types import Array1D, Array2D


# region Main Functions
def infer_gene_set_entropy(
    expression_data: Union[Array2D, pd.DataFrame],
    sample_group1,
    sample_group2,
    gene_network,
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str | float | Callable[[gaussian_kde], float]]] = None,
    iterations: int = 1_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=1,
) -> Tuple[float, float]:
    """Calculate the difference in information entropy of ranks, and it's significance

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
        Tuple of the difference in information entropy of ranks, and the
        significance level found via bootstrapping
    """
    return _bootstrap_rank_entropy_p_value(
        samples_array=expression_data,
        sample_groups=[sample_group1, sample_group2],
        gene_network=gene_network,
        rank_entropy_fun=_infer_differential_entropy,  # type: ignore
        rank_fun=_rank_array,
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


# endregion Main Functions

# region Helper Functions


def _pairwise_rank_entropy(rank_array: Array2D) -> float:
    # Get the DIRAC rank template matrix, and calculate the frequency
    # of 1's in each column
    freqs = np.mean(rank_array, axis=0)
    return cast(float, np.mean(special.entr(freqs) + special.entr(1 - freqs)))


def _gene_contributions(in_array: Array2D) -> Array1D:
    # Get the rank array
    rank_array = _rank_array(in_array)
    freqs = np.mean(rank_array, axis=0)
    entropies = special.entr(freqs) + special.entr(1 - freqs)
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
        res_array[idx] = entropies[comparison_mask_array[idx]].mean()
    # Each gene comparison is double counted, so divide by 2 so that the
    # sum of the gene contributions adds up to the rank entropy (calculated
    # via mismatches)
    return res_array / 2.0


def _infer_differential_entropy(rank_array_a: Array2D, rank_array_b: Array2D) -> float:
    return np.abs(
        _pairwise_rank_entropy(rank_array_a) - _pairwise_rank_entropy(rank_array_b)
    )


# endregion Helper Functions
