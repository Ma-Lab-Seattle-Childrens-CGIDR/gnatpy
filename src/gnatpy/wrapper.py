"""
Main functions for interfacing with GNATpy
"""

from gnatpy._bootstrap_pvalue import _bootstrap_rank_entropy_p_value

import itertools
from typing import Any, Callable, Hashable, Literal, Optional, Union

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


from gnatpy import dirac_functions as dirac
from gnatpy import infer_functions as infer
from gnatpy import crane_functions as crane
from gnatpy import race_functions as race
from gnatpy.gnatpy_types import Array2D


# NOTE:
# - One function to calculate rank entropy across gene networks and sample groups
# - One function to calculate the p-value difference between groups (which will also do multiway)
# - One function for classification rate? (Less needed, but maybe good for p-values)


def rank_entropy(
    data: Union[Array2D, pd.DataFrame, ad.AnnData],
    sample_groups: dict[Hashable, Any],
    gene_networks: dict[Hashable, Any],
    method: Literal["DIRAC", "INFER", "RACE", "CRANE"] = "DIRAC",
    layer: Optional[str] = None,
    processes=-1,
) -> pd.DataFrame:
    """
    Calculate the rank entropy for the sample groups across the gene networks

    Parameters
    ----------
    data : Array2D or AnnData
        The data to calculate the rank entropy values for
    sample_groups : dict
        Groups of samples to evaluate the rank entropy for. Keys will be
        used to name the rows in the returned dataframe, and the values
        should be able to index the rows of `data`.
    gene_networks : dict
        Dict of gene networks to evaluate the rank entropy for,
        the keys will be used to name the columns of the returned
        dataframe, and the values should be able to index the columns
        of `data`.
    method : {'DIRAC', 'INFER', 'RACE', 'CRANE'}
        The method to use for calculating the rank entropy
    layer : str, optional
        The layer of the AnnData instance to calculate the rank entropy for,
        ignored if data isn't an AnnData instance
    processes : int
        Number of processes to use during the bootstrapping, default 1

    Returns
    -------
    rank_entropy : pd.DataFrame
        DataFrame with the rank entropy valu es, indexed by sample groups with
        columns for each gene network
    """
    # Create the Results DataFrame
    rank_entropy_df = pd.DataFrame(
        np.nan,
        index=pd.Index(sample_groups.keys()),
        columns=pd.Index(gene_networks.keys()),
    )
    # Create the method to find the rank entropy
    if method == "DIRAC":

        def rank_entropy_method(data: Array2D) -> float:
            return dirac._rank_entropy(dirac._rank_array(data))
    elif method == "RACE":

        def rank_entopy_method(data: Array2D) -> float:
            return race._rank_correlation_mean(data)
    elif method == "CRANE":

        def rank_entropy_method(data: Array2D) -> float:
            return crane._rank_grouping_score(crane._rank_array(data))
    elif method == "INFER":

        def rank_entropy_method(data: Array2D) -> float:
            return infer._pairwise_rank_entropy(infer._rank_array(data))
    else:
        raise ValueError(
            f"Invalid method, expected 'DIRAC', 'INFER', 'RACE', or 'CRANE' but received {method}"
        )
    for row, col, value in joblib.Parallel(
        n_jobs=processes, return_as="generator_unordered"
    )(
        joblib.delayed(_rank_entropy_worker)(
            data=data,
            sample_group=sg,
            gene_network=gn,
            sample_group_name=sg_name,
            gene_network_name=gn_name,
            rank_entropy_method=rank_entopy_method,
            layer=layer,
        )
        for (gn, gn_name), (sg, sg_name) in itertools.product(
            gene_networks.items(), sample_groups.items()
        )
    ):
        rank_entropy_df.loc[row, col] = value

    return rank_entropy_df


def rank_entropy_comparison(
    data: Union[Array2D, pd.DataFrame, ad.AnnData],
    sample_group1,
    sample_group2,
    gene_networks: dict[Hashable, Any],
    method: Literal["DIRAC", "INFER", "RACE", "CRANE"] = "DIRAC",
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str, float, Callable[[gaussian_kde], float]]] = None,
    iterations: int = 1_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=1,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate the differential rank entropy between two sample groups for
    each gene network in `gene_networks`

    Parameters
    ----------
    data : Array2D or AnnData
        The data to calculate the rank entropy values for
    sample_group1, sample_group2
        Which samples belong to each group. If expression_data is a numpy
        array, this should be a something able to index the rows of the
        array. If expression_data is a pandas dataframe, this should be
        something that can index rows of a dataframe inside a .loc (see
        pandas documentation for details)
    gene_networks : dict
        Dict of gene networks to evaluate the rank entropy for,
        the keys will be used as the indexes of the returned
        series, and the values should be able to index the columns
        of `data`.
    method : {'DIRAC', 'INFER', 'RACE', 'CRANE'}
        The method to use for calculating the rank entropy
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
    tuple of pd.Series,pd.Series
        Tuple of the difference in rank conservation index, and the
        significance level found via bootstrapping for all of the
        gene networks. The series are indexed by the keys of
        the `gene_networks` dict
    """
    if method == "DIRAC":
        rank_entropy_diff_fun = dirac._dirac_differential_entropy
        rank_fun = dirac._rank_array

    elif method == "RACE":
        rank_entropy_diff_fun = race._rank_correlation_mean

        def rank_fun(data: Array2D) -> Array2D:
            return data
    elif method == "CRANE":
        rank_entropy_diff_fun = (crane._crane_differential_entropy,)
        rank_fun = crane._rank_array

    elif method == "INFER":
        rank_entropy_diff_fun = infer._pairwise_rank_entropy
        rank_fun = infer._rank_array

    else:
        raise ValueError(
            f"Invalid method, expected 'DIRAC', 'INFER', 'RACE', or 'CRANE' but received {method}"
        )
    gene_network_index = pd.Index(gene_networks.keys())
    rank_entropy_difference = pd.Series(np.nan, index=gene_network_index)
    rank_entropy_difference_pvalue = pd.Series(np.nan, index=gene_network_index)

    for gene_network_name, gene_network in gene_networks.items():
        stat, pvalue = _bootstrap_rank_entropy_p_value(
            samples_array=data,
            sample_groups=[sample_group1, sample_group2],
            gene_network=gene_network,
            rank_entropy_fun=rank_entropy_diff_fun,  # type: ignore
            rank_fun=rank_fun,
            kernel_density_estimate=kernel_density_estimate,
            bw_method=bw_method,
            iterations=iterations,
            replace=replace,
            seed=seed,
            processes=processes,
        )
        rank_entropy_difference[gene_network_name] = stat
        rank_entropy_difference_pvalue[gene_network_name] = pvalue

    return rank_entropy_difference, rank_entropy_difference_pvalue


def _rank_entropy_worker(
    data: Union[Array2D, pd.DataFrame, ad.AnnData],
    sample_group,
    gene_network,
    sample_group_name,
    gene_network_name,
    rank_entropy_method: Callable[[Array2D], float],
    layer: Optional[str],
) -> tuple[Hashable, Hashable, float]:
    if isinstance(data, pd.DataFrame):
        rank_entropy = rank_entropy_method(
            data.loc[sample_group, gene_network].to_numpy()
        )
    elif isinstance(data, ad.AnnData):
        rank_entropy = rank_entropy_method(
            data[sample_group, gene_network].to_df(layer=layer).to_numpy()
        )
    else:
        rank_entropy = rank_entropy_method(np.array(data))
    return sample_group_name, gene_network_name, rank_entropy
