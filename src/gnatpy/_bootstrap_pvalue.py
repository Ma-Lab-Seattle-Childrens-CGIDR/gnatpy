# The code for calculating the p-value empirically is modified from SciPy,
# licensed under the BSD-3-Clause license. See the LICENSE file for more information
"""Bootstrap p-values for the various rank entropy methods"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Callable, Hashable, Iterable, Optional, Tuple, Union

# External Imports
import anndata as ad
import joblib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ecdf

# Local Imports
from gnatpy.gnatpy_types import Array1D, Array2D, EntropyFunction


# region Main Function
def _bootstrap_rank_entropy_p_value(
    samples_array: Array2D | np.typing.ArrayLike | pd.DataFrame | ad.AnnData,
    sample_groups: Iterable[Array1D] | Iterable[Iterable[Hashable]],
    gene_network: Array1D | Iterable[Hashable],
    rank_entropy_fun: EntropyFunction,
    rank_fun: Optional[Callable[[Array2D], Array2D]] = None,
    layer: Optional[str] = None,
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str, float, Callable[[gaussian_kde], float]]] = None,
    iterations: int = 1_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=-1,
) -> Tuple[float, float]:
    """Generate a rank entropy value from the rank_entropy_fun function, and bootstrap a p-value for it

    Parameters
    ----------
    samples_array : NDArray[int|float] | pd.DataFrame
        Gene expression data, either a numpy array or a pandas
        DataFrame, with rows representing different samples, and columns
        representing different genes
    sample_groups: Iterable of Array1D
        Which samples belong to each group. If expression_data is a numpy
        array, this should be a 1D ArrayLike of integers. If
        expression_data is a pandas dataframe, must be compatible with
        `pandas.Index.get_indexer <https://pandas.pydata.org/docs/reference/api/pandas.Index.get_indexer.html#pandas.Index.get_indexer>`_
    gene_network: Array1D or Iterable[Hashable]
        List of indices for genes in the gene network
    rank_entropy_fun : Callable[Array2D, Array2D], float]
        Function used to calculate the rank entropy difference between
        two sample groups, should take two np.ndarrays as arguments and
        return a float
    rank_fun : Callable[[Array2D], Array2D], optional
        Optional function which will create the ranked arrays prior to
        perfoming the boostrapping. It is called on the inputs
    layer : str, optional
        Layer containing the expression data in an AnnData object
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
        Number of processes to use during the bootstrapping, default will
        use all available

    Returns
    -------
    Tuple[float, float]
        Tuple of the return value from rank_entropy_fun(sample_group1
        array, sample_group2 array), and the p-value found by
        bootstrapping
    """
    # Begin by converting the expression data into the proper form
    # Convert dataframe into numpy array
    if isinstance(samples_array, pd.DataFrame):
        # Convert sample groups of labels into integer positions
        sample_groups = [
            samples_array.index.get_indexer(s).ravel() for s in sample_groups
        ]
        sample_group_sizes = list(map(len, sample_groups))
        # Filter the array for only the gene network
        samples_array = samples_array.loc[:, gene_network]
        # Get a numpy array from the dataframe
        samples_array = samples_array.to_numpy()
    elif isinstance(samples_array, ad.AnnData):
        sample_arrays_list = []
        sample_groups = []
        sample_group_sizes = []
        starting_idx = 0
        for sg in sample_groups:
            array = samples_array[sg, gene_network].to_df(layer=layer).to_numpy()
            size = array.shape[0]
            sample_arrays_list.append(array)
            sample_group_sizes.append(size)
            sample_groups.append(np.array(range(starting_idx, starting_idx + size)))
            starting_idx += size
        samples_array = np.concatenate(*sample_arrays_list, axis=0)

    else:
        samples_array = np.array(samples_array)
        sample_groups = [np.array(s).ravel() for s in sample_groups]
        gene_network = np.array(gene_network)
        sample_group_sizes = list(map(len, sample_groups))
        # Filter the samples array for only the gene network of interest
        samples_array = samples_array[:, gene_network]
    # Combine sample group indices
    sample_indices = np.concatenate(sample_groups)
    if rank_fun is None:

        def rank_fun(data: Array2D) -> Array2D:
            return data

    rank_array = rank_fun(samples_array)

    # Create a numpy rng
    rng = np.random.default_rng(seed=seed)
    # Create an array to hold the results
    rank_entropy_samples = np.empty((iterations,), dtype=float)
    # NOTE: For the null distribution the order of the entropy values doesn't matter
    for idx, entropy in enumerate(
        joblib.Parallel(n_jobs=processes, return_as="generator_unordered")(
            joblib.delayed(_pvalue_worker)(
                rank_entropy_fun=rank_entropy_fun,
                rank_array=rank_array,
                sample_indices=sample_indices,
                sample_group_sizes=sample_group_sizes,
                replace=replace,
                seed=rng.integers(low=0, high=np.iinfo(np.intp).max),
            )
            for _ in range(iterations)
        )
    ):
        rank_entropy_samples[idx] = entropy

    # Calculate the value for the unshuffled array
    sample_group_rank_arrays = [rank_array[sg] for sg in sample_groups]
    rank_entropy = rank_entropy_fun(*sample_group_rank_arrays)
    if not kernel_density_estimate:
        # Apply an adjustment based on 'Permutation p-values should never be zero:
        # calculating exact p-values when permutations are randomly drawn'
        # First find eps, which is floating point tolerance
        # Based on Scipy's permutation test implementation
        # Licensed under the BSD-3-Clause
        eps = (
            0
            if not np.isdtype(rank_entropy_samples.dtype, "real floating")
            else np.finfo(rank_entropy_samples.dtype).eps * 100
        )
        gamma = np.abs(eps * rank_entropy)
        pvalue = float(
            np.count_nonzero(rank_entropy_samples >= rank_entropy - gamma) + 1
        ) / float(iterations + 1)

        empirical_cdf = ecdf(rank_entropy_samples)
        pvalue = empirical_cdf.sf.evaluate(rank_entropy)[()]
        # End of code modified from SciPy
    else:
        kde = gaussian_kde(rank_entropy_samples, bw_method=bw_method)
        pvalue = kde.integrate_box_1d(rank_entropy, np.inf)
    return rank_entropy, pvalue


# endregion Main Function


def _pvalue_worker(
    rank_entropy_fun: EntropyFunction,
    rank_array: Array2D,
    sample_indices: Array1D,
    sample_group_sizes: Array1D,
    replace: bool,
    seed: int,
) -> float:
    """
    Worker for boostrapping a p-value, takes a samples array, breaks it into two
    groups based on the size of the sample groups, and uses the rank_entropy_fun
    to calculate the rank entropy

    Parameters
    ----------
    rank_entropy_fun : EntropyFunction
        Function which takes two numpy array representing the rank arrays and
        returns a single float
    rank_array : Array2D
        The array containing the samples
    sample_indices : Array1D
        The indices for samples. This will be split into
        two groups, and the the samples_array will be split
        using these indices. Each index specifies a row in
        the samples_array
    sample_group_sizes : Iterable[int]
        The size of the two sample groups
    replace : bool
        Whether to sample with replacement
    seed : int
        The seed for the RNG used for randomly splitting the samples indices
        into two groups

    Returns
    -------
    rank_entropy : float
        The rank entropy value of the rank_array
    """
    # Create the random number generator from the seed
    rng = np.random.default_rng(seed=seed)
    # Split the samples array
    if replace:
        sample_groups = [
            rng.choice(sample_indices, size=s, replace=replace)
            for s in sample_group_sizes
        ]
    else:
        shuffled_sample_indices = rng.permuted(sample_indices)
        sample_groups = []
        cur_idx = 0
        for sample_group_size in sample_group_sizes:
            sample_groups.append(
                shuffled_sample_indices[cur_idx : cur_idx + sample_group_size]
            )
            cur_idx += sample_group_size
    rank_arrays = [rank_array[sg] for sg in sample_groups]
    return rank_entropy_fun(*rank_arrays)
