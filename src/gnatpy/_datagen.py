"""Generate data for testing the rank entropy functions"""

# Imports
# Standard Library Imports
from __future__ import annotations

import functools
from typing import Union, Optional, Tuple


# External Imports
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, rv_discrete

# Local Imports
from gnatpy.gnatpy_types import Array2D

# Typing information
Distribution = Union[rv_continuous, rv_discrete]


# region Main Function


def _generate_rank_entropy_data(
    n_ordered_samples: int,
    n_unordered_samples: int,
    n_ordered_genes: int,
    n_unordered_genes: int,
    dist: Distribution,
    noise_dist: Optional[Distribution] = None,
    noise_swaps: Optional[Union[float, int]] = None,
    shuffle_genes: bool = True,
    shuffle_samples: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate data with ordered and disordered genes/samples

    Parameters
    ----------
    n_ordered_samples : int
        Number of ordered samples
    n_unordered_samples : int
        Number of unordered samples
    n_ordered_genes : int
        Number of ordered genes
    n_unordered_genes : int
        Number of unordered genes
    dist : Distribution
        Distribution to use for sampling, should be a scipy
        rv_continuous, or rv_discrete (or at least have a rvs method
        which matches the SciPy API). The seed to this function
        will be used to create an RNG which is passed as the
        random state to all calls to the rvs function.
    noise_dist : Distribution, optional
        Distribution to use for adding noise to the samples,
        used to generate random values for each entry in the expression
        array, and these values are added to the array
    noise_swaps : float or int, optional
        Number of random swaps to perform. Randomly selects a row, and then
        swaps two elements in that row. Can be an integer
        in which case that is the number of swaps, or a float
        between 0 and 1 in which case it is the proportion of the
        number of values in the resulting array to swap
        (note that it randomly swaps, so it could repeat some swaps
        so this proportion won't be the number of elements that are swapped,
        only the number of swaps that are performed). The swapping using the
        seed argument to seed its rng.
    shuffle_genes : bool
        Whether the order of the genes should be shuffled
    shuffle_samples : bool
        Whether the order of the samples should be shuffled
    seed : Optional[int]
        Seed to use for the random number generator used whe shuffling
        (doesn't change the sampling from the provided dist)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple of np.ndarrays, representing: 1. the generated expression
        data, with rows representing samples, and columns representing
        genes. 2. the indices of the ordered samples 3. the indices of
        the unordered samples 4. the indices of the ordered genes 5. the
        indices of the unordered genes
    """
    rng = np.random.default_rng(seed=seed)
    ordered_array = _ordered_array(
        nrow=n_ordered_samples,
        ncol=n_ordered_genes,
        dist=dist,
        col_shuffle=shuffle_genes,
        rng=rng,
    )
    unordered_array = _unordered_array(
        nrow=n_unordered_samples,
        ncol=n_ordered_genes,
        dist=dist,
        rng=rng,
    )
    ordered_genes_array = np.vstack((ordered_array, unordered_array))
    if shuffle_samples:
        samples_shuffled = rng.permuted(
            list(range(n_ordered_samples + n_unordered_samples))
        )
        ordered_samples = samples_shuffled[:n_ordered_samples]
        unordered_samples = samples_shuffled[n_ordered_samples:]
        ordered_genes_array[ordered_samples, :] = ordered_array
        ordered_genes_array[unordered_samples, :] = unordered_array
    else:
        ordered_samples = np.array(range(n_ordered_samples))
        unordered_samples = np.array(
            range(n_ordered_samples, n_unordered_samples + n_ordered_samples)
        )
    unordered_genes_array = _unordered_array(
        nrow=n_ordered_samples + n_unordered_samples,
        ncol=n_unordered_genes,
        dist=dist,
        rng=rng,
    )
    res_array = np.hstack((ordered_genes_array, unordered_genes_array))
    if shuffle_genes:
        genes_shuffled = rng.permuted(list(range(n_ordered_genes + n_unordered_genes)))
        ordered_genes = genes_shuffled[:n_ordered_genes]
        unordered_genes = genes_shuffled[n_ordered_genes:]
        res_array[:, ordered_genes] = ordered_genes_array
        res_array[:, unordered_genes] = unordered_genes_array
    else:
        ordered_genes = np.array(range(n_ordered_genes))
        unordered_genes = np.array(
            range(n_ordered_genes, n_unordered_genes + n_ordered_genes)
        )
    if noise_dist is not None:
        res_array = _noise_addition(res_array, dist=noise_dist, rng=rng)
    if noise_swaps is not None:
        res_array = _noise_swap(res_array, swaps=noise_swaps, rng=rng)
    return (
        res_array,
        ordered_samples,
        unordered_samples,
        ordered_genes,
        unordered_genes,
    )


def _generate_rank_entropy_anndata(
    n_ordered_samples: int,
    n_unordered_samples: int,
    n_ordered_genes: int,
    n_unordered_genes: int,
    dist: Distribution,
    noise_dist: Optional[Distribution] = None,
    noise_swaps: Optional[Union[float, int]] = None,
    shuffle_genes: bool = True,
    shuffle_samples: bool = True,
    seed: Optional[int] = None,
) -> ad.AnnData:
    """Generate data with ordered and disordered genes/samples

    Parameters
    ----------
    n_ordered_samples : int
        Number of ordered samples
    n_unordered_samples : int
        Number of unordered samples
    n_ordered_genes : int
        Number of ordered genes
    n_unordered_genes : int
        Number of unordered genes
    dist : Distribution
        Distribution to use for sampling, should be a scipy
        rv_continuous, or rv_discrete (or at least have a rvs method
        which matches the SciPy API). The seed to this function
        will be used to create an RNG which is passed as the
        random state to all calls to the rvs function.
    noise_dist : Distribution, optional
        Distribution to use for adding noise to the samples,
        used to generate random values for each entry in the expression
        array, and these values are added to the array
    noise_swaps : float or int, optional
        Number of random swaps to perform. Randomly selects a row, and then
        swaps two elements in that row. Can be an integer
        in which case that is the number of swaps, or a float
        between 0 and 1 in which case it is the proportion of the
        number of values in the resulting array to swap
        (note that it randomly swaps, so it could repeat some swaps
        so this proportion won't be the number of elements that are swapped,
        only the number of swaps that are performed). The swapping using the
        seed argument to seed its rng.
    shuffle_genes : bool
        Whether the order of the genes should be shuffled
    shuffle_samples : bool
        Whether the order of the samples should be shuffled
    seed : Optional[int]
        Seed to use for the random number generator used whe shuffling
        (doesn't change the sampling from the provided dist)

    Returns
    -------
    ad.AnnData
        AnnData with the generated expression data. The observations have metadata ('sample_type')
        which is 'ordered'/'unordered' to indicate the samples which are ordered. Simmilarly, the
        vars has metadata ('gene_type') which is 'ordered'/'unordered' to indicate which genes
        are ordered.
    """
    expr_data, ordered_samples, unordered_samples, ordered_genes, unordered_genes = (
        _generate_rank_entropy_data(
            n_ordered_samples=n_ordered_samples,
            n_unordered_samples=n_unordered_samples,
            n_ordered_genes=n_ordered_genes,
            n_unordered_genes=n_unordered_genes,
            dist=dist,
            noise_dist=noise_dist,
            noise_swaps=noise_swaps,
            shuffle_genes=shuffle_genes,
            shuffle_samples=shuffle_samples,
            seed=seed,
        )
    )
    adata = ad.AnnData(expr_data)
    # Add in cell and gene names (just by index)
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
    # Add in annotations for which samples are ordered vs unordered
    sample_types = np.empty((adata.n_obs), dtype=np.dtypes.StringDType)
    sample_types[ordered_samples] = "ordered"
    sample_types[unordered_samples] = "unordered"
    adata.obs["sample_type"] = pd.Categorical(sample_types)
    # Add in annotations for which genes are ordered vs unordered
    gene_types = np.empty((adata.n_vars), dtype=np.dtypes.StringDType)
    gene_types[ordered_genes] = "ordered"
    gene_types[unordered_genes] = "unordered"
    adata.var["gene_type"] = pd.Categorical(gene_types)
    # Return the final anndata
    return adata


# endregion Main Function


def _noise_addition(
    array: Array2D,
    dist: Distribution,
    rng: Optional[np.random.Generator] = None,
) -> Array2D:
    """
    Add some random noise to the array
    """
    noise = dist.rvs(size=array.shape, random_state=rng)
    return array + noise


def _noise_swap(
    array: Array2D, swaps: Union[int, float], rng: np.random.Generator
) -> Array2D:
    """
    Randomly swap entried in the array
    """
    if swaps <= 0:
        return array
    if 0 < swaps <= 1:
        swaps: int = functools.reduce(lambda a, b: a * b, array.shape)
    # get the size of the array
    rows, cols = array.shape
    for _ in range(int(swaps)):
        # Pick a random row
        row = rng.integers(0, rows)
        # Pick two elements to swap
        elems = rng.integers(0, cols, size=2)
        a, b = elems[0], elems[1]
        # Perform the swap
        array[row, a], array[row, b] = array[row, b], array[row, a]
    return array


# region Unordered
def _unordered_vector(
    size: int,
    dist: Distribution,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    return dist.rvs(size, random_state=rng)


def _unordered_array(
    nrow: int,
    ncol: int,
    dist: Distribution,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    res_array = np.zeros((nrow, ncol), dtype=dist.rvs(0).dtype)
    for row in range(nrow):
        res_array[row, :] = _unordered_vector(size=ncol, dist=dist, rng=rng)
    return res_array


# endregion Unordered


# region Ordered
def _ordered_vector(
    size: int,
    dist: Distribution,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    return np.sort(_unordered_vector(size, dist, rng=rng))


def _ordered_array(
    nrow: int,
    ncol: int,
    dist: Distribution,
    col_shuffle: bool = True,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    res_array = np.zeros((nrow, ncol), dtype=dist.rvs(0).dtype)
    for row in range(nrow):
        res_array[row, :] = _ordered_vector(size=ncol, dist=dist, rng=rng)
    if col_shuffle and ncol != 0:
        new_col_order = rng.permuted(list(range(ncol)))
        res_array = res_array[:, new_col_order]
    return res_array


# endregion Ordered
