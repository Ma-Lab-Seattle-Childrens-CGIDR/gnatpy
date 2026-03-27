"""
Some utility functions for calculating statistics
"""


# Kendall Tau:

# This code is taken from SciPy Stats module, licensed under a
# BSD-3-Clause license (reproduced in the LICENSE file of the
# GNATpy repository (https://github.com/Ma-Lab-Seattle-Childrens-CGIDR/gnatpy/blob/main/LICENSE),
# with the original in the SciPy repository here:
# https://github.com/scipy/scipy/blob/main/LICENSE.txt), modified to
# remove the code to calculate the p-value since the p-value isn't needed
# for GNATpy.
# The copyright included in the
# scipy.stats._stats_py.py file where this code is from is reproduced here:
# Copyright 2002 Gary Strangman.  All rights reserved
# Copyright 2002-2016 The SciPy Developers
#
# The original code from Gary Strangman was heavily adapted for
# use in SciPy by Travis Oliphant.  The original code came with the
# following disclaimer:
#
# This software is provided "as-is".  There are no expressed or implied
# warranties of any kind, including, but not limited to, the warranties
# of merchantability and fitness for a given application.  In no event
# shall Gary Strangman be liable for any direct, indirect, incidental,
# special, exemplary or consequential damages (including, but not limited
# to, loss of use, data or profits, or business interruption) however
# caused and on any theory of liability, whether in contract, strict
# liability or tort (including negligence or otherwise) arising in any way
# out of the use of this software, even if advised of the possibility of
# such damage.

import numpy as np

from scipy._lib._util import _get_nan
from scipy.stats._stats_py import _kendall_dis


def kendalltau(
    x,
    y,
    *,
    nan_policy="propagate",
    variant="b",
):
    r"""Calculate Kendall's tau, a correlation measure for ordinal data.

    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, and values close to -1
    indicate strong disagreement. This implements two variants of Kendall's
    tau: tau-b (the default) and tau-c (also known as Stuart's tau-c). These
    differ only in how they are normalized to lie within the range -1 to 1.
    Kendall's original tau-a is not implemented separately because both tau-b
    and tau-c reduce to tau-a in the absence of ties.

    Although a naive implementation has O(n^2) complexity, this implementation
    uses a Fenwick tree to do the computation in O(n log(n)) complexity.

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values


    variant : {'b', 'c'}, optional
        Defines which variant of Kendall's tau is returned. Default is 'b'.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the rank correlation is nonzero
        * 'less': the rank correlation is negative (less than zero)
        * 'greater': the rank correlation is positive (greater than zero)

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : float
           The tau statistic.
        pvalue : float
           The p-value for a hypothesis test whose null hypothesis is
           an absence of association, tau = 0.

    Raises
    ------
    ValueError
        If `nan_policy` is 'omit' and `variant` is not 'b' or
        if `method` is 'exact' and there are ties between `x` and `y`.

    See Also
    --------
    spearmanr : Calculates a Spearman rank-order correlation coefficient.
    theilslopes : Computes the Theil-Sen estimator for a set of points (x, y).
    weightedtau : Computes a weighted version of Kendall's tau.
    :ref:`hypothesis_kendalltau` : Extended example

    Notes
    -----
    This function taken (and modified to remove the p-value calculation) from SciPy,
    licensed under a
    [BSD-3-Clause license](https://github.com/scipy/scipy/blob/main/LICENSE.txt).

    The definition of Kendall's tau that is used is [2]_::

      tau_b = (P - Q) / sqrt((P + Q + T) * (P + Q + U))

      tau_c = 2 (P - Q) / (n**2 * (m - 1) / m)

    where P is the number of concordant pairs, Q the number of discordant
    pairs, T the number of tied pairs only in `x`, and U the number of tied pairs only
    in `y`.  If a tie occurs for the same pair in both `x` and `y`, it is not
    added to either T or U. n is the total number of samples, and m is the
    number of unique values in either `x` or `y`, whichever is smaller.

    References
    ----------
    .. [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.
    .. [2] Maurice G. Kendall, "The treatment of ties in ranking problems",
           Biometrika Vol. 33, No. 3, pp. 239-251. 1945.
    .. [3] Gottfried E. Noether, "Elements of Nonparametric Statistics", John
           Wiley & Sons, 1967.
    .. [4] Peter M. Fenwick, "A new data structure for cumulative frequency
           tables", Software: Practice and Experience, Vol. 24, No. 3,
           pp. 327-336, 1994.
    .. [5] Maurice G. Kendall, "Rank Correlation Methods" (4th Edition),
           Charles Griffin & Co., 1970.

    Examples
    --------

    >>> from gnatpy.statistical_utils import kendalltau
    >>> x1 = [12, 2, 1, 12, 2]
    >>> x2 = [1, 4, 7, 1, 0]
    >>> res = kendalltau(x1, x2)
    >>> res
    0.2827454599327748

    For a more detailed example, see :ref:`hypothesis_kendalltau`.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("Array shapes are incompatible for broadcasting.")
    elif not x.size or not y.size:
        # Return NaN if arrays are empty
        NaN = _get_nan(x, y)
        return NaN

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype("int64", copy=False)
        cnt = cnt[cnt > 1]
        # Python ints to avoid overflow down the line
        return (
            int((cnt * (cnt - 1) // 2).sum()),
            int((cnt * (cnt - 1.0) * (cnt - 2)).sum()),
            int((cnt * (cnt - 1.0) * (2 * cnt + 5)).sum()),
        )

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind="mergesort")
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype("int64", copy=False)

    ntie = int((cnt * (cnt - 1) // 2).sum())  # joint ties
    xtie, x0, x1 = count_rank_tie(x)  # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)  # ties in y, stats

    tot = (size * (size - 1)) // 2

    if xtie == tot or ytie == tot:
        NaN = _get_nan(x, y)
        return NaN

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    if variant == "b":
        tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
    elif variant == "c":
        minclasses = min(len(set(x)), len(set(y)))
        tau = 2 * con_minus_dis / (size**2 * (minclasses - 1) / minclasses)
    else:
        raise ValueError(
            f"Unknown variant of the method chosen: {variant}. "
            "variant must be 'b' or 'c'."
        )

    # Limit range to fix computational errors
    tau = np.minimum(1.0, max(-1.0, tau))

    return tau


# end Kendall Tau
