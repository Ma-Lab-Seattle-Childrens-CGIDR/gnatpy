"""
Types for GNATpy
"""

from __future__ import annotations

from typing import Tuple, Union, Protocol

import numpy as np

# Array Types
Array2D = np.ndarray[Tuple[int, int], np.dtype[Union[np.float32, np.float64, np.int_]]]
Array1D = np.ndarray[Tuple[int,], np.dtype[Union[np.float32, np.float64, np.int_]]]


# Entropy Function Type
class EntropyFunction(Protocol):
    def __call__(self, *arrays: Array2D) -> float: ...
