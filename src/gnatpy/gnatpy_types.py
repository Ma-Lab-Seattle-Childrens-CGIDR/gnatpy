"""
Types for GNATpy
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

# Type definitions
Array2D = np.ndarray[Tuple[int, int], np.dtype[Union[np.float32, np.float64, np.int_]]]
Array1D = np.ndarray[Tuple[int,], np.dtype[Union[np.float32, np.float64, np.int_]]]
