from __future__ import annotations

import typing  # for `SupportsIndex`, which is Python3.8+ only
from typing import Union
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def L2_norm(
    array: npt.ArrayLike,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    """
    Compute the L2 norm (a.k.a. the Euclidean Norm).

    :param array: the input array
    :param axis: the axis/axes along which to aggregate; if `None`, the L2 norm
        is computed along the flattened array
    :param keepdims: if `True`, the axes which are reduced are left in the
        result as dimensions with size one; if `False` (default), the reduced
        axes are removed
    :return: an array containing the computed values
    """
    return np.sqrt(np.sum(np.abs(array) ** 2, axis=axis, keepdims=keepdims))


def max_abs(
    array: npt.ArrayLike,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    """
    Compute the maximum of the absolute value of an array.

    This function should be equivalent to, but generally use less memory than
    `np.amax(np.abs(array))`.

    Specifically, it generates the absolute-value maximum from `np.amax(array)`
    and `-np.amin(array)`. Thus instead of allocating space for the intermediate
    array `np.abs(array)`, it allocates for the axis-collapsed smaller arrays
    `np.amax(array)` & `np.amin(array)`.

    Note - this method does **not** work on complex-valued arrays.

    :param array: the input data
    :param axis: the axis/axes along which to aggregate; if `None`, the
        absolute maximum is computed along the flattened array
    :param keepdims: if `True`, the axes which are reduced are left in the
        result as dimensions with size one; if `False` (default), the reduced
        axes are removed
    :return: an array containing the computed values
    """
    # Forbid complex-valued data
    if np.iscomplexobj(array):
        raise ValueError("`max_abs` does not accept complex arrays")

    return np.maximum(
        np.amax(array, initial=-np.inf, axis=axis, keepdims=keepdims),
        -np.amin(array, initial=np.inf, axis=axis, keepdims=keepdims),
    )


def rms(
    array: npt.ArrayLike,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
):
    """
    Calculate the root-mean-square (RMS) along a given axis.

    :param array: the input array
    :param axis: the axis/axes along which to aggregate; if `None`, the RMS is
        computed along the flattened array
    :param keepdims: if `True`, the axes which are reduced are left in the
        result as dimensions with size one; if `False` (default), the reduced
        axes are removed
    :return: an array containing the computed values
    """
    return np.sqrt(np.mean(np.abs(array) ** 2, axis=axis, keepdims=keepdims))
