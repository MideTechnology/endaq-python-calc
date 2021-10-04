from __future__ import annotations

import typing  # for `SupportsIndex`, which is Python3.8+ only
from typing import Union
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.ndimage


def L2_norm(
    data: np.ndarray,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
):
    """Compute the L2 norm (a.k.a. the Euclidean Norm)."""
    return np.sqrt(np.sum(np.abs(data) ** 2, axis=axis, keepdims=keepdims))


def max_abs(
    array: np.ndarray,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
):
    """
    Compute the maximum of the absolute value of an array.

    This function should be equivalent to, but generally use less memory than
    `np.amax(np.abs(array))`.

    Specifically, it generates the absolute-value maximum from `np.amax(array)`
    and `-np.amin(array)`. Thus instead of allocating space for the intermediate
    array `np.abs(array)`, it allocates for the axis-collapsed smaller arrays
    `np.amax(array)` & `np.amin(array)`.

    Note - this method does not work on complex-valued arrays.
    """
    # Forbid complex-valued data
    if np.iscomplexobj(array):
        raise ValueError("`max_abs` does not accept complex arrays")

    return np.maximum(
        np.amax(array, initial=-np.inf, axis=axis, keepdims=keepdims),
        -np.amin(array, initial=np.inf, axis=axis, keepdims=keepdims),
    )


def rms(
    data: np.ndarray,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
):
    """Calculate the root-mean-square (RMS) along a given axis."""
    return np.sqrt(np.mean(np.abs(data) ** 2, axis=axis, keepdims=keepdims))


def rolling_rms(
    df: pd.DataFrame,
    nperseg: int = 256,
    axis: typing.SupportsIndex = -1,
) -> pd.DataFrame:
    """Calculate a rolling RMS along a given axis."""
    dt = (df.index[-1] - df.index[0]) / (len(df.index) - 1)
    if isinstance(dt, (np.timedelta64, pd.Timedelta)):
        dt = dt / np.timedelta64(1, "s")

    # RMS = √(1/T ∫|x(t)|² dt)
    #     ≈ √(∆t/T ∑|x[n]|²)
    #     = √(1/N ∑|x[n]|²)
    sq = df.values ** 2
    window = np.full(nperseg, 1 / nperseg)
    mean_sq = scipy.ndimage.convolve1d(sq, window, axis=axis, mode="mirror")

    return pd.DataFrame(np.sqrt(mean_sq), index=df.index, columns=df.columns)
