from __future__ import annotations

from typing import List, Optional, Iterable
import functools

import numpy as np
import pandas as pd
import scipy.integrate

from . import filters


def _integrate(df: pd.DataFrame) -> pd.DataFrame:
    """Integrate data over an axis."""
    dt = (df.index[-1] - df.index[0]) / (len(df.index) - 1)
    if isinstance(dt, (np.timedelta64, pd.Timedelta)):
        dt = dt / np.timedelta64(1, "s")

    result = df.apply(
        functools.partial(scipy.integrate.cumulative_trapezoid, dx=dt, initial=0),
        axis=0,
        raw=True,
    )
    # In lieu of explicit initial offset, set integration bias to remove mean
    # -> avoids trend artifacts after successive integrations
    result = result - result.mean(axis=0, keepdims=True)

    return result


def iter_integrals(
    df: pd.DataFrame, highpass_cutoff=None, filter_half_order=3
) -> Iterable[pd.DataFrame]:
    """Iterate over conditioned integrals of the given original data."""
    df = filters.butterworth(
        df,
        half_order=filter_half_order,
        low_cutoff=highpass_cutoff,
        high_cutoff=None,
    )

    while True:
        yield df.copy()  # otherwise, edits to the yielded item would alter the results
        df = _integrate(df)


def integrals(
    df: pd.DataFrame, n: int = 1, highpass_cutoff: Optional[float] = None
) -> List[np.ndarray]:
    """
    Calculate `n` integrations of the given data.

    :param array: the data to integrate
    :param dt: the time between adjacent samples; assumes a uniform sampling rate
    :param n: the number of integrals to calculate
    :param axis: the axis of `array` over which to integrate
    :param highpass_cutoff: the cutoff frequency for the initial highpass filter;
        this is used to remove artifacts caused by DC trends
    :return: a length `n+1` list of the kth order integrals from 0 to n (inclusive)
    """
    return [
        integ
        for _, integ in zip(
            range(n + 1),
            iter_integrals(
                df,
                highpass_cutoff=highpass_cutoff,
                filter_half_order=n // 2 + 1,  # ensures zero DC content in nth integral
            ),
        )
    ]
