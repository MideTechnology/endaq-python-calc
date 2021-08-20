from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy.integrate

from . import filters


def _integrate(array, dt, axis=-1):
    """Integrate data over an axis."""
    result = scipy.integrate.cumulative_trapezoid(array, dx=dt, initial=0, axis=axis)
    # In lieu of explicit initial offset, set integration bias to remove mean
    # -> avoids trend artifacts after successive integrations
    result = result - result.mean(axis=axis, keepdims=True)

    return result


def iter_integrals(array, dt, axis=-1, highpass_cutoff=None, filter_half_order=3):
    """Iterate over conditioned integrals of the given original data."""
    array = filters.bandpass(
        array,
        fs=1 / dt,
        half_order=filter_half_order,
        low_cutoff=highpass_cutoff,
        high_cutoff=None,
        axis=axis,
    )
    while True:
        array.setflags(write=False)  # should NOT mutate shared data
        yield array
        array.setflags(write=True)  # array will be replaced below -> now ok to edit
        array = _integrate(array, dt, axis=axis)


def integrals(
    array: np.ndarray,
    dt: float,
    n: int = 1,
    axis: int = -1,
    highpass_cutoff: Optional[float] = None,
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
    result = [
        integ
        for _, integ in zip(
            range(n + 1),
            iter_integrals(
                array,
                dt,
                axis=axis,
                highpass_cutoff=highpass_cutoff,
                filter_half_order=n // 2 + 1,  # ensures zero DC content in nth integral
            ),
        )
    ]
    result[-1].setflags(write=True)  # iterator is gone -> now ok to edit

    return result
