from __future__ import annotations

from typing import Optional, Union, Tuple

import numpy as np
import scipy.signal


def bandpass(
    array: np.ndarray,
    fs: float,
    low_cutoff: Optional[float] = 1.0,
    high_cutoff: Optional[float] = None,
    half_order: int = 3,
    axis: int = -1,
) -> np.ndarray:
    """
    Apply a lowpass and/or a highpass filter to an array.

    This function uses Butterworth filter designs, and implements the filter(s)
    as bi-directional digital biquad filters, split into second-order sections.
    """
    cutoff_freqs: Union[float, Tuple[float, float]]
    filter_type: str

    if low_cutoff is not None and high_cutoff is not None:
        cutoff_freqs = (low_cutoff, high_cutoff)
        filter_type = "bandpass"
    elif low_cutoff is not None:
        cutoff_freqs = low_cutoff
        filter_type = "highpass"
    elif high_cutoff is not None:
        cutoff_freqs = high_cutoff
        filter_type = "lowpass"
    else:
        return array

    sos_coeffs = scipy.signal.butter(
        N=half_order,
        Wn=cutoff_freqs,
        btype=filter_type,
        fs=fs,
        output="sos",
    )

    for b, a in zip(*np.split(sos_coeffs, [3], axis=-1)):
        array = scipy.signal.filtfilt(
            b, a, array, axis=axis, method="gust", irlen=5 * 10 ** 4
        )

    return array
