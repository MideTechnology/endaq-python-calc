from __future__ import annotations

from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import scipy.signal


def butterworth(
    df: pd.DataFrame,
    low_cutoff: Optional[float] = 1.0,
    high_cutoff: Optional[float] = None,
    half_order: int = 3,
) -> pd.DataFrame:
    """
    Apply a lowpass and/or a highpass Butterworth filter to an array.

    This function uses Butterworth filter designs, and implements the filter(s)
    as bi-directional digital biquad filters, split into second-order sections.

    :param df: the input data; cutoff frequencies are relative to the
        timestamps in `df.index`
    :param low_cutoff: the low-frequency cutoff, if any; frequencies below this
        value are rejected, and frequencies above this value are preserved
    :param high_cutoff: the high-frequency cutoff, if any; frequencies above
        this value are rejected, and frequencies below this value are preserved
    :param half_order: half of the order of the filter; higher orders provide
        more aggressive stopband reduction
    :return: the filtered data
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
        return df

    dt = (df.index[-1] - df.index[0]) / (len(df.index) - 1)
    if isinstance(dt, (np.timedelta64, pd.Timedelta)):
        dt = dt / np.timedelta64(1, "s")

    sos_coeffs = scipy.signal.butter(
        N=half_order,
        Wn=cutoff_freqs,
        btype=filter_type,
        fs=1 / dt,
        output="sos",
    )

    array = df.to_numpy()

    for b, a in zip(*np.split(sos_coeffs, [3], axis=-1)):
        array = scipy.signal.filtfilt(
            b, a, array, axis=0, method="gust", irlen=5 * 10 ** 4
        )

    return pd.DataFrame(
        array,
        index=df.index,
        columns=df.columns,
    )
