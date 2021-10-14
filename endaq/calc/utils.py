# -*- coding: utf-8 -*-

from __future__ import annotations

import typing
from typing import Optional
import warnings

import numpy as np
import pandas as pd


def sample_spacing(
    df: pd.DataFrame, convert: typing.Literal[None, "to_seconds"] = "to_seconds"
):
    """
    Calculate the average spacing between individual samples.

    For time indices, this calculates the sampling period `dt`.

    :param df: the input data
    :param convert: if `"to_seconds"` (default), convert any time objects into
        floating-point seconds
    """
    dt = (df.index[-1] - df.index[0]) / (len(df.index) - 1)
    if convert == "to_seconds" and isinstance(dt, (np.timedelta64, pd.Timedelta)):
        dt = dt / np.timedelta64(1, "s")

    return dt


def logfreqs(
    df: pd.DataFrame, init_freq: Optional[float] = None, bins_per_octave: float = 12
) -> np.ndarray:
    """
    Calculate a sequence of log-spaced frequencies for a given dataframe.

    :param df: the input data
    :param init_freq: the initial frequency in the sequence; if None (default),
        use the frequency corresponding to the data's duration
    :param bins_per_octave: the number of frequencies per octave
    :return: an array of log-spaced frequencies
    """
    dt = sample_spacing(df)
    T = dt * len(df.index)

    if init_freq is None:
        init_freq = 1 / T
    elif 1 / init_freq > T:
        warnings.warn(
            "the data's duration is too short to accurately represent an"
            f" initial frequency of {init_freq:.3f} Hz",
            RuntimeWarning,
        )

    return 2 ** np.arange(
        np.log2(init_freq),
        np.log2(1 / dt) - 1,
        1 / bins_per_octave,
    )


def to_dB(data: np.ndarray, reference: float, squared=False):
    """
    Scale data into units of decibels.

    ..note:: Decibels can**NOT** be calculated from negative values.
        E.g., to calculate dB on arbitrary time-series data, typically data is
        first aggregated via a total or a rolling RMS or PSD, and the result is
        then scaled into decibels.

    :param data: the input data
    :param reference: the reference value corresponding to 0dB
    :param squared: whether the input data & reference value are pre-squared;
        defaults to `False`
    """
    if reference <= 0:
        raise ValueError("reference value must be strictly positive")

    data = np.asarray(data)
    if np.any(data < 0):
        raise ValueError(
            "cannot compute decibels from negative values (see the docstring"
            " for details)"
        )

    return (10 if squared else 20) * (np.log10(data) - np.log10(reference))


dB_refs = {
    "SPL": 2e-5,  # Pascal
    "audio_intensity": 1e-12,  # W/m²
}
