# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple
from collections import namedtuple
import functools
import warnings

import numpy as np
import pandas as pd
import scipy.signal


def rel_displ(df: pd.DataFrame, omega: float, damp: float = 0) -> pd.DataFrame:
    """Calculate the relative velocity for a SDOF system."""
    # Generate the transfer function
    #   H(s) = L{z(t)}(s) / L{y"(t)}(s) = (1/s²)(Z(s)/Y(s))
    # for the PDE
    #   z" + (2ζω)z' + (ω^2)z = -y"
    dt = (df.index[-1] - df.index[0]) / (len(df.index) - 1)
    if isinstance(dt, (np.timedelta64, pd.Timedelta)):
        dt = dt / np.timedelta64(1, "s")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.signal.BadCoefficients)

        tf = scipy.signal.TransferFunction(
            [-1],
            [1, 2 * damp * omega, omega ** 2],
        ).to_discrete(dt=dt)

    return df.apply(
        functools.partial(scipy.signal.lfilter, tf.num, tf.den, axis=0),
        raw=True,
    )


def pseudo_velocity(
    df: pd.DataFrame,
    freqs,
    damp: float = 0,
    two_sided: bool = False,
) -> pd.DataFrame:
    """The pseudo velocity of an acceleration signal."""
    freqs = np.asarray(freqs)
    if freqs.ndim != 1:
        raise ValueError("target frequencies must be in a 1D-array")
    omega = 2 * np.pi * freqs

    results = np.empty((2,) + freqs.shape + df.shape[1:], dtype=np.float64)

    for i_nd in np.ndindex(freqs.shape):
        rd = rel_displ(df, omega[i_nd], damp)

        results[(0,) + i_nd] = -omega[i_nd] * rd.min(axis=0)
        results[(1,) + i_nd] = omega[i_nd] * rd.max(axis=0)

    if not two_sided:
        return pd.DataFrame(
            np.maximum(results[0], results[1]),
            index=pd.Series(freqs, name="frequency (Hz)"),
            columns=df.columns,
        )

    return namedtuple("PseudoVelocityResults", "neg pos")(
        pd.DataFrame(
            r, index=pd.Series(freqs, name="frequency (Hz)"), columns=df.columns
        )
        for r in results
    )


def half_sine_shock_envelope(
    df_pvss: pd.DataFrame,
    damp: float = 0,
) -> Tuple[pd.Series, pd.Series]:
    """Characterize a half-sine pulse whose PVSS envelopes the input."""

    def amp_factor_approx(damp):
        """
        Approximate the PVSS amplitude attenuation on a half-sine pulse from the
        damping coefficient.

        The PVSS of a half-sine pulse differs based on the damping coefficient
        used. While the high-frequency rolloff is relatively consistent, the
        flat low-frequency amplitude is attenuated at higher damping values.
        This function approximates the attenuation for a given damping
        coefficient.

        This approximation was derived via trial-and-error, and can likely be
        improved to be (slightly) more optimal w/o extra complexity.
        """
        return 1 / (1 + (np.e - 1) * damp ** 1.04)  # the 1.04 can be more optimal

    max_pvss = df_pvss.max()
    max_f_pvss = df_pvss.mul(df_pvss.index, axis=0).max()

    return namedtuple("HalfSinePulseParameters", "amplitude, period")(
        amplitude=2 * np.pi * max_f_pvss,
        period=max_pvss / (4 * amp_factor_approx(damp) * max_f_pvss),
    )
