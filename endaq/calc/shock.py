# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple
from collections import namedtuple
import functools
import warnings

import numpy as np
import pandas as pd
import scipy.signal


def _pvss_transfer_func(omega: float, damp: float = 0, dt: float = 1):
    """
    Generate the transfer function
       H(s) = L{z(t)}(s) / L{y"(t)}(s) = (1/s²)(Z(s)/Y(s))
    for the PDE
       z" + (2ζω)z' + (ω^2)z = -y"
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.signal.BadCoefficients)

        return scipy.signal.TransferFunction(
            [-1],
            [1, 2 * damp * omega, omega ** 2],
        ).to_discrete(dt=dt)


def rel_displ(df: pd.DataFrame, omega: float, damp: float = 0) -> pd.DataFrame:
    """Calculate the relative displacement for a SDOF system."""
    dt = (df.index[-1] - df.index[0]) / (len(df.index) - 1)
    if isinstance(dt, (np.timedelta64, pd.Timedelta)):
        dt = dt / np.timedelta64(1, "s")

    tf = _pvss_transfer_func(omega, damp, dt)

    return df.apply(
        functools.partial(scipy.signal.lfilter, tf.num, tf.den, axis=0),
        raw=True,
    )


def _minmax_sos_zeros(a1, a2, z0, z1):
    """Calculate the extrema when zero-extending a biquad SOS filter."""
    r = np.sqrt(a1 ** 2 - 4 * a2 + 0j)
    a1_r_com = (a1 + r) / 2
    a1_r_diff = (a1 - r) / 2

    def realish_or_nan(x, rtol=1e-6, atol=1e-10):
        """Verify that the input is nearly real-valued; if not, raise Error."""
        result = np.where(
            (np.abs(np.imag(x)) < atol) | (np.abs(np.imag(x) / np.real(x)) <= rtol),
            np.real(x),
            np.nan,
        )
        assert np.all(~np.isnan(result))
        return result

    def z0_n(n):
        return realish_or_nan(
            (
                -a2 * z0 * ((-a1_r_com) ** (n + 1) - (-a1_r_diff) ** (n + 1))
                + z1
                * (
                    (-a1_r_com) ** (n + 1) * a1_r_diff
                    - (-a1_r_diff) ** (n + 1) * a1_r_com
                )
            )
            / (a2 * r)
        )

    def z1_n(n):
        return realish_or_nan(
            (
                -a2 * z0 * (-((-a1_r_com) ** n) + (-a1_r_diff) ** n)
                + z1 * ((-a1_r_com) ** n * (-a1_r_diff) + (-a1_r_diff) ** n * a1_r_com)
            )
            / r
        )

    def n_opt(z0, z1):
        return np.real(
            np.log(
                np.log(-a1_r_com)
                * (z0 * a1_r_com - z1)
                / (np.log(-a1_r_diff) * (z0 * a1_r_diff - z1))
            )
            / np.log(a1_r_diff / a1_r_com)
        )

    def n_zero(z0, z1):
        return np.real(
            np.log((z0 * a1_r_com - z1) / (z0 * a1_r_diff - z1))
            / np.log(a1_r_diff / a1_r_com)
        )

    def n_half_period(n_opt1):
        return np.pi / np.angle(a1_r_com)

    n1 = n_opt(z0, z1)
    N_half = n_half_period(n1)
    n1 = n1 % N_half
    n2 = n1 + N_half

    y_n1, y_n2 = z0_n(n1), z0_n(n2)

    return tuple(sorted([y_n1, y_n2]))


def _minmax_rel_displ(df: pd.DataFrame, omega: float, damp: float = 0) -> pd.DataFrame:
    """Calculate the relative displacement extremes for a SDOF system."""
    dt = (df.index[-1] - df.index[0]) / (len(df.index) - 1)
    if isinstance(dt, (np.timedelta64, pd.Timedelta)):
        dt = dt / np.timedelta64(1, "s")

    tf = _pvss_transfer_func(omega, damp, dt)

    result = pd.DataFrame(index=["min", "max"], columns=df.columns)
    for col in df.columns:
        rd, zf = scipy.signal.lfilter(tf.num, tf.den, df[col].values, zi=[])
        min0s, max0s = _minmax_sos_zeros(tf.den[1], tf.den[2], zi[0], zi[1])
        result.loc["min", col] = np.minimum(rd.min(axis="rows"), min0s)
        result.loc["max", col] = np.maximum(rd.max(axis="rows"), min0s)

    return result


def pseudo_velocity(
    df: pd.DataFrame,
    freqs: np.ndarray,
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


def enveloping_half_sine(
    df_pvss: pd.DataFrame,
    damp: float = 0,
) -> Tuple[pd.Series, pd.Series]:
    """Characterize a half-sine pulse whose PVSS envelopes the input."""

    def amp_factor(damp):
        """
        Calculate the PVSS amplitude attenuation on a half-sine pulse from the
        damping coefficient.

        The PVSS of a half-sine pulse differs based on the damping coefficient
        used. While the high-frequency rolloff is relatively consistent, the
        flat low-frequency amplitude is attenuated at higher damping values.
        This function calculates this attenuation for a given damping
        coefficient.
        """
        # This calculates the PVSS value as ω->0. However, since it necessarily
        # computes the maximum of a function *over time*, and ω is only found
        # therein in the multiplicative factor (ωt), it is mathematically
        # equivalent to compute this maximum for any arbitrary ω>0. Thus we
        # choose ω=1 for convenience, w/o loss of generality.
        a = np.exp(1j * np.arccos(-damp))  # = -damp + 1j * np.sqrt(1 - damp**2)
        # From WolframAlpha: https://www.wolframalpha.com/input/?i=D%5BPower%5Be%2C%5C%2840%29-d+*t%5C%2841%29%5D+sin%5C%2840%29Sqrt%5B1-Power%5Bd%2C2%5D%5D*t%5C%2841%29%2Ct%5D+%3D+0&assumption=%22ListOrTimes%22+-%3E+%22Times%22&assumption=%7B%22C%22%2C+%22e%22%7D+-%3E+%7B%22NamedConstant%22%7D&assumption=%7B%22C%22%2C+%22d%22%7D+-%3E+%7B%22Variable%22%7D&assumption=%22UnitClash%22+-%3E+%7B%22d%22%2C+%7B%22Days%22%7D%7D
        t_max = (2 / np.imag(a)) * np.arctan2(np.imag(a), 1 - np.real(a))
        PVSS_max = (1 / np.imag(a)) * np.imag(np.exp(a * t_max))
        return PVSS_max

    max_pvss = df_pvss.max()
    max_f_pvss = df_pvss.mul(df_pvss.index, axis=0).max()

    return namedtuple("HalfSinePulseParameters", "amplitude, period")(
        amplitude=2 * np.pi * max_f_pvss,
        period=max_pvss / (4 * amp_factor(damp) * max_f_pvss),
    )
