from collections import namedtuple
import warnings

import numpy as np
import scipy.signal


def rel_displ(accel, omega, dt=1, damp=0, axis=-1):
    """Calculate the relative velocity for a SDOF system."""
    # Generate the transfer function
    #   H(s) = L{z(t)}(s) / L{y"(t)}(s) = (1/s²)(Z(s)/Y(s))
    # for the PDE
    #   z" + (2ζω)z' + (ω^2)z = -y"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.signal.BadCoefficients)

        tf = scipy.signal.TransferFunction(
            [-1],
            [1, 2 * damp * omega, omega ** 2],
        ).to_discrete(dt=dt)

    return scipy.signal.lfilter(tf.num, tf.den, accel, axis=axis)


def rolling_rel_vel(accel, freqs, dt=1, damp=0, nperseg=256, axis=-1):
    """Calculate a rolling windowed relative velocity for a SDOF system."""
    axis = (axis % accel.ndim) - accel.ndim  # index axis from end
    accel = np.moveaxis(accel, axis, -1)
    freqs = np.asarray(freqs)

    t = dt * np.arange(nperseg)
    η = np.sqrt(1 - damp ** 2)
    omega = 2 * np.pi * freqs[..., np.newaxis]
    filt = -np.imag(np.exp(omega * t * (-damp + 1j * η))) / η

    result = np.empty(freqs.shape + accel.shape)
    for i_nd in np.ndindex(*freqs.shape):
        for j_nd in np.ndindex(*accel.shape[:-1]):
            result[i_nd + j_nd] = scipy.signal.oaconvolve(
                accel[j_nd], filt[i_nd], mode="full", axes=-1
            )[: (1 - nperseg) or None]
    return np.moveaxis(result, -1, axis)


def pseudo_velocity(accel, freqs, dt=1, damp=0, two_sided=False, axis=-1):
    """The pseudo velocity of an acceleration signal."""
    freqs = np.asarray(freqs)
    omega = 2 * np.pi * freqs

    accel = np.moveaxis(accel, axis, -1)

    results = np.empty((2,) + freqs.shape + accel.shape[:-1], dtype=np.float64)

    for i_nd in np.ndindex(freqs.shape):
        rd = rel_displ(accel, omega[i_nd], dt, damp)

        results[(0,) + i_nd] = -omega[i_nd] * rd.min(axis=-1)
        results[(1,) + i_nd] = omega[i_nd] * rd.max(axis=-1)

    # Move any frequency axes in place of the specified acceleration axis
    results = np.moveaxis(
        results,
        np.arange(1, omega.ndim + 1),
        np.arange(1, omega.ndim + 1) + (axis % accel.ndim),
    )

    if not two_sided:
        return np.maximum(results[0], results[1])

    return namedtuple("PseudoVelocityResults", "neg pos")(*results)


def half_sine_shock_envelope(f, pvss, damp=0, axis=-1):
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

    max_pvss = np.amax(pvss, axis=axis)
    max_f_pvss = np.amax(pvss * f, axis=axis)

    return namedtuple("HalfSinePulseParameters", "amplitude, period")(
        amplitude=2 * np.pi * max_f_pvss,
        period=max_pvss / (4 * amp_factor_approx(damp) * max_f_pvss),
    )
