import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np
import numpy as np

import itertools

from endaq.calc import shock


@pytest.mark.parametrize(
    "freq, damp",
    itertools.product(
        [100, 300, 500, 1000],
        [0.0, 0.05, 0.5],
    ),
)
def test_rel_displ(freq, damp):
    """
    This test uses a step-function input acceleration. In a SDOF spring system,
    the spring should be relaxed in the first portion where `a(t < t0) = 0`.
    Once the acceleration flips on (`a(t > t0) = 1`), the mass should begin to
    oscillate.

    (This scenario is mathematically identical to having the mass pulled out
    some distance and held steady with a constant force at `t=0`, then
    releasing the mass at `t > t0` and letting it oscillate freely.)

    This system is tested over a handful of different oscillation parameter
    (i.e., frequency & damping rate) configurations.
    """
    # Data parameters
    signal = np.zeros(1000, dtype=float)
    signal[200:] = 1
    fs = 10 ** 4  # Hz
    # Other parameters
    omega = 2 * np.pi * freq

    # Calculate result
    calc_result = shock.rel_displ(signal, omega=omega, dt=1 / fs, damp=damp)

    # Calculate expected result
    t = np.arange(1, 801) / fs
    atten = omega * (-damp + 1j * np.sqrt(1 - damp ** 2))

    expt_result = np.zeros_like(signal)
    expt_result[200:] = (-1 / np.imag(atten)) * np.imag(
        np.exp(t * atten) / atten
    ) - 1 / omega ** 2

    # Test results
    assert np.allclose(calc_result, expt_result)


@hyp.given(
    accel_pvss=hyp_np.arrays(
        dtype=np.float64,
        shape=(40,),
        elements=hyp_st.floats(
            1e-20, 1e20, allow_nan=False, allow_infinity=False, exclude_min=True
        ),
    ),
    damp=hyp_st.floats(0, 0.2, exclude_max=True),
)
def test_half_sine_shock_envelope(accel_pvss, damp):
    n = len(accel_pvss)
    freqs = np.fft.rfftfreq(2 * n, d=1 / n)[1:]  # remove f=0

    ampl, T = shock.half_sine_shock_envelope(freqs, accel_pvss, damp=damp)
    hyp.note(f"pulse amplitude: {ampl}")
    hyp.note(f"pulse duration: {T}")

    dt = min(
        1 / (2 * freqs[-1]), T / 20
    )  # guarantee sufficient # of samples to represent pulse
    fs = 1 / dt
    times = np.arange(int(fs * (T + 1 / freqs[0]))) / fs
    pulse = np.zeros_like(times)
    pulse[: int(T * fs)] = ampl * np.sin((np.pi / T) * times[: int(T * fs)])
    pulse_pvss = shock.pseudo_velocity(pulse, freqs, dt=dt, damp=damp)

    # This is an approximation -> give the result a fudge-factor for correctness
    assert np.amax(accel_pvss / pulse_pvss) < 1.2