import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np
import numpy as np

from endaq.calc import shock


@hyp.given(
    freq=hyp_st.floats(12.5, 1000, allow_nan=False),
    damp=hyp_st.floats(0, 1, exclude_max=True, allow_nan=False),
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
    atten = omega * np.exp(1j * np.arccos(-damp))
    assert np.real(atten) == pytest.approx(-omega * damp)

    expt_result = np.zeros_like(signal)
    expt_result[200:] = (-1 / np.imag(atten)) * np.imag(
        np.exp(t * atten) / atten
    ) - 1 / omega ** 2

    # Test results
    assert np.allclose(calc_result, expt_result)


@hyp.given(
    accel=hyp_np.arrays(
        dtype=np.float64,
        shape=(40,),
        elements=hyp_st.floats(1e-20, 1e20, allow_nan=False, allow_infinity=False),
    ),
    freq=hyp_st.floats(250, 1000, allow_nan=False),
    damp=hyp_st.floats(0, 1, exclude_max=True, allow_nan=False),
)
def test_pseudo_velocity_inversion(accel, freq, damp):
    fs = 10 ** 4
    assert shock.pseudo_velocity(
        accel, freq, dt=1 / fs, damp=damp
    ) == shock.pseudo_velocity(-accel, freq, dt=1 / fs, damp=damp)


@hyp.given(
    accel_pvss=hyp_np.arrays(
        dtype=np.float64,
        shape=(40,),
        elements=hyp_st.floats(
            1e-20, 1e20, allow_nan=False, allow_infinity=False, exclude_min=True
        ),
    ),
    damp=hyp_st.floats(0, 0.2, exclude_max=True, allow_nan=False),
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
