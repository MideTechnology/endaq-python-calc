import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np

import numpy as np
import pandas as pd

from endaq.calc import shock


@hyp.given(
    freq=hyp_st.floats(12.5, 1000),
    damp=hyp_st.floats(0, 1, exclude_max=True),
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
    calc_result = (
        shock.rel_displ(
            pd.DataFrame(signal, index=np.arange(len(signal)) / fs),
            omega=omega,
            damp=damp,
        )
        .to_numpy()
        .flatten()
    )

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
    df_accel=hyp_np.arrays(
        dtype=np.float64,
        shape=(40,),
        elements=hyp_st.floats(1e-20, 1e20),
    ).map(lambda array: pd.DataFrame(array, index=np.arange(40) * 1e-4)),
    freq=hyp_st.floats(250, 1000),
    damp=hyp_st.floats(0, 1, exclude_max=True),
)
def test_pseudo_velocity_inversion(df_accel, freq, damp):
    pd.testing.assert_frame_equal(
        shock.pseudo_velocity(df_accel, [freq], damp=damp),
        shock.pseudo_velocity(-df_accel, [freq], damp=damp),
    )


@hyp.given(
    df_pvss=hyp_np.arrays(
        dtype=np.float64,
        shape=(40,),
        elements=hyp_st.floats(1e-20, 1e20),
    ).map(lambda array: pd.DataFrame(array, index=np.arange(1, 41))),
    damp=hyp_st.floats(0, 0.2),
)
def test_half_sine_shock_envelope(df_pvss, damp):
    ampl, T = shock.half_sine_shock_envelope(df_pvss, damp=damp)
    hyp.note(f"pulse amplitude: {ampl}")
    hyp.note(f"pulse duration: {T}")

    dt = min(
        1 / (2 * df_pvss.index[-1]), T / 20
    )  # guarantee sufficient # of samples to represent pulse
    fs = 1 / dt
    times = np.arange(int(fs * (T + 1 / df_pvss.index[0]))) / fs
    pulse = np.zeros_like(times)
    pulse[: int(T * fs)] = ampl * np.sin((np.pi / T) * times[: int(T * fs)])
    pulse_pvss = shock.pseudo_velocity(
        pd.DataFrame(pulse, index=df_pvss.index), damp=damp
    )

    # This is an approximation -> give the result a fudge-factor for correctness
    assert (df_pvss / pulse_pvss).max().max() < 1.2
