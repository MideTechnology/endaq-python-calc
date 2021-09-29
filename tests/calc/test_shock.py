import warnings

import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np

import numpy as np
import pandas as pd
import scipy.signal

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


@hyp_st.composite
def ai_zi(draw):
    a1 = draw(hyp_st.floats(-2, -1e-7, exclude_min=True))

    # Generate values of `a2` s.t. the filter's period > 10
    period = draw(hyp_st.floats(12, 500))
    a2 = (a1 / 2 / np.cos(2 * np.pi / period)) ** 2
    r = np.sqrt(a1 ** 2 - 4 * a2 + 0j)
    eigan_pos = (-a1 + r) / 2
    assert 2 * np.pi / np.angle(eigan_pos) > 10  # period > 10

    # Generate sufficiently non-zero `zi` values
    z0 = draw(hyp_st.floats(1e-7, 1)) * draw(hyp_st.sampled_from([-1, 1]))
    z1 = draw(hyp_st.floats(1e-7, 1)) * draw(hyp_st.sampled_from([-1, 1]))

    return (a1, a2, z0, z1)


@hyp.given(ai_zi=ai_zi())
def test_minmax_sos_zeros(ai_zi):
    a1, a2, z0, z1 = ai_zi
    # The square-root term
    r = np.sqrt(a1 ** 2 - 4 * a2 + 0j)
    # The filter's eiganvalues
    e_pos = (-a1 + np.sqrt(a1 ** 2 - 4 * a2 + 0j)) / 2
    e_neg = (-a1 - np.sqrt(a1 ** 2 - 4 * a2 + 0j)) / 2

    hyp.note(r)
    hyp.note(e_pos)
    hyp.note(e_neg)

    is_decaying = np.abs(e_pos) < 1

    array = np.zeros(1000)
    array_filt, _ = scipy.signal.lfilter([1], [1, a1, a2], array, zi=[z0, z1])

    sign_changes = np.flatnonzero(np.diff(np.diff(array_filt) > 0))
    hyp.assume(len(sign_changes) >= 2)
    hyp.assume(sign_changes[1] - sign_changes[0] > 10)

    calc_result = shock.BiquadProperties(a1, a2).extrema(z0, z1)

    i1, i2 = int(np.floor(calc_result.imin)), int(np.ceil(calc_result.imin))
    # Check that the peak is "correct"
    assert calc_result.min <= array_filt[[i1, i2]].min()
    if is_decaying:
        assert calc_result.min <= array_filt[i1:].min()
    else:
        assert calc_result.min <= array_filt[:i2].min()
    # Check that the peak is in fact at a local minimum
    if i1 > 0:
        # data should be decreasing before the peak
        assert array_filt[i1 - 1] > array_filt[i1]
    if i2 < len(array_filt) - 1:
        # data should be increasing after the peak
        assert array_filt[i2] < array_filt[i2 + 1]

    i1, i2 = int(np.floor(calc_result.imax)), int(np.ceil(calc_result.imax))
    # Check that the peak is "correct"
    assert calc_result.max >= array_filt[[i1, i2]].max()
    if is_decaying:
        assert calc_result.max >= array_filt[i1:].max()
    else:
        assert calc_result.max >= array_filt[:i2].max()
    # Check that the peak is in fact at a local maximum
    if i1 > 0:
        # data should be increasing before the peak
        assert array_filt[i1 - 1] < array_filt[i1]
    if i2 < len(array_filt) - 1:
        # data should be decreasing after the peak
        assert array_filt[i2] > array_filt[i2 + 1]


@hyp.given(
    df_accel=hyp_np.arrays(
        dtype=np.float64,
        shape=(40,),
        elements=hyp_st.floats(1e-20, 1e20),
    ).map(lambda array: pd.DataFrame(array, index=np.arange(40) * 1e-4)),
    freq=hyp_st.floats(1, 20),
    damp=hyp_st.floats(0, 1, exclude_max=True),
    factor=hyp_st.floats(-1e2, 1e2),
)
def test_pseudo_velocity_linearity(df_accel, freq, damp, factor):
    pd.testing.assert_frame_equal(
        shock.pseudo_velocity(factor * df_accel, [freq], damp=damp),
        (factor * shock.pseudo_velocity(df_accel, [freq], damp=damp)).abs(),
    )


@hyp.given(
    df_pvss=hyp_np.arrays(
        dtype=np.float64,
        shape=(40,),
        elements=hyp_st.floats(1e-20, 1e20),
    ).map(lambda array: pd.DataFrame(array, index=np.arange(1, 41))),
    damp=hyp_st.floats(0, 0.2),
)
@hyp.settings(deadline=None)  # this test tends to timeout
def test_enveloping_half_sine(df_pvss, damp):
    ampl, T = shock.enveloping_half_sine(df_pvss, damp=damp)
    hyp.note(f"pulse amplitude: {ampl}")
    hyp.note(f"pulse duration: {T}")

    ampl = ampl[0]
    T = T[0]

    dt = min(
        1 / (2 * df_pvss.index[-1]), T / 20
    )  # guarantee sufficient # of samples to represent pulse
    fs = 1 / dt
    times = np.arange(int(fs * (T + 1 / df_pvss.index[0]))) / fs
    pulse = np.zeros_like(times)
    pulse[: int(T * fs)] = ampl * np.sin((np.pi / T) * times[: int(T * fs)])
    pulse_pvss = shock.pseudo_velocity(
        pd.DataFrame(pulse, index=times), freqs=df_pvss.index, damp=damp
    )

    # This is an approximation -> give the result a fudge-factor for correctness
    assert (df_pvss / pulse_pvss).max().max() < 1.2
