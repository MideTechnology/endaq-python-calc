import timeit

import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np
import numpy as np

from nre_utils.calc import psd, stats


@pytest.mark.parametrize(
    "array, nfft, dt, axis",
    [
        [np.array([0, 1, 1, 0, 2] * 2000), 1024, None, -1],
        [np.array([0, 1, 1, 0, 2] * 2000), 500, None, -1],
        [np.array([0, 1, 1, 0, 2] * 2000), 123, None, -1],
        [np.array([0, 1, 1, 0, 2] * 2000), 1024, 0.05, -1],
        [np.array([0, 1, 1, 0, 2] * 2000), 500, 0.05, -1],
        [np.array([0, 1, 1, 0, 2] * 2000), 123, 0.05, -1],
        # Check dimension assertions
        [np.zeros(99), 48, None, -1],
        [np.zeros(99), 49, None, -1],
        [np.zeros(99), 50, None, -1],
        [np.zeros(100), 49, None, -1],
        [np.zeros(100), 50, None, -1],
        [np.zeros(100), 51, None, -1],
    ],
)
def test_welch_L2_norm(array, nfft, dt, axis):
    """Test whether PSD function obeys Parseval's Theorem."""
    if dt is not None:
        calc_freq, calc_psd = psd.welch(array, nfft=nfft, dt=dt, axis=axis)

        np.testing.assert_allclose(calc_freq, np.fft.rfftfreq(nfft, d=dt))
    else:
        calc_psd = psd.welch(array, nfft=nfft, dt=dt, axis=axis)
        dt = 1

    np.testing.assert_allclose(
        np.sum(calc_psd, axis=axis) / nfft,
        dt * stats.rms(array, axis=axis) ** 2,
    )


@pytest.mark.parametrize(
    "array, fs, dn, nfft, axis",
    [
        [np.array([0, 1, 1, 0, 2] * 2000), 1, 0, 1024, -1],
        [np.array([0, 1, 1, 0, 2] * 2000), 1, 0, 500, -1],
        [np.array([0, 1, 1, 0, 2] * 2000), 1, 0, 123, -1],
    ],
)
def test_rms_literal(array, fs, dn, nfft, axis):
    """Test `rms_psd` against literal definition of RMS."""
    # RMS-by-PSD automatically removes DC content -> strip first for fair
    # comparison
    array = array - array.mean(axis, keepdims=True)
    np.testing.assert_allclose(
        psd.rms(array, fs, dn, nfft=nfft, axis=axis),
        stats.rms(array, axis),
        rtol=1e-4,
    )


@hyp.given(
    psd_array=hyp_np.arrays(
        dtype=np.float64,
        shape=(20, 3),
        elements=hyp_st.floats(
            0,
            1e20,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
    freq_diffs=hyp_np.arrays(
        dtype=np.float64,
        shape=(8,),
        elements=hyp_st.floats(
            0,
            20,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=True,
        ),
    ),
)
@pytest.mark.parametrize(
    "agg1, agg2",
    [
        ("mean", lambda x, axis=-1: np.nan_to_num(np.mean(x, axis=axis))),
        ("sum", np.sum),
    ],
)
def test_to_jagged_modes(psd_array, freq_diffs, agg1, agg2):
    """Test `to_jagged(..., mode='mean')` against the equivalent `mode=np.mean`."""
    axis = 0
    f = np.arange(psd_array.shape[axis]) * 10
    freq_splits = np.cumsum(freq_diffs)
    hyp.assume(np.all(np.diff(freq_splits, prepend=0) > 0))

    result1 = psd.to_jagged(f, psd_array, freq_splits, axis=axis, agg=agg1).values
    result2 = psd.to_jagged(f, psd_array, freq_splits, axis=axis, agg=agg2).values

    np.testing.assert_allclose(
        result1,
        result2,
        atol=np.amin(psd_array * 1e-7),
    )


def test_to_jagged_mode_times():
    setup = """
from nre_utils.calc import psd
import numpy as np

n = 10 ** 4

axis = -1
psd_array = np.random.random((3, n))
f = np.arange(n) / 3
#freq_splits = np.logspace(0, np.log2(n), num=100, base=2)
freq_splits = f[1:-1]
    """

    t_direct = timeit.timeit(
        "psd.to_jagged(f, psd_array, freq_splits, axis=axis, agg=np.sum)",
        setup=setup,
        number=3,
    )
    t_hist = timeit.timeit(
        "psd.to_jagged(f, psd_array, freq_splits, axis=axis, agg='sum')",
        setup=setup,
        number=3,
    )

    print(f"direct form time: {t_direct}")
    print(f"histogram time: {t_hist}")
    assert t_hist < t_direct
