import pytest
import numpy as np

from endaq.calc import psd, stats


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
