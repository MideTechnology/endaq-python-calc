import numpy as np
import scipy.signal


def highpass(array, fs, cutoff=1.0, half_order=3, axis=-1):
    """Apply a highpass filter to an array."""
    sos_coeffs = scipy.signal.butter(
        N=half_order,
        Wn=cutoff,
        btype="highpass",
        fs=fs,
        output="sos",
    )

    for b, a in zip(*np.split(sos_coeffs, [3], axis=-1)):
        array = scipy.signal.filtfilt(
            b, a, array, axis=axis, method="gust", irlen=5 * 10 ** 4
        )

    return array
