from collections import namedtuple
import warnings

import numpy as np


SpectrumStruct = namedtuple("SpectrumStruct", "freqs, values")


def _np_histogram_nd(array, bins=10, weights=None, axis=-1, **kwargs):
    """Compute histograms over a specified axis."""
    array = np.asarray(array)
    bins = np.asarray(bins)
    weights = np.asarray(weights) if weights is not None else None

    # Collect all ND inputs
    nd_params = {}
    (nd_params if array.ndim > 1 else kwargs)["a"] = array
    (nd_params if bins.ndim > 1 else kwargs)["bins"] = bins
    if weights is not None:
        (nd_params if weights.ndim > 1 else kwargs)["weights"] = weights

    if len(nd_params) == 0:
        return np.histogram(**kwargs)[0]

    # Move the desired axes to the back
    for k, v in nd_params.items():
        nd_params[k] = np.moveaxis(v, axis, -1)

    # Broadcast ND arrays to the same shape
    ks, vs = zip(*nd_params.items())
    vs_broadcasted = np.broadcast_arrays(*vs)
    for k, v in zip(ks, vs_broadcasted):
        nd_params[k] = v

    # Generate output
    bins = nd_params.get("bins", bins)

    result_shape = ()
    if len(nd_params) != 0:
        result_shape = v.shape[:-1]
    if bins.ndim >= 1:
        result_shape = result_shape + (bins.shape[-1] - 1,)
    else:
        result_shape = result_shape + (bins,)

    result = np.empty(
        result_shape,
        dtype=(weights.dtype if weights is not None else int),
    )
    loop_shape = v.shape[:-1]

    for nd_i in np.ndindex(*loop_shape):
        nd_kwargs = {k: v[nd_i] for k, v in nd_params.items()}
        result[nd_i], edges = np.histogram(**nd_kwargs, **kwargs)

    result = np.moveaxis(result, -1, axis)
    return result


def differentiate(f, psd, n=1):
    """Perform time-domain differentiation on periodogram data."""
    # Involves a division by zero for n<0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        factor = (2 * np.pi * f) ** (2 * n)  # divide by zero
    if n < 0:
        factor[f == 0] = 0

    return SpectrumStruct(freqs=f, values=psd * factor)


def to_jagged(f, psd, freq_splits, axis=-1, agg="sum"):
    """
    Calculate a periodogram over non-uniformly spaced frequency bins.

    :param f, psd: the returned values from `scipy.signal.welch`
    :param freq_splits: the boundaries of the frequency bins; must be strictly
        increasing
    :param axis: same as the axis parameter provided to `scipy.signal.welch`
    :param agg: the method for aggregating values into bins; 'mean' preserves
        the PSD's area-under-the-curve, 'sum' preserves the PSD's "energy"
    """
    f = np.asarray(f)
    if f.ndim != 1 or not np.all(np.diff(freq_splits, prepend=0) > 0):
        raise ValueError

    # Check that PSD samples do not skip any frequency bins
    spacing_test = np.diff(np.searchsorted(freq_splits, f))
    if np.any(spacing_test > 1):
        warnings.warn(
            "empty frequency bins in re-binned PSD; "
            "original PSD's frequency spacing is too coarse",
            RuntimeWarning,
        )

    if isinstance(agg, str):
        if agg not in ("sum", "mean"):
            raise ValueError(f'invalid aggregation mode "{agg}"')

        # Reshape frequencies for histogram function
        f_ndim = np.broadcast_to(
            f,
            (psd.shape[:axis] + psd.shape[axis + 1 or psd.ndim :] + (psd.shape[axis],)),
        )
        f_ndim = np.moveaxis(f_ndim, -1, axis)

        # Calculate sum via histogram function
        psd_jagged = _np_histogram_nd(f_ndim, bins=freq_splits, weights=psd, axis=axis)

        # Adjust values for mean calculation
        if agg == "mean":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # x/0

                psd_jagged = np.moveaxis(psd_jagged, axis, -1)
                psd_jagged = np.nan_to_num(  # <- fix divisions by zero
                    psd_jagged / np.histogram(f, bins=freq_splits)[0]
                )
                psd_jagged = np.moveaxis(psd_jagged, -1, axis)

    else:
        psd_binned = np.split(psd, np.searchsorted(f, freq_splits), axis=axis)[1:-1]
        psd_jagged = np.stack([agg(a, axis=axis) for a in psd_binned], axis=axis)

    f = (freq_splits[1:] + freq_splits[:-1]) / 2

    return SpectrumStruct(freqs=f, values=psd_jagged)


def to_octave(f, psd, fstart=1, octave_bins=12, **kwargs):
    """Calculate a periodogram over log-spaced frequency bins."""
    max_f = f.max()

    octave_step = 1 / octave_bins
    center_freqs = 2 ** np.arange(
        np.log2(fstart),
        np.log2(max_f) - octave_step / 2,
        octave_step,
    )
    freq_splits = 2 ** np.arange(
        np.log2(fstart) - octave_step / 2,
        np.log2(max_f),
        octave_step,
    )
    assert len(center_freqs) + 1 == len(freq_splits)

    return SpectrumStruct(
        freqs=center_freqs,
        values=to_jagged(f, psd, freq_splits=freq_splits, **kwargs).values,
    )


def vc_curves(f, psd, fstart=1, octave_bins=12, axis=-1):
    """
    Calculate Vibration Criterion (VC) curves from an acceration periodogram.

    # Theory behind the calculation

    Let x(t) be a real-valued time-domain signal, and X(2πf) = F{x(t)}(2πf)
    be the Fourier Transform of that signal. By Parseval's Theorem,

        ∫x(t)^2 dt = ∫|X(2πf)|^2 df

    (see https://en.wikipedia.org/wiki/Parseval%27s_theorem#Notation_used_in_physics)

    Rewriting the right side of that equation in the discrete form becomes

        ∫x(t)^2 dt ≈ ∑ |X[k]|^2 • ∆f

    where ∆f = fs/N = (1/∆t) / N = 1/T.
    Limiting the right side to a range of discrete frequencies (k_0, k_1):

        ∫x(t)^2 dt ≈ [∑; k=k_0 -> k≤k_1] |X[k]|^2 • ∆f

    The VC curve calculation is the RMS over the time-domain. If T is the
    duration of the time-domain signal, then:

        √((1/T) ∫x(t)^2 dt)
            ≈ √((1/T) [∑; k=k_0 -> k≤k_1] |X[k]|^2 • ∆f)
            = ∆f • √([∑; k=k_0 -> k≤k_1] |X[k]|^2)

    If the time-series data is acceleration, then the signal needs to first
    be integrated into velocity. This can be done in the frequency domain
    by replacing |X(2πf)|^2 with (1/2πf)^2 |X(2πf)|^2.
    """
    f, v_psd = differentiate(f, psd, n=-1)
    f_oct, v_psd_oct = to_octave(
        f,
        v_psd,
        fstart=fstart,  # Hz
        octave_bins=octave_bins,
        agg=np.sum,
        axis=axis,
    )

    return SpectrumStruct(
        freqs=f_oct,
        values=np.sqrt(f[1] * v_psd_oct),  # the PSD must already scale by ∆f?
    )
