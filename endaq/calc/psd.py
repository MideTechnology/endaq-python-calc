from collections import namedtuple
import warnings

import numpy as np
import scipy.signal


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

    return f, psd * factor


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

    return namedtuple("JaggedPSDResults", "center_freqs, values")(f, psd_jagged)


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

    return namedtuple("OctavePSDResults", "center_freqs, values")(
        center_freqs,
        to_jagged(f, psd, freq_splits=freq_splits, **kwargs).values,
    )
