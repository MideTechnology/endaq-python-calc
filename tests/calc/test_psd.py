import timeit

import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np
import numpy as np

from endaq.calc import psd


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
    """
    Check that a situation exists where the histogram method is more
    performant.
    """
    setup = """
from endaq.calc import psd
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
