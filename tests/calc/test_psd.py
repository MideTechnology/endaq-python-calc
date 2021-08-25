from collections import namedtuple
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
        elements=hyp_st.floats(0, 1e20),
    ),
    freq_splits=hyp_np.arrays(
        dtype=np.float64,
        shape=(8,),
        elements=hyp_st.floats(0, 200, exclude_min=True),
        unique=True,
    ).map(lambda array: np.sort(array)),
)
@pytest.mark.parametrize(
    "agg1, agg2",
    [
        ("mean", lambda x, axis=-1: np.nan_to_num(np.mean(x, axis=axis))),
        ("sum", np.sum),
    ],
)
def test_to_jagged_modes(psd_array, freq_splits, agg1, agg2):
    """Test `to_jagged(..., mode='mean')` against the equivalent `mode=np.mean`."""
    axis = 0
    f = np.arange(psd_array.shape[axis]) * 10

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


TestStruct = namedtuple("TestStruct", "f, array, agg, expt_f, expt_array")


@pytest.mark.parametrize(
    ", ".join(TestStruct._fields),
    [
        TestStruct(
            f=list(range(8)),
            array=[1, 0, 0, 0, 0, 0, 0, 0],
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 0, 0, 0],
        ),
        TestStruct(
            f=list(range(8)),
            array=[0, 1, 0, 0, 0, 0, 0, 0],
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[1, 0, 0, 0],
        ),
        TestStruct(
            f=list(range(8)),
            array=[0, 0, 1, 0, 0, 0, 0, 0],
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 1, 0, 0],
        ),
        TestStruct(
            f=list(range(8)),
            array=[0, 0, 0, 1, 1, 1, 0, 0],
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 0, 3, 0],
        ),
        TestStruct(
            f=list(range(8)),
            array=[0, 0, 0, 0, 0, 0, 1, 1],
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 0, 0, 2],
        ),
    ],
)
def test_to_octave(f, array, agg, expt_f, expt_array):
    calc_f, calc_array = psd.to_octave(f, array, fstart=1, octave_bins=1, agg=agg)
    assert calc_f.tolist() == expt_f
    assert calc_array.tolist() == expt_array
