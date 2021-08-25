from collections import namedtuple
import timeit

import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np

import numpy as np
import pandas as pd

from endaq.calc import psd


@hyp.given(
    psd_df=hyp_np.arrays(
        dtype=np.float64,
        shape=(20, 3),
        elements=hyp_st.floats(0, 1e20),
    ).map(lambda array: pd.DataFrame(array, index=np.arange(len(array)) * 10)),
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
def test_to_jagged_modes(psd_df, freq_splits, agg1, agg2):
    """Test `to_jagged(..., mode='mean')` against the equivalent `mode=np.mean`."""
    result1 = psd.to_jagged(psd_df, freq_splits, agg=agg1)
    result2 = psd.to_jagged(psd_df, freq_splits, agg=agg2)

    assert np.all(result1.index == result2.index)
    np.testing.assert_allclose(
        result1.to_numpy(),
        result2.to_numpy(),
        atol=psd_df.min().min() * 1e-7,
    )


def test_to_jagged_mode_times():
    """
    Check that a situation exists where the histogram method is more
    performant.
    """
    setup = """
from endaq.calc import psd
import numpy as np
import pandas as pd

n = 10 ** 4

axis = -1
psd_array = np.random.random((3, n))
f = np.arange(n) / 3
psd_df = pd.DataFrame(psd_array.T, index=f)
#freq_splits = np.logspace(0, np.log2(n), num=100, base=2)
freq_splits = f[1:-1]
    """

    t_direct = timeit.timeit(
        "psd.to_jagged(psd_df, freq_splits, agg=np.sum)",
        setup=setup,
        number=3,
    )
    t_hist = timeit.timeit(
        "psd.to_jagged(psd_df, freq_splits, agg='sum')",
        setup=setup,
        number=3,
    )

    print(f"direct form time: {t_direct}")
    print(f"histogram time: {t_hist}")
    assert t_hist < t_direct


_TestStruct = namedtuple("_TestStruct", "psd_df, agg, expt_f, expt_array")


@pytest.mark.parametrize(
    ", ".join(_TestStruct._fields),
    [
        _TestStruct(
            psd_df=pd.DataFrame([1, 0, 0, 0, 0, 0, 0, 0]),
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 0, 0, 0],
        ),
        _TestStruct(
            psd_df=pd.DataFrame([0, 1, 0, 0, 0, 0, 0, 0]),
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[1, 0, 0, 0],
        ),
        _TestStruct(
            psd_df=pd.DataFrame([0, 0, 1, 0, 0, 0, 0, 0]),
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 1, 0, 0],
        ),
        _TestStruct(
            psd_df=pd.DataFrame([0, 0, 0, 1, 1, 1, 0, 0]),
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 0, 3, 0],
        ),
        _TestStruct(
            psd_df=pd.DataFrame([0, 0, 0, 0, 0, 0, 1, 1]),
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 0, 0, 2],
        ),
    ],
)
def test_to_octave(psd_df, agg, expt_f, expt_array):
    calc_df = psd.to_octave(psd_df, fstart=1, octave_bins=1, agg=agg)
    assert calc_df.index.to_numpy().tolist() == expt_f
    assert calc_df.to_numpy().flatten().tolist() == expt_array
