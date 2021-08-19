import pytest
import numpy as np
import sympy as sp

from endaq.calc import integrate


def test_integrate():
    """Test `_integrate` via differentiation."""
    n = 20
    array = np.array([sp.symbols(f"x:{n}"), sp.symbols(f"y:{n}")])
    dt = sp.symbols("dt")

    # Ensure derivative looks correct
    calc_result = integrate._integrate(array, dt, axis=-1)
    expt_result_diff = 0.5 * dt * (array[..., :-1] + array[..., 1:])
    assert np.all(np.diff(calc_result, axis=-1) - expt_result_diff == 0)

    # Ensure offset results in zero-mean data
    # Note: symbols cannot be directly tested, since scalar factors are floats
    assert calc_result.mean().subs(
        [(dt, 1)] + list(zip(array.flatten(), np.ones(array.size)))
    ) == pytest.approx(0)


def test_integrals():
    n = 20
    array = np.array([sp.symbols(f"x:{n}"), sp.symbols(f"y:{n}")])
    dt = sp.symbols("dt")

    calc_result = integrate.integrals(array, dt, n=2, axis=-1)

    assert len(calc_result) == 3
    for dx_dt, x in zip(calc_result[:-1], calc_result[1:]):
        assert np.all(x == integrate._integrate(dx_dt, dt, axis=-1))
