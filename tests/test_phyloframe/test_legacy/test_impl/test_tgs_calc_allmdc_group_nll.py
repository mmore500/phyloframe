import numpy as np
import pytest

from phyloframe._auxlib._tgs_fit import _tgs_calc_nll
from phyloframe.legacy._impl import tgs_calc_allmdc_group_nll

GROUP_DATA = (
    np.array([100.0, 200.0, 50.0]),
    np.array([1.0, 7.0, 4.0]),
    np.array([2.0, 10.0, 40.0]),
)


def test_matches_pendant_without_shift():
    res = tgs_calc_allmdc_group_nll(
        log_gain_rate=np.log(1e-3),
        rate=-0.02,
        shift=0.0,
        group_data=GROUP_DATA,
        max_growth_exponent=600.0,
    )
    expected = _tgs_calc_nll(
        *GROUP_DATA, 1e-3, -0.02, max_growth_exponent=600.0
    )
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("shift", [0.0, 5.0, 30.0])
def test_gradient(shift: float):
    def nll(params):
        return tgs_calc_allmdc_group_nll(
            log_gain_rate=params[0],
            rate=params[1],
            shift=shift,
            group_data=GROUP_DATA,
            max_growth_exponent=600.0,
        )

    params = np.array([np.log(1e-3), -0.02])
    __, dlog_gain, drate = nll(params)
    eps = 1e-6
    numeric = [
        (nll(params + step)[0] - nll(params - step)[0]) / (2 * eps)
        for step in (np.array([eps, 0]), np.array([0, eps]))
    ]
    np.testing.assert_allclose([dlog_gain, drate], numeric, rtol=1e-4)


def test_shift_lowers_gain_under_cost():
    kwargs = dict(
        log_gain_rate=np.log(1e-3),
        rate=-0.05,
        max_growth_exponent=600.0,
        group_data=(np.array([1e6]), np.array([0.0]), np.array([10.0])),
    )
    # with no gains observed, a larger shift (lower gain probability)
    # explains the data better
    assert (
        tgs_calc_allmdc_group_nll(shift=20.0, **kwargs)[0]
        < tgs_calc_allmdc_group_nll(shift=0.0, **kwargs)[0]
    )
