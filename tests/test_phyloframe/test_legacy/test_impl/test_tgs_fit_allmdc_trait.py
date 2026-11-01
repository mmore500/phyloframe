import numpy as np
import pytest

from phyloframe.legacy._impl import (
    tgs_calc_allmdc_group_data,
    tgs_fit_allmdc_trait,
)

from ._simulate_allmdc_groups import SHIFT, simulate_allmdc_groups


@pytest.fixture(scope="module")
def data():
    groups = simulate_allmdc_groups()
    return tgs_calc_allmdc_group_data(
        groups=groups,
        unit_weights=[np.ones(len(g.partitions)) for g in groups],
        neutral_traits=[1, 2],
        calibrate=False,
        num_traits=3,
    )


@pytest.mark.parametrize("start", [None, (np.log(1e-3), -0.01)])
def test_recovers_focal(data, start):
    log_gain_rate, selection, nll, lrt = tgs_fit_allmdc_trait(
        trait_data=data[0],
        shifts=np.array([0.0, SHIFT]),
        loss_rate=0.0,
        start=start,
        selection_starts=(0.0, -0.3, -1.8),
        max_growth_exponent=600.0,
    )
    assert np.exp(log_gain_rate) == pytest.approx(2e-3, rel=0.15)
    assert selection == pytest.approx(-0.05, abs=0.015)
    assert lrt > 10
    assert np.isfinite(nll)


def test_nan_shift_group_ignored(data):
    with_both = tgs_fit_allmdc_trait(
        trait_data=data[1],
        shifts=np.array([0.0, np.nan]),
        loss_rate=0.0,
        start=None,
        selection_starts=(0.0,),
        max_growth_exponent=600.0,
    )
    leaves_only = tgs_fit_allmdc_trait(
        trait_data=[data[1][0], None],
        shifts=np.array([0.0, 0.0]),
        loss_rate=0.0,
        start=None,
        selection_starts=(0.0,),
        max_growth_exponent=600.0,
    )
    np.testing.assert_allclose(with_both, leaves_only)
