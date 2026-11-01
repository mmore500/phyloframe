import numpy as np
import pytest

from phyloframe.legacy._impl import (
    tgs_calc_allmdc_group_data,
    tgs_fit_allmdc_shift,
)

from ._simulate_allmdc_groups import SHIFT, TRAITS, simulate_allmdc_groups


def test_recovers_shift():
    groups = simulate_allmdc_groups()
    data = tgs_calc_allmdc_group_data(
        groups=groups,
        unit_weights=[np.ones(len(g.partitions)) for g in groups],
        neutral_traits=[1, 2],
        calibrate=False,
        num_traits=3,
    )
    params = np.array([(np.log(gain), sel) for gain, sel in TRAITS])
    shift = tgs_fit_allmdc_shift(
        group_data=[trait_data[1] for trait_data in data],
        params=params,
        loss_rate=0.0,
        max_shift=100.0,
        max_growth_exponent=600.0,
    )
    assert shift == pytest.approx(SHIFT, abs=6.0)


def test_skips_unfit_traits():
    groups = simulate_allmdc_groups((4_000, 2_000))
    data = tgs_calc_allmdc_group_data(
        groups=groups,
        unit_weights=[np.ones(len(g.partitions)) for g in groups],
        neutral_traits=[1, 2],
        calibrate=False,
        num_traits=3,
    )
    params = np.array(
        [(np.log(2e-3), -0.05), (np.nan, np.nan), (np.nan, np.nan)]
    )
    shift = tgs_fit_allmdc_shift(
        group_data=[data[0][1], None, data[2][1]],
        params=params,
        loss_rate=0.0,
        max_shift=100.0,
        max_growth_exponent=600.0,
    )
    assert 0.0 <= shift <= 100.0
