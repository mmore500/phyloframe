import numpy as np

from phyloframe.legacy._impl import tgs_calc_neutral_reference_gain_rates


def test_leave_one_out():
    res = tgs_calc_neutral_reference_gain_rates(
        gain_rates=np.array([4.0, 1.0, 3.0]), neutral_traits=[1, 2]
    )
    # trait 0's reference is mean(1, 3); trait 1's, 3; trait 2's, 1
    np.testing.assert_allclose(res, [2.0, 3.0, 1.0])


def test_undefined():
    res = tgs_calc_neutral_reference_gain_rates(
        gain_rates=np.array([4.0, 0.0]), neutral_traits=[1]
    )
    # trait 0's reference never gains; trait 1 has no reference
    assert np.isnan(res).all()
    res = tgs_calc_neutral_reference_gain_rates(
        gain_rates=np.array([4.0]), neutral_traits=[]
    )
    assert np.isnan(res).all()
