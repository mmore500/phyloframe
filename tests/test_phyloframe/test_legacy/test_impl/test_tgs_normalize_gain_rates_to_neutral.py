import numpy as np

from phyloframe.legacy._impl import tgs_normalize_gain_rates_to_neutral


def test_leave_one_out():
    res = tgs_normalize_gain_rates_to_neutral(
        gain_rates=np.array([4.0, 1.0, 3.0]), neutral_traits=[1, 2]
    )
    # trait 0 against mean(1, 3) = 2; trait 1 against 3; trait 2 against 1
    np.testing.assert_allclose(res, [2.0, 1 / 3, 3.0])


def test_undefined():
    res = tgs_normalize_gain_rates_to_neutral(
        gain_rates=np.array([4.0, 0.0]), neutral_traits=[1]
    )
    assert np.isnan(res).all()
    res = tgs_normalize_gain_rates_to_neutral(
        gain_rates=np.array([4.0]), neutral_traits=[]
    )
    assert np.isnan(res).all()
