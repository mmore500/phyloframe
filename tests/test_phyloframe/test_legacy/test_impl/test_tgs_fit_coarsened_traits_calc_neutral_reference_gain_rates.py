import numpy as np
import pytest

from phyloframe.legacy._impl._tgs_fit_coarsened_traits import (
    _tgs_calc_neutral_reference_gain_rates,
)


def test_leave_one_out():
    res = _tgs_calc_neutral_reference_gain_rates(
        gain_rates=np.array([4.0, 1.0, 3.0]), neutral_traits=[1, 2]
    )
    # trait 0's reference is mean(1, 3); trait 1's, 3; trait 2's, 1
    np.testing.assert_allclose(res, [2.0, 3.0, 1.0])


def test_undefined():
    res = _tgs_calc_neutral_reference_gain_rates(
        gain_rates=np.array([4.0, 0.0, 0.0]), neutral_traits=[1, 2]
    )
    # no trait's reference gains
    assert np.isnan(res).all()


@pytest.mark.parametrize("neutral_traits", [[], [1], [1, 1]])
def test_too_few_neutral_traits_raises(neutral_traits):
    with pytest.raises(ValueError, match="at least two"):
        _tgs_calc_neutral_reference_gain_rates(
            gain_rates=np.array([4.0, 1.0]), neutral_traits=neutral_traits
        )
