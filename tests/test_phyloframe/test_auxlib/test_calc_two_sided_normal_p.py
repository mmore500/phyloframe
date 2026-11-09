import numpy as np

from phyloframe._auxlib._calc_two_sided_normal_p import calc_two_sided_normal_p


def test_values():
    res = calc_two_sided_normal_p(np.array([0.0, 1.959963984540054, -1.96]))
    np.testing.assert_allclose(res, [1.0, 0.05, 0.04999579], rtol=1e-6)


def test_symmetric():
    z = np.linspace(-4, 4, 9)
    np.testing.assert_array_equal(
        calc_two_sided_normal_p(z), calc_two_sided_normal_p(-z)
    )


def test_nan():
    assert np.isnan(calc_two_sided_normal_p(np.array([np.nan]))).all()
