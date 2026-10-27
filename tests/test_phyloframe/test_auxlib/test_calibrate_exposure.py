import numpy as np
import pytest

from phyloframe._auxlib._calibrate_exposure import calibrate_exposure


def test_uniform_rate():
    exposure = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = calibrate_exposure(0.1 * exposure, exposure, np.array([0, 1]))
    np.testing.assert_allclose(res, np.ones((2, 2)))


def test_recovers_distortion():
    true_exposure = np.array([10.0, 10.0, 10.0, 10.0])
    measured = true_exposure * np.array([2.0, 2.0, 0.5, 0.5])
    num_successes = 0.01 * true_exposure
    factor = calibrate_exposure(num_successes, measured, [0, 0, 1, 1])
    corrected = measured * factor
    # corrected exposure is proportional to true exposure
    np.testing.assert_allclose(corrected / corrected[0], np.ones(4))


def test_cells_pool_bins():
    factor = calibrate_exposure([1.0, 3.0, 0.0], [1.0, 1.0, 2.0], [5, 5, 7])
    # cell 5 rate 2.0, cell 7 rate 0.0, overall rate 1.0
    np.testing.assert_allclose(factor, [2.0, 2.0, 0.0])


def test_no_exposure_cell():
    factor = calibrate_exposure([1.0, 0.0], [1.0, 0.0], [0, 1])
    np.testing.assert_allclose(factor, [1.0, 1.0])


def test_no_successes():
    factor = calibrate_exposure([0.0, 0.0], [1.0, 2.0], [0, 1])
    np.testing.assert_allclose(factor, [1.0, 1.0])


def test_preserves_shape():
    num_successes = np.ones((3, 4))
    cell_ids = np.arange(12).reshape(3, 4) % 2
    factor = calibrate_exposure(num_successes, np.ones((3, 4)), cell_ids)
    assert factor.shape == (3, 4)
    assert factor == pytest.approx(np.ones((3, 4)))
