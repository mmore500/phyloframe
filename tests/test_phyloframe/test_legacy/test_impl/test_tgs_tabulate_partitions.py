import numpy as np
import pytest
from scipy import sparse

from phyloframe.legacy._impl import TgsCounts, tgs_tabulate_partitions


def _counts():
    return TgsCounts(
        num_leaves=np.array([[2.0, 0.0], [4.0, 1.0]]),
        num_carriers=np.array([[1.0, 0.0], [2.0, 1.0]]),
        pendant_length_sums=np.array([[6.0, 0.0], [8.0, 5.0]]),
    )


def test_uncalibrated():
    res = tgs_tabulate_partitions(
        counts=_counts(),
        neutral_traits=[],
        calibration_neighborhoods=None,
        max_rate_ratio=np.inf,
    )
    assert res["trait"].tolist() == [0, 0, 1, 1]
    assert list(res["trait"].cat.categories) == [0, 1]
    np.testing.assert_array_equal(res["num_trials"], [2, 0, 4, 1])
    np.testing.assert_array_equal(res["num_successes"], [1, 0, 2, 1])
    # mean length, zero without branches
    np.testing.assert_allclose(res["length"], [3.0, 0.0, 2.0, 5.0])


def test_calibrated_with_uniform_rates_unchanged():
    counts = _counts()
    # neutral trait 1 gains in proportion to exposure everywhere
    counts = counts._replace(
        num_carriers=counts.num_carriers * [[1.0], [0.0]]
        + [[0.0], [1.0]] * counts.pendant_length_sums / 10
    )
    # trait 1, the only neutral trait, has no reference, so no exposure
    with pytest.warns(UserWarning, match="no exposure"):
        res = tgs_tabulate_partitions(
            counts=counts,
            neutral_traits=[1],
            calibration_neighborhoods=sparse.identity(2, format="csr"),
            max_rate_ratio=np.inf,
        )
    np.testing.assert_allclose(res["length"][:2], [3.0, 0.0])
