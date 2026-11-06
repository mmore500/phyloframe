import numpy as np
import polars as pl
from scipy import sparse

from phyloframe.legacy._impl import tgs_calc_calibration_factors


def _counts(num_successes) -> pl.DataFrame:
    num_successes = np.asarray(num_successes, dtype=float)
    num_traits, num_bins = num_successes.shape
    return pl.DataFrame(
        {
            "trait": np.repeat(np.arange(num_traits), num_bins),
            "bin": np.tile(np.arange(num_bins), num_traits),
            "num_trials": 100.0,
            "num_successes": num_successes.ravel(),
            "length_sum": 1000.0,
        },
    )


def test_rate_ratios_of_other_neutral_traits():
    # neutral traits 1 and 2 gain twice as often per unit length in bin 1
    counts = _counts([[5, 5], [10, 20], [10, 20]])
    kwargs = dict(
        trait_bin_counts_df=counts,
        neutral_traits=[1, 2],
        calibration_neighborhoods=sparse.identity(2, format="csr"),
        max_rate_ratio=np.inf,
    )
    for trait in range(3):
        res = tgs_calc_calibration_factors(trait=trait, **kwargs)
        np.testing.assert_allclose(res, [2 / 3, 4 / 3])


def test_one_neighborhood_is_uncalibrated():
    counts = _counts([[5, 5], [10, 20], [10, 20]])
    res = tgs_calc_calibration_factors(
        trait_bin_counts_df=counts,
        trait=0,
        neutral_traits=[1, 2],
        calibration_neighborhoods=sparse.csr_matrix(np.ones((2, 2))),
        max_rate_ratio=np.inf,
    )
    np.testing.assert_allclose(res, [1.0, 1.0])


def test_max_rate_ratio_caps():
    counts = _counts([[5, 5], [1, 99], [1, 99]])
    res = tgs_calc_calibration_factors(
        trait_bin_counts_df=counts,
        trait=0,
        neutral_traits=[1, 2],
        calibration_neighborhoods=sparse.identity(2, format="csr"),
        max_rate_ratio=1.5,
    )
    assert res.max() == 1.5
