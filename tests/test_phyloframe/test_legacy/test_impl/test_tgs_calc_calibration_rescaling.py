import numpy as np
import polars as pl
from scipy import sparse

from phyloframe.legacy._impl import tgs_calc_calibration_rescaling


def _counts(num_successes) -> pl.DataFrame:
    num_successes = np.asarray(num_successes, dtype=float)
    num_traits, num_bins = num_successes.shape
    return pl.DataFrame(
        {
            "trait_id": np.repeat(np.arange(num_traits), num_bins),
            "bin_id": np.tile(np.arange(num_bins), num_traits),
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
        neutral_trait_ids=[1, 2],
        calibration_neighborhoods=sparse.identity(2, format="csr"),
        max_rate_ratio=np.inf,
    )
    # (observed + 1) / (expected + 1), from pseudocount 1; trait 0's
    # reference is both neutral traits, each neutral trait's the other
    for trait, expected in [
        (0, [21 / 31, 41 / 31]),
        (1, [11 / 16, 21 / 16]),
        (2, [11 / 16, 21 / 16]),
    ]:
        res = tgs_calc_calibration_rescaling(trait_id=trait, **kwargs)
        np.testing.assert_allclose(res, expected)


def test_one_neighborhood_is_uncalibrated():
    counts = _counts([[5, 5], [10, 20], [10, 20]])
    res = tgs_calc_calibration_rescaling(
        trait_bin_counts_df=counts,
        trait_id=0,
        neutral_trait_ids=[1, 2],
        calibration_neighborhoods=sparse.csr_matrix(np.ones((2, 2))),
        max_rate_ratio=np.inf,
    )
    np.testing.assert_allclose(res, [1.0, 1.0])


def test_max_rate_ratio_caps():
    counts = _counts([[5, 5], [1, 99], [1, 99]])
    res = tgs_calc_calibration_rescaling(
        trait_bin_counts_df=counts,
        trait_id=0,
        neutral_trait_ids=[1, 2],
        calibration_neighborhoods=sparse.identity(2, format="csr"),
        max_rate_ratio=1.5,
    )
    assert res.max() == 1.5


def test_neighborhood_without_reference_gains_shrunk():
    # bin 1's neighborhood has no gains of neutral traits 1 and 2, so the
    # pseudocount keeps its factor positive
    counts = _counts([[5, 5], [10, 0], [10, 0]])
    res = tgs_calc_calibration_rescaling(
        trait_bin_counts_df=counts,
        trait_id=0,
        neutral_trait_ids=[1, 2],
        calibration_neighborhoods=sparse.identity(2, format="csr"),
        max_rate_ratio=np.inf,
    )
    np.testing.assert_allclose(res, [21 / 11, 1 / 11])
