import numpy as np
import polars as pl

from phyloframe.legacy._impl import tgs_count_trait_bin_exposures


def _exposure_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "branch_length": [1.0, 2.0, 3.0, 4.0],
            "trait0_starts_absent": [True, True, True, False],
            "trait0_gain": [False, True, False, False],
            "trait1_starts_absent": [True, True, True, True],
            "trait1_gain": [False, False, True, True],
        },
    )


def test_counts():
    res = tgs_count_trait_bin_exposures(
        exposure_df=_exposure_df(),
        num_traits=2,
        bin_ids=np.array([0, 0, 1, 1]),
        num_bins=3,
        weights=np.array([1.0, 2.0, 1.0, 1.0]),
    )
    assert res.select("trait_id", "bin_id").rows() == [
        (t, b) for t in range(2) for b in range(3)
    ]
    np.testing.assert_array_equal(res["num_trials"], [3, 1, 0, 3, 2, 0])
    np.testing.assert_array_equal(res["num_successes"], [2, 0, 0, 0, 2, 0])
    np.testing.assert_array_equal(res["length_sum"], [5, 3, 0, 5, 7, 0])
