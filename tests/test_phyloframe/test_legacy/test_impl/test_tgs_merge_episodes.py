import numpy as np
import polars as pl
import pytest

from phyloframe.legacy._impl import tgs_merge_episodes


def _exposure_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "branch_length": [1.0, 1.0001, 2.0, 2.0],
            "survival_duration": [0.0, 0.0, 0.0, 5.0],
            "trait0_starts_absent": [True, True, True, True],
            "trait0_gain": [False, True, True, False],
            "trait1_starts_absent": [False, True, True, False],
            "trait1_gain": [False, False, True, False],
        },
    )


def _merge(**kwargs):
    distinct, indices = np.unique([0.0, 0.0, 0.0, 5.0], return_inverse=True)
    return tgs_merge_episodes(
        **{
            **dict(
                exposure_df=_exposure_df(),
                bins=np.array([0, 0, 0, 1]),
                calibration_factors=np.ones((2, 2)),
                weights=np.ones(4),
                length_rtol=1e-3,
                survival_duration_indices=indices,
                num_survival_durations=len(distinct),
            ),
            **kwargs,
        },
    )


def test_merges_within_tolerance():
    partition_df, shares = _merge()
    # trait 0: lengths ~1 merge; length 2 splits by survival duration
    assert partition_df["trait"].to_list() == [0, 0, 0, 1, 1]
    np.testing.assert_array_equal(partition_df["num_trials"], [2, 1, 1, 1, 1])
    np.testing.assert_array_equal(
        partition_df["num_successes"], [1, 1, 0, 0, 1]
    )
    np.testing.assert_allclose(
        partition_df["length"], [1.00005, 2.0, 2.0, 1.0001, 2.0]
    )
    np.testing.assert_array_equal(
        shares.toarray(), [[1, 0], [1, 0], [0, 1], [1, 0], [1, 0]]
    )


def test_no_merging():
    partition_df, __ = _merge(length_rtol=0.0)
    assert len(partition_df) == 4 + 2


def test_calibration_factors_scale_lengths():
    partition_df, __ = _merge(calibration_factors=np.array([[1, 3], [1, 1]]))
    assert partition_df["length"].to_list()[:3] == pytest.approx(
        [1.00005, 2.0, 6.0]
    )


def test_zero_weight_dropped():
    partition_df, shares = _merge(weights=np.array([1.0, 1.0, 0.0, 1.0]))
    # episode 2's rows, one per trait, have no weight left
    assert partition_df["trait"].to_list() == [0, 0, 1]
    assert partition_df["num_trials"].to_list() == [2, 1, 1]
    assert shares.shape == (3, 2)
