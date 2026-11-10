import numpy as np
import polars as pl
import pytest

from phyloframe.legacy._impl._tgs_fit_traits import _tgs_coarsegrain_episodes


def _exposure_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "bin_id": [0, 0, 0, 1],
            "episode_length": [1.0002, 1.0004, 2.0, 2.0],
            "survival_duration": [0.0, 0.0, 0.0, 5.0],
            "survival_duration_id": [0, 0, 0, 1],
            "trait0_starts_absent": [True, True, True, True],
            "trait0_gain": [False, True, True, False],
            "trait1_starts_absent": [False, True, True, False],
            "trait1_gain": [False, False, True, False],
        },
    )


def _coarsegrain(trait: int, weights=np.ones(4), **kwargs):
    return _tgs_coarsegrain_episodes(
        _exposure_df().with_columns(weight=weights),
        **{
            **dict(
                trait_id=trait,
                calibration_factors=np.ones(2),
                length_rtol=1e-3,
                num_survival_durations=2,
            ),
            **kwargs,
        },
    )


def test_merges_within_tolerance():
    # trait 0: lengths ~1 merge; length 2 splits by survival duration
    coarsened_df, shares = _coarsegrain(0)
    assert coarsened_df.columns == [
        "num_trials",
        "num_successes",
        "mean_episode_length",
    ]
    np.testing.assert_array_equal(coarsened_df["num_trials"], [2, 1, 1])
    np.testing.assert_array_equal(coarsened_df["num_successes"], [1, 1, 0])
    np.testing.assert_allclose(
        coarsened_df["mean_episode_length"], [1.0003, 2.0, 2.0]
    )
    np.testing.assert_array_equal(shares.toarray(), [[1, 0], [1, 0], [0, 1]])
    # trait 1: only branches 1 and 2 start without it
    coarsened_df, shares = _coarsegrain(1)
    np.testing.assert_array_equal(coarsened_df["num_trials"], [1, 1])
    np.testing.assert_array_equal(coarsened_df["num_successes"], [0, 1])
    np.testing.assert_allclose(
        coarsened_df["mean_episode_length"], [1.0004, 2.0]
    )
    np.testing.assert_array_equal(shares.toarray(), [[1, 0], [1, 0]])


def test_no_merging():
    coarsened_df, __ = _coarsegrain(0, length_rtol=0.0)
    assert len(coarsened_df) == 4


def test_calibration_factors_scale_lengths():
    coarsened_df, __ = _coarsegrain(
        0, calibration_factors=np.array([1.0, 3.0])
    )
    assert coarsened_df["mean_episode_length"].to_list() == pytest.approx(
        [1.0003, 2.0, 6.0]
    )


def test_zero_weight_dropped():
    coarsened_df, shares = _coarsegrain(
        0, weights=np.array([1.0, 1.0, 0.0, 1.0])
    )
    # episode 2 has no weight left
    assert coarsened_df["num_trials"].to_list() == [2, 1]
    assert shares.shape == (2, 2)


def test_no_episodes():
    # no branch starts without the trait
    coarsened_df, shares = _coarsegrain(
        1, weights=np.array([1.0, 0.0, 0.0, 1.0])
    )
    assert coarsened_df.is_empty()
    assert shares.shape == (0, 2)
