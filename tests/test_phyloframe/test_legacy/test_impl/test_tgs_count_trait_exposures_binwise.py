import numpy as np
import polars as pl

from phyloframe.legacy._impl import tgs_count_trait_exposures_binwise


def _exposure_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "bin_id": [0, 0, 2, 2],
            "episode_duration": [1.0, 2.0, 3.0, 4.0],
            "trait0_episode_starts_absent": [True, True, True, False],
            "trait0_episode_ends_absent": [True, False, True, False],
            "trait1_episode_starts_absent": [True, True, True, True],
            "trait1_episode_ends_absent": [True, True, False, False],
        },
    )


def test_counts():
    res = tgs_count_trait_exposures_binwise(
        _exposure_df(),
        trait_ids=[0, 1],
        weights=np.array([1.0, 2.0, 1.0, 1.0]),
    )
    assert res.select("trait_id", "bin_id").rows() == [
        (t, b) for t in range(2) for b in range(3)
    ]
    np.testing.assert_array_equal(res["num_episodes"], [3, 0, 1, 3, 0, 2])
    np.testing.assert_array_equal(res["num_gains"], [2, 0, 0, 0, 0, 2])
    np.testing.assert_array_equal(
        res["total_episode_duration"], [5, 0, 3, 5, 0, 7]
    )
