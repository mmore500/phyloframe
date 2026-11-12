import polars as pl

from phyloframe.legacy._impl import tgs_tally_exposure_episodes


def test_count():
    exposure_df = pl.DataFrame(
        {
            "clade_size": [1, 1, 2, 1, 5],
            "trait0_episode_starts_absent": [True] * 5,
            "trait0_episode_ends_absent": [True, False, False, True, True],
            "trait1_episode_starts_absent": [True, False, True, False, False],
            "trait1_episode_ends_absent": [True, False, True, False, False],
        },
    )
    res = tgs_tally_exposure_episodes(exposure_df, trait_names=["a", "b"])
    assert res["trait_name"].to_list() == ["a", "b"]
    assert res.columns == [
        "num_leaf_episodes",
        "num_leaf_gains",
        "num_inner_episodes",
        "num_inner_gains",
        "trait_name",
    ]
    assert res["num_leaf_episodes"].to_list() == [3, 1]
    assert res["num_leaf_gains"].to_list() == [1, 0]
    assert res["num_inner_episodes"].to_list() == [2, 1]
    assert res["num_inner_gains"].to_list() == [1, 0]
