import polars as pl

from phyloframe.legacy._impl import tgs_count_episodes


def test_count():
    exposure_df = pl.DataFrame(
        {
            "clade_size": [1, 1, 2, 1, 5],
            "trait0_starts_absent": [True] * 5,
            "trait0_gain": [False, True, True, False, False],
            "trait1_starts_absent": [True, False, True, False, False],
            "trait1_gain": [False] * 5,
        },
    )
    res = tgs_count_episodes(trait_names=["a", "b"], exposure_df=exposure_df)
    assert res.columns == [
        "trait",
        "num_leaves",
        "num_trait_present",
        "num_inner_branches",
        "num_inner_gains",
    ]
    assert res["num_leaves"].to_list() == [3, 1]
    assert res["num_trait_present"].to_list() == [1, 0]
    assert res["num_inner_branches"].to_list() == [2, 1]
    assert res["num_inner_gains"].to_list() == [1, 0]


def test_no_traits():
    res = tgs_count_episodes(
        trait_names=[], exposure_df=pl.DataFrame({"clade_size": [1, 2]})
    )
    assert res.columns[0] == "trait"
    assert res.is_empty()
