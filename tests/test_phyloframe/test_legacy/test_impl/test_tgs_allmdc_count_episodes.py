import pandas as pd

from phyloframe.legacy._impl import tgs_allmdc_count_episodes


def test_count():
    exposure_df = pd.DataFrame(
        {
            "origin_time": 0.0,
            "branch_length": 1.0,
            "survival_time": 0.0,
            "clade_size": [1, 1, 2, 1, 5],
            "trait0_starts_absent": True,
            "trait0_gain": [False, True, True, False, False],
            "trait1_starts_absent": [True, False, True, False, False],
            "trait1_gain": False,
        },
    )
    res = tgs_allmdc_count_episodes(
        trait_names=["a", "b"], exposure_df=exposure_df
    )
    assert res.columns.tolist() == [
        "trait",
        "num_leaves",
        "num_trait_present",
        "num_clades",
        "num_clade_gains",
    ]
    assert res["num_leaves"].tolist() == [3, 1]
    assert res["num_trait_present"].tolist() == [1, 0]
    assert res["num_clades"].tolist() == [2, 1]
    assert res["num_clade_gains"].tolist() == [1, 0]
