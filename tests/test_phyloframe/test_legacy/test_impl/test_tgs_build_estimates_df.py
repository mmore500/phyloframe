import polars as pl

from phyloframe.legacy._impl import tgs_build_estimates_df


def _episode_tallies() -> pl.DataFrame:
    return pl.DataFrame({"trait": ["a", "b"], "num_leaf_episodes": [10, 20]})


def _point() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "gain_rate": [1e-3, 2e-3],
            "selection_coef": [-0.05, 0.0],
            "selection_coef_lrt_naive": [9.0, 0.1],
            "gain_ratio": [0.5, 2.0],
            "gain_ratio_lrt_naive": [3.0, 4.0],
        },
    )


_COLUMNS = [
    "trait",
    "num_leaf_episodes",
    "gain_rate",
    "selection_coef",
    "gain_ratio",
    "selection_coef_lrt_naive",
    "gain_ratio_lrt_naive",
]


def test_joins_tallies_and_fits():
    out = tgs_build_estimates_df(_episode_tallies(), fits_df=_point())
    assert out.columns == _COLUMNS
    assert out["trait"].to_list() == ["a", "b"]
    assert out["gain_ratio"].to_list() == [0.5, 2.0]
