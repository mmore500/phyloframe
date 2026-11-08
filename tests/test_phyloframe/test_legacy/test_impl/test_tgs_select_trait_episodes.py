import polars as pl

from phyloframe.legacy._impl import (
    tgs_count_exposure_traits,
    tgs_select_trait_episodes,
)


def _exposure_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "clade_size": [1, 1, 2],
            "trait0_starts_absent": [True, False, True],
            "trait0_gain": [True, False, False],
            "trait1_starts_absent": [True, True, True],
            "trait1_gain": [False, False, True],
        },
    )


def test_count_exposure_traits():
    assert tgs_count_exposure_traits(_exposure_df()) == 2
    assert tgs_count_exposure_traits(pl.DataFrame({"clade_size": [1]})) == 0


def test_episodes():
    weights = pl.Series([1.0, 2.0, 1.0])
    res = tgs_select_trait_episodes(
        _exposure_df(), trait_id=0, weights=weights
    )
    assert res.schema == pl.Schema(
        {"exposure_id": pl.Int64, "gain": pl.Boolean, "weight": pl.Float64}
    )
    # branches starting without the trait only
    assert res.rows() == [(0, True, 1.0), (2, False, 1.0)]
    res = tgs_select_trait_episodes(
        _exposure_df(), trait_id=1, weights=weights
    )
    assert res.rows() == [(0, False, 1.0), (1, False, 2.0), (2, True, 1.0)]


def test_zero_weight_dropped():
    res = tgs_select_trait_episodes(
        _exposure_df(), trait_id=1, weights=pl.Series([1.0, 0.0, 3.0])
    )
    assert res["exposure_id"].to_list() == [0, 2]


def test_empty():
    exposure_df = pl.DataFrame(
        schema={"trait0_starts_absent": pl.Boolean, "trait0_gain": pl.Boolean},
    )
    assert tgs_select_trait_episodes(
        exposure_df, trait_id=0, weights=pl.Series([], dtype=pl.Float64)
    ).is_empty()
