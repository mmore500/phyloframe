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
    res = tgs_select_trait_episodes(_exposure_df(), trait=0)
    assert res.schema == pl.Schema(
        {"exposure_row": pl.Int64, "gain": pl.Boolean}
    )
    # branches starting without the trait only
    assert res.rows() == [(0, True), (2, False)]
    res = tgs_select_trait_episodes(_exposure_df(), trait=1)
    assert res.rows() == [(0, False), (1, False), (2, True)]


def test_empty():
    exposure_df = pl.DataFrame(
        schema={"trait0_starts_absent": pl.Boolean, "trait0_gain": pl.Boolean},
    )
    assert tgs_select_trait_episodes(exposure_df, trait=0).is_empty()
