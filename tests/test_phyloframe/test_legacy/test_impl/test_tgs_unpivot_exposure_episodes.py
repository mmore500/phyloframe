import polars as pl

from phyloframe.legacy._impl import (
    tgs_count_exposure_traits,
    tgs_unpivot_exposure_episodes,
)


def test_episodes():
    exposure_df = pl.DataFrame(
        {
            "clade_size": [1, 1, 2],
            "trait0_starts_absent": [True, False, True],
            "trait0_gain": [True, False, False],
            "trait1_starts_absent": [True, True, True],
            "trait1_gain": [False, False, True],
        },
    )
    assert tgs_count_exposure_traits(exposure_df) == 2
    res = tgs_unpivot_exposure_episodes(exposure_df)
    assert res.schema == pl.Schema(
        {"trait": pl.Int64, "branch": pl.Int64, "gain": pl.Boolean}
    )
    # trait-major, branches starting without each trait only
    assert res.rows() == [
        (0, 0, True),
        (0, 2, False),
        (1, 0, False),
        (1, 1, False),
        (1, 2, True),
    ]


def test_empty():
    exposure_df = pl.DataFrame(
        schema={"trait0_starts_absent": pl.Boolean, "trait0_gain": pl.Boolean},
    )
    assert tgs_count_exposure_traits(exposure_df) == 1
    assert tgs_unpivot_exposure_episodes(exposure_df).is_empty()


def test_no_traits():
    res = tgs_unpivot_exposure_episodes(pl.DataFrame({"clade_size": [1, 2]}))
    assert res.columns == ["trait", "branch", "gain"]
    assert res.is_empty()
