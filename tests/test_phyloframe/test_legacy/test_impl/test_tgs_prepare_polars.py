import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy._impl import tgs_prepare_asexual, tgs_prepare_polars

from .test_tgs_prepare_asexual import _make_tree


@pytest.mark.parametrize("lazy", [False, True])
def test_matches_asexual(lazy: bool):
    phylogeny_df = _make_tree()
    df_pl = pl.from_pandas(phylogeny_df).with_columns(
        t=pl.Series([False, True, False, True])
    )
    res = tgs_prepare_polars(
        df_pl.lazy() if lazy else df_pl,
        trait_masks=["t", ~pl.col("t")],
        trait_names=["t", "not_t"],
        caller="test",
    )
    expected = tgs_prepare_asexual(
        phylogeny_df,
        False,
        trait_masks=[
            np.array([False, True, False, True]),
            np.array([True, False, True, False]),
        ],
        trait_names=["t", "not_t"],
    )
    for column in (
        "ancestor_id",
        "origin_time",
        "__alifestd_tgs_branch_length",
        "__alifestd_tgs_trait0",
        "__alifestd_tgs_trait1",
    ):
        assert res[column].to_list() == expected[column].tolist()


def test_empty():
    df_pl = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
            "origin_time": pl.Series([], dtype=pl.Float64),
            "t": pl.Series([], dtype=pl.Boolean),
        },
    )
    res = tgs_prepare_polars(
        df_pl, trait_masks=["t"], trait_names=["t"], caller="test"
    )
    assert res.is_empty()
    assert "__alifestd_tgs_branch_length" in res.columns


def test_invalid():
    df_pl = pl.from_pandas(_make_tree()).with_columns(
        t=pl.Series([None, True, False, True], dtype=pl.Boolean)
    )
    with pytest.raises(NotImplementedError):
        tgs_prepare_polars(
            df_pl, trait_masks=["t"], trait_names=["t"], caller="test"
        )
    sexual = pd.DataFrame(
        {"id": [0, 1], "ancestor_list": ["[none]", "[0]"], "t": [0, 1]}
    ).assign(ancestor_list=["[none]", "[0, 1]"])
    with pytest.raises(NotImplementedError):
        tgs_prepare_polars(
            pl.from_pandas(sexual).with_columns(origin_time=pl.lit(0.0)),
            trait_masks=["t"],
            trait_names=["t"],
            caller="test",
        )
