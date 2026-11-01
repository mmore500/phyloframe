import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy._impl import tgs_prepare_asexual, tgs_prepare_polars

from .test_tgs_prepare_asexual import _make_tree


@pytest.mark.parametrize("lazy", [False, True])
def test_matches_asexual(lazy: bool):
    phylogeny_df = _make_tree().assign(u=[1, 0, 0, 0])
    df_pl = pl.from_pandas(phylogeny_df)
    res = tgs_prepare_polars(
        df_pl.lazy() if lazy else df_pl,
        trait_transitions=["t", "u"],
        caller="test",
    )
    expected = tgs_prepare_asexual(
        phylogeny_df,
        False,
        trait_transitions=["t", "u"],
    )
    pd.testing.assert_frame_equal(res, expected)


def test_empty():
    df_pl = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
            "origin_time": pl.Series([], dtype=pl.Float64),
            "t": pl.Series([], dtype=pl.Int8),
        },
    )
    res = tgs_prepare_polars(df_pl, trait_transitions=["t"], caller="test")
    assert res.empty
    assert {"branch_length", "trait0_starts_absent", "trait0_gain"} <= set(res)


def test_null_transitions_raise():
    df_pl = pl.from_pandas(_make_tree()).with_columns(
        t=pl.Series([None, 1, 0, 0], dtype=pl.Int8)
    )
    with pytest.raises(NotImplementedError):
        tgs_prepare_polars(df_pl, trait_transitions=["t"], caller="test")


def test_missing_column_raises():
    with pytest.raises(ValueError):
        tgs_prepare_polars(
            pl.from_pandas(_make_tree()),
            trait_transitions=["x"],
            caller="test",
        )
